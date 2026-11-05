//! Local browser remote: `GET /` controls page, `GET /api/status`, `POST /api/set`.
//!
//! Each connection carries one request on its own thread (at most [`MAX_CONNECTIONS`]) and must
//! deliver it within [`REQUEST_DEADLINE`], so a stalled or oversized request costs only that
//! connection. The camera backend (not `Send`) and the preview stay on the thread that called
//! [`serve`] and are reached through a channel.

use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use crate::backend::{CameraBackend, Control, ControlKind};
use crate::media::{Preview, VideoConfig};

const MAX_HEAD: usize = 8192;
const MAX_BODY: usize = 4096;
const MAX_VALUE: usize = 64;
const MAX_TOKEN: usize = 128;
const MAX_CONNECTIONS: usize = 32;
const REQUEST_DEADLINE: Duration = Duration::from_secs(10);
const BACKEND_TIMEOUT: Duration = Duration::from_secs(30);
const CSP: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; \
                   connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

struct Config {
    listen: SocketAddr,
    token: String,
}

enum Job {
    Status,
    Set { key: String, value: String },
}

struct Reply {
    code: u16,
    content_type: &'static str,
    body: String,
    allow: Option<&'static str>,
}

type Msg = (Job, Sender<Reply>);

pub fn serve(device: Option<String>, listen: SocketAddr, token: Option<String>) -> Result<()> {
    let token = match token {
        Some(t) => {
            if !valid_token(&t) {
                bail!("token must be 1-{MAX_TOKEN} characters of A-Z a-z 0-9 - . _ ~");
            }
            t
        }
        None if !listen.ip().is_loopback() => {
            bail!("listening on {listen} (not loopback) requires an explicit non-empty --token")
        }
        None => random_token()?,
    };

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }

    let backend = CameraBackend::open(device.as_deref())?;
    let dev = backend.device();
    let listener = TcpListener::bind(listen).map_err(|e| anyhow!("binding {listen}: {e}"))?;
    // Polled, so the acceptor notices `stop` without a wake-up connection.
    listener
        .set_nonblocking(true)
        .with_context(|| format!("configuring {listen}"))?;

    // Some vendor settings only apply while the camera streams, so the remote session owns a preview.
    eprintln!(
        "PRIVACY: starting camera stream on {} ({}) for the whole remote session; \
         anyone holding the token can change camera settings.",
        dev.name, dev.id
    );
    let preview = match Preview::start(
        &dev.id,
        &VideoConfig {
            width: 1280,
            height: 720,
            fps: 30,
        },
    ) {
        Ok(p) => {
            if let Err(e) = p.wait_first_frame(Duration::from_secs(10)) {
                eprintln!("warning: preview produced no frame: {e:#}");
            }
            Some(p)
        }
        Err(e) => {
            eprintln!("warning: preview not started, stream-dependent settings may fail: {e:#}");
            None
        }
    };

    let cfg = Arc::new(Config { listen, token });
    let (tx, rx) = mpsc::channel::<Msg>();
    let acceptor = {
        let (cfg, stop) = (cfg.clone(), stop.clone());
        thread::spawn(move || accept_loop(&listener, &cfg, &tx, &stop))
    };

    if !listen.ip().is_loopback() {
        eprintln!("warning: plain HTTP on {listen}; traffic and token are visible on the network.");
    }
    println!("Remote control: http://{listen}/?token={}", cfg.token);
    println!("Press Ctrl-C to stop.");

    while !stop.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok((job, reply)) => {
                let _ = reply.send(handle(&backend, preview.as_ref(), job));
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    stop.store(true, Ordering::SeqCst);
    let _ = acceptor.join();
    eprintln!("Remote stopped.");
    drop(preview); // stops camera capture
    Ok(())
}

fn accept_loop(listener: &TcpListener, cfg: &Arc<Config>, tx: &Sender<Msg>, stop: &AtomicBool) {
    let active = Arc::new(AtomicUsize::new(0));
    while !stop.load(Ordering::SeqCst) {
        let stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(e) => {
                if e.kind() != io::ErrorKind::WouldBlock {
                    eprintln!("remote: accepting a connection failed: {e}");
                }
                thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        // Over the cap a connection is closed unanswered until a slot frees up.
        if active.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            active.fetch_sub(1, Ordering::SeqCst);
            continue;
        }
        let slot = Slot(active.clone());
        let (cfg, tx) = (cfg.clone(), tx.clone());
        // A failed spawn drops the closure, which closes the stream and frees the slot.
        let _ = thread::Builder::new().spawn(move || {
            let _slot = slot;
            connection(stream, &cfg, &tx, Instant::now() + REQUEST_DEADLINE);
        });
    }
}

/// A counted connection, released even if its thread panics.
struct Slot(Arc<AtomicUsize>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn connection(mut stream: TcpStream, cfg: &Config, tx: &Sender<Msg>, deadline: Instant) {
    // Some systems hand out accepted sockets in the listener's non-blocking mode.
    if stream.set_nonblocking(false).is_err() {
        return;
    }
    let reply = match Request::read(&mut stream, deadline) {
        Ok(mut req) => route(&mut req, cfg, tx),
        Err(reply) => reply,
    };
    respond(&mut stream, &reply);
}

/// One request: the parsed head, plus the stream for a body read only when a route needs it.
struct Request<'a> {
    stream: &'a mut TcpStream,
    deadline: Instant,
    method: String,
    url: String,
    headers: Vec<(String, String)>,
    /// Bytes received after the head: the start of the body.
    rest: Vec<u8>,
}

impl<'a> Request<'a> {
    /// Request line and headers: at most [`MAX_HEAD`] bytes, received before `deadline`.
    fn read(stream: &'a mut TcpStream, deadline: Instant) -> Result<Self, Reply> {
        let mut buf = Vec::new();
        let end = loop {
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break end;
            }
            if buf.len() == MAX_HEAD {
                return Err(error(431, "request header too large"));
            }
            read_some(stream, &mut buf, MAX_HEAD, deadline)?;
        };
        let rest = buf.split_off(end + 4);
        let malformed = || error(400, "malformed request");
        let head = std::str::from_utf8(&buf[..end]).map_err(|_| malformed())?;
        let mut lines = head.split("\r\n");
        let mut start = lines.next().unwrap_or_default().split(' ');
        let (Some(method), Some(url), Some(version), None) =
            (start.next(), start.next(), start.next(), start.next())
        else {
            return Err(malformed());
        };
        if !is_token(method) || !url.starts_with('/') || !matches!(version, "HTTP/1.0" | "HTTP/1.1")
        {
            return Err(malformed());
        }
        let mut headers = Vec::new();
        for line in lines {
            let (name, value) = line.split_once(':').ok_or_else(malformed)?;
            let value = value.trim_matches([' ', '\t']);
            if !is_token(name) || value.bytes().any(|b| b.is_ascii_control() && b != b'\t') {
                return Err(malformed());
            }
            headers.push((name.to_owned(), value.to_owned()));
        }
        Ok(Request {
            stream,
            deadline,
            method: method.to_owned(),
            url: url.to_owned(),
            headers,
            rest,
        })
    }

    /// Exactly `len` body bytes, received before the deadline.
    fn body(&mut self, len: usize) -> Result<Vec<u8>, Reply> {
        let mut body = std::mem::take(&mut self.rest);
        body.truncate(len);
        while body.len() < len {
            read_some(self.stream, &mut body, len, self.deadline)?;
        }
        Ok(body)
    }
}

/// Appends the next bytes that arrive (keeping `buf` within `limit`); fails after `deadline`.
fn read_some(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    limit: usize,
    deadline: Instant,
) -> Result<(), Reply> {
    let late = || error(408, "request not received in time");
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() || stream.set_read_timeout(Some(left)).is_err() {
        return Err(late());
    }
    let mut chunk = [0; 1024];
    let want = chunk.len().min(limit - buf.len());
    match stream.read(&mut chunk[..want]) {
        Ok(0) => Err(error(400, "incomplete request")),
        Ok(n) => {
            buf.extend_from_slice(&chunk[..n]);
            Ok(())
        }
        Err(e) if e.kind() == io::ErrorKind::Interrupted => Ok(()),
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
        {
            Err(late())
        }
        Err(_) => Err(error(400, "incomplete request")),
    }
}

/// RFC 9110 token, as used for methods and header names.
fn is_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b))
}

fn respond(stream: &mut TcpStream, reply: &Reply) {
    let allow = reply
        .allow
        .map(|a| format!("Allow: {a}\r\n"))
        .unwrap_or_default();
    // shortcut: no Date header (every reply is no-store); add one if a cache or proxy ever sits in front.
    let head = format!(
        "HTTP/1.1 {} \r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\
         Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\n\
         X-Frame-Options: DENY\r\nContent-Security-Policy: {CSP}\r\n{allow}\r\n",
        reply.code,
        reply.content_type,
        reply.body.len()
    );
    let sent = stream.set_write_timeout(Some(REQUEST_DEADLINE)).is_ok()
        && stream.write_all(head.as_bytes()).is_ok()
        && stream.write_all(reply.body.as_bytes()).is_ok();
    if sent {
        // Closing with unread request bytes would reset the connection and could drop the reply.
        let _ = stream.shutdown(Shutdown::Write);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
        let _ = io::copy(
            &mut (&*stream).take((MAX_HEAD + MAX_BODY) as u64),
            &mut io::sink(),
        );
    }
}

fn json_reply(code: u16, body: String) -> Reply {
    Reply {
        code,
        content_type: "application/json",
        body,
        allow: None,
    }
}

fn error(code: u16, msg: &str) -> Reply {
    json_reply(code, serde_json::json!({ "error": msg }).to_string())
}

/// Single header value; `Err` when it appears more than once (ambiguous).
fn header<'r>(req: &'r Request<'_>, name: &str) -> Result<Option<&'r str>, ()> {
    let mut found = None;
    for (field, value) in &req.headers {
        if field.eq_ignore_ascii_case(name) {
            if found.is_some() {
                return Err(());
            }
            found = Some(value.as_str());
        }
    }
    Ok(found)
}

fn route(req: &mut Request, cfg: &Config, tx: &Sender<Msg>) -> Reply {
    let Ok(Some(host)) = header(req, "Host").map(|h| h.map(str::to_owned)) else {
        return error(400, "missing or duplicate Host header");
    };
    if !host_ok(&host, cfg.listen) {
        return error(403, "Host not allowed");
    }
    let url = req.url.clone();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (url.as_str(), None),
    };
    let allow = match path {
        "/" | "/api/status" => "GET",
        "/api/set" => "POST",
        _ => return error(404, "not found"),
    };
    if req.method != allow {
        return Reply {
            allow: Some(allow),
            ..error(405, "method not allowed")
        };
    }

    if path == "/" {
        let given = query.and_then(|q| q.strip_prefix("token="));
        if !given.is_some_and(|t| ct_eq(t, &cfg.token)) {
            return error(
                401,
                "open the URL printed at server startup (missing or invalid token)",
            );
        }
        return Reply {
            code: 200,
            content_type: "text/html; charset=utf-8",
            body: PAGE.to_owned(),
            allow: None,
        };
    }
    if query.is_some() {
        return error(400, "query string not accepted");
    }
    let Ok(auth) = header(req, "Authorization") else {
        return error(400, "duplicate Authorization header");
    };
    if !bearer_ok(auth, &cfg.token) {
        return error(401, "missing or invalid bearer token");
    }
    if path == "/api/status" {
        return dispatch(tx, Job::Status);
    }

    // POST /api/set
    let (Ok(origin), Ok(ctype), Ok(clen), Ok(te)) = (
        header(req, "Origin"),
        header(req, "Content-Type"),
        header(req, "Content-Length"),
        header(req, "Transfer-Encoding"),
    ) else {
        return error(400, "duplicate header");
    };
    if !origin_ok(origin, &host) {
        return error(403, "cross-origin request rejected");
    }
    if !ctype.is_some_and(is_json) {
        return error(415, "Content-Type must be application/json");
    }
    let Some(len) = clen
        .filter(|v| te.is_none() && !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|v| v.parse::<usize>().ok())
    else {
        return error(411, "valid Content-Length required");
    };
    if len > MAX_BODY {
        return error(413, "request body too large");
    }
    let body = match req.body(len) {
        Ok(body) => body,
        Err(reply) => return reply,
    };
    match parse_set(&body) {
        Ok((key, value)) => dispatch(tx, Job::Set { key, value }),
        Err(msg) => error(400, &msg),
    }
}

fn dispatch(tx: &Sender<Msg>, job: Job) -> Reply {
    let (rtx, rrx) = mpsc::channel();
    if tx.send((job, rtx)).is_err() {
        return error(503, "server is shutting down");
    }
    rrx.recv_timeout(BACKEND_TIMEOUT)
        .unwrap_or_else(|_| error(504, "camera did not answer in time"))
}

/// Runs on the backend thread.
fn handle(backend: &CameraBackend, preview: Option<&Preview>, job: Job) -> Reply {
    let snapshot = match backend.snapshot() {
        Ok(s) => s,
        Err(e) => return error(500, &format!("{e:#}")),
    };
    match job {
        Job::Status => {
            let preview = preview.map(|p| {
                let s = p.status();
                serde_json::json!({ "frames": s.frames, "error": s.error })
            });
            let body = serde_json::json!({ "snapshot": snapshot, "preview": preview });
            json_reply(200, body.to_string())
        }
        Job::Set { key, value } => {
            let Some(control) = snapshot.controls.iter().find(|c| c.key == key) else {
                return error(404, "unknown control");
            };
            if let Err(msg) = validate(control, &value) {
                return error(422, &msg);
            }
            match backend.set(&key, &value) {
                Ok(()) => json_reply(200, r#"{"ok":true}"#.to_owned()),
                Err(e) => error(422, &format!("{e:#}")),
            }
        }
    }
}

fn validate(c: &Control, value: &str) -> Result<(), String> {
    match c.kind {
        ControlKind::Number => {
            let v: f64 = value
                .parse()
                .map_err(|_| "value must be a number".to_owned())?;
            if !v.is_finite() || v < c.min || v > c.max {
                return Err(format!("value must be between {} and {}", c.min, c.max));
            }
        }
        ControlKind::Choice if !c.options.iter().any(|o| o == value) => {
            return Err("value is not one of the offered options".to_owned());
        }
        _ => {}
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetRequest {
    key: String,
    value: serde_json::Value,
}

fn parse_set(body: &[u8]) -> Result<(String, String), String> {
    let req: SetRequest = serde_json::from_slice(body)
        .map_err(|_| "body must be JSON {\"key\",\"value\"}".to_owned())?;
    let value = match req.value {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        _ => return Err("value must be a string, number or boolean".to_owned()),
    };
    if req.key.is_empty()
        || req.key.len() > MAX_VALUE
        || value.len() > MAX_VALUE
        || value.chars().any(char::is_control)
    {
        return Err("key or value is empty, too long or has control characters".to_owned());
    }
    Ok((req.key, value))
}

fn valid_token(t: &str) -> bool {
    !t.is_empty()
        && t.len() <= MAX_TOKEN
        && t.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._~".contains(&b))
}

fn random_token() -> Result<String> {
    let mut raw = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut raw))
        .context("reading /dev/urandom for the access token")?;
    Ok(raw.iter().map(|b| format!("{b:02x}")).collect())
}

fn ct_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn bearer_ok(header: Option<&str>, token: &str) -> bool {
    header
        .and_then(|h| h.strip_prefix("Bearer "))
        .is_some_and(|t| ct_eq(t, token))
}

/// Same-origin only: a present Origin must be exactly the page's own scheme+Host (rejects `null`).
fn origin_ok(origin: Option<&str>, host: &str) -> bool {
    origin.is_none_or(|o| o.strip_prefix("http://") == Some(host))
}

/// Loopback listeners only answer loopback host names (DNS-rebinding guard); others rely on the token.
fn host_ok(host: &str, listen: SocketAddr) -> bool {
    if !listen.ip().is_loopback() {
        return true;
    }
    let name = host
        .strip_suffix(&format!(":{}", listen.port()))
        .unwrap_or(host);
    matches!(name, "localhost" | "127.0.0.1" | "[::1]")
}

fn is_json(ctype: &str) -> bool {
    let (essence, params) = ctype.split_once(';').unwrap_or((ctype, ""));
    essence.trim().eq_ignore_ascii_case("application/json")
        && (params.trim().is_empty() || params.trim().eq_ignore_ascii_case("charset=utf-8"))
}

const PAGE: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera remote</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#16181b;color:#e6e8ea}
main{max-width:46rem;margin:0 auto;padding:1rem}
h1{font-size:1.2rem;margin:.2rem 0}
#info,#warn{color:#aab2bb;font-size:.9rem}
#msg{min-height:1.4rem;color:#7fd18b} #msg.bad{color:#ff8a80}
fieldset{border:1px solid #3a4047;border-radius:6px;margin:.8rem 0;padding:.4rem .8rem}
legend{padding:0 .4rem;color:#9ab}
.row{display:flex;align-items:center;gap:.7rem;padding:.35rem 0}
.row label{flex:0 0 11rem}
.row input[type=range]{flex:1;min-width:6rem}
.row output{min-width:3.5rem;text-align:right;font-variant-numeric:tabular-nums}
select,button{font:inherit;background:#23272c;color:inherit;border:1px solid #4a525a;border-radius:4px;padding:.2rem .5rem}
</style></head><body><main>
<h1>Camera remote</h1>
<div id="info" role="status"></div>
<div id="warn"></div>
<div id="msg" role="alert"></div>
<button id="refresh" type="button">Refresh</button>
<div id="controls"></div>
</main>
<script>
"use strict";
const token = new URLSearchParams(location.search).get("token") || "";
const root = document.getElementById("controls");
const msg = document.getElementById("msg");
function el(tag, parent, text) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function say(text, bad) { msg.textContent = text; msg.className = bad ? "bad" : ""; }
function human(s) { return s.replace(/-/g, " "); }
async function api(path, opts) {
  const r = await fetch(path, Object.assign({ cache: "no-store", headers: {
    "Authorization": "Bearer " + token, "Content-Type": "application/json" } }, opts));
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}
async function send(key, value) {
  try { await api("/api/set", { method: "POST", body: JSON.stringify({ key: key, value: String(value) }) });
        say("Set " + human(key) + " = " + value); }
  catch (e) { say(human(key) + ": " + e.message, true); }
  await load();
}
function render(s) {
  const snap = s.snapshot, d = snap.device;
  const p = s.preview;
  document.getElementById("info").textContent = d.name + " (" + d.model + ") firmware " + snap.firmware +
    " serial " + snap.serial + " | " + (snap.streaming ? "streaming" : "not streaming") +
    (p ? " | preview frames " + p.frames + (p.error ? " error: " + p.error : "") : " | preview not running");
  document.getElementById("warn").textContent = snap.warnings.join(" | ");
  const focused = root.contains(document.activeElement) ? document.activeElement.id : "";
  root.replaceChildren();
  const groups = new Map();
  for (const c of snap.controls) {
    if (!groups.has(c.group)) groups.set(c.group, []);
    groups.get(c.group).push(c);
  }
  for (const [name, list] of groups) {
    const fs = el("fieldset", root);
    el("legend", fs, name);
    for (const c of list) {
      const row = el("div", fs);
      row.className = "row";
      const id = "c-" + c.key;
      el("label", row, c.label).htmlFor = id;
      let input;
      if (c.kind === "Number") {
        input = el("input", row);
        input.type = "range"; input.min = c.min; input.max = c.max; input.step = c.step > 0 ? c.step : "any";
        input.value = c.value;
        const out = el("output", row, c.value === "" ? "unknown" : c.value);
        input.addEventListener("input", () => { out.textContent = input.value; });
        input.addEventListener("change", () => send(c.key, input.value));
      } else if (c.kind === "Toggle") {
        input = el("input", row);
        input.type = "checkbox"; input.checked = c.value === "true";
        input.addEventListener("change", () => send(c.key, input.checked));
      } else {
        input = el("select", row);
        for (const o of c.options) el("option", input, human(o)).value = o;
        input.value = c.value;
        input.addEventListener("change", () => send(c.key, input.value));
      }
      input.id = id;
    }
  }
  if (focused) { const f = document.getElementById(focused); if (f) f.focus(); }
}
async function load() {
  try { render(await api("/api/status")); }
  catch (e) { say("Status: " + e.message, true); }
}
document.getElementById("refresh").addEventListener("click", load);
load();
</script></body></html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn loopback() -> SocketAddr {
        "127.0.0.1:9360".parse().unwrap()
    }

    #[test]
    fn bearer_must_match_exactly() {
        assert!(bearer_ok(Some("Bearer abc"), "abc"));
        assert!(!bearer_ok(Some("Bearer abd"), "abc"));
        assert!(!bearer_ok(Some("Bearer abcd"), "abc"));
        assert!(!bearer_ok(Some("abc"), "abc"));
        assert!(!bearer_ok(None, "abc"));
    }

    #[test]
    fn origin_must_be_same_origin() {
        assert!(origin_ok(None, "localhost:9360"));
        assert!(origin_ok(Some("http://localhost:9360"), "localhost:9360"));
        assert!(!origin_ok(Some("http://evil.example"), "localhost:9360"));
        assert!(!origin_ok(Some("null"), "localhost:9360"));
    }

    #[test]
    fn loopback_rejects_rebound_hosts() {
        assert!(host_ok("localhost:9360", loopback()));
        assert!(host_ok("[::1]:9360", loopback()));
        assert!(!host_ok("evil.example:9360", loopback()));
        assert!(!host_ok("localhost.evil.example", loopback()));
    }

    #[test]
    fn set_body_is_strict() {
        assert_eq!(
            parse_set(br#"{"key":"zoom","value":1.5}"#).unwrap(),
            ("zoom".into(), "1.5".into())
        );
        assert!(parse_set(br#"{"key":"zoom","value":"1","extra":1}"#).is_err());
        assert!(parse_set(br#"{"key":"zoom","value":[1]}"#).is_err());
        assert!(parse_set(b"not json").is_err());
        assert!(
            parse_set(format!(r#"{{"key":"k","value":"{}"}}"#, "x".repeat(65)).as_bytes()).is_err()
        );
    }

    #[test]
    fn content_type_allows_only_json() {
        assert!(is_json("application/json"));
        assert!(is_json("application/json; charset=utf-8"));
        assert!(!is_json("text/plain"));
        assert!(!is_json("application/json; boundary=x"));
    }

    /// The real accept loop on an ephemeral loopback port, answered by a stand-in camera thread.
    fn spawn_remote(token: &str) -> SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let listen = listener.local_addr().unwrap();
        let cfg = Arc::new(Config {
            listen,
            token: token.into(),
        });
        let (tx, rx) = mpsc::channel::<Msg>();
        thread::spawn(move || {
            for (_, reply) in rx {
                let _ = reply.send(json_reply(200, "{}".into()));
            }
        });
        thread::spawn(move || accept_loop(&listener, &cfg, &tx, &AtomicBool::new(false)));
        listen
    }

    fn send(addr: SocketAddr, raw: &str) -> TcpStream {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(raw.as_bytes()).unwrap();
        stream
    }

    fn reply(mut stream: TcpStream) -> String {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut text = String::new();
        let _ = stream.read_to_string(&mut text);
        text
    }

    #[test]
    fn stalled_or_oversized_requests_cost_only_their_connection() {
        let addr = spawn_remote("t");
        let host = format!("Host: localhost:{}", addr.port());
        let json = "Authorization: Bearer t\r\nContent-Type: application/json";
        // More stalls than the old four request workers: silent sockets and unfinished bodies.
        let stalled: Vec<TcpStream> = (0..3)
            .map(|_| send(addr, ""))
            .chain((0..3).map(|_| {
                let head = format!("POST /api/set HTTP/1.1\r\n{host}\r\n{json}\r\n");
                send(addr, &format!("{head}Content-Length: 2000\r\n\r\n{{"))
            }))
            .collect();
        // Discarding a huge announced body used to abort the whole process.
        let huge = "Content-Length: 99999999999999\r\n\r\n";
        let anonymous = reply(send(
            addr,
            &format!("POST /api/set HTTP/1.1\r\n{host}\r\n{huge}"),
        ));
        assert!(anonymous.starts_with("HTTP/1.1 401 "), "{anonymous}");
        let authorized = reply(send(
            addr,
            &format!("POST /api/set HTTP/1.1\r\n{host}\r\n{json}\r\n{huge}"),
        ));
        assert!(authorized.starts_with("HTTP/1.1 413 "), "{authorized}");

        let started = Instant::now();
        let status = reply(send(
            addr,
            &format!("GET /api/status HTTP/1.1\r\n{host}\r\nAuthorization: Bearer t\r\n\r\n"),
        ));
        assert!(
            status.starts_with("HTTP/1.1 200 ") && status.ends_with("\r\n\r\n{}"),
            "{status}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "answered while others stall"
        );
        drop(stalled);
    }

    #[test]
    fn a_withheld_body_is_cut_off_at_the_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let listen = listener.local_addr().unwrap();
        let client = send(
            listen,
            &format!(
                "POST /api/set HTTP/1.1\r\nHost: localhost:{}\r\nAuthorization: Bearer t\r\n\
                 Content-Type: application/json\r\nContent-Length: 2000\r\n\r\n{{",
                listen.port()
            ),
        );
        let (stream, _) = listener.accept().unwrap();
        let cfg = Config {
            listen,
            token: "t".into(),
        };
        let (tx, _rx) = mpsc::channel::<Msg>();
        let deadline = Instant::now() + Duration::from_millis(300);
        let server = thread::spawn(move || connection(stream, &cfg, &tx, deadline));
        let text = reply(client);
        assert!(text.starts_with("HTTP/1.1 408 "), "{text}");
        server.join().unwrap();
    }
}
