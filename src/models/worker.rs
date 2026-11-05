//! One local inference at a time; cancellation invalidates results and terminates owned work.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, anyhow, ensure};
use serde_json::Value;

use super::history::Snapshot;
use super::{Analysis, Client, ModelInfo, RuntimeConfig, interpret, model_spec, validate_options};
use crate::media::Frame;

#[derive(Clone, Default)]
pub struct EngineStatus {
    pub catalog: Option<Arc<Vec<ModelInfo>>>,
    pub busy: bool,
    pub stopping: bool,
    pub message: String,
    pub error: Option<String>,
    pub result: Option<Arc<Analysis>>,
    pub snapshot: Option<Arc<Snapshot>>,
    pub completed: u64,
}

struct State {
    status: EngineStatus,
    generation: u64,
    cancel: Option<Arc<AtomicBool>>,
}

impl State {
    fn cancel(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if let Some(flag) = &self.cancel {
            flag.store(true, Ordering::Release);
        }
        self.status.result = None;
        self.status.snapshot = None;
        self.status.error = None;
        self.status.message = if self.status.busy {
            "Stopping the owned local inference process…".into()
        } else {
            "Analysis stopped".into()
        };
    }

    fn stop(&mut self) -> bool {
        if self.status.stopping {
            return false;
        }
        self.cancel();
        self.status.stopping = true;
        self.status.busy = true;
        self.status.message = "Stopping and unloading the owned local AI worker…".into();
        true
    }

    fn finish(&mut self, generation: u64, result: Result<(Analysis, Arc<Snapshot>)>) {
        self.status.busy = self.status.stopping;
        self.cancel = None;
        self.status.completed = self.status.completed.wrapping_add(1);
        if self.status.stopping {
            return;
        }
        if generation != self.generation {
            self.status.message = "Analysis stopped; late result discarded".into();
            if let Ok((analysis, _)) = &result {
                self.status.error = analysis.output.warning.clone();
            }
            if let Err(e) = result {
                if e.downcast_ref::<super::Cancelled>().is_none() {
                    self.status.error = Some(format!("{e:#}"));
                }
            }
            return;
        }
        match result {
            Ok((analysis, snapshot)) => {
                self.status.message = format!(
                    "{} · frame {} · {:.0} ms",
                    analysis.model,
                    analysis.source.sequence,
                    analysis.output.elapsed.as_secs_f64() * 1000.0
                );
                self.status.error = analysis.output.warning.clone();
                self.status.result = Some(Arc::new(analysis));
                self.status.snapshot = Some(snapshot);
            }
            Err(error) => {
                self.status.result = None;
                self.status.snapshot = None;
                self.status.message =
                    "Analysis failed; retry explicitly after correcting the error".into();
                self.status.error = Some(format!("{error:#}"));
            }
        }
    }

    fn refresh(&mut self, catalog: Result<Vec<ModelInfo>>, stopped: bool) {
        if self.status.stopping && !stopped {
            if let Ok(models) = catalog {
                self.status.catalog = Some(Arc::new(models));
            }
            return;
        }
        self.status.stopping = false;
        self.status.busy = false;
        self.status.error = None;
        match catalog {
            Ok(models) => {
                self.status.catalog = Some(Arc::new(models));
                self.status.message = "Local model library ready".into();
            }
            Err(error) => {
                self.status.catalog = None;
                self.status.error = Some(format!("{error:#}"));
                self.status.message = "Cannot read the local model library".into();
            }
        }
    }
}

struct Work {
    generation: u64,
    model: String,
    options: Value,
    frame: Arc<Frame>,
    cancel: Arc<AtomicBool>,
}

enum Request {
    Infer(Work),
    Refresh,
    Stop,
}

fn refresh_library(client: &Client, shared: &Mutex<State>, stopped: bool) {
    let catalog = client.catalog();
    shared.lock().unwrap().refresh(catalog, stopped);
}

pub struct Engine {
    shared: Arc<Mutex<State>>,
    requests: Option<mpsc::Sender<Request>>,
    worker: Option<JoinHandle<()>>,
}

impl Engine {
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        let client = Client::new(config)?;
        let shared = Arc::new(Mutex::new(State {
            status: EngineStatus {
                busy: true,
                message: "Inspecting the local model library…".into(),
                ..EngineStatus::default()
            },
            generation: 0,
            cancel: None,
        }));
        let (requests, receiver) = mpsc::channel::<Request>();
        let state = shared.clone();
        let worker = thread::Builder::new()
            .name("local-analysis".into())
            .spawn(move || {
                refresh_library(&client, &state, false);
                while let Ok(request) = receiver.recv() {
                    let work = match request {
                        Request::Infer(work) => work,
                        Request::Refresh => {
                            refresh_library(&client, &state, false);
                            continue;
                        }
                        Request::Stop => {
                            client.stop();
                            refresh_library(&client, &state, true);
                            continue;
                        }
                    };
                    let result = client
                        .run(&work.model, &work.options, &work.frame, &work.cancel)
                        .and_then(|raw| interpret(&work.model, work.frame, raw))
                        .and_then(|analysis| {
                            let snapshot = Snapshot::capture(&analysis)
                                .context("cannot preserve analysis in result history")?;
                            Ok((analysis, Arc::new(snapshot)))
                        });
                    let refresh = {
                        let state = state.lock().unwrap();
                        result.is_err()
                            || !state.status.catalog.as_ref().is_some_and(|models| {
                                models.iter().any(|m| m.id == work.model && m.loaded)
                            })
                    };
                    let catalog = refresh.then(|| client.catalog()).and_then(Result::ok);
                    let mut state = state.lock().unwrap();
                    state.finish(work.generation, result);
                    if let Some(models) = catalog {
                        state.status.catalog = Some(Arc::new(models));
                    }
                }
            })
            .context("cannot start model analysis worker")?;
        Ok(Self {
            shared,
            requests: Some(requests),
            worker: Some(worker),
        })
    }

    pub fn status(&self) -> EngineStatus {
        self.shared.lock().unwrap().status.clone()
    }

    pub fn submit(&self, model: &str, options: Value, frame: Arc<Frame>) -> Result<()> {
        model_spec(model)?;
        validate_options(model, &options, frame.width, frame.height)?;
        ensure!(
            frame.rgb.len() as u64 == u64::from(frame.width) * u64::from(frame.height) * 3,
            "invalid RGB frame"
        );
        let mut state = self.shared.lock().unwrap();
        ensure!(
            !state.status.busy,
            "analysis is already running or draining; no additional job was queued"
        );
        let catalog = state
            .status
            .catalog
            .as_ref()
            .context("initialize the local model library first")?;
        let info = catalog
            .iter()
            .find(|m| m.id == model)
            .with_context(|| format!("unknown local model {model}"))?;
        ensure!(
            info.available != Some(false),
            "{model} is unavailable: {}",
            info.unavailable_reason
                .as_deref()
                .unwrap_or("download the model and install its local runtime first")
        );
        let cancel = Arc::new(AtomicBool::new(false));
        let generation = state.generation;
        let work = Work {
            generation,
            model: model.into(),
            options,
            frame,
            cancel: cancel.clone(),
        };
        // Holding State until busy is set prevents concurrent callers bypassing single-flight.
        self.requests
            .as_ref()
            .context("analysis worker is stopped")?
            .send(Request::Infer(work))
            .map_err(|_| {
                anyhow!("local analysis worker is unavailable; refresh the model library")
            })?;
        state.cancel = Some(cancel);
        state.status.busy = true;
        state.status.error = None;
        state.status.message = format!("Running {model} on this machine");
        Ok(())
    }

    pub fn refresh_catalog(&self) -> Result<()> {
        let mut state = self.shared.lock().unwrap();
        ensure!(
            !state.status.busy,
            "wait for the current local model operation"
        );
        self.requests
            .as_ref()
            .context("analysis worker is stopped")?
            .send(Request::Refresh)
            .map_err(|_| anyhow!("local model worker is unavailable"))?;
        state.status.busy = true;
        state.status.message = "Refreshing the local model library…".into();
        Ok(())
    }

    pub fn cancel(&self) {
        self.shared.lock().unwrap().cancel();
    }

    /// Latches immediately; unloading and waiting for real process exit happen off the UI thread.
    pub fn stop(&self) {
        let mut state = self.shared.lock().unwrap();
        if !state.stop() {
            return;
        }
        if self
            .requests
            .as_ref()
            .is_none_or(|requests| requests.send(Request::Stop).is_err())
        {
            state.status.error =
                Some("local analysis worker is unavailable during shutdown".into());
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop();
        self.requests.take();
        if let Some(worker) = self.worker.take() {
            // A driver-stalled child must not hold window shutdown hostage; its reaper keeps the lease.
            if worker.is_finished() {
                let _ = worker.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{RawOutput, interpret};
    use std::time::Duration;

    #[test]
    fn cancel_invalidates_late_result_without_opening_a_second_job_slot() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut state = State {
            status: EngineStatus {
                busy: true,
                ..EngineStatus::default()
            },
            generation: 7,
            cancel: Some(flag.clone()),
        };
        let frame = Arc::new(Frame {
            width: 2,
            height: 2,
            rgb: vec![0; 12],
            sequence: 9,
        });
        let raw = RawOutput {
            job_id: "a".repeat(32),
            content_type: "application/json".into(),
            bytes: Arc::new(
                br#"{"image":{"width":2,"height":2},"text":"a room","task":"caption"}"#.to_vec(),
            ),
            elapsed: Duration::ZERO,
            warning: None,
        };
        let result = interpret("florence-2-base", frame, raw).unwrap();
        let snapshot = Arc::new(Snapshot::capture(&result).unwrap());
        state.status.result = Some(Arc::new(result.clone()));
        state.status.snapshot = Some(snapshot.clone());
        state.cancel();
        assert!(flag.load(Ordering::Acquire));
        assert!(
            state.status.busy,
            "cancelled work occupies its slot until the owned process exits"
        );
        assert!(state.status.result.is_none());
        assert!(state.status.snapshot.is_none());
        state.finish(7, Ok((result, snapshot)));
        assert!(!state.status.busy);
        assert!(
            state.status.result.is_none(),
            "cancelled source frame must not reappear"
        );
        assert!(
            state.status.snapshot.is_none(),
            "cancelled source snapshot must not enter history"
        );
        assert_eq!(state.status.completed, 1);
    }

    #[test]
    fn stop_latches_through_initial_refresh_and_late_completion() {
        for busy in [false, true] {
            let flag = Arc::new(AtomicBool::new(false));
            let shared = Arc::new(Mutex::new(State {
                status: EngineStatus {
                    busy,
                    ..EngineStatus::default()
                },
                generation: 4,
                cancel: busy.then(|| flag.clone()),
            }));
            let (requests, receiver) = mpsc::channel();
            let engine = Engine {
                shared: shared.clone(),
                requests: Some(requests),
                worker: None,
            };
            engine.stop();
            engine.stop();
            assert!(matches!(receiver.try_recv(), Ok(Request::Stop)));
            assert!(
                receiver.try_recv().is_err(),
                "repeated Stop must not queue cleanup"
            );
            assert_eq!(flag.load(Ordering::Acquire), busy);
            assert!(engine.status().busy && engine.status().stopping);
            let frame = Arc::new(Frame {
                width: 2,
                height: 2,
                rgb: vec![0; 12],
                sequence: 1,
            });
            assert!(
                engine
                    .submit("yolo11n", serde_json::json!({}), frame)
                    .is_err()
            );
            assert!(engine.refresh_catalog().is_err());
            let mut state = shared.lock().unwrap();
            state.refresh(Ok(Vec::new()), false);
            assert!(state.status.busy && state.status.stopping);
            if busy {
                state.finish(4, Err(super::super::Cancelled { job_id: None }.into()));
                assert!(state.status.busy && state.status.stopping);
            }
            state.refresh(Ok(Vec::new()), true);
            assert!(!state.status.busy && !state.status.stopping);
            assert!(state.status.result.is_none());
            assert!(state.status.snapshot.is_none());
        }
    }

    #[test]
    fn stop_cancels_queued_work_and_reopens_one_slot_only_after_cleanup() {
        let model: ModelInfo = serde_json::from_value(serde_json::json!({
            "id": "yolo11n", "input": "image", "output": "application/json",
            "installed": true, "available": true
        }))
        .unwrap();
        let shared = Arc::new(Mutex::new(State {
            status: EngineStatus {
                catalog: Some(Arc::new(vec![model.clone()])),
                ..EngineStatus::default()
            },
            generation: 0,
            cancel: None,
        }));
        let (requests, receiver) = mpsc::channel();
        let engine = Engine {
            shared: shared.clone(),
            requests: Some(requests),
            worker: None,
        };
        let frame = Arc::new(Frame {
            width: 2,
            height: 2,
            rgb: vec![0; 12],
            sequence: 1,
        });
        engine
            .submit("yolo11n", serde_json::json!({}), frame.clone())
            .unwrap();
        engine.stop();
        assert!(
            engine
                .submit("yolo11n", serde_json::json!({}), frame.clone())
                .is_err()
        );
        let Request::Infer(work) = receiver.try_recv().unwrap() else {
            panic!("queued inference")
        };
        assert!(work.cancel.load(Ordering::Acquire));
        assert!(matches!(receiver.try_recv(), Ok(Request::Stop)));
        shared.lock().unwrap().finish(
            work.generation,
            Err(super::super::Cancelled { job_id: None }.into()),
        );
        assert!(engine.status().busy && engine.status().stopping);
        shared.lock().unwrap().refresh(Ok(vec![model]), true);
        engine
            .submit("yolo11n", serde_json::json!({}), frame.clone())
            .unwrap();
        assert!(
            engine
                .submit("yolo11n", serde_json::json!({}), frame)
                .is_err()
        );
        assert!(matches!(receiver.try_recv(), Ok(Request::Infer(_))));
        assert!(receiver.try_recv().is_err());
    }
}
