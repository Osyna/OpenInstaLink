"""link-camera local AI worker, started only by src/models/local.rs inside the managed runtime.

    worker.py serve                  framed IPC on stdin/stdout, one resident model, no network (seccomp)
    worker.py install MODEL DIR      explicit download into the empty DIR, then writes DIR/.installed.json
    worker.py metadata MODEL         prints the plugin's source/revision/license as JSON
    worker.py probe                  seccomp-confined: which runtime packages import, which accelerators exist
    worker.py netcheck               installs the network filter and prints the socket() self-check result

Frame, both directions: >II (header length, body length), a JSON header, a binary body.
Request header {id, op:"infer", model, dir, options, width, height}, body = width*height*3 raw RGB bytes.
Reply header {id, ok, loaded, device, ...}: ok -> body is the result and content_type is set;
not ok -> error text and kind ("input" for bad options/image, "runtime" for anything else). Diagnostics go
to stderr; fd 1 is reserved for frames. Only the standard library is imported at module level.
"""
import ctypes
import ctypes.util
import errno
import gc
import importlib
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_HEADER = 1 << 20
MAX_PIXELS = 40_000_000
MAX_OUTPUT = 32 * 1024 * 1024
CONTENT_TYPES = ("application/json", "image/png")
MARKER = ".installed.json"
# model id -> plugin module next to this file
PLUGINS = {
    "mediapipe-hand": "landmarks", "mediapipe-pose": "landmarks", "mediapipe-face": "landmarks",
    "yolo11n": "detection",
    "depth-anything-v2-small": "scene", "marigold-depth-v1-1": "scene",
    "sam2.1-tiny": "scene", "florence-2-base": "scene",
}


def plugin(model):
    name = PLUGINS.get(model)
    if name is None:
        raise ValueError(f"unsupported model {model!r}")
    if HERE not in sys.path:  # -I keeps the script directory off sys.path
        sys.path.insert(0, HERE)
    module = importlib.import_module(name)
    if model not in module.MODEL_IDS:
        raise RuntimeError(f"plugin {name} does not provide {model}")
    return module


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError("stdin closed in the middle of a frame")
    return data


def write_frame(stream, header, body=b""):
    raw = json.dumps(header, separators=(",", ":")).encode()
    stream.write(struct.pack(">II", len(raw), len(body)) + raw)
    stream.write(body)
    stream.flush()


def block_network():
    """Defence in depth: inference never needs the network, so Python-level connects fail loudly."""
    import socket

    def deny(*_args, **_kwargs):
        raise OSError("network access is disabled during local inference")

    original = {name: getattr(socket.socket, name) for name in ("connect", "connect_ex", "sendto")}

    def guard(name):
        def wrapper(self, *args, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                deny()
            return original[name](self, *args, **kwargs)
        return wrapper

    for name in original:
        setattr(socket.socket, name, guard(name))
    socket.getaddrinfo = deny
    socket.create_connection = deny


# Address families inference must never open: IPv4, IPv6, raw packet. AF_UNIX stays (CUDA, X11-less GPU stacks).
DENIED_FAMILIES = (2, 10, 17)
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO_EPERM = 0x00050000 | 1  # SCMP_ACT_ERRNO(EPERM)
SCMP_CMP_MASKED_EQ = 7


class _ArgCmp(ctypes.Structure):  # struct scmp_arg_cmp
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def _libseccomp():
    last = None
    for name in ("libseccomp.so.2", ctypes.util.find_library("seccomp")):
        if not name:
            continue
        try:
            return ctypes.CDLL(name, use_errno=True)
        except OSError as error:
            last = error
    raise RuntimeError(
        "libseccomp.so.2 is required to guarantee that local AI inference cannot use the network, and it "
        "was not found: install libseccomp2 (Debian/Ubuntu) or libseccomp (Arch, Fedora)"
        + (f" [{last}]" if last else ""))


def enforce_offline():
    """OS-level network ban for this process and everything it later starts or loads.

    A seccomp filter makes socket(AF_INET/AF_INET6/AF_PACKET) fail with EPERM, which also covers native
    code (mediapipe's C++ core reports usage metrics to Google; the socket monkeypatch cannot see that).
    Unprivileged: libseccomp sets no_new_privs. There is deliberately no fallback: if the filter cannot be
    installed and verified, this raises and the worker does not run.
    """
    lib = _libseccomp()
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
                                           ctypes.POINTER(_ArgCmp)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("libseccomp could not create a filter")
    try:
        number = lib.seccomp_syscall_resolve_name(b"socket")
        if number < 0:
            raise RuntimeError("libseccomp does not know the socket system call on this architecture")
        for family in DENIED_FAMILIES:
            # 32-bit masked compare: the upper register half of an int argument is not significant.
            rule = _ArgCmp(0, SCMP_CMP_MASKED_EQ, 0xFFFFFFFF, family)
            code = lib.seccomp_rule_add_array(context, SCMP_ACT_ERRNO_EPERM, number, 1, ctypes.byref(rule))
            if code != 0:
                raise RuntimeError(f"libseccomp rejected the network rule ({code})")
        code = lib.seccomp_load(context)
        if code != 0:
            raise RuntimeError(f"the kernel refused the network filter ({code}); seccomp is required")
    finally:
        lib.seccomp_release(context)
    result = _socket_probe()
    if result["inet"] != errno.EPERM or result["inet6"] != errno.EPERM or not result["unix"]:
        raise RuntimeError(f"the network filter self-check failed: {result}")


def _socket_probe():
    """Calls libc socket() directly (not Python's socket module): errno of AF_INET/AF_INET6, AF_UNIX works."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    libc.close.argtypes = [ctypes.c_int]
    result = {}
    for key, family in (("inet", 2), ("inet6", 10)):
        ctypes.set_errno(0)
        fd = libc.socket(family, 1, 0)  # SOCK_STREAM
        if fd >= 0:
            libc.close(fd)
            result[key] = 0
        else:
            result[key] = ctypes.get_errno()
    fd = libc.socket(1, 1, 0)  # AF_UNIX
    result["unix"] = fd >= 0
    if fd >= 0:
        libc.close(fd)
    return result


def netcheck():
    enforce_offline()
    print(json.dumps(_socket_probe()))


class Resident:
    """The one loaded model; switching models releases the previous one first."""

    def __init__(self):
        self.model = None
        self.fn = None
        self.device = "unknown"

    def release(self):
        fn, self.fn, self.model, self.device = self.fn, None, None, "unknown"
        close = getattr(fn, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                logger.exception("Resident model cleanup failed")
        del fn, close
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    def load(self, model, directory):
        self.release()
        if not (directory / MARKER).is_file():
            raise ValueError(f"{model} is not installed")
        fn = plugin(model).load(model, directory)
        device = getattr(fn, "device", None)
        label = device if isinstance(device, str) and device else "unknown"
        if label.startswith("cuda"):
            torch = sys.modules.get("torch")
            try:
                label = f"cuda ({torch.cuda.get_device_name(0)})" if torch else "cuda"
            except (AssertionError, RuntimeError) as error:
                logger.warning("CUDA device name is unavailable: %s", str(error)[:600])
                label = "cuda"
        self.model, self.fn, self.device = model, fn, label


def infer(resident, header, body):
    model = header["model"]
    width, height = header["width"], header["height"]
    options = header.get("options", {})
    if not (isinstance(width, int) and isinstance(height, int) and isinstance(options, dict)):
        raise TypeError("malformed request")
    if not 0 < width * height <= MAX_PIXELS or len(body) != width * height * 3:
        raise ValueError("the frame does not match its declared size")
    timing = {}
    if resident.model != model:
        started = time.monotonic()
        resident.load(model, Path(header["dir"]))
        timing["load_ms"] = round((time.monotonic() - started) * 1000)
    from PIL import Image

    image = Image.frombuffer("RGB", (width, height), body, "raw", "RGB", 0, 1)
    started = time.monotonic()
    content_type, data = resident.fn(image, options)
    timing["infer_ms"] = round((time.monotonic() - started) * 1000)
    if content_type not in CONTENT_TYPES or not isinstance(data, (bytes, bytearray)):
        raise RuntimeError("the model returned an unsupported result")
    if len(data) > MAX_OUTPUT:
        raise RuntimeError("the model result is too large")
    return {"content_type": content_type, **timing}, bytes(data)


def serve():
    enforce_offline()  # first, before any thread or native library exists
    block_network()
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # stray prints from libraries must never corrupt the frame stream
    stdin = sys.stdin.buffer
    resident = Resident()
    while True:
        prefix = stdin.read(8)
        if not prefix:
            return 0  # parent closed the pipe
        if len(prefix) != 8:
            return 2
        header_len, body_len = struct.unpack(">II", prefix)
        if header_len > MAX_HEADER or body_len > MAX_PIXELS * 3:
            print("frame exceeds the protocol limits", file=sys.stderr)
            return 2
        header = json.loads(read_exact(stdin, header_len))
        body = read_exact(stdin, body_len)
        reply = {"id": header.get("id")}
        data = b""
        try:
            if header.get("op") != "infer":
                raise ValueError("unknown operation")
            extra, data = infer(resident, header, body)
            reply.update(ok=True, **extra)
        except (ValueError, KeyError, TypeError) as error:
            reply.update(ok=False, kind="input", error=str(error)[:2000])
        except Exception as error:
            logger.exception("Local model inference failed")
            reply.update(ok=False, kind="runtime", error=f"{type(error).__name__}: {error}"[:2000])
        reply.update(loaded=resident.model, device=resident.device)
        write_frame(out, reply, data)
        del body, data


def install(model, directory):
    directory = Path(directory)
    module = plugin(model)
    metadata = {k: str(v) for k, v in module.metadata(model).items()}
    module.download(model, directory)
    pending = directory / (MARKER + ".part")
    pending.write_text(json.dumps({"format": 1, "model": model, "metadata": metadata}))
    pending.replace(directory / MARKER)  # the marker is written last: it proves the download completed


def probe():
    report = {"ok": True, "python": sys.version.split()[0], "versions": {}, "missing": {}}
    try:
        enforce_offline()  # before any native import: mediapipe's C++ core could otherwise phone home
    except RuntimeError as error:
        report["ok"] = False
        report["missing"]["libseccomp"] = str(error)[:600]
        print(json.dumps(report))
        return
    for name in ("numpy", "PIL", "cv2", "mediapipe", "onnxruntime", "torch", "torchvision",
                 "transformers", "diffusers", "huggingface_hub", "safetensors"):
        try:
            module = importlib.import_module(name)
            report["versions"][name] = str(getattr(module, "__version__", "?"))
        except Exception as error:
            logger.exception("Runtime package import failed: %s", name)
            report["ok"] = False
            report["missing"][name] = f"{type(error).__name__}: {error}"[:300]
    torch = sys.modules.get("torch")
    report["cuda"] = bool(torch and torch.cuda.is_available())
    report["gpu"] = torch.cuda.get_device_name(0) if report["cuda"] else None
    ort = sys.modules.get("onnxruntime")
    report["onnx_providers"] = list(ort.get_available_providers()) if ort else []
    print(json.dumps(report))


HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    mode, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if mode == "serve" and not args:
        sys.exit(serve())
    elif mode == "install" and len(args) == 2:
        install(*args)
    elif mode == "metadata" and len(args) == 1:
        print(json.dumps(plugin(args[0]).metadata(args[0])))
    elif mode == "probe" and not args:
        probe()
    elif mode == "netcheck" and not args:
        netcheck()
    else:
        sys.exit("usage: worker.py serve | install MODEL DIR | metadata MODEL | probe | netcheck")
