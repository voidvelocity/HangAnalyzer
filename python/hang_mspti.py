"""Optional worker bootstrap for libhangmspti.so; import from sitecustomize or app startup."""
from __future__ import annotations

import atexit
import ctypes
import os
import threading
from pathlib import Path

_lib = None
_stop = threading.Event()
_thread = None
_owner_pid = None


def start(directory: str, library: str, *, interval_s: float = 0.5,
          capacity: int = 262144, control_file: str | None = None) -> Path:
    """Start in the final worker process. Control file '0' delays activation."""
    global _lib, _thread, _owner_pid
    if _lib is not None:
        raise RuntimeError("already started")
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    path = directory_path / f"mspti-{os.getpid()}.msflight"
    lib = ctypes.CDLL(library)
    lib.hang_mspti_start.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.hang_mspti_start.restype = ctypes.c_int
    lib.hang_mspti_flush.restype = ctypes.c_int
    lib.hang_mspti_set_enabled.argtypes = [ctypes.c_int]
    lib.hang_mspti_set_enabled.restype = ctypes.c_int
    lib.hang_mspti_stop.restype = None
    _lib = lib
    _owner_pid = os.getpid()

    active_at_start = not control_file or _enabled(control_file)
    if active_at_start:
        result = lib.hang_mspti_start(os.fsencode(path), capacity)
        if result:
            _lib = None
            raise RuntimeError(f"hang_mspti_start failed: {result}; check LD_PRELOAD=libmspti.so")

    def worker() -> None:
        active = active_at_start
        collecting = active_at_start
        while not _stop.is_set():
            enabled = not control_file or _enabled(control_file)
            if enabled and not active:
                result = lib.hang_mspti_start(os.fsencode(path), capacity)
                if result:
                    # Keep this process alive and leave a visible error for deployment.
                    (directory_path / f"mspti-{os.getpid()}.error").write_text(
                        f"hang_mspti_start failed: {result}\n")
                    return
                active = True
                collecting = True
            elif active and enabled != collecting:
                result = lib.hang_mspti_set_enabled(int(enabled))
                if result:
                    (directory_path / f"mspti-{os.getpid()}.error").write_text(
                        f"hang_mspti_set_enabled failed: {result}\n")
                    return
                collecting = enabled
            if collecting:
                lib.hang_mspti_flush()
            _stop.wait(interval_s)
        if active:
            lib.hang_mspti_stop()

    _thread = threading.Thread(target=worker, name="hang-mspti-flush", daemon=True)
    _thread.start()
    atexit.register(stop)
    return path


def _enabled(path: str) -> bool:
    try:
        return Path(path).read_text().strip() == "1"
    except FileNotFoundError:
        return False


def stop() -> None:
    if _owner_pid != os.getpid():
        return
    _stop.set()
    if _thread is not None and _thread.is_alive() and _thread is not threading.current_thread():
        _thread.join(timeout=2.0)


if __name__ == "__main__":
    import argparse
    from flightrecorder.mspti_ring import report
    parser = argparse.ArgumentParser()
    parser.add_argument("ring")
    parser.add_argument("--tail", type=int, default=40)
    args = parser.parse_args()
    print(report(args.ring, tail=args.tail))
