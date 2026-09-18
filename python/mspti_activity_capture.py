"""In-process msPTI 26.1 Python Monitor capture with periodic durable JSONL.

Import ``MsptiCapture`` into each NPU worker, or run ``--demo`` under torchrun.
Only msPTI's Python API is used for device activity collection.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _record(kind: str, data: Any) -> dict[str, Any]:
    """Copy callback-owned fields before the next msPTI buffer is recycled."""
    start = int(data.start)
    end = int(data.end)
    item = {
        "kind": kind,
        "device_id": int(data.device_id),
        "stream_id": int(data.stream_id),
        "correlation_id": int(data.correlation_id),
        "name": str(data.name),
        "start_ns": start,
        "end_ns": end,
        "timestamp_valid": start > 0 and end >= start,
        "duration_ns": end - start if start > 0 and end >= start else None,
    }
    if kind == "kernel":
        item["kernel_type"] = str(data.type)
    else:
        item.update({
            "comm_name": str(data.comm_name),
            "alg_type": str(data.alg_type),
            "count": int(data.count),
            "data_type": _enum_value(data.data_type),
        })
    return item


class MsptiCapture:
    """One instance per NPU worker/process. Start after selecting its NPU."""

    def __init__(self, output: str | Path, *, interval_s: float = 0.5,
                 buffer_mb: int = 64, queue_size: int = 100_000,
                 enable_file: str | Path | None = "/home/enable_prof"):
        if interval_s <= 0 or buffer_mb <= 0 or queue_size <= 0:
            raise ValueError("interval_s, buffer_mb and queue_size must be positive")
        self.output = Path(output)
        self.interval_s = interval_s
        self.buffer_mb = buffer_mb
        self.enable_file = Path(enable_file) if enable_file is not None else None
        self.items: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=queue_size)
        self.stop_requested = threading.Event()
        self.writer_done = threading.Event()
        self.flush_thread: threading.Thread | None = None
        self.writer_thread: threading.Thread | None = None
        self.kernel_monitor: Any = None
        self.comm_monitor: Any = None
        self.started = False
        self.monitor_active = False
        self.transitions = 0
        self.dropped = 0
        self.callback_errors = 0
        self.flush_errors = 0
        self.written = Counter()
        self.last_error: str | None = None

    def _enqueue(self, kind: str, data: Any) -> None:
        try:
            self.items.put_nowait(_record(kind, data))
        except queue.Full:
            self.dropped += 1
        except Exception as exc:  # never throw through a profiler callback
            self.callback_errors += 1
            self.last_error = f"callback {type(exc).__name__}: {exc}"

    def _write_loop(self) -> None:
        # One writer owns the file. Each 0.5 s batch is flushed and fsynced.
        try:
            with self.output.open("a", encoding="utf-8", buffering=1) as handle:
                deadline = time.monotonic() + self.interval_s
                while not self.writer_done.is_set() or not self.items.empty():
                    timeout = max(0.0, min(0.1, deadline - time.monotonic()))
                    try:
                        item = self.items.get(timeout=timeout)
                    except queue.Empty:
                        item = None
                    if item is not None:
                        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                        self.written[item["kind"]] += 1
                    if time.monotonic() >= deadline or self.writer_done.is_set():
                        handle.flush()
                        os.fsync(handle.fileno())
                        deadline = time.monotonic() + self.interval_s
        except Exception as exc:
            self.last_error = f"writer {type(exc).__name__}: {exc}"

    @staticmethod
    def _success(result: Any) -> bool:
        return _enum_value(result) == 0

    def _put_control(self, item: dict[str, Any]) -> None:
        try:
            self.items.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _enabled_by_file(self) -> bool:
        if self.enable_file is None:
            return True
        try:
            value = self.enable_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return False
        except OSError as exc:
            self.last_error = f"enable file {type(exc).__name__}: {exc}"
            return False
        if value not in ("0", "1"):
            self.last_error = f"enable file must contain 0 or 1: {self.enable_file}"
            return False
        return value == "1"

    def _set_monitor_active(self, enabled: bool) -> None:
        if enabled == self.monitor_active:
            return
        if enabled:
            result = self.kernel_monitor.start(lambda data: self._enqueue("kernel", data))
            if not self._success(result):
                raise RuntimeError(f"KernelMonitor.start failed: {result}")
            result = self.comm_monitor.start(lambda data: self._enqueue("communication", data))
            if not self._success(result):
                self.kernel_monitor.stop()
                raise RuntimeError(f"CommunicationMonitor.start failed: {result}")
            self.monitor_active = True
        else:
            # stop() performs the final flush before removing each callback.
            for label, monitor in (("kernel", self.kernel_monitor), ("communication", self.comm_monitor)):
                result = monitor.stop()
                if not self._success(result):
                    self.last_error = f"{label} stop returned {result}"
            self.monitor_active = False
        self.transitions += 1
        self._put_control({"kind": "capture_state", "host_time_ns": time.time_ns(),
                           "enabled": enabled})

    def _flush_loop(self) -> None:
        # This thread owns the msPTI start/stop lifecycle, including final stop.
        while not self.stop_requested.is_set():
            try:
                self._set_monitor_active(self._enabled_by_file())
                flush_ok: bool | None = None
                if self.monitor_active:
                    # flush_all is global in msPTI 26.1; one call flushes both kinds.
                    result = self.kernel_monitor.flush_all()
                    flush_ok = self._success(result)
                    if not flush_ok:
                        self.flush_errors += 1
                        self.last_error = f"flush_all returned {result}"
                self._put_control({"kind": "flush_tick", "host_time_ns": time.time_ns(),
                                   "enabled": self.monitor_active, "flush_ok": flush_ok,
                                   "dropped": self.dropped})
            except Exception as exc:
                self.flush_errors += 1
                self.last_error = f"monitor loop {type(exc).__name__}: {exc}"
            self.stop_requested.wait(self.interval_s)
        if self.monitor_active:
            try:
                self._set_monitor_active(False)
            except Exception as exc:
                self.last_error = f"final stop {type(exc).__name__}: {exc}"

    def start(self) -> "MsptiCapture":
        if self.started:
            raise RuntimeError("capture already started")
        if self.writer_done.is_set():
            raise RuntimeError("capture has stopped; create a new instance")
        from mspti import KernelMonitor, CommunicationMonitor

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.kernel_monitor = KernelMonitor()
        self.comm_monitor = CommunicationMonitor()
        if not self._success(self.kernel_monitor.set_buffer_size(self.buffer_mb)):
            raise RuntimeError("msPTI set_buffer_size failed")
        self.writer_thread = threading.Thread(target=self._write_loop, name="mspti-writer", daemon=True)
        self.writer_thread.start()
        self.started = True
        self.flush_thread = threading.Thread(target=self._flush_loop, name="mspti-flush", daemon=True)
        self.flush_thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if not self.started:
            return self.summary()
        self.stop_requested.set()
        assert self.flush_thread is not None and self.writer_thread is not None
        self.flush_thread.join(timeout=10)
        if self.flush_thread.is_alive():
            raise RuntimeError("msPTI flush thread did not stop; monitor remains active")
        self.writer_done.set()
        self.writer_thread.join(timeout=10)
        self.started = False
        summary = self.summary()
        self.output.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary

    def summary(self) -> dict[str, Any]:
        return {
            "output": str(self.output), "pid": os.getpid(),
            "interval_s": self.interval_s, "counts": dict(self.written),
            "enable_file": str(self.enable_file) if self.enable_file else None,
            "monitor_active": self.monitor_active, "transitions": self.transitions,
            "dropped": self.dropped, "callback_errors": self.callback_errors,
            "flush_errors": self.flush_errors, "last_error": self.last_error,
            "pipe_utilization": None,
            "pipe_utilization_note": "msPTI 26.1 Python Kernel/CommunicationMonitor does not expose AICore/MTE/Vector counters",
        }

    def __enter__(self) -> "MsptiCapture":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()


def run_demo(args: argparse.Namespace) -> None:
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    devices = [int(part) for part in args.devices.split(",")]
    if local_rank >= len(devices):
        raise ValueError("--devices must have one physical device per local rank")
    device = devices[local_rank]
    torch.npu.set_device(device)
    if world_size > 1:
        dist.init_process_group("hccl")
    output = args.output / f"rank{rank}-pid{os.getpid()}.jsonl"
    try:
        with MsptiCapture(output, interval_s=args.interval, buffer_mb=args.buffer_mb,
                          enable_file=args.enable_file) as capture:
            x = torch.randn((args.width, args.width), dtype=torch.float16, device=f"npu:{device}")
            y = torch.randn_like(x)
            for _ in range(args.iterations):
                x = x @ y
                x = x + y
                if world_size > 1:
                    dist.all_reduce(x)
                if args.pause:
                    time.sleep(args.pause)
            torch.npu.synchronize(device)
            # Give the periodic flush thread at least one tick during smoke tests.
            time.sleep(args.interval * 1.2)
        print(json.dumps(capture.summary(), ensure_ascii=False), flush=True)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run a real torch_npu compute/HCCL smoke workload")
    parser.add_argument("--output", type=Path, required=True, help="JSONL path or demo output directory")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--buffer-mb", type=int, default=64)
    parser.add_argument("--enable-file", type=Path, default=Path("/home/enable_prof"),
                        help="0/missing disables; 1 enables capture (checked every interval)")
    parser.add_argument("--devices", default="0", help="comma-separated physical NPU IDs for demo local ranks")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--pause", type=float, default=0.1)
    args = parser.parse_args()
    if not args.demo:
        parser.error("import MsptiCapture from this file inside the NPU worker; --demo runs a smoke workload")
    args.output.mkdir(parents=True, exist_ok=True)
    run_demo(args)


if __name__ == "__main__":
    main()
