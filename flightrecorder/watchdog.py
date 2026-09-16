"""External, capture-only watchdog. Run on each node before sending the request."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from .format import read_header


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        # A zombie has exited even though kill(pid, 0) succeeds.
        status = Path(f"/proc/{pid}/status")
        if status.exists() and "State:\tZ" in status.read_text(errors="replace"):
            return False
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def collect_proc(pid: int, dest: Path) -> None:
    proc = Path(f"/proc/{pid}")
    out = dest / f"pid{pid}"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("status", "cmdline", "limits", "maps", "wchan"):
        try:
            data = (proc / name).read_bytes()[:4_000_000].replace(b"\0", b" ")
            (out / name).write_bytes(data)
        except OSError as e:
            (out / (name + ".error")).write_text(str(e))
    task_root = proc / "task"
    try:
        tids = list(task_root.iterdir())
    except OSError:
        tids = []
    for task in tids:
        for name in ("stack", "wchan", "status"):
            try:
                (out / f"tid{task.name}.{name}").write_bytes((task / name).read_bytes()[:100_000])
            except OSError as e:
                (out / f"tid{task.name}.{name}.error").write_text(str(e))


def snapshot(files: list[Path], output: Path, reason: str, logs: list[Path], npu_smi: bool = False) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    dest = output / f"{stamp}-{reason}"
    partial = output / f".{stamp}-{reason}.partial-{os.getpid()}"
    partial.mkdir(parents=True)
    metadata = {"reason": reason, "captured_utc": stamp, "ranks": [], "errors": []}
    for file in files:
        try:
            h = read_header(file)
            target = partial / file.name
            shutil.copyfile(file, target)
            metadata["ranks"].append({**h, "snapshot_file": target.name})
            collect_proc(h["pid"], partial)
        except (OSError, ValueError) as e:
            metadata["errors"].append(f"{file}: {e}")
    for log in logs:
        try:
            with log.open("rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 1_000_000))
                (partial / (log.name + ".tail")).write_bytes(f.read())
        except OSError as e:
            metadata["errors"].append(f"{log}: {e}")
    if npu_smi:
        try:
            completed = subprocess.run(["npu-smi", "info"], capture_output=True,
                                       timeout=5, check=False)
            (partial / "npu-smi.txt").write_bytes(completed.stdout + b"\nSTDERR\n" + completed.stderr)
            metadata["npu_smi_returncode"] = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as e:
            metadata["errors"].append(f"npu-smi: {e}")
    (partial / "metadata.json").write_text(json.dumps(metadata, indent=2))
    partial.rename(dest)
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=15, help="No recorded event for this many seconds")
    p.add_argument("--interval", type=float, default=0.2)
    p.add_argument("--pid", type=int, action="append", default=[], help="Monitored PID; may be repeated")
    p.add_argument("--log", type=Path, action="append", default=[])
    p.add_argument("--npu-smi", action="store_true", help="Capture npu-smi info during snapshot")
    args = p.parse_args()
    if args.timeout <= 0 or args.interval <= 0:
        p.error("timeout and interval must be positive")
    seen_stalls: set[tuple[int, int]] = set()
    seen_exit: set[int] = set()
    print("watchdog ready", flush=True)
    try:
        while True:
            files = sorted(args.directory.glob("*.flight"))
            headers = []
            for file in files:
                try:
                    headers.append(read_header(file))
                except (OSError, ValueError):
                    continue
            now = time.monotonic_ns()
            for h in headers:
                pid = h["pid"]
                # started_mono_ns is from this node's monotonic clock.
                if alive(pid) and (now - h["last_ns"]) / 1e9 >= args.timeout:
                    key = (pid, h["write_seq"])
                    if key not in seen_stalls:
                        seen_stalls.add(key)
                        print(snapshot(files, args.output, f"stalled-pid{pid}", args.log, args.npu_smi), flush=True)
                elif not alive(pid) and pid not in seen_exit:
                    seen_exit.add(pid)
                    print(snapshot(files, args.output, f"exit-pid{pid}", args.log, args.npu_smi), flush=True)
            if args.pid and all(not alive(pid) for pid in args.pid):
                if not headers:
                    print("monitored PIDs exited before recorder files appeared", flush=True)
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
