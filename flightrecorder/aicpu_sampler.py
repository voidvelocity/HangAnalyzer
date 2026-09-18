"""Independent device utilization snapshots; no profiler subscription required."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIELDS = {
    "Aicpu Usage Rate(%)": "aicpu_pct",
    "Aicore Usage Rate(%)": "aicore_pct",
    "Aivector Usage Rate(%)": "aivector_pct",
    "Ctrlcpu Usage Rate(%)": "ctrlcpu_pct",
}


def parse_usages(output: str) -> dict[str, int]:
    values = {}
    for label, key in FIELDS.items():
        match = re.search(r"^\s*" + re.escape(label) + r"\s*:\s*(\d+)\s*$", output, re.M)
        if match:
            values[key] = int(match.group(1))
    return values


def sample(device: int, timeout_s: float) -> dict:
    started = time.time_ns()
    try:
        result = subprocess.run(["npu-smi", "info", "-t", "usages", "-i", str(device)],
                                capture_output=True, text=True, timeout=timeout_s, check=True)
        values = parse_usages(result.stdout)
        if "aicpu_pct" not in values:
            raise ValueError("npu-smi output has no Aicpu Usage Rate")
        return dict(device=device, started_wall_ns=started, ended_wall_ns=time.time_ns(),
                    **values)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return dict(device=device, started_wall_ns=started, ended_wall_ns=time.time_ns(),
                    error=f"{type(exc).__name__}: {exc}")


def capture(output: Path, devices: list[int], interval_s: float,
            duration_s: float | None = None) -> None:
    if not devices or interval_s <= 0:
        raise ValueError("devices required and interval must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_s if duration_s is not None else None
    with ThreadPoolExecutor(max_workers=len(devices)) as pool, output.open("a", encoding="utf-8") as file:
        while deadline is None or time.monotonic() < deadline:
            started = time.monotonic()
            futures = [pool.submit(sample, device, max(2.0, interval_s * 4)) for device in devices]
            for future in futures:
                file.write(json.dumps(future.result(), separators=(",", ":")) + "\n")
            file.flush()
            os.fsync(file.fileno())
            remaining = interval_s - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


def summarize(path: Path) -> str:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    lines = [f"file: {path}  samples: {len(rows)}"]
    for device in sorted({row["device"] for row in rows}):
        group = [row for row in rows if row["device"] == device]
        valid = [row for row in group if "aicpu_pct" in row]
        if not valid:
            lines.append(f"device {device}: no valid samples; last error: {group[-1].get('error')}")
            continue
        peak = max(row["aicpu_pct"] for row in valid)
        last = valid[-1]
        mean = sum(row["aicpu_pct"] for row in valid) / len(valid)
        lines.append(f"device {device}: AICPU mean={mean:.1f}% peak={peak}% "
                     f"last={last['aicpu_pct']}% valid={len(valid)}/{len(group)} "
                     f"last_wall_ns={last['ended_wall_ns']}")
    lines.append("Device-wide utilization only; it does not identify a process, task, "
                 "HCCL operation, or the reason for a wait.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("capture")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--devices", required=True, help="comma separated physical device IDs")
    run.add_argument("--interval", type=float, default=0.5)
    run.add_argument("--duration", type=float)
    view = sub.add_parser("summarize")
    view.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.output, [int(item) for item in args.devices.split(",")],
                args.interval, args.duration)
    else:
        print(summarize(args.path))
