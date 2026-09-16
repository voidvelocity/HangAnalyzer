"""Summarize torch_npu profiler kernel_details.csv as a readable operator list."""
from __future__ import annotations
import csv
from pathlib import Path


def summarize(profiles: Path, output: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    lines = ["Ascend device kernels from completed profiler windows",
             "These are executed device kernels recorded before the controlled hang.",
             "They do not prove completion of tasks submitted after profiler stop."]
    for rank_dir in sorted(profiles.glob("rank*")):
        try:
            rank = int(rank_dir.name[4:])
        except ValueError:
            continue
        files = sorted(rank_dir.rglob("kernel_details.csv"))
        rows = []
        for path in files:
            with path.open(newline="", errors="replace") as f:
                rows.extend(csv.DictReader(f))
        counts[rank] = len(rows)
        lines += ["", f"RANK {rank}: device_kernel_count={len(rows)}"]
        for row in rows:
            lines.append(f"  device={row.get('Device_id', '?')} "
                         f"start_us={row.get('Start Time(us)', '').strip()} "
                         f"duration_us={row.get('Duration(us)', '').strip()} "
                         f"name={row.get('Name', '?')}")
        if not rows:
            lines.append("  no parsed kernel_details.csv found")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return counts
