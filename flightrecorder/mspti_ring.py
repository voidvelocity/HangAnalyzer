"""Read the unified msPTI mmap ring without loading msPTI or touching the NPU."""
from __future__ import annotations

import mmap
import struct
from collections import defaultdict
from pathlib import Path

MAGIC = 0x315450534D474E48
HEADER = struct.Struct("<QIIQIIQQQQ")
EVENT = struct.Struct("<QQQQQIIIIHHI64s")
LABELS = {
    1: "runtime_enter", 2: "runtime_exit", 3: "hccl_enter",
    4: "hccl_exit", 5: "kernel_done", 6: "hccl_done", 7: "status",
    8: "runtime_api_done",
}


def read_ring(path: str | Path) -> tuple[dict, list[dict]]:
    with Path(path).open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if len(data) < 128:
            raise ValueError("ring header is truncated")
        magic, version, size, capacity, pid, _, written, wall, mono, dropped = HEADER.unpack_from(data)
        if magic != MAGIC or version != 1 or size != EVENT.size:
            raise ValueError("unsupported msPTI ring format")
        if len(data) < 128 + capacity * size:
            raise ValueError("ring is truncated")
        first = max(1, written - capacity + 1)
        events = []
        for seq in range(first, written + 1):
            offset = 128 + ((seq - 1) % capacity) * size
            before = struct.unpack_from("<Q", data, offset)[0]
            if before != seq:
                continue
            raw = EVENT.unpack_from(data, offset)
            after = struct.unpack_from("<Q", data, offset)[0]
            if raw[0] != seq or after != seq:
                continue  # writer has claimed, but not published, this slot
            name = raw[-1].split(b"\0", 1)[0].decode("utf-8", "replace")
            events.append(dict(seq=seq, observed_ns=raw[1], start_ns=raw[2],
                               end_ns=raw[3], correlation_id=raw[4], pid=raw[5],
                               tid=raw[6], device=raw[7], stream=raw[8],
                               kind=LABELS.get(raw[9], f"unknown_{raw[9]}"),
                               flags=raw[10], code=raw[11], name=name))
        return dict(pid=pid, capacity=capacity, written=written, wall_ns=wall,
                    mono_ns=mono, dropped=dropped, overwritten=max(0, written - capacity)), events


def report(path: str | Path, *, tail: int = 40) -> str:
    header, events = read_ring(path)
    count = defaultdict(int)
    for event in events:
        count[event["kind"]] += 1
    lines = [f"file: {path}", f"pid: {header['pid']}  records: {len(events)}"
             f"  overwritten: {header['overwritten']}  dropped_buffers: {header['dropped']}",
             "counts: " + ", ".join(f"{key}={count[key]}" for key in LABELS.values())]
    # Callback correlation ID is useful, but HCCL Activity has no such ID in msPTI 26.1.
    pending: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        if event["kind"] in ("runtime_enter", "hccl_enter"):
            key = (event["kind"].split("_")[0], event["tid"], event["code"])
            pending[key].append(event)
        elif event["kind"] in ("runtime_exit", "hccl_exit"):
            key = (event["kind"].split("_")[0], event["tid"], event["code"])
            if pending[key]:
                pending[key].pop()
    open_calls = sorted((e for stack in pending.values() for e in stack), key=lambda e: e["seq"])
    lines.append(f"callback entries without exits in retained window: {len(open_calls)}")
    for e in open_calls[-20:]:
        lines.append(f"  seq={e['seq']} {e['kind']} tid={e['tid']} corr={e['correlation_id']} {e['name']}")
    api_by_correlation = {e["correlation_id"]: e for e in events
                          if e["kind"] == "runtime_api_done" and e["correlation_id"]}
    kernels = [e for e in events if e["kind"] == "kernel_done"]
    joined = [e for e in kernels if e["correlation_id"] in api_by_correlation]
    lines.append(f"completed kernels with matching Runtime API Activity correlation: "
                 f"{len(joined)}/{len(kernels)}")
    last_by_stream = {}
    for e in events:
        if e["kind"] in ("kernel_done", "hccl_done"):
            last_by_stream[(e["device"], e["stream"])] = e
    lines.append("last delivered completed activity per stream:")
    for (device, stream), e in sorted(last_by_stream.items()):
        lines.append(f"  device={device} stream={stream} #{e['seq']} "
                     f"{e['kind']} corr={e['correlation_id']} {e['name']}")
    if not last_by_stream:
        lines.append("  none")
    if events and events[0]["seq"] > 1:
        lines.append("WARNING: ring wrapped; unmatched entries can be older than retained data")
    lines.append("recent observations (device start/end use msPTI clock, not host monotonic clock):")
    for e in events[-tail:]:
        rel_ms = (e["observed_ns"] - header["mono_ns"]) / 1e6
        dev = "" if e["device"] == 0xFFFFFFFF else f" device={e['device']} stream={e['stream']}"
        lines.append(f"  +{rel_ms:10.3f}ms #{e['seq']:>8} {e['kind']:<16}"
                     f" corr={e['correlation_id']}{dev} code={e['code']} {e['name']}")
    lines.append("Interpretation: *_done means a completed activity record was delivered to host; "
                 "an absent record cannot prove that a kernel never ran. Callback exit means API returned, "
                 "not device completion. Callback correlation IDs are not assumed reliable; "
                 "match enter/exit by thread and callback ID.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Human-readable unified msPTI flight ring")
    parser.add_argument("ring", type=Path)
    parser.add_argument("--tail", type=int, default=40)
    args = parser.parse_args()
    print(report(args.ring, tail=args.tail))
