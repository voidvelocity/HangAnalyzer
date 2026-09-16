"""Version 1 flight file layout. Host timestamps are monotonic nanoseconds."""
from __future__ import annotations
import struct
from pathlib import Path

MAGIC = 0x31524647534E4341
HEADER = struct.Struct("<QIIQIHHQQQQ")
# Matches the C++ ABI: alignas(64), with 4 bytes of padding before uint64 correlation_id.
EVENT = struct.Struct("<QQIIHHIHH4xQQQ")
assert EVENT.size == 64
NAMES = [
    "UNKNOWN", "REQUEST_BEGIN", "REQUEST_END", "SCHEDULER_BEGIN", "SCHEDULER_END",
    "MODEL_BEGIN", "MODEL_END", "GRAPH_BEGIN", "GRAPH_END", "KV_BEGIN", "KV_END",
    "PP_SEND_BEGIN", "PP_SEND_END", "PP_RECV_BEGIN", "PP_RECV_END", "HCCL_BEGIN",
    "HCCL_END", "EVENT_RECORD", "EVENT_WAIT", "STREAM_SYNC_BEGIN", "STREAM_SYNC_END",
    "DEVICE_SYNC_BEGIN", "DEVICE_SYNC_END", "CHECKPOINT", "DEVICE_CONFIRMED",
]
TYPES = {name: i for i, name in enumerate(NAMES)}


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        raw = f.read(128)
    if len(raw) < 128:
        raise ValueError(f"short header: {path}")
    magic, version, event_size, capacity, pid, rank, device, write_seq, last_ns, wall, mono = HEADER.unpack_from(raw)
    if magic != MAGIC or version != 1 or event_size != 64 or capacity < 2:
        raise ValueError(f"unsupported flight format: {path}")
    return dict(path=str(path), capacity=capacity, pid=pid, rank=rank, device=device,
                write_seq=write_seq, last_ns=last_ns, started_wall_ns=wall, started_mono_ns=mono)


def read_events(path: Path) -> tuple[dict, list[dict]]:
    h = read_header(path)
    raw = path.read_bytes()
    count = min(h["capacity"], max(0, (len(raw) - 128) // 64))
    lower = max(1, h["write_seq"] - h["capacity"] + 1)
    result = []
    for slot in range(count):
        t, seq, pid, tid, device, rank, stream, kind, flags, corr, a0, a1 = EVENT.unpack_from(raw, 128 + slot * 64)
        if not (lower <= seq <= h["write_seq"] and (seq - 1) % h["capacity"] == slot):
            continue
        result.append(dict(timestamp_ns=t, seq=seq, pid=pid, tid=tid, device=device,
                           rank=rank, stream=stream, type=NAMES[kind] if kind < len(NAMES) else f"TYPE_{kind}",
                           flags=flags, correlation_id=corr, arg0=a0, arg1=a1))
    return h, sorted(result, key=lambda e: e["seq"])
