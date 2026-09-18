"""A2 smoke: real Runtime callback plus completed device activity in one ring."""
import argparse
import ctypes
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from flightrecorder.mspti_ring import read_ring, report

parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", type=int, default=7)
args = parser.parse_args()
torch.npu.set_device(args.device)
lib = ctypes.CDLL(args.library)
lib.hang_mspti_start.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
lib.hang_mspti_start.restype = ctypes.c_int
lib.hang_mspti_flush.restype = ctypes.c_int
assert lib.hang_mspti_start(str(args.output).encode(), 16384) == 0
for _ in range(10):
    a = torch.randn((256, 256), device=f"npu:{args.device}")
    b = a @ a
    torch.npu.synchronize(args.device)
    assert b.shape == a.shape
    time.sleep(0.05)
assert lib.hang_mspti_flush() == 0
time.sleep(0.5)
header, events = read_ring(args.output)
print(report(args.output, tail=12))
assert any(e["kind"] == "runtime_enter" for e in events), "no Runtime callback"
assert any(e["kind"] == "runtime_exit" for e in events), "no Runtime exit"
assert any(e["kind"] == "kernel_done" for e in events), "no completed kernel activity"
assert any(e["device"] == args.device for e in events if e["kind"] == "kernel_done")
lib.hang_mspti_stop()
print(f"A2 unified msPTI smoke passed: {len(events)} events, pid={header['pid']}")
