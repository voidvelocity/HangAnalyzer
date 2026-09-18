"""Verify zero application-source bootstrap through sitecustomize and LD_PRELOAD."""
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from flightrecorder.mspti_ring import read_ring

device = 7
torch.npu.set_device(device)
for _ in range(12):
    a = torch.randn((256, 256), device=f"npu:{device}")
    b = a @ a
    torch.npu.synchronize(device)
    assert b.numel() == a.numel()
    time.sleep(0.05)
time.sleep(1.0)
ring = Path(os.environ["HANG_MSPTI_DIR"]) / f"mspti-{os.getpid()}.msflight"
_, events = read_ring(ring)
assert any(e["kind"] == "runtime_enter" for e in events)
assert any(e["kind"] == "kernel_done" for e in events)
print(f"sitecustomize bootstrap smoke passed: {ring}, events={len(events)}")
