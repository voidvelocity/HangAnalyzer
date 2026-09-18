"""Verify 0 -> 1 -> 0 control transitions on a real A2 NPU."""
import argparse
import tempfile
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from flightrecorder.mspti_ring import read_ring
from hang_mspti import start, stop

parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
parser.add_argument("--directory", required=True)
args = parser.parse_args()
torch.npu.set_device(7)
with tempfile.TemporaryDirectory() as directory:
    control = Path(directory) / "enable_prof"
    control.write_text("0")
    path = start(args.directory, args.library, interval_s=0.2, control_file=str(control))
    time.sleep(0.4)
    assert not path.exists(), "recorder started while disabled"
    control.write_text("1")
    time.sleep(0.6)
    for _ in range(12):
        a = torch.randn((256, 256), device="npu:7")
        b = a @ a
        torch.npu.synchronize(7)
        assert b.numel() == a.numel()
        time.sleep(0.05)
    time.sleep(1.0)
    control.write_text("0")
    time.sleep(0.6)
    stop()
    _, events = read_ring(path)
    states = [e["name"] for e in events if e["kind"] == "status"]
    assert "enabled" in states and "disabled" in states, states
    assert any(e["kind"] == "kernel_done" for e in events)
    print(f"toggle smoke passed: {path}, records={len(events)}, states={states}")
