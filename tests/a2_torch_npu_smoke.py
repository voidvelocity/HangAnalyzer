"""Run a real asynchronous torch_npu operation, synchronize, then record confirmation."""
import argparse
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from flightrecorder import Recorder
from flightrecorder.format import read_events

p = argparse.ArgumentParser()
p.add_argument("--library", required=True)
p.add_argument("--directory", required=True)
p.add_argument("--device", type=int, default=7)
args = p.parse_args()
torch.npu.set_device(args.device)
r = Recorder(args.directory, rank=0, device=args.device, library=args.library, capacity=128)
r.record("REQUEST_BEGIN", correlation=101, arg0=1024)
r.record("MODEL_BEGIN", stream=0, correlation=201)
a = torch.randn((256, 256), device=f"npu:{args.device}")
b = torch.randn((256, 256), device=f"npu:{args.device}")
c = a @ b
r.record("MODEL_END", stream=0, correlation=201)
torch.npu.synchronize(args.device)
r.record("DEVICE_CONFIRMED", stream=0, correlation=201, arg0=c.numel())
r.record("REQUEST_END", correlation=101)
r.close()
header, events = read_events(r.path)
assert [e["type"] for e in events] == ["REQUEST_BEGIN", "MODEL_BEGIN", "MODEL_END",
                                         "DEVICE_CONFIRMED", "REQUEST_END"]
assert events[-2]["arg0"] == 256 * 256
print(f"A2 torch_npu smoke passed: device={args.device} file={r.path} events={len(events)}")
