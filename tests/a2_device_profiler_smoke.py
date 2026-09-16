"""Validate actual torch_npu device operator export on one A2 card."""
import argparse
from pathlib import Path
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--device", type=int, default=7)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
torch.npu.set_device(args.device)
args.output.mkdir(parents=True, exist_ok=True)
handler = torch_npu.profiler.tensorboard_trace_handler(str(args.output), worker_name="smoke",
                                                       analyse_flag=True)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    on_trace_ready=handler,
    record_shapes=False, profile_memory=False, with_stack=False,
):
    compute = torch.npu.Stream(device=args.device)
    transfer = torch.npu.Stream(device=args.device)
    with torch.npu.stream(compute):
        x = torch.randn((128, 128), device=f"npu:{args.device}")
        y = x @ x
        ready = torch.npu.Event()
        ready.record(compute)
    with torch.npu.stream(transfer):
        transfer.wait_event(ready)
        z = y.clone()
    torch.npu.synchronize(args.device)

files = list(args.output.rglob("kernel_details.csv"))
print(f"kernel_details_count={len(files)} files={[str(f) for f in files]}")
if files:
    print(files[0].read_text(errors="replace")[:2000])
