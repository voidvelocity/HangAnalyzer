"""Real CANN Event pool/query/generation test on an Ascend A2 stream."""
import argparse
import shutil
import time
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from flightrecorder import Recorder, CannCheckpointManager
from flightrecorder.analyzer import analyze, write_outputs

p = argparse.ArgumentParser()
p.add_argument("--recorder-library", required=True)
p.add_argument("--checkpoint-library", required=True)
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--device", type=int, default=7)
args = p.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
torch.npu.set_device(args.device)
stream = torch.npu.Stream(device=args.device)
transfer = torch.npu.Stream(device=args.device)
stream_id = int(stream.npu_stream) & 0xFFFFFFFF
transfer_id = int(transfer.npu_stream) & 0xFFFFFFFF
r = Recorder(str(args.directory / "live"), rank=args.device, device=args.device,
             library=args.recorder_library, capacity=1024)
m = CannCheckpointManager(args.checkpoint_library, slots_per_stream=1)
m.register_stream(stream_id, int(stream.npu_stream))
m.register_stream(transfer_id, int(transfer.npu_stream))

# Warm up compilation outside the measured queue.
with torch.npu.stream(stream):
    x = torch.randn((1024, 1024), device=f"npu:{args.device}")
    x = x @ x
torch.npu.synchronize(args.device)

# Enqueue enough real work that the first nonblocking query observes NOT_READY.
with torch.npu.stream(stream):
    for _ in range(30):
        x = x @ x
submitted1 = r.record("MODEL_END", stream=stream_id, correlation=9001)
generation1 = m.submit(stream_id, submitted1, checkpoint_id=5001)
initial = m.poll(1)
if initial != 0:
    raise AssertionError("checkpoint completed before the first poll; increase queued work")

# Snapshot while the actual device checkpoint is unconfirmed.
pending_file = args.directory / "pending.flight"
shutil.copyfile(r.path, pending_file)
pending = analyze([pending_file])["ranks"][args.device]["device_checkpoint_progress"][stream_id]
assert pending["first_unconfirmed"]["arg0"] == submitted1
m.start_poller(interval_us=500, budget=8)

def confirmed(checkpoint_id):
    from flightrecorder.format import read_events
    return any(e["type"] == "DEVICE_CONFIRMED" and e["correlation_id"] == checkpoint_id
               for e in read_events(r.path)[1])

deadline = time.monotonic() + 30
while not confirmed(5001):
    if time.monotonic() > deadline:
        raise TimeoutError("CANN checkpoint did not complete")
    time.sleep(0.001)

# A second physical stream has an independent one-slot pool and watermark.
with torch.npu.stream(transfer):
    copied = x.clone()
submitted_transfer = r.record("KV_END", stream=transfer_id, correlation=9100,
                              arg0=copied.numel())
transfer_generation = m.submit(transfer_id, submitted_transfer, checkpoint_id=6001)
deadline = time.monotonic() + 30
while not confirmed(6001):
    if time.monotonic() > deadline:
        raise TimeoutError("transfer-stream checkpoint did not complete")
    time.sleep(0.001)

# Reuse the single physical Event: generation must prevent aliasing with checkpoint 1.
with torch.npu.stream(stream):
    x = x + 1
submitted2 = r.record("MODEL_END", stream=stream_id, correlation=9002)
generation2 = m.submit(stream_id, submitted2, checkpoint_id=5002)
assert generation1 == 1 and generation2 == 2
assert transfer_generation == 1
deadline = time.monotonic() + 30
while not confirmed(5002):
    if time.monotonic() > deadline:
        raise TimeoutError("reused CANN checkpoint did not complete")
    time.sleep(0.001)

m.close(); r.close()
final = analyze([r.path])
progress = final["ranks"][args.device]["device_checkpoint_progress"][stream_id]
assert progress["confirmed_through"]["arg0"] == submitted2
assert progress["unconfirmed_count"] == 0
transfer_progress = final["ranks"][args.device]["device_checkpoint_progress"][transfer_id]
assert transfer_progress["confirmed_through"]["arg0"] == submitted_transfer
assert transfer_progress["unconfirmed_count"] == 0
assert not final["ranks"][args.device]["stale_or_unmatched_confirmations"]
write_outputs(analyze([pending_file]), args.directory / "pending-report", [pending_file])
write_outputs(final, args.directory / "final-report", [r.path])
print(f"A2 checkpoint smoke passed compute={stream_id} generations={generation1},{generation2} "
      f"transfer={transfer_id} generation={transfer_generation}")
