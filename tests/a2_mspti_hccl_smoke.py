"""Run with torchrun --nproc-per-node 2 on physical A2 devices 4 and 5."""
import ctypes
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

from flightrecorder.mspti_ring import read_ring, report

rank = int(os.environ["LOCAL_RANK"])
device = 4 + rank
torch.npu.set_device(device)
dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
                        rank=rank, world_size=2)
directory = Path(os.environ["HANG_TEST_OUTPUT"])
directory.mkdir(parents=True, exist_ok=True)
path = directory / f"rank{rank}-{os.getpid()}.msflight"
lib = ctypes.CDLL(os.environ["HANG_MSPTI_LIBRARY"])
lib.hang_mspti_start.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
lib.hang_mspti_start.restype = ctypes.c_int
lib.hang_mspti_flush.restype = ctypes.c_int
assert lib.hang_mspti_start(str(path).encode(), 16384) == 0
for _ in range(3):
    tensor = torch.ones(1024, device=f"npu:{device}") * (rank + 1)
    dist.all_reduce(tensor)
    torch.npu.synchronize(device)
    assert float(tensor[0]) == 3.0
assert lib.hang_mspti_flush() == 0
time.sleep(0.5)
_, events = read_ring(path)
print(report(path, tail=6), flush=True)
assert any(e["kind"] == "hccl_enter" for e in events), "no HCCL callback"
assert any(e["kind"] == "hccl_exit" for e in events), "no HCCL exit"
assert any(e["kind"] == "hccl_done" for e in events), "no completed HCCL activity"
assert any(e["detail"] for e in events if e["kind"] == "hccl_done"), "no HCCL group name"
lib.hang_mspti_stop()
dist.destroy_process_group()
print(f"HCCL unified ring rank {rank} passed", flush=True)
