"""Small two-rank HCCL baseline for a separate msprof AICPU experiment."""
import os

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
device = 4 + rank
torch.npu.set_device(device)
dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
                        rank=rank, world_size=2)
for _ in range(10):
    tensor = torch.ones(1024 * 1024, device=f"npu:{device}") * (rank + 1)
    dist.all_reduce(tensor)
    torch.npu.synchronize(device)
    assert float(tensor[0]) == 3.0
dist.destroy_process_group()
print(f"HCCL baseline rank {rank} passed", flush=True)
