"""Run with torchrun --standalone --nproc_per_node=4 on an idle allocation."""

import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
print("DEVICE", rank, torch.cuda.get_device_name(), flush=True)
print(
    "DRIVER_LIB",
    rank,
    [line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/libcuda." in line][:1],
    flush=True,
)
dist.init_process_group("nccl", timeout=timedelta(seconds=60))
assert dist.get_world_size() == 4, "This probe expects four ranks"
x = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
assert x.item() == 10, x
out = [None] * 4
dist.all_gather_object(out, {"rank": rank})
assert sorted(d["rank"] for d in out) == [0, 1, 2, 3]


@triton.jit
def add_one(x, y, B: tl.constexpr):
    i = tl.arange(0, B)
    tl.store(y + i, tl.load(x + i) + 1)


x = torch.arange(256, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
add_one[(1,)](x, y, 256)
torch.cuda.synchronize()
assert torch.equal(y, x + 1)
print("NCCL_TRITON_PASS", rank, flush=True)
dist.destroy_process_group()
