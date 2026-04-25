from __future__ import annotations

import os
from typing import List, Sequence, TypeVar

import torch
import torch.distributed as dist

from refactor_v2.common import DistInfo

T = TypeVar("T")


def get_dist_info() -> DistInfo:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def maybe_init_distributed(backend: str | None = None) -> DistInfo:
    info = get_dist_info()
    if info.world_size > 1 and not dist.is_initialized():
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=info.rank,
            world_size=info.world_size,
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    return info


def barrier_if_needed() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(info: DistInfo) -> bool:
    return info.rank == 0


def split_items_for_rank(items: Sequence[T], info: DistInfo) -> List[T]:
    if info.world_size <= 1:
        return list(items)
    return [item for idx, item in enumerate(items) if idx % info.world_size == info.rank]
