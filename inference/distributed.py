from __future__ import annotations

import os
from typing import List, Sequence, TypeVar
import datetime
import torch
import torch.distributed as dist

from utils.common import DistInfo

T = TypeVar("T")


def get_dist_info() -> DistInfo:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return DistInfo(rank=rank, world_size=world_size, local_rank=local_rank)


def maybe_init_distributed(backend: str | None = None) -> DistInfo:
    info = get_dist_info()
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Running more ranks than GPUs packs several ranks onto one card, which is
    # worth doing here because projection is CPU-bound and would otherwise leave
    # the machine idle. NCCL does not support that layout, but the only
    # collective this pipeline performs is a barrier, so gloo covers it.
    oversubscribed = device_count > 0 and info.world_size > device_count

    if info.world_size > 1 and not dist.is_initialized():
        if backend is None or oversubscribed:
            backend = "gloo" if (oversubscribed or not torch.cuda.is_available()) else "nccl"
        dist.init_process_group(
            backend=backend,
            rank=info.rank,
            world_size=info.world_size,
            timeout=datetime.timedelta(hours=2)
        )
    if device_count:
        torch.cuda.set_device(info.local_rank % device_count)
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
