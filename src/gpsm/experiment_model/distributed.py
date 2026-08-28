"""
Multi-GPU (DistributedDataParallel) helpers for training on more than one
GPU — e.g. Kaggle's dual T4 (T4 x2) notebooks.

SMTrainer/SFTTrainer already had `_is_ddp()`/`_module()`/`_rank()` helpers
that assumed `self.model` might be a DistributedDataParallel instance, but
nothing ever actually launched multiple processes or wrapped the model —
that wiring lives here, plus the loud per-GPU confirmation logging needed to
actually see that training is running on every GPU, not just GPU 0.
"""

import os
import torch
import torch.distributed as dist
from src.gpsm.utils.logger import logger


def setup_ddp(rank: int, world_size: int, master_addr: str = "127.0.0.1", master_port: str = "29500") -> None:
    """Initialize this process's slot in the distributed group and pin it to
    its own GPU. Must be called once at the very start of each spawned
    worker process, before building the model/optimizer."""
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Loud, per-rank confirmation that this specific GPU is actually in the
    # training group. Uses plain print() (flushed) in addition to the
    # logger, since notebook environments don't always surface log sinks
    # from spawned child processes the same way they do the main process —
    # print() to the inherited stdout is the more reliable of the two here.
    name = torch.cuda.get_device_name(rank)
    msg  = f"[DDP] rank {rank}/{world_size - 1} ready on cuda:{rank} ({name})"
    print(msg, flush=True)
    logger.info(msg)

    dist.barrier()  # every rank waits here so the per-rank lines above all land before training starts
    if rank == 0:
        confirm = f"[DDP] all {world_size} ranks initialized — training is running across {world_size} GPUs."
        print(confirm, flush=True)
        logger.info(confirm)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def reduce_mean(value: float, device: torch.device) -> float:
    """
    All-reduce (average) a per-rank scalar (e.g. this epoch's local mean
    loss) across every rank in the distributed group, so every rank ends up
    logging the identical, correct *global* value instead of its own local
    shard's approximation.

    This matters beyond just accurate logging: early stopping compares this
    value against a running best and decides whether to `break` out of the
    training loop. If ranks compared un-synced local values they could reach
    different decisions on different epochs — one rank stops iterating while
    the others keep calling forward/backward, and since DDP's gradient
    all-reduce requires every rank to participate in every step, that
    deadlocks the whole job. Syncing the metric keeps the decision, and
    hence the loop, identical across all ranks.

    No-ops (returns `value` unchanged) when distributed training isn't
    active, so it's safe to call unconditionally from single-GPU code too.
    """
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / dist.get_world_size()
