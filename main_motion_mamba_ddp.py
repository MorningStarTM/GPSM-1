"""
Distributed (multi-GPU) MotionMamba pretraining entrypoint — for Kaggle's
dual T4 (T4 x2) or any multi-GPU machine.

Mirrors sft_main_ddp.py's design exactly (same file-sharding rationale, same
deadlock fix) — see that file's module docstring for the full explanation.
Short version: each GPU gets a disjoint, contiguous slice of the *cached*
file list (not a DistributedSampler splitting samples), so no rank ever
touches data another rank already owns, and every rank is capped to the
cross-rank MINIMUM batch count per epoch so all ranks call backward() the
same number of times — otherwise the rank that runs out of batches first
leaves the others hanging on the next gradient all-reduce and the job
deadlocks.

Loads train.pt / val.pt written by preprocess_motion_mamba_cache.py — no
.npz scanning here, so this never burns GPU-session time on CPU-bound work
(see that script's docstring for the full reasoning). Point CACHE_DIR at a
Kaggle dataset published by that script (e.g. "motion-mamba-cache").

How to confirm both GPUs are actually being used, and which files each one
got: watch the cell output for lines like —

    [DDP] rank 0/1 ready on cuda:0 (Tesla T4)
    [DDP] rank 1/1 ready on cuda:1 (Tesla T4)
    [DDP] all 2 ranks initialized — training is running across 2 GPUs.
    [DDP] rank 0 (cuda:0, Tesla T4): assigned files[0:12] -> 930 samples
    [DDP] rank 1 (cuda:1, Tesla T4): assigned files[12:24] -> 912 samples
    [DDP] per-rank train batch counts: [7, 7] -> capped to 7 steps/epoch

Usage (Kaggle notebook cell):
    !python main_motion_mamba_ddp.py
"""

import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.gpsm.ssm.motion_mamba_trainer import MotionMambaTrainer
from src.gpsm.experiment_model.distributed import setup_ddp, cleanup_ddp
from src.gpsm.utils.cached_dataset import CachedWindowDataset
from src.gpsm.utils.plotting import plot_training_history
from src.gpsm.utils.logger import logger

CACHE_DIR = "/kaggle/input/motion-mamba-cache"
CKPT_DIR = "/kaggle/working/checkpoints"
EPOCHS = 15
BATCH_SIZE_PER_GPU = 128
ACC_THRESHOLD = 0.5  # see MotionMambaTrainer._threshold_accuracy's docstring before trusting the default

CONFIG = {
    "n_embd": 256, "n_layers": 4, "learning_rate": 1e-4, "dropout": 0.1,
    "loss_scale": 1.0, "d_state": 16, "d_conv": 4, "expand": 2,
    "dt_rank": math.ceil(256 / 16),
}


def split_range(start: int, end: int, world_size: int, rank: int):
    """Split [start, end) into `world_size` contiguous, near-equal chunks
    and return the one for `rank`. Any remainder files go to the first few
    ranks, one each, so chunk sizes differ by at most 1 file. Same helper
    as sft_main_ddp.py's, duplicated rather than cross-imported since these
    are meant to be self-contained entry points."""
    total = end - start
    per_rank = total // world_size
    remainder = total % world_size
    rank_start = start + rank * per_rank + min(rank, remainder)
    rank_end = rank_start + per_rank + (1 if rank < remainder else 0)
    return rank_start, rank_end


def _gathered_batch_counts(loader, rank: int, world_size: int):
    """All-gather every rank's local batch count, for logging and for
    computing the cross-rank minimum (see module docstring)."""
    local = torch.tensor(len(loader), dtype=torch.long, device=f"cuda:{rank}")
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [int(t.item()) for t in gathered]


def train_worker(rank: int, world_size: int) -> None:
    setup_ddp(rank, world_size)  # prints/logs the per-GPU "ready" banner

    train_path = f"{CACHE_DIR}/train.pt"
    val_path = f"{CACHE_DIR}/val.pt"

    # How many files each dataset actually has, so the split is sized
    # correctly per split rather than assuming train/val have equal counts.
    probe_train = CachedWindowDataset(train_path)
    probe_val = CachedWindowDataset(val_path)
    train_range = split_range(0, len(probe_train._files), world_size, rank)
    val_range = split_range(0, len(probe_val._files), world_size, rank)
    del probe_train, probe_val

    train_ds = CachedWindowDataset(train_path, file_range=train_range)
    val_ds = CachedWindowDataset(val_path, file_range=val_range)

    msg = (f"[DDP] rank {rank} (cuda:{rank}, {torch.cuda.get_device_name(rank)}): "
           f"train files{list(train_range)} -> {len(train_ds)} samples, "
           f"val files{list(val_range)} -> {len(val_ds)} samples")
    print(msg, flush=True)
    logger.info(msg)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_PER_GPU, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, drop_last=True)

    # Different file ranges -> likely different sample counts -> different
    # local batch counts. Gather every rank's count (for logging) and sync
    # to the cross-rank minimum so every rank runs the same number of
    # steps per epoch (see module docstring).
    train_counts = _gathered_batch_counts(train_loader, rank, world_size)
    val_counts = _gathered_batch_counts(val_loader, rank, world_size)
    max_train_steps = min(train_counts) if min(train_counts) > 0 else None
    max_val_steps = min(val_counts) if min(val_counts) > 0 else None

    if rank == 0:
        logger.info(f"[DDP] per-rank train batch counts: {train_counts} -> capped to {max_train_steps} steps/epoch")
        logger.info(f"[DDP] per-rank val batch counts: {val_counts} -> capped to {max_val_steps} steps/epoch")

    config = dict(CONFIG)
    config["state_dim"] = train_ds.state_dim

    trainer = MotionMambaTrainer(config, ckpt_dir=CKPT_DIR)
    trainer.device = torch.device(f"cuda:{rank}")
    trainer.model.device = trainer.device
    trainer.model.to(trainer.device)
    trainer.model = DDP(trainer.model, device_ids=[rank])

    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        log_every=1,
        best_path=f"{CKPT_DIR}/best_motion_mamba",
        acc_threshold=ACC_THRESHOLD,
        max_steps_per_epoch=max_train_steps,
    )

    if rank == 0:
        plot_training_history(
            history, out_path="/kaggle/working/training_history.png",
            title="MotionMamba pretraining (DDP, 2x T4)", acc_threshold=ACC_THRESHOLD,
        )

    cleanup_ddp()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(
            f"Found {world_size} GPU(s) -- this script is for multi-GPU distributed "
            "training. Use main_motion_mamba.py for single-GPU/CPU."
        )
    print(
        f"[DDP] launching {world_size} processes (1 per GPU): "
        f"{[torch.cuda.get_device_name(i) for i in range(world_size)]}",
        flush=True,
    )
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
