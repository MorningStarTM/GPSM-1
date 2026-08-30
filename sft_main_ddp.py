"""
Distributed (multi-GPU) SFT training entrypoint — for Kaggle's dual T4
(T4 x2) or any multi-GPU machine.

Unlike a typical DDP setup where every rank builds the *same* dataset and a
DistributedSampler splits sample indices across ranks, this assigns each GPU
a distinct, disjoint FILE range — GPU 0 trains on files[a:b], GPU 1 on
files[b:c], etc. — so no rank ever has to scan/index files another rank is
also going to use. FILE_START/FILE_END (see sft_main.py for the single-GPU
equivalent) picks the overall range for this run; it's split evenly across
however many GPUs are visible.

Because each rank's file range can produce a different number of samples,
every rank is capped to the cross-rank MINIMUM batch count per epoch (see
SFTTrainer.train()'s max_steps_per_epoch) — required so every rank calls
backward() the same number of times per epoch; otherwise the rank that runs
out of batches first leaves the others hanging on the next gradient
all-reduce and the whole job deadlocks.

How to confirm both GPUs are actually being used, and which files each one
got: watch the cell output for lines like —

    [DDP] rank 0/1 ready on cuda:0 (Tesla T4)
    [DDP] rank 1/1 ready on cuda:1 (Tesla T4)
    [DDP] all 2 ranks initialized — training is running across 2 GPUs.
    [DDP] rank 0 (cuda:0, Tesla T4): assigned files[0:250] -> 3312 samples
    [DDP] rank 1 (cuda:1, Tesla T4): assigned files[250:500] -> 3488 samples
    [DDP] per-rank batch counts: [51, 54] -> capped to 51 steps/epoch

Usage (Kaggle notebook cell):
    !python sft_main_ddp.py
"""

import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from src.gpsm.experiment_model.sm_gpt_sft_trainer import SFTTrainer
from src.gpsm.experiment_model.distributed import setup_ddp, cleanup_ddp
from src.gpsm.utils.dataset import MultiSMPLXNPZSFTDataset
from src.gpsm.utils.logger import logger

BASE_CHECKPOINT    = "checkpoints/best_sm"        # only used for the very first chunk
SFT_CHECKPOINT     = "checkpoints/best_sm_sft"     # every later chunk resumes from + overwrites this
N_FRAMES            = 8                              # rollout length supervised per sample
EPOCHS               = 20
BATCH_SIZE_PER_GPU   = 64
DATA_DIR             = "data"
FILE_START           = 0                              # first file index for this whole run (all GPUs combined)
FILE_END             = 500                            # one-past-last file index for this whole run


def split_range(start: int, end: int, world_size: int, rank: int):
    """Split [start, end) into `world_size` contiguous, near-equal chunks
    and return the one for `rank`. Any remainder files go to the first few
    ranks, one each, so chunk sizes differ by at most 1 file."""
    total     = end - start
    per_rank  = total // world_size
    remainder = total % world_size
    rank_start = start + rank * per_rank + min(rank, remainder)
    rank_end   = rank_start + per_rank + (1 if rank < remainder else 0)
    return rank_start, rank_end


def train_worker(rank: int, world_size: int) -> None:
    setup_ddp(rank, world_size)   # prints/logs the per-GPU "ready" banner

    resume_from = SFT_CHECKPOINT if os.path.exists(f"{SFT_CHECKPOINT}.safetensors") else BASE_CHECKPOINT
    if rank == 0:
        logger.info(f"[DDP] resuming from: {resume_from}")

    with open(f"{resume_from}.json", "r") as f:
        config = json.load(f)["config"]
    assert config["block_size"] >= N_FRAMES + 1, (
        f"block_size={config['block_size']} must be >= N_FRAMES+1={N_FRAMES + 1}"
    )

    rank_start, rank_end = split_range(FILE_START, FILE_END, world_size, rank)

    ds = MultiSMPLXNPZSFTDataset(
        DATA_DIR,
        n_frames=N_FRAMES,
        feature_set="poses+trans",        # must match what the pretrained checkpoint used
        include_betas=False,
        include_expression=False,
        include_face=False,
        prefer_joints_if_available=False,
        normalize=True,
        preload=False,
        strict_dim=False,
        file_range=(rank_start, rank_end),
    )
    if ds.D != config["state_dim"]:
        raise ValueError(
            f"Dataset feature width D={ds.D} does not match pretrained "
            f"state_dim={config['state_dim']} — check feature_set matches what "
            "the base checkpoint was trained with."
        )

    msg = (f"[DDP] rank {rank} (cuda:{rank}, {torch.cuda.get_device_name(rank)}): "
           f"assigned files[{rank_start}:{rank_end}] -> {len(ds)} samples")
    print(msg, flush=True)
    logger.info(msg)

    # No DistributedSampler needed — each rank's dataset already only covers
    # its own file range, so the data is disjoint across ranks by construction.
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, shuffle=True, drop_last=True)

    # Different file ranges -> likely different sample counts -> different
    # local batch counts. Gather every rank's count (for logging) and sync
    # to the cross-rank minimum so every rank runs the same number of
    # training steps per epoch (see module docstring).
    local_batches = torch.tensor(len(train_loader), dtype=torch.long, device=f"cuda:{rank}")
    gathered      = [torch.zeros_like(local_batches) for _ in range(world_size)]
    dist.all_gather(gathered, local_batches)
    max_steps_per_epoch = int(min(t.item() for t in gathered))

    if rank == 0:
        logger.info(f"[DDP] per-rank batch counts: {[t.item() for t in gathered]} "
                    f"-> capped to {max_steps_per_epoch} steps/epoch")

    trainer = SFTTrainer(config, pretrained_checkpoint=resume_from)

    # setup_ddp() already called torch.cuda.set_device(rank), so the model
    # trainer just built should already be on this rank's GPU — reassigned
    # explicitly here anyway so it's unambiguous rather than relying on
    # "current device" resolution.
    trainer.device = torch.device(f"cuda:{rank}")
    trainer.model.device = trainer.device
    trainer.model.to(trainer.device)
    trainer.model = DDP(trainer.model, device_ids=[rank])

    trainer.train(
        train_loader,
        epochs=EPOCHS,
        log_every=1,
        best_path=SFT_CHECKPOINT,
        max_steps_per_epoch=max_steps_per_epoch,
    )

    cleanup_ddp()


def main() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(
            f"Found {world_size} GPU(s) — this script is for multi-GPU distributed "
            "training. Use sft_main.py for single-GPU/CPU."
        )
    print(
        f"[DDP] launching {world_size} processes (1 per GPU): "
        f"{[torch.cuda.get_device_name(i) for i in range(world_size)]}",
        flush=True,
    )
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
