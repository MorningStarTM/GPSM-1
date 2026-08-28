"""
Distributed (multi-GPU) SFT training entrypoint — for Kaggle's dual T4
(T4 x2) or any multi-GPU machine. Spawns one process per GPU
(torch.multiprocessing.spawn), each driving its own SFTTrainer with a
DistributedDataParallel-wrapped model on its own device, and a
DistributedSampler splitting the dataset across GPUs.

How to confirm both GPUs are actually being used: watch the cell output for
one line per GPU on startup —

    [DDP] rank 0/1 ready on cuda:0 (Tesla T4)
    [DDP] rank 1/1 ready on cuda:1 (Tesla T4)
    [DDP] all 2 ranks initialized — training is running across 2 GPUs.

If you only ever see rank 0, something is wrong (e.g. only 1 GPU visible) —
see distributed.py for how these lines are produced.

Usage (Kaggle notebook cell):
    !python sft_main_ddp.py
"""

import json

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.gpsm.experiment_model.sm_gpt_sft_trainer import SFTTrainer
from src.gpsm.experiment_model.distributed import setup_ddp, cleanup_ddp
from src.gpsm.utils.dataset import MultiSMPLXNPZSFTDataset
from src.gpsm.utils.logger import logger

PRETRAINED_CHECKPOINT = "checkpoints/best_sm"   # base model to fine-tune from
N_FRAMES               = 8                        # rollout length supervised per sample
EPOCHS                  = 20
BATCH_SIZE_PER_GPU      = 64   # global batch size = this * world_size (number of GPUs)


def train_worker(rank: int, world_size: int) -> None:
    setup_ddp(rank, world_size)   # prints/logs the per-GPU "ready" banner

    with open(f"{PRETRAINED_CHECKPOINT}.json", "r") as f:
        config = json.load(f)["config"]
    assert config["block_size"] >= N_FRAMES + 1, (
        f"block_size={config['block_size']} must be >= N_FRAMES+1={N_FRAMES + 1}"
    )

    ds = MultiSMPLXNPZSFTDataset(
        "data",
        n_frames=N_FRAMES,
        feature_set="poses+trans",        # must match what the pretrained checkpoint used
        include_betas=False,
        include_expression=False,
        include_face=False,
        prefer_joints_if_available=False,
        normalize=True,
        preload=False,
        strict_dim=False,
    )
    if ds.D != config["state_dim"]:
        raise ValueError(
            f"Dataset feature width D={ds.D} does not match pretrained "
            f"state_dim={config['state_dim']} — check feature_set matches what "
            "the base checkpoint was trained with."
        )

    # Splits the dataset across ranks so each GPU trains on a different
    # shard every epoch — this (plus the DDP-wrapped model below) is what
    # actually makes this distributed rather than world_size independent
    # single-GPU runs.
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, sampler=sampler, drop_last=True)

    trainer = SFTTrainer(config, pretrained_checkpoint=PRETRAINED_CHECKPOINT)

    # setup_ddp() already called torch.cuda.set_device(rank), so the model
    # trainer just built should already be on this rank's GPU — reassigned
    # explicitly here anyway so it's unambiguous rather than relying on
    # "current device" resolution.
    trainer.device = torch.device(f"cuda:{rank}")
    trainer.model.device = trainer.device
    trainer.model.to(trainer.device)
    trainer.model = DDP(trainer.model, device_ids=[rank])

    if rank == 0:
        logger.info(
            f"[DDP] global batch size = {BATCH_SIZE_PER_GPU} x {world_size} GPUs "
            f"= {BATCH_SIZE_PER_GPU * world_size}"
        )

    trainer.train(
        train_loader,
        epochs=EPOCHS,
        log_every=1,
        best_path="checkpoints/best_sm_sft",
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
