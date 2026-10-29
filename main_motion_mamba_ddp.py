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
(see that script's docstring for the full reasoning).

How to confirm both GPUs are actually being used, and which files each one
got: watch the cell output for lines like —

    [DDP] rank 0/1 ready on cuda:0 (Tesla T4)
    [DDP] rank 1/1 ready on cuda:1 (Tesla T4)
    [DDP] all 2 ranks initialized — training is running across 2 GPUs.
    [DDP] rank 0 (cuda:0, Tesla T4): train files[0:12] -> 930 samples
    [DDP] rank 1 (cuda:1, Tesla T4): train files[12:24] -> 912 samples
    [DDP] per-rank train batch counts: [7, 7] -> capped to 7 steps/epoch

USAGE (Kaggle notebook cell)
------------------------------
    !python main_motion_mamba_ddp.py \\
        --cache-dir /kaggle/input/motion-mamba-cache \\
        --hf-repo-id <your-username>/<gpsm-checkpoint-repo-name>

--hf-repo-id is optional; omit it to just checkpoint locally. It needs an
HF_TOKEN Kaggle Secret (write access) -- see src/gpsm/utils/hub.py.
Run `python main_motion_mamba_ddp.py --help` for every option.
"""
from __future__ import annotations

import argparse

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
from src.gpsm.utils.utils import load_yaml_config

# configs/motion_mamba.yaml, not hardcoded here -- single source of truth,
# shared with main_motion_mamba.py / main_motion_mamba_sft.py /
# main_motion_mamba_sft_ddp.py.
MODEL_CONFIG = load_yaml_config("configs/motion_mamba.yaml")


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


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_ddp(rank, world_size)  # prints/logs the per-GPU "ready" banner

    train_path = f"{args.cache_dir}/train.pt"
    val_path = f"{args.cache_dir}/val.pt"

    # How many files each split actually has, so the split is sized
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size_per_gpu, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_per_gpu, shuffle=False, drop_last=True)

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

    config = dict(MODEL_CONFIG)
    config["state_dim"] = train_ds.state_dim
    config["control_dim"] = train_ds.control_dim

    trainer = MotionMambaTrainer(config, ckpt_dir=args.ckpt_dir)
    trainer.device = torch.device(f"cuda:{rank}")
    trainer.model.device = trainer.device
    trainer.model.to(trainer.device)
    trainer.model = DDP(trainer.model, device_ids=[rank])

    best_path = f"{args.ckpt_dir}/best_motion_mamba"

    # Push on EVERY new-best checkpoint, not just once after all epochs --
    # a session that dies mid-run would otherwise leave nothing on HF at
    # all, even though every earlier epoch's improvement was saved locally
    # the whole time. Same reasoning as src/gpsm/utils/hub.py keeping
    # push_checkpoint_to_hub apart from the trainer itself: publishing
    # stays opt-in (--hf-repo-id) and the trainer never needs to know
    # anything about HF, just that a callback exists.
    on_checkpoint_saved = None
    if args.hf_repo_id:
        from src.gpsm.utils.hub import push_checkpoint_to_hub

        def on_checkpoint_saved(path):  # noqa: F811
            push_checkpoint_to_hub(path, args.hf_repo_id, private=not args.hf_public)

    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        log_every=1,
        best_path=best_path,
        acc_threshold=args.acc_threshold,
        max_steps_per_epoch=max_train_steps,
        on_checkpoint_saved=on_checkpoint_saved,
    )

    if rank == 0:
        plot_training_history(
            history, out_path=f"{args.ckpt_dir}/../training_history.png",
            title="MotionMamba pretraining (DDP)", acc_threshold=args.acc_threshold,
        )

    cleanup_ddp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir", default="/kaggle/input/motion-mamba-cache",
        help="Folder holding train.pt/val.pt, written by "
             "preprocess_motion_mamba_cache.py (default: %(default)s)",
    )
    parser.add_argument(
        "--ckpt-dir", default="/kaggle/working/checkpoints",
        help="Where to write checkpoints locally (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-repo-id", default=None,
        help="If given, push the best checkpoint here (<username>/<name>) "
             "after training -- rank 0 only. Needs an HF_TOKEN Kaggle "
             "Secret with write access. Omit to skip publishing.",
    )
    parser.add_argument(
        "--hf-public", action="store_true",
        help="Create the Hugging Face repo as public. Default is private.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size-per-gpu", type=int, default=128)
    parser.add_argument(
        "--acc-threshold", type=float, default=0.5,
        help="Tolerance for the threshold-accuracy metric -- see "
             "MotionMambaTrainer._threshold_accuracy's docstring before "
             "trusting the default (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
