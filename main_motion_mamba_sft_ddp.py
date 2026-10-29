"""
Distributed (multi-GPU) MotionMamba SFT entrypoint — for Kaggle's dual T4
(T4 x2) or any multi-GPU machine.

Adapts main_motion_mamba_ddp.py's design to the SFT phase (free-running
multi-step rollout) the way sft_main_ddp.py adapts its own non-SFT sibling
-- see src/gpsm/ssm/motion_mamba_sft_trainer.py's module docstring for why
multi-step rollout is a different training objective from single-step
teacher forcing.

Does NOT require a pretrained checkpoint -- pretraining was decided to be
skipped in favor of starting SFT directly from a fresh, randomly-
initialized model (architecture from configs/motion_mamba.yaml). Pass
--base-checkpoint if you ever DO want to warm-start from an existing
pretrained checkpoint instead; resuming an in-progress SFT run (e.g. a
later chunk/session) still works automatically either way.

Same file-sharding rationale and deadlock fix as main_motion_mamba_ddp.py:
each GPU gets a disjoint, contiguous slice of the *cached* file list, and
every rank is capped to the cross-rank MINIMUM batch count per epoch so all
ranks call backward() the same number of times.

Loads train.pt / val.pt written by preprocess_motion_mamba_cache.py -- no
.npz scanning here.

How to confirm both GPUs are actually being used, and which files each one
got: watch the cell output for lines like —

    [DDP] rank 0/1 ready on cuda:0 (Tesla T4)
    [DDP] rank 1/1 ready on cuda:1 (Tesla T4)
    [DDP] all 2 ranks initialized — training is running across 2 GPUs.
    [DDP] rank 0 (cuda:0, Tesla T4): train files[0, 12] -> 930 samples
    [DDP] rank 1 (cuda:1, Tesla T4): train files[12, 24] -> 912 samples
    [DDP] per-rank train batch counts: [7, 7] -> capped to 7 steps/epoch

USAGE (Kaggle notebook cell)
------------------------------
    !python main_motion_mamba_sft_ddp.py \\
        --cache-dir /kaggle/input/motion-mamba-cache \\
        --train-file train_1.pt \\
        --hf-repo-id <your-username>/<gpsm-sft-checkpoint-repo-name>

--train-file defaults to "train.pt" -- if the cache was built with
--frames-per-shard and you only have one shard, you MUST pass --train-file
explicitly, or it looks for a plain "train.pt" that doesn't exist. If you
forget, the error now lists which train*.pt files are actually there.

No --base-checkpoint needed for a fresh start. Add it only to warm-start
from an existing pretrained checkpoint instead.

--hf-repo-id is optional; omit it to just checkpoint locally. It needs an
HF_TOKEN Kaggle Secret (write access) -- see src/gpsm/utils/hub.py.
Run `python main_motion_mamba_sft_ddp.py --help` for every option.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src.gpsm.ssm.motion_mamba_sft_trainer import MotionMambaSFTTrainer
from src.gpsm.experiment_model.distributed import setup_ddp, cleanup_ddp
from src.gpsm.utils.cached_dataset import CachedSFTDataset
from src.gpsm.utils.logger import logger
from src.gpsm.utils.utils import load_yaml_config

# Run-setting DEFAULTS: configs/motion_mamba_sft.yaml, not hardcoded here --
# shared with main_motion_mamba_sft.py. Still overridable per-run via the
# CLI flags below (yaml sets the default, flags take precedence).
SFT_CONFIG = load_yaml_config("configs/motion_mamba_sft.yaml")


def _resolve_train_path(cache_dir: str, train_file: str) -> str:
    """Full path to the requested train file, or a clear, actionable error
    listing what's actually in --cache-dir instead of a bare
    FileNotFoundError -- this exact confusion (--train-file left at its
    "train.pt" default when only a shard like "train_1.pt" was downloaded)
    has happened once already and gave no hint what to fix."""
    path = Path(cache_dir) / train_file
    if path.exists():
        return str(path)
    available = sorted(p.name for p in Path(cache_dir).glob("train*.pt"))
    if available:
        raise FileNotFoundError(
            f"{path} not found. Files actually in {cache_dir}: {available}. "
            f"If the cache was built with --frames-per-shard, pass the exact "
            f"shard via --train-file, e.g. --train-file {available[0]}."
        )
    raise FileNotFoundError(
        f"{path} not found, and no train*.pt files exist in {cache_dir} at "
        f"all -- check --cache-dir points at the right folder."
    )


def split_range(start: int, end: int, world_size: int, rank: int):
    """Split [start, end) into `world_size` contiguous, near-equal chunks
    and return the one for `rank`. Any remainder files go to the first few
    ranks, one each, so chunk sizes differ by at most 1 file. Same helper
    as main_motion_mamba_ddp.py's, duplicated rather than cross-imported
    since these are meant to be self-contained entry points."""
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

    sft_checkpoint = f"{args.ckpt_dir}/best_motion_mamba_sft"
    if os.path.exists(f"{sft_checkpoint}.safetensors"):
        # Resuming an in-progress SFT run -- reuse ITS saved config so the
        # fresh model MotionMambaSFTTrainer builds matches the saved
        # weights exactly (load_safetensors requires an identical config
        # and raises otherwise).
        resume_from = sft_checkpoint
        with open(f"{resume_from}.json", "r") as f:
            config = json.load(f)["config"]
        if rank == 0:
            logger.info(f"[DDP] resuming SFT from: {resume_from}")
    elif args.base_checkpoint and os.path.exists(f"{args.base_checkpoint}.safetensors"):
        # Optional warm start from an existing pretrained checkpoint, if
        # --base-checkpoint was given.
        resume_from = args.base_checkpoint
        with open(f"{resume_from}.json", "r") as f:
            config = json.load(f)["config"]
        if rank == 0:
            logger.info(f"[DDP] starting SFT from pretrained checkpoint: {resume_from}")
    else:
        # Fresh start, no pretraining phase: architecture comes from
        # configs/motion_mamba.yaml, state_dim filled in below once the
        # dataset is loaded. No block_size check here, unlike
        # sft_main_ddp.py's StateMachineGPT version -- an SSM has no such
        # concept.
        from src.gpsm.utils.utils import load_yaml_config
        resume_from = None
        config = load_yaml_config("configs/motion_mamba.yaml")
        if rank == 0:
            logger.info("[DDP] starting SFT from scratch (no pretrained checkpoint) -- "
                        "architecture from configs/motion_mamba.yaml")

    train_path = _resolve_train_path(args.cache_dir, args.train_file)

    probe_train = CachedSFTDataset(train_path, n_frames=args.n_frames)
    train_range = split_range(0, len(probe_train._files), world_size, rank)
    del probe_train

    train_ds = CachedSFTDataset(train_path, n_frames=args.n_frames, file_range=train_range)

    if "state_dim" not in config:
        config["state_dim"] = train_ds.state_dim
    if "control_dim" not in config:
        config["control_dim"] = train_ds.control_dim

    train_loader = DataLoader(train_ds, batch_size=args.batch_size_per_gpu, shuffle=True, drop_last=True)
    train_counts = _gathered_batch_counts(train_loader, rank, world_size)
    max_train_steps = min(train_counts) if min(train_counts) > 0 else None

    # val.pt is NOT sharded (see preprocess_motion_mamba_cache.py), so it
    # can be far bigger than any single train shard -- disproportionately
    # expensive to even load, let alone evaluate, every epoch while
    # training in small chunks. --no-val skips constructing it (and its
    # own DDP file-range split / batch-count sync) entirely, not just
    # skips scoring it. Do a separate validation pass once all chunks are
    # trained instead.
    val_loader = None
    if not args.no_val:
        val_path = f"{args.cache_dir}/val.pt"
        probe_val = CachedSFTDataset(val_path, n_frames=args.n_frames)
        val_range = split_range(0, len(probe_val._files), world_size, rank)
        del probe_val
        val_ds = CachedSFTDataset(val_path, n_frames=args.n_frames, file_range=val_range)
        if val_ds.state_dim != config["state_dim"] or val_ds.control_dim != config["control_dim"]:
            raise ValueError(
                f"Cache state_dim={val_ds.state_dim}/control_dim={val_ds.control_dim} does not "
                f"match the resumed checkpoint's state_dim={config['state_dim']}/"
                f"control_dim={config['control_dim']} -- check val.pt was built from the same "
                "preprocessing run (control on/off, same horizons) as the train shard."
            )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size_per_gpu, shuffle=False, drop_last=True)
        val_counts = _gathered_batch_counts(val_loader, rank, world_size)

        msg = (f"[DDP] rank {rank} (cuda:{rank}, {torch.cuda.get_device_name(rank)}): "
               f"train files{list(train_range)} -> {len(train_ds)} samples, "
               f"val files{list(val_range)} -> {len(val_ds)} samples")
        print(msg, flush=True)
        logger.info(msg)
        if rank == 0:
            logger.info(f"[DDP] per-rank train batch counts: {train_counts} -> capped to {max_train_steps} steps/epoch")
            logger.info(f"[DDP] per-rank val batch counts: {val_counts} -> capped to {min(val_counts)} steps/epoch")
    else:
        msg = (f"[DDP] rank {rank} (cuda:{rank}, {torch.cuda.get_device_name(rank)}): "
               f"train files{list(train_range)} -> {len(train_ds)} samples, val SKIPPED (--no-val)")
        print(msg, flush=True)
        logger.info(msg)
        if rank == 0:
            logger.info(f"[DDP] per-rank train batch counts: {train_counts} -> capped to {max_train_steps} steps/epoch")

    trainer = MotionMambaSFTTrainer(config, ckpt_dir=args.ckpt_dir, pretrained_checkpoint=resume_from)

    # setup_ddp() already called torch.cuda.set_device(rank), so the model
    # trainer just built should already be on this rank's GPU -- reassigned
    # explicitly here anyway so it's unambiguous rather than relying on
    # "current device" resolution.
    trainer.device = torch.device(f"cuda:{rank}")
    trainer.model.device = trainer.device
    trainer.model.to(trainer.device)
    trainer.model = DDP(trainer.model, device_ids=[rank])

    # Push on EVERY new-best checkpoint, not just once after all epochs --
    # a session that dies mid-run (the exact failure mode a 20-epoch,
    # hours-long job is prone to) would otherwise leave nothing on HF at
    # all, even though every earlier epoch's improvement was saved locally
    # the whole time. push_checkpoint_to_hub() only runs rank 0 ever calls
    # it (same guard the trainer already uses for local saves).
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
        best_path=sft_checkpoint,
        max_steps_per_epoch=max_train_steps,
        on_checkpoint_saved=on_checkpoint_saved,
    )

    cleanup_ddp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir", default="/kaggle/input/motion-mamba-cache",
        help="Folder holding the train shard(s) + val.pt, written by "
             "preprocess_motion_mamba_cache.py (default: %(default)s)",
    )
    parser.add_argument(
        "--train-file", default="train.pt",
        help="Which train file under --cache-dir to use this run -- if the "
             "cache was built with --frames-per-shard, pass one shard, e.g. "
             "'train_3.pt', to train on that portion only (move to the next "
             "shard next session). Defaults to 'train.pt' for an "
             "unsharded/single-file cache.",
    )
    parser.add_argument(
        "--base-checkpoint", default=None,
        help="Optional: a pretrained checkpoint prefix to warm-start SFT "
             "from. Only used if no SFT checkpoint already exists at "
             "--ckpt-dir to resume from. Omit to start from scratch "
             "(architecture from configs/motion_mamba.yaml) -- pretraining "
             "is not required.",
    )
    parser.add_argument(
        "--ckpt-dir", default="/kaggle/working/checkpoints",
        help="Where SFT checkpoints are written locally, and where an "
             "existing best_motion_mamba_sft is looked for to resume "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--hf-repo-id", default=None,
        help="If given, push the best SFT checkpoint here (<username>/<name>) "
             "after training -- rank 0 only. Needs an HF_TOKEN Kaggle "
             "Secret with write access. Omit to skip publishing.",
    )
    parser.add_argument(
        "--hf-public", action="store_true",
        help="Create the Hugging Face repo as public. Default is private.",
    )
    parser.add_argument("--n-frames", type=int, default=SFT_CONFIG["n_frames"],
                         help="Rollout length supervised per sample (default: %(default)s, "
                              "from configs/motion_mamba_sft.yaml)")
    parser.add_argument("--epochs", type=int, default=SFT_CONFIG["epochs"])
    parser.add_argument("--batch-size-per-gpu", type=int, default=SFT_CONFIG["batch_size_per_gpu"])
    parser.add_argument(
        "--no-val", action="store_true",
        help="Skip loading and evaluating val.pt entirely during training. "
             "val.pt is NOT sharded, so it can be much bigger than a single "
             "train shard -- do a separate validation pass once all chunks "
             "are trained instead. With this set, checkpointing falls back "
             "to train_loss as the 'best' metric (MotionMambaSFTTrainer's "
             "existing behaviour whenever val_loader is None).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(
            f"Found {world_size} GPU(s) -- this script is for multi-GPU distributed "
            "training. Use main_motion_mamba_sft.py for single-GPU/CPU."
        )
    print(
        f"[DDP] launching {world_size} processes (1 per GPU): "
        f"{[torch.cuda.get_device_name(i) for i in range(world_size)]}",
        flush=True,
    )
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
