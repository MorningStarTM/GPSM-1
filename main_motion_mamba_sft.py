"""SFT phase: fine-tune MotionMamba on multi-step autoregressive rollout,
instead of single-step teacher forcing. See MotionMambaSFTTrainer /
CachedSFTDataset for why this is a distinct training objective, and
src/gpsm/ssm/motion_mamba_sft_trainer.py's module docstring for the full
reasoning.

Does NOT require a pretrained checkpoint -- pretraining (main_motion_mamba.py
/ main_motion_mamba_ddp.py) was decided to be skipped in favor of starting
SFT directly from a fresh, randomly-initialized model. If no SFT checkpoint
exists yet either, this starts from scratch using configs/motion_mamba.yaml
for the architecture. Pass --base-checkpoint if you ever DO want to warm-start
from an existing pretrained checkpoint instead.

Resuming works the same way either way: if best_motion_mamba_sft already
exists (e.g. a later chunk/session), this picks it back up automatically --
that part is unaffected by skipping pretraining.

Reads from the same train.pt / val.pt cache written by
scripts/preprocess_motion_mamba_cache.py. No hardcoded Kaggle path --
--cache-dir takes whatever directory you actually downloaded it to, so this
runs the same way on Kaggle, Colab, or local.

USAGE
------
    !python main_motion_mamba_sft.py \\
        --cache-dir /content/motion-mamba-cache \\
        --train-file train_1.pt \\
        --hf-repo-id <your-username>/<gpsm-sft-checkpoint-repo-name>

--train-file defaults to "train.pt" -- if the cache was built with
--frames-per-shard (scripts/preprocess_motion_mamba_cache.py) and you only
downloaded one shard, you MUST pass --train-file explicitly (e.g.
"train_1.pt"), or it will look for a plain "train.pt" that was never
downloaded and fail. If you forget, the error now lists which train*.pt
files actually exist in --cache-dir, instead of a bare "not found".

--hf-repo-id is optional; omit it to just checkpoint locally. It needs an
HF_TOKEN available the way src/gpsm/utils/hub.py's _hub_token() resolves it
(Kaggle Secret, or an HF_TOKEN env var -- set that yourself on Colab, there
is no Kaggle-Secrets equivalent there).
Run `python main_motion_mamba_sft.py --help` for every option.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from torch.utils.data import DataLoader

from src.gpsm.ssm.motion_mamba_sft_trainer import MotionMambaSFTTrainer
from src.gpsm.utils.cached_dataset import CachedSFTDataset
from src.gpsm.utils.utils import load_yaml_config

# Run-setting DEFAULTS: configs/motion_mamba_sft.yaml, not hardcoded here --
# shared with main_motion_mamba_sft_ddp.py. Still overridable per-run via
# the CLI flags below (yaml sets the default, flags take precedence).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir", required=True,
        help="Folder holding the train shard(s) + val.pt, written by "
             "preprocess_motion_mamba_cache.py -- wherever you actually "
             "downloaded it to (e.g. /kaggle/input/motion-mamba-cache on "
             "Kaggle, /content/motion-mamba-cache on Colab). No default: "
             "a wrong silent default is exactly the bug this replaces.",
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
        "--ckpt-dir", default="checkpoints",
        help="Where SFT checkpoints are written locally, and where an "
             "existing best_motion_mamba_sft is looked for to resume "
             "(default: %(default)s)",
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
        "--hf-repo-id", default=None,
        help="If given, push the best SFT checkpoint here (<username>/<name>) "
             "after training. Needs an HF token -- see src/gpsm/utils/hub.py's "
             "_hub_token(). Omit to skip publishing.",
    )
    parser.add_argument(
        "--hf-public", action="store_true",
        help="Create the Hugging Face repo as public. Default is private.",
    )
    parser.add_argument("--n-frames", type=int, default=SFT_CONFIG["n_frames"],
                         help="Rollout length supervised per sample (default: %(default)s, "
                              "from configs/motion_mamba_sft.yaml)")
    parser.add_argument("--epochs", type=int, default=SFT_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=SFT_CONFIG["batch_size"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sft_checkpoint = f"{args.ckpt_dir}/best_motion_mamba_sft"

    if os.path.exists(f"{sft_checkpoint}.safetensors"):
        # Resuming an in-progress SFT run -- reuse ITS saved config so the
        # fresh model MotionMambaSFTTrainer builds matches the saved
        # weights exactly (load_safetensors requires an identical config
        # and raises otherwise).
        resume_from = sft_checkpoint
        with open(f"{resume_from}.json", "r") as f:
            config = json.load(f)["config"]
        print(f"Resuming SFT from: {resume_from}")
    elif args.base_checkpoint and os.path.exists(f"{args.base_checkpoint}.safetensors"):
        # Optional warm start from an existing pretrained checkpoint, if
        # --base-checkpoint was given.
        resume_from = args.base_checkpoint
        with open(f"{resume_from}.json", "r") as f:
            config = json.load(f)["config"]
        print(f"Starting SFT from pretrained checkpoint: {resume_from}")
    else:
        # Fresh start, no pretraining phase: architecture comes from
        # configs/motion_mamba.yaml, state_dim filled in below once the
        # dataset is loaded. No block_size check here, unlike sft_main.py's
        # StateMachineGPT version -- an SSM has no such concept.
        resume_from = None
        config = load_yaml_config("configs/motion_mamba.yaml")
        print("Starting SFT from scratch (no pretrained checkpoint) -- "
              "architecture from configs/motion_mamba.yaml")

    train_ds = CachedSFTDataset(_resolve_train_path(args.cache_dir, args.train_file), n_frames=args.n_frames)
    val_ds = CachedSFTDataset(f"{args.cache_dir}/val.pt", n_frames=args.n_frames)

    if "state_dim" not in config:
        config["state_dim"] = train_ds.state_dim
    elif train_ds.state_dim != config["state_dim"]:
        raise ValueError(
            f"Cache state_dim={train_ds.state_dim} does not match the resumed "
            f"checkpoint's state_dim={config['state_dim']} -- check the cache "
            "was built from the same data that checkpoint was trained on."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"SFT dataset: state_dim={train_ds.state_dim}, n_frames={args.n_frames}, "
          f"train={len(train_ds)} samples across {train_ds.file_stats()['num_files']} files, "
          f"val={len(val_ds)} samples across {val_ds.file_stats()['num_files']} files")

    trainer = MotionMambaSFTTrainer(config, ckpt_dir=args.ckpt_dir, pretrained_checkpoint=resume_from)
    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        log_every=1,
        best_path=sft_checkpoint,
    )

    print("train_loss per epoch:", [f"{v:.6f}" for v in history["train_loss"]])
    print("val_loss per epoch:", [f"{v:.6f}" for v in history["val_loss"]])

    if args.hf_repo_id:
        # Separate, optional step -- publishing a checkpoint is a decision
        # distinct from training one, same reasoning as
        # main_motion_mamba_ddp.py / main_motion_mamba_sft_ddp.py.
        from src.gpsm.utils.hub import push_checkpoint_to_hub
        push_checkpoint_to_hub(sft_checkpoint, args.hf_repo_id, private=not args.hf_public)


if __name__ == "__main__":
    main()
