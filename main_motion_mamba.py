"""Training entrypoint for MotionMamba — mirrors main.py (StateMachineGPT's
entrypoint), with a separate checkpoint prefix so the two don't overwrite
each other. See Task 1.6 in doc/Mamba_Implementation_Task_Breakdown.docx:
this is a correctness check ("does the loss curve behave sensibly?"), not
the final, full-budget training run.

No hardcoded data path -- --data-dir takes whatever directory you actually
have the data in, so this runs the same way on Kaggle, Colab, or local.

USAGE
------
    !python main_motion_mamba.py \\
        --data-dir /content/data \\
        --hf-repo-id <your-username>/<gpsm-checkpoint-repo-name>

--hf-repo-id is optional; omit it to just checkpoint locally. It needs an
HF token available the way src/gpsm/utils/hub.py's _hub_token() resolves it
(Kaggle Secret, or an HF_TOKEN env var -- set that yourself on Colab, there
is no Kaggle-Secrets equivalent there).
Run `python main_motion_mamba.py --help` for every option.
"""
from __future__ import annotations

import argparse

from src.gpsm.ssm.motion_mamba_trainer import MotionMambaTrainer
from torch.utils.data import DataLoader
from src.gpsm.utils.dataset import MultiSMPLXNPZNextPoseDataset
from src.gpsm.utils.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Folder of raw mocap .npz, searched recursively -- wherever "
             "you actually have the data (e.g. /kaggle/input/... on "
             "Kaggle, /content/data on Colab). No default: a wrong silent "
             "default is exactly the bug this replaces.",
    )
    parser.add_argument(
        "--ckpt-dir", default="checkpoints",
        help="Where the checkpoint is written locally (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-repo-id", default=None,
        help="If given, push the best checkpoint here (<username>/<name>) "
             "after training. Needs an HF token -- see "
             "src/gpsm/utils/hub.py's _hub_token(). Omit to skip publishing.",
    )
    parser.add_argument(
        "--hf-public", action="store_true",
        help="Create the Hugging Face repo as public. Default is private.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Architecture + core hyperparameters: configs/motion_mamba.yaml, not
    # hardcoded here -- single source of truth, shared with
    # main_motion_mamba_ddp.py / main_motion_mamba_sft.py /
    # main_motion_mamba_sft_ddp.py, so they can't drift apart from each other.
    config = load_yaml_config("configs/motion_mamba.yaml")

    # Same dataset call as main.py, so this trains on the same data
    # (strict_dim=False keeps only the files matching the first-loaded file's
    # width — in practice the three state_dim=159 files; see
    # doc/GPSM_variable_dimension_handling.docx for why that happens).
    ds = MultiSMPLXNPZNextPoseDataset(
        args.data_dir,
        k=8,
        feature_set="poses+trans",
        include_betas=False,
        include_expression=False,
        include_face=False,
        prefer_joints_if_available=False,
        normalize=True,
        preload=False,
        strict_dim=False,
    )

    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    config["state_dim"] = ds.D
    print(f"Using state_dim={ds.D} (from dataset), {len(ds)} samples across {ds.file_stats()['num_files']} files")

    best_path = f"{args.ckpt_dir}/best_motion_mamba"
    trainer = MotionMambaTrainer(config, ckpt_dir=args.ckpt_dir)
    history = trainer.train(
        train_loader,
        val_loader=train_loader,
        epochs=args.epochs,
        log_every=1,
        best_path=best_path,
    )

    print("train_loss per epoch:", [f"{v:.6f}" for v in history["train_loss"]])

    if args.hf_repo_id:
        # Separate, optional step -- publishing a checkpoint is a decision
        # distinct from training one, same reasoning as the DDP scripts.
        from src.gpsm.utils.hub import push_checkpoint_to_hub
        push_checkpoint_to_hub(best_path, args.hf_repo_id, private=not args.hf_public)


if __name__ == "__main__":
    main()
