"""Train MotionMamba on MotionPrep's output, conditioned on control.

Where main_motion_mamba.py trains the plain next-pose model straight from
raw .npz files, this trains the steerable version: each input frame is the
pose *plus* the control that says where the motion should go from there, and
the model predicts the next pose.

    input  frame t : concat(pose_t, control_t)   69 + 14 = 83 numbers
    output         : pose_{t+1}                  69 numbers

Run it:
    python main_motion_mamba_control.py
    python main_motion_mamba_control.py --epochs 20 --no-control   # baseline
"""

import argparse
import math

from torch.utils.data import DataLoader

from src.gpsm.ssm.motion_mamba_trainer import MotionMambaTrainer
from src.gpsm.utils.control_dataset import ControlWindowDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train control-conditioned MotionMamba.")
    parser.add_argument("--clips", default="processed/with_control", help="MotionPrep Stage 2 output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--k", type=int, default=8, help="Past frames per window")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-control", dest="use_control", action="store_false",
                         help="Train the plain next-pose model instead, as a baseline")
    parser.add_argument("--checkpoint", default="checkpoints/best_motion_mamba_control")
    args = parser.parse_args()

    dataset = ControlWindowDataset(args.clips, k=args.k, use_control=args.use_control)
    print(f"Dataset: {dataset.summary()}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = {
        "state_dim": dataset.pose_dim,        # what the model predicts: pose only
        "control_dim": dataset.control_dim if args.use_control else 0,
        "n_embd": args.n_embd,
        "n_layers": args.n_layers,
        "learning_rate": args.learning_rate,
        "dropout": 0.1,
        "loss_scale": 1.0,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dt_rank": math.ceil(args.n_embd / 16),
    }
    print(f"Model: predicts {config['state_dim']} pose numbers from "
          f"{config['state_dim']} + {config['control_dim']} = "
          f"{config['state_dim'] + config['control_dim']} inputs per frame")

    trainer = MotionMambaTrainer(config)
    history = trainer.train(
        loader,
        val_loader=loader,   # small dataset: this reports fit, not generalization
        epochs=args.epochs,
        log_every=1,
        best_path=args.checkpoint,
    )

    losses = history["train_loss"]
    print(f"\ntrain_loss: first={losses[0]:.5f}  last={losses[-1]:.5f}  "
          f"({100 * (1 - losses[-1] / losses[0]):.1f}% lower)")


if __name__ == "__main__":
    main()
