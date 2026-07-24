"""
End-to-end inference test for StateMachineGPT: seed the model with the first
frame of a mocap .npz file, autoregressively roll out N future frames via
StateMachineGPT.rollout(), then simulate/visualize the seed + predicted
frames.

Pipeline:
    npz file --(frame_loader)--> normalized first frame
                                        |
                                        v
                          StateMachineGPT.rollout(seed, n_steps)
                                        |
                                        v
                    denormalize back to real pose/trans units
                                        |
            -----------------------------------------------------
            |                                                     |
   always: pose-value heatmap                      --model-folder given: also run
   (visualize_npz.plot_pose_curves)                 through the real SMPL-X body
                                                     model -> animated joints GIF

Usage:
    python -m src.gpsm.tests.rollout_inference data/12_L_2_stageii.npz \
        --checkpoint checkpoints/best_sm --n-steps 8

    # with real SMPL-X visualization
    python -m src.gpsm.tests.rollout_inference data/12_L_2_stageii.npz \
        --checkpoint checkpoints/best_sm --n-steps 8 \
        --model-folder model/SMPLX_FEMALE.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.gpsm.experiment_model.sm_gpt import StateMachineGPT
from src.gpsm.tests.frame_loader import load_sequence_features
from src.gpsm.tests.visualize_npz import load_npz, animate_joints, plot_pose_curves
from src.gpsm.tests.simulate_smplx import build_forward_kwargs, resolve_gender


# ---------------------------------------------------------------------------
# Model + rollout
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path: str, device: Optional[str] = None) -> StateMachineGPT:
    """Rebuild StateMachineGPT using the config saved alongside the safetensors
    checkpoint, then load the trained weights (optimizer state not needed for
    inference)."""
    base = Path(checkpoint_path).with_suffix("")
    meta_path = base.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = StateMachineGPT(meta["config"])
    if device is not None:
        model.device = torch.device(device)
    model.to(model.device)  # StateMachineGPT.__init__ sets self.device but doesn't move params itself
    model.load_safetensors(str(base), device=model.device, load_optimizer=False)
    model.eval()
    return model


def rollout_from_first_frame(
    npz_path: str,
    checkpoint_path: str,
    n_steps: int = 8,
    device: Optional[str] = None,
    feature_set: str = "poses+trans",
) -> dict:
    """
    Seed the model with the first frame of npz_path and autoregressively
    predict n_steps future frames.

    Returns a dict with:
        seed_frame_denorm : (D,)            real-units first frame
        predicted_denorm  : (n_steps, D)    real-units autoregressive predictions
        sequence_denorm   : (n_steps+1, D)  seed frame followed by predictions
        mean, std          : (D,)           per-file normalization stats used
        model               : the loaded StateMachineGPT
    """
    model = load_model_from_checkpoint(checkpoint_path, device=device)
    state_dim = model.config["state_dim"]

    X, mean, std = load_sequence_features(npz_path, feature_set=feature_set, normalize=True)
    if X.shape[1] != state_dim:
        raise ValueError(
            f"Feature width mismatch: npz produces D={X.shape[1]} but checkpoint "
            f"expects state_dim={state_dim}. Check feature_set matches training."
        )

    seed = torch.from_numpy(X[:1]).float().to(model.device)  # (1, D) normalized

    with torch.no_grad():
        predicted = model.rollout(seed, n_steps=n_steps)  # (n_steps, D) normalized
    predicted_np = predicted.detach().cpu().numpy()

    if not np.isfinite(predicted_np).all():
        raise ValueError("Model produced non-finite (NaN/Inf) predictions.")

    seed_denorm      = X[0] * std + mean
    predicted_denorm = predicted_np * std + mean
    sequence_denorm  = np.concatenate([seed_denorm[None, :], predicted_denorm], axis=0)

    return {
        "seed_frame_denorm": seed_denorm,
        "predicted_denorm": predicted_denorm,
        "sequence_denorm": sequence_denorm,
        "mean": mean,
        "std": std,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def simulate_prediction(
    npz_path: str,
    checkpoint_path: str,
    n_steps: int = 8,
    model_folder: Optional[str] = None,
    gender: Optional[str] = None,
    num_betas: Optional[int] = None,
    use_pca: bool = False,
    flat_hand_mean: bool = True,
    fps: int = 8,
    out_dir: str = "src/gpsm/tests/output",
    device: Optional[str] = None,
    keep_root_motion: bool = False,
) -> None:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    stem = Path(npz_path).stem

    result = rollout_from_first_frame(npz_path, checkpoint_path, n_steps=n_steps, device=device)
    sequence = result["sequence_denorm"]  # (n_steps+1, D) = poses(165) + trans(3)

    print(f"Seed frame + {n_steps} autoregressive predictions "
          f"(sequence shape {sequence.shape})")
    print(f"  predicted value range: "
          f"[{result['predicted_denorm'].min():.4f}, {result['predicted_denorm'].max():.4f}]")

    # Always produce a lightweight, model-free sanity plot.
    plot_pose_curves(sequence, out_dir_p / f"{stem}_rollout_pose.png",
                      title=f"{stem} — seed + {n_steps}-step rollout")

    if model_folder is None:
        print("No --model-folder given — skipping SMPL-X animation "
              "(pose-curve plot above is the only output).")
        return

    if sequence.shape[1] != 168:
        raise ValueError(
            f"SMPL-X visualization expects poses(165)+trans(3)=168-dim features, "
            f"got D={sequence.shape[1]}. Re-run without --model-folder to still "
            "get the pose-curve plot."
        )

    try:
        import smplx
    except ImportError as e:
        raise ImportError("The `smplx` package is not installed. Run: pip install smplx") from e

    poses = sequence[:, :165]
    trans = sequence[:, 165:168]

    npz_orig = load_npz(Path(npz_path))
    fake_npz = {"poses": poses, "trans": trans}
    if "betas" in npz_orig:
        fake_npz["betas"] = npz_orig["betas"]
    if "gender" in npz_orig:
        fake_npz["gender"] = npz_orig["gender"]

    resolved_gender = resolve_gender(fake_npz, gender)
    resolved_num_betas = num_betas or (
        int(np.asarray(fake_npz["betas"]).reshape(-1).shape[0]) if "betas" in fake_npz else 10
    )

    model_folder_p = Path(model_folder)
    if model_folder_p.is_dir():
        resolved_model_path = model_folder_p / "smplx" / f"SMPLX_{resolved_gender.upper()}.npz"
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Expected SMPL-X model file not found: {resolved_model_path}")
    elif model_folder_p.is_file():
        resolved_model_path = model_folder_p
        name = model_folder_p.stem.upper()
        for g in ("NEUTRAL", "FEMALE", "MALE"):  # FEMALE before MALE ("MALE" is a substring of "FEMALE")
            if g in name:
                resolved_gender = g.lower()
                break
    else:
        raise FileNotFoundError(f"--model-folder does not exist: {model_folder_p}")

    num_frames = sequence.shape[0]
    fwd_kwargs = build_forward_kwargs(fake_npz, num_frames=num_frames,
                                       num_betas=resolved_num_betas, device="cpu")

    body_model = smplx.create(
        str(resolved_model_path),
        model_type="smplx",
        gender=resolved_gender,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        num_betas=resolved_num_betas,
        ext="npz",
        batch_size=num_frames,
    )

    with torch.no_grad():
        output = body_model(return_verts=False, **fwd_kwargs)

    joints = output.joints.detach().cpu().numpy()
    if not keep_root_motion:
        joints = joints - joints[:, [0], :]

    animate_joints(
        joints, out_dir_p / f"{stem}_rollout_joints.gif", fps=fps,
        title=f"{stem} — seed + {n_steps}-step rollout (SMPL-X, {resolved_gender})",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed StateMachineGPT with the first frame of a mocap .npz file, "
                    "autoregressively roll out N frames, and visualize the result."
    )
    parser.add_argument("npz_path", help="Path to a single motion .npz file")
    parser.add_argument("--checkpoint", default="checkpoints/best_sm",
                         help="Path prefix to a safetensors checkpoint (expects <prefix>.safetensors + <prefix>.json)")
    parser.add_argument("--n-steps", type=int, default=8, help="Number of autoregressive future frames to predict")
    parser.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    parser.add_argument("--model-folder", default=None,
                         help="Optional SMPL-X model file/folder — if given, also renders a real SMPL-X animation")
    parser.add_argument("--gender", default=None, choices=[None, "neutral", "male", "female"])
    parser.add_argument("--num-betas", type=int, default=None)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--no-flat-hand-mean", dest="flat_hand_mean", action="store_false")
    parser.add_argument("--fps", type=int, default=8, help="Animation fps (kept low — only a few predicted frames)")
    parser.add_argument("--out", default="src/gpsm/tests/output")
    parser.add_argument("--keep-root-motion", action="store_true")
    args = parser.parse_args()

    simulate_prediction(
        args.npz_path,
        checkpoint_path=args.checkpoint,
        n_steps=args.n_steps,
        model_folder=args.model_folder,
        gender=args.gender,
        num_betas=args.num_betas,
        use_pca=args.use_pca,
        flat_hand_mean=args.flat_hand_mean,
        fps=args.fps,
        out_dir=args.out,
        device=args.device,
        keep_root_motion=args.keep_root_motion,
    )


if __name__ == "__main__":
    main()
