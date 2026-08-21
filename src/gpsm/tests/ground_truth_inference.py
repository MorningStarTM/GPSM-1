"""
Ground-truth counterpart to rollout_inference.py.

rollout_inference.py takes the first real frame of an .npz file, feeds it to
the trained model, and autoregressively predicts the following frames — no
ground truth after frame 0. This module instead takes the *actual recorded*
frames `[start : start + n_frames]` straight from the same .npz file — no
model involved at all — and renders them through the same SMPL-X pipeline
(joints + mesh GIFs), so the two outputs can be played side by side: what
really happened vs. what the model predicted, over the identical frame range.

Usage:
    python -m src.gpsm.tests.ground_truth_inference data/12_L_2_stageii.npz \
        --n-frames 9 --model-folder model/SMPLX_FEMALE.npz --fps 4

    # to directly compare against a rollout_inference.py run with
    # --n-steps 8 (1 seed frame + 8 predicted = 9 total frames), request the
    # same total frame count here so both GIFs cover the identical window:
    python -m src.gpsm.tests.rollout_inference data/12_L_2_stageii.npz \
        --checkpoint checkpoints/best_sm --n-steps 8 \
        --model-folder model/SMPLX_FEMALE.npz --mesh --fps 4
    python -m src.gpsm.tests.ground_truth_inference data/12_L_2_stageii.npz \
        --n-frames 9 --model-folder model/SMPLX_FEMALE.npz --fps 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.gpsm.tests.visualize_npz import load_npz, animate_joints, summarize
from src.gpsm.tests.simulate_smplx import build_forward_kwargs, resolve_gender, animate_mesh


def extract_clip(npz: dict, start: int, n_frames: int) -> dict:
    """Slice every per-frame array in npz down to [start:start+n_frames];
    static (non per-frame) arrays like betas/gender pass through unchanged."""
    total = int(np.asarray(npz.get("trans", npz.get("poses"))).shape[0])
    end = min(start + n_frames, total)
    clip = {
        k: (v[start:end] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == total else v)
        for k, v in npz.items()
    }
    return clip, end - start


def simulate_ground_truth(
    npz_path: str,
    model_folder: str,
    n_frames: int = 9,
    start: int = 0,
    gender: Optional[str] = None,
    num_betas: Optional[int] = None,
    use_pca: bool = False,
    flat_hand_mean: bool = True,
    fps: int = 8,
    out_dir: str = "src/gpsm/tests/output",
    keep_root_motion: bool = False,
) -> None:
    try:
        import smplx
    except ImportError as e:
        raise ImportError("The `smplx` package is not installed. Run: pip install smplx") from e

    npz_path_p = Path(npz_path)
    out_dir_p  = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    stem = npz_path_p.stem

    npz = load_npz(npz_path_p)
    summarize(npz, npz_path_p)

    total = int(np.asarray(npz.get("trans", npz.get("poses"))).shape[0])
    if start < 0 or start >= total:
        raise ValueError(f"--start={start} is out of range for a {total}-frame sequence.")

    clip, actual_n = extract_clip(npz, start, n_frames)
    if actual_n < n_frames:
        print(f"Note: sequence only has {total} frames — clipping to {actual_n} frames "
              f"(requested {n_frames} starting at {start}).")
    print(f"Ground-truth clip: frames [{start}:{start + actual_n}] ({actual_n} frames) of {stem}")

    resolved_gender = resolve_gender(clip, gender)
    resolved_num_betas = num_betas or (
        int(np.asarray(clip["betas"]).reshape(-1).shape[0]) if "betas" in clip else 10
    )

    # Same file-vs-folder --model-folder resolution as simulate_smplx.py /
    # rollout_inference.py (smplx.create() supports either).
    model_folder_p = Path(model_folder)
    if model_folder_p.is_dir():
        resolved_model_path = model_folder_p / "smplx" / f"SMPLX_{resolved_gender.upper()}.npz"
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Expected SMPL-X model file not found: {resolved_model_path}\n"
                "Download it (after registering) from https://smpl-x.is.tue.mpg.de "
                "and place it at that path, or pass --model-folder pointing directly "
                "at a specific model file."
            )
    elif model_folder_p.is_file():
        resolved_model_path = model_folder_p
        name = model_folder_p.stem.upper()
        for g in ("NEUTRAL", "FEMALE", "MALE"):  # FEMALE before MALE ("MALE" is a substring of "FEMALE")
            if g in name:
                resolved_gender = g.lower()
                break
    else:
        raise FileNotFoundError(f"--model-folder does not exist: {model_folder_p}")

    fwd_kwargs = build_forward_kwargs(clip, num_frames=actual_n, num_betas=resolved_num_betas, device="cpu")

    body_model = smplx.create(
        str(resolved_model_path),
        model_type="smplx",
        gender=resolved_gender,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        num_betas=resolved_num_betas,
        ext="npz",
        batch_size=actual_n,
    )

    with torch.no_grad():
        output = body_model(return_verts=True, **fwd_kwargs)

    joints = output.joints.detach().cpu().numpy()
    root = joints[:, [0], :]
    if not keep_root_motion:
        joints = joints - root

    title = f"{stem} — ground truth [{start}:{start + actual_n}] (SMPL-X, {resolved_gender})"

    animate_joints(joints, out_dir_p / f"{stem}_groundtruth_joints.gif", fps=fps, title=title)

    vertices = output.vertices.detach().cpu().numpy()
    if not keep_root_motion:
        vertices = vertices - root
    animate_mesh(vertices, body_model.faces, out_dir_p / f"{stem}_groundtruth_mesh.gif", fps=fps, title=title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the real recorded frames of a mocap .npz file through SMPL-X "
                    "(joints + mesh GIFs) — the ground-truth counterpart to rollout_inference.py."
    )
    parser.add_argument("npz_path", help="Path to a single motion .npz file")
    parser.add_argument("--model-folder", required=True,
                         help="SMPL-X model file/folder (see simulate_smplx.py docstring for how to obtain)")
    parser.add_argument("--n-frames", type=int, default=9,
                         help="Number of contiguous real frames to render (default 9 = "
                              "matches a rollout_inference.py run with --n-steps 8)")
    parser.add_argument("--start", type=int, default=0, help="Index of the first frame in the clip")
    parser.add_argument("--gender", default=None, choices=[None, "neutral", "male", "female"])
    parser.add_argument("--num-betas", type=int, default=None)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--no-flat-hand-mean", dest="flat_hand_mean", action="store_false")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", default="src/gpsm/tests/output")
    parser.add_argument("--keep-root-motion", action="store_true")
    args = parser.parse_args()

    simulate_ground_truth(
        args.npz_path,
        model_folder=args.model_folder,
        n_frames=args.n_frames,
        start=args.start,
        gender=args.gender,
        num_betas=args.num_betas,
        use_pca=args.use_pca,
        flat_hand_mean=args.flat_hand_mean,
        fps=args.fps,
        out_dir=args.out,
        keep_root_motion=args.keep_root_motion,
    )


if __name__ == "__main__":
    main()
