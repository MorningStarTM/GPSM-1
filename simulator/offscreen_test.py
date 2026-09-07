"""
Headless self-test / sanity-check for the 3D simulator: renders a
StateMachineGPT rollout through the exact same scene-building code used by
live_viewer.py, using pyrender's OffscreenRenderer (no window/display
required), and saves the frames as a GIF plus one full-resolution PNG.

This is the automated regression test for the simulator package — run it any
time after changing simulator/smplx_scene.py or simulator/model_runner.py to
confirm the rendering pipeline still produces a valid, non-empty, finite,
visibly-a-body image before trusting the interactive live_viewer, which
cannot be checked by an automated script the same way (it opens a real
window).

Usage:
    python -m simulator.offscreen_test data/12_L_2_stageii.npz \
        --checkpoint checkpoints/best_sm --model-folder model/SMPLX_FEMALE.npz --n-steps 20

    # ground-truth playback instead of a model rollout
    python -m simulator.offscreen_test data/12_L_2_stageii.npz \
        --model-folder model/SMPLX_FEMALE.npz --ground-truth --n-steps 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import pyrender

from simulator.model_runner import MotionSequence, generate_rollout_sequence, generate_ground_truth_sequence
from simulator.smplx_scene import build_scene, update_body_mesh


def render_sequence_offscreen(
    sequence: MotionSequence,
    out_gif: Path,
    out_png: Path,
    viewport_size=(640, 480),
) -> None:
    if sequence.n_frames == 0:
        raise ValueError("Cannot render an empty motion sequence.")

    scene, body_node, _cam_node = build_scene(sequence.vertices[0], sequence.faces, viewport_size=viewport_size)
    renderer = pyrender.OffscreenRenderer(viewport_width=viewport_size[0], viewport_height=viewport_size[1])

    frames = []
    try:
        for t in range(sequence.n_frames):
            if t > 0:
                body_node = update_body_mesh(scene, body_node, sequence.vertices[t], sequence.faces)
            color, _depth = renderer.render(scene)

            if not np.isfinite(color.astype(np.float32)).all():
                raise ValueError(f"Non-finite pixels rendered at frame {t}.")
            if color.std() < 1e-3:
                raise ValueError(
                    f"Rendered frame {t} is (near-)blank (pixel std={color.std():.6f}) — "
                    "scene/camera/mesh is probably empty or mis-positioned."
                )
            frames.append(color)
    finally:
        renderer.delete()

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    # newer imageio/Pillow GIF plugin dropped the `fps` kwarg in favor of
    # per-frame `duration` in milliseconds
    imageio.mimsave(str(out_gif), frames, duration=1000.0 / sequence.fps)
    imageio.imwrite(str(out_png), frames[len(frames) // 2])
    print(f"Rendered {len(frames)} frames without error (all finite, none blank).")
    print(f"  GIF -> {out_gif}")
    print(f"  PNG (mid-sequence frame) -> {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless render test for the 3D simulator (no display/window required)."
    )
    parser.add_argument("npz_path")
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_sm")
    parser.add_argument("--ground-truth", action="store_true",
                         help="Render the real recorded npz frames instead of a model rollout")
    parser.add_argument("--n-steps", type=int, default=20,
                         help="Rollout: autoregressive frames to predict. Ground-truth: frames to render.")
    parser.add_argument("--start", type=int, default=0, help="Ground-truth only: starting frame index")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gender", default=None, choices=[None, "neutral", "male", "female"])
    parser.add_argument("--num-betas", type=int, default=None)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--no-flat-hand-mean", dest="flat_hand_mean", action="store_false")
    parser.add_argument("--keep-root-motion", action="store_true")
    parser.add_argument("--out-dir", default="simulator/output")
    args = parser.parse_args()

    if args.ground_truth:
        sequence = generate_ground_truth_sequence(
            args.npz_path, model_folder=args.model_folder, n_frames=args.n_steps, start=args.start,
            gender=args.gender, num_betas=args.num_betas, use_pca=args.use_pca,
            flat_hand_mean=args.flat_hand_mean, keep_root_motion=args.keep_root_motion, fps=args.fps,
        )
        tag = "groundtruth"
    else:
        sequence = generate_rollout_sequence(
            args.npz_path, checkpoint_path=args.checkpoint, model_folder=args.model_folder,
            n_steps=args.n_steps, device=args.device, gender=args.gender, num_betas=args.num_betas,
            use_pca=args.use_pca, flat_hand_mean=args.flat_hand_mean,
            keep_root_motion=args.keep_root_motion, fps=args.fps,
        )
        tag = "rollout"

    stem = Path(args.npz_path).stem
    out_dir = Path(args.out_dir)
    render_sequence_offscreen(
        sequence,
        out_gif=out_dir / f"{stem}_{tag}_3d.gif",
        out_png=out_dir / f"{stem}_{tag}_3d_mid.png",
    )


if __name__ == "__main__":
    main()
