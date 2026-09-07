"""
Interactive real-time 3D viewer for GPSM: plays a StateMachineGPT
autoregressive rollout back as a posed SMPL-X body in a live pyrender
window, using a game-engine-editor-style fly camera (like Unity's Scene
view or Unreal's editor viewport) instead of pyrender's default trackball
— the camera moves through the world; the ground and body never rotate.
This is the 3D counterpart to the GIF outputs produced by
src/gpsm/tests/rollout_inference.py.

This opens a real OpenGL window (via pyglet) — run it on a machine with a
display, not headless CI. For an automated, no-display sanity check of the
same rendering pipeline, use offscreen_test.py in this folder instead.

Controls (see simulator/fly_camera.py):
    left-drag           look around (camera rotates in place)
    W / A / S / D        move forward / left / backward / right
    Space / Left-Ctrl     move up / down
    scroll               adjust move speed
    q / ESC              quit
    (other pyrender hotkeys — f fullscreen, l lighting, z reset view — still work)

Usage:
    python -m simulator.live_viewer data/12_L_2_stageii.npz \
        --checkpoint checkpoints/best_sm --model-folder model/SMPLX_FEMALE.npz --n-steps 90

    # ground-truth playback instead of a model rollout (no --checkpoint needed)
    python -m simulator.live_viewer data/12_L_2_stageii.npz \
        --model-folder model/SMPLX_FEMALE.npz --ground-truth --n-steps 150
"""
from __future__ import annotations

import argparse
import time
from typing import Optional

from simulator.fly_camera import GameViewer
from simulator.model_runner import MotionSequence, generate_rollout_sequence, generate_ground_truth_sequence
from simulator.smplx_scene import build_scene, update_body_mesh


def play(
    sequence: MotionSequence,
    window_title: str,
    loop: bool = True,
    auto_close_seconds: Optional[float] = None,
    move_speed: float = 3.0,
) -> None:
    """Opens a GameViewer (pyrender.Viewer + fly camera) and plays `sequence`
    back in real time, swapping the posed body mesh in under the viewer's
    render lock every frame so it never races the render thread."""
    if sequence.n_frames == 0:
        raise ValueError("Cannot play an empty motion sequence.")

    scene, body_node, _cam_node = build_scene(sequence.vertices[0], sequence.faces)

    viewer = GameViewer(
        scene,
        run_in_thread=True,
        use_raymond_lighting=False,
        viewport_size=(960, 720),
        window_title=window_title,
        move_speed=move_speed,
    )

    dt = 1.0 / max(sequence.fps, 1)
    start_time = time.time()
    frame_idx = 0
    try:
        while viewer.is_active:
            if auto_close_seconds is not None and (time.time() - start_time) > auto_close_seconds:
                break

            viewer.render_lock.acquire()
            try:
                body_node = update_body_mesh(scene, body_node, sequence.vertices[frame_idx], sequence.faces)
            finally:
                viewer.render_lock.release()

            frame_idx += 1
            if frame_idx >= sequence.n_frames:
                if not loop:
                    time.sleep(dt)
                    break
                frame_idx = 0

            time.sleep(dt)
    finally:
        if viewer.is_active:
            viewer.close_external()
        while viewer.is_active:
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live 3D viewer: play back a StateMachineGPT rollout (or ground-truth mocap) "
                    "as a posed SMPL-X body."
    )
    parser.add_argument("npz_path", help="Seed / source motion .npz file")
    parser.add_argument("--model-folder", required=True, help="SMPL-X model file/folder (e.g. model/SMPLX_FEMALE.npz)")
    parser.add_argument("--checkpoint", default="checkpoints/best_sm",
                         help="StateMachineGPT checkpoint prefix (ignored with --ground-truth)")
    parser.add_argument("--ground-truth", action="store_true",
                         help="Play back the real recorded npz frames instead of a model rollout")
    parser.add_argument("--n-steps", type=int, default=90,
                         help="Rollout: autoregressive frames to predict. Ground-truth: frames to play.")
    parser.add_argument("--start", type=int, default=0, help="Ground-truth only: starting frame index")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", default=None, help="cpu/cuda (default: auto)")
    parser.add_argument("--gender", default=None, choices=[None, "neutral", "male", "female"])
    parser.add_argument("--num-betas", type=int, default=None)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--no-flat-hand-mean", dest="flat_hand_mean", action="store_false")
    parser.add_argument("--keep-root-motion", action="store_true",
                         help="Don't recenter on the pelvis each frame — shows true world-space translation")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Play once instead of looping")
    parser.add_argument("--auto-close-seconds", type=float, default=None,
                         help="Automatically close the viewer after N seconds (mainly for automated testing)")
    parser.add_argument("--move-speed", type=float, default=3.0,
                         help="Fly-camera starting move speed in metres/second (scroll in-viewer to adjust live)")
    args = parser.parse_args()

    if args.ground_truth:
        sequence = generate_ground_truth_sequence(
            args.npz_path, model_folder=args.model_folder, n_frames=args.n_steps, start=args.start,
            gender=args.gender, num_betas=args.num_betas, use_pca=args.use_pca,
            flat_hand_mean=args.flat_hand_mean, keep_root_motion=args.keep_root_motion, fps=args.fps,
        )
        title = f"GPSM simulator -- ground truth: {args.npz_path}"
    else:
        sequence = generate_rollout_sequence(
            args.npz_path, checkpoint_path=args.checkpoint, model_folder=args.model_folder,
            n_steps=args.n_steps, device=args.device, gender=args.gender, num_betas=args.num_betas,
            use_pca=args.use_pca, flat_hand_mean=args.flat_hand_mean,
            keep_root_motion=args.keep_root_motion, fps=args.fps,
        )
        title = f"GPSM simulator -- rollout ({args.checkpoint}): {args.npz_path}"

    print(f"Loaded {sequence.n_frames} frames ({sequence.vertices.shape[1]} vertices each). Opening viewer...")
    play(sequence, window_title=title, loop=args.loop, auto_close_seconds=args.auto_close_seconds,
         move_speed=args.move_speed)
    print("Viewer closed.")


if __name__ == "__main__":
    main()
