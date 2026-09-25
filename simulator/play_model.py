"""Drive a trained model with the keyboard and watch the character move.

This is the playable test harness: press a key, the model predicts the next
pose, the SMPL-X body is posed with it, and the camera follows. It is the
closest thing in this repo to "what a game would actually do", and the point
of it is to *see* whether the character obeys commands rather than trusting
a loss number.

    arrow keys ─> control vector ─> model ─> pose ─> SMPL-X body ─> screen
                                      ^                               │
                                      └───────── next frame ──────────┘

Controls (third-person, the way a game does it — the *character* moves,
not the camera, and the camera follows along behind whichever way it faces)
    W / S            walk forward / backwards
    A / D            step left / right
    mouse            steer: move it right and the character turns right
    (nothing held)   ease back to standing
    q / ESC          quit

The cursor is captured while playing so the mouse can keep turning past the
edge of the window; pass --no-mouse-grab to keep a normal cursor.

Run it:
    python -m simulator.play_model --checkpoint checkpoints/best_motion_mamba_control

Or without a display, which renders the same loop to a GIF and checks it:
    python -m simulator.play_model --headless --script forward,left,forward
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.gpsm.smplx.config import find_smplx_model
from simulator.control_input import ControlDriver, PlayerIntent
from simulator.fly_camera import GameViewer
from simulator.live_model import LiveMotionSession, SmplxPoser, load_model
from simulator.smplx_scene import (
    body_mesh_from_vertices,
    build_scene,
    follow_camera_pose,
    make_checkerboard_ground,
    update_body_mesh,
)

DEFAULT_FPS = 30.0

#: Named intents the headless mode can be scripted with, so the same loop
#: can be exercised without anyone at the keyboard.
SCRIPTED_INTENTS = {
    "idle": PlayerIntent(0.0, 0.0),
    "forward": PlayerIntent(1.0, 0.0),
    "back": PlayerIntent(-1.0, 0.0),
    "left": PlayerIntent(1.0, 1.0),      # walking while turning left
    "right": PlayerIntent(1.0, -1.0),
    "turn_left": PlayerIntent(0.0, 1.0),  # turning on the spot
}


def character_heading(pose: np.ndarray) -> float:
    """Which way the character faces, in radians, from its root rotation.

    Reuses the same convention the training control was built with (see
    src/gpsm/control/root_trajectory.py), so "facing" means the same thing
    here as it did in the data.
    """
    from src.gpsm.control.root_trajectory import _axis_angle_to_matrix, heading_from_left_vector

    rotation = _axis_angle_to_matrix(pose[3:6][None, :])
    left_in_world = rotation @ np.array([1.0, 0.0, 0.0])
    return float(heading_from_left_vector(left_in_world[:, :2])[0])


def _ease_angle(current: float, target: float, factor: float) -> float:
    """Ease an angle toward another, taking the short way round the circle."""
    difference = float(np.angle(np.exp(1j * (target - current))))
    return current + difference * factor


def default_seed_clip(clips_dir: str = "processed/with_control") -> str:
    """Pick a clip to start the character from (see live_model's docstring
    for why a seed is needed at all)."""
    clips = sorted(Path(clips_dir).glob("*.npz"))
    if not clips:
        raise FileNotFoundError(
            f"No clips in {clips_dir}. Run MotionPrep first:\n"
            "  python -m src.gpsm.motionprep.unify data --out processed/unified\n"
            "  python -m src.gpsm.motionprep.extract_control processed/unified "
            "--out processed/with_control"
        )
    return str(clips[0])


class CharacterRunner:
    """Ties the model, the control driver and the body model together.

    One call to :meth:`advance` produces one frame's worth of everything the
    renderer needs.
    """

    def __init__(self, checkpoint: str, seed_clip: str, smplx_model: str, fps: float = DEFAULT_FPS):
        self.model = load_model(checkpoint)
        self.session = LiveMotionSession(self.model, seed_clip)
        self.poser = SmplxPoser(smplx_model, betas=self.session.betas)
        self.control = ControlDriver()
        self.fps = fps

        if self.control.control_dim != self.session.control_dim:
            raise ValueError(
                f"The keyboard produces {self.control.control_dim} control numbers but the "
                f"checkpoint expects {self.session.control_dim}. They must describe the same "
                "thing — check the look-ahead horizons."
            )

    def advance(self, intent: PlayerIntent, dt: float) -> "tuple[np.ndarray, np.ndarray]":
        """Step one frame.

        Returns:
            ``(vertices, pose)`` — the body to draw, and the raw pose behind it.
        """
        control = self.control.update(intent, dt)
        pose = self.session.step(control)
        return self.poser.vertices(pose), pose


# ---------------------------------------------------------------------------
# Headless: same loop, rendered to a file, no display needed
# ---------------------------------------------------------------------------

def run_headless(runner: CharacterRunner, script: List[str], seconds_each: float,
                  out_path: Path, viewport=(640, 480)) -> dict:
    """Run the loop offscreen under a scripted sequence of intents.

    This is what makes the simulator testable: it exercises exactly the same
    model → pose → body → render path as the live window, but produces a GIF
    and a set of numbers instead of needing someone to watch it.

    Returns:
        A summary dict (frames rendered, distance travelled, net turn).
    """
    import pyrender

    vertices, pose = runner.advance(SCRIPTED_INTENTS["idle"], 1.0 / runner.fps)
    scene, body_node, camera_node = build_scene(
        vertices, runner.poser.faces, viewport_size=viewport, with_ground=False
    )
    scene.add(make_checkerboard_ground(z=float(vertices[:, 2].min())))

    renderer = pyrender.OffscreenRenderer(*viewport)
    frames, path_xy, headings = [], [], []
    dt = 1.0 / runner.fps

    try:
        for name in script:
            intent = SCRIPTED_INTENTS[name]
            for _ in range(int(seconds_each * runner.fps)):
                vertices, pose = runner.advance(intent, dt)
                path_xy.append(pose[:2].copy())
                headings.append(pose[3:6].copy())

                body_node = update_body_mesh(scene, body_node, vertices, runner.poser.faces)
                scene.set_pose(camera_node, follow_camera_pose(vertices.mean(axis=0)))

                color, _ = renderer.render(scene)
                if not np.isfinite(color.astype(np.float32)).all():
                    raise ValueError("renderer produced non-finite pixels")
                if color.std() < 1e-3:
                    raise ValueError("rendered frame is blank — scene or camera is wrong")
                frames.append(color)
    finally:
        renderer.delete()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.mimsave(str(out_path), frames, duration=1000.0 / runner.fps)

    path_xy = np.asarray(path_xy)
    distance = float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum()) if len(path_xy) > 1 else 0.0
    return {"frames": len(frames), "distance_m": distance, "output": str(out_path)}


# ---------------------------------------------------------------------------
# Live: a real window, driven by the keyboard
# ---------------------------------------------------------------------------

class PlayerViewer(GameViewer):
    """A window where the keyboard and mouse drive the *character*.

    ``GameViewer`` normally spends W/A/S/D and the mouse on flying the
    camera around. In a third-person game those belong to the character
    instead, and the camera simply follows — so this subclass takes the
    input over (``external_camera=True`` tells the base class to stop
    touching the camera or the movement keys) and accumulates mouse
    movement as steering.
    """

    #: Radians of turn requested per pixel of mouse movement. The result is
    #: clamped to +/-1 and fed to ControlDriver as a fraction of full turn rate.
    MOUSE_TURN_SENSITIVITY = 0.02

    def __init__(self, *args, **kwargs):
        self._turn_request = 0.0
        super().__init__(*args, external_camera=True, **kwargs)

    def on_mouse_motion(self, x, y, dx, dy):
        """Moving the mouse right steers right, left steers left."""
        self._turn_request = float(np.clip(-dx * self.MOUSE_TURN_SENSITIVITY, -1.0, 1.0))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.on_mouse_motion(x, y, dx, dy)

    def take_turn_request(self) -> float:
        """Read the accumulated steering and reset it.

        Reset-on-read is what makes the character stop turning as soon as
        the mouse stops: with no new movement there is no new request, so
        the turn rate eases back to zero.
        """
        value, self._turn_request = self._turn_request, 0.0
        return value


def run_live(runner: CharacterRunner, auto_close_seconds: Optional[float] = None,
              grab_mouse: bool = True) -> None:
    """Open a window and let the player drive the character.

    Args:
        runner:             The character to drive.
        auto_close_seconds: Close automatically after this long (smoke tests).
        grab_mouse:         Capture the cursor so the mouse can be moved
                            without limit, the way a game does. Turning this
                            off keeps a normal cursor, but steering then
                            stops at the window edge.
    """
    import pyglet

    vertices, pose = runner.advance(SCRIPTED_INTENTS["idle"], 1.0 / runner.fps)
    scene, body_node, camera_node = build_scene(vertices, runner.poser.faces, with_ground=False)
    scene.add(make_checkerboard_ground(z=float(vertices[:, 2].min())))

    viewer = PlayerViewer(
        scene, run_in_thread=True, use_raymond_lighting=False,
        viewport_size=(960, 720),
        window_title="GPSM — WASD moves the character, mouse steers, ESC quits",
    )
    if grab_mouse:
        try:
            viewer.set_exclusive_mouse(True)
        except Exception:      # not fatal: steering still works, just bounded
            grab_mouse = False

    key = pyglet.window.key
    dt = 1.0 / runner.fps
    started = last = time.time()
    camera_heading = character_heading(pose)

    try:
        while viewer.is_active:
            if auto_close_seconds is not None and time.time() - started > auto_close_seconds:
                break

            keys = viewer.held_keys
            intent = PlayerIntent(
                forward=(1.0 if keys[key.W] else 0.0) - (1.0 if keys[key.S] else 0.0),
                strafe=(1.0 if keys[key.A] else 0.0) - (1.0 if keys[key.D] else 0.0),
                turn=viewer.take_turn_request(),
            )

            now = time.time()
            vertices, pose = runner.advance(intent, min(now - last, 0.1))
            last = now

            # The camera sits behind whichever way the character faces, eased
            # so that a jittery predicted heading does not shake the view.
            camera_heading = _ease_angle(camera_heading, character_heading(pose), 0.25)

            viewer.render_lock.acquire()
            try:
                body_node = update_body_mesh(scene, body_node, vertices, runner.poser.faces)
                scene.set_pose(camera_node, follow_camera_pose(
                    vertices.mean(axis=0),
                    behind=np.array([-np.cos(camera_heading), -np.sin(camera_heading), 0.0]),
                ))
            finally:
                viewer.render_lock.release()

            time.sleep(max(0.0, dt - (time.time() - now)))
    finally:
        if viewer.is_active:
            if grab_mouse:
                try:
                    viewer.set_exclusive_mouse(False)
                except Exception:
                    pass
            viewer.close_external()
        while viewer.is_active:
            time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive a trained MotionMamba with the keyboard.")
    parser.add_argument("--checkpoint", default="checkpoints/best_motion_mamba_control")
    parser.add_argument("--seed-clip", default=None, help="Clip to start from (default: first available)")
    parser.add_argument("--smplx-model", default=None, help="SMPL-X .npz (default: found automatically)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--headless", action="store_true", help="Render a scripted run to a GIF, no window")
    parser.add_argument("--script", default="idle,forward,left,forward,right,back",
                         help="Headless only: comma-separated intents " + "|".join(SCRIPTED_INTENTS))
    parser.add_argument("--seconds-each", type=float, default=1.5, help="Headless only: seconds per intent")
    parser.add_argument("--out", default="simulator/output/play_model.gif")
    parser.add_argument("--auto-close-seconds", type=float, default=None,
                         help="Live only: close the window automatically (for smoke tests)")
    parser.add_argument("--no-mouse-grab", action="store_true",
                         help="Live only: keep a normal cursor instead of capturing it. "
                              "Steering then stops at the window edge.")
    args = parser.parse_args()

    smplx_model = args.smplx_model or find_smplx_model()
    if smplx_model is None:
        raise SystemExit("No SMPL-X model found. Put it in model/ or set GPSM_SMPLX_MODEL.")

    runner = CharacterRunner(
        checkpoint=args.checkpoint,
        seed_clip=args.seed_clip or default_seed_clip(),
        smplx_model=str(smplx_model),
        fps=args.fps,
    )

    if args.headless:
        script = [name.strip() for name in args.script.split(",") if name.strip()]
        unknown = [n for n in script if n not in SCRIPTED_INTENTS]
        if unknown:
            raise SystemExit(f"Unknown intents {unknown}; choose from {sorted(SCRIPTED_INTENTS)}")
        summary = run_headless(runner, script, args.seconds_each, Path(args.out))
        print(f"Rendered {summary['frames']} frames -> {summary['output']}")
        print(f"Character travelled {summary['distance_m']:.2f} m")
    else:
        print("WASD moves the character, the mouse steers it, ESC or q quits.")
        run_live(runner, auto_close_seconds=args.auto_close_seconds,
                  grab_mouse=not args.no_mouse_grab)


if __name__ == "__main__":
    main()
