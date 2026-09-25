"""Tests for the playable model simulator (simulator/play_model.py and friends).

Two layers:

* the keyboard -> control translation, which is pure maths and can be
  checked exactly, and
* the live loop (model -> pose -> SMPL-X body -> rendered frame), which is
  checked end to end headlessly, since the interactive window cannot be
  driven by a test.

The model-dependent tests skip when there is no trained checkpoint or no
SMPL-X body model on the machine.

Run with:
    pytest tests/test_play_simulator.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from simulator.control_input import BACK_SPEED, TURN_RATE, WALK_SPEED, ControlDriver, PlayerIntent

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "checkpoints" / "best_motion_mamba_control"
CLIPS_DIR = REPO_ROOT / "processed" / "with_control"


def column(driver: ControlDriver, control: np.ndarray, name: str) -> float:
    return float(control[driver.names.index(name)])


# ---------------------------------------------------------------------------
# Keyboard -> control
# ---------------------------------------------------------------------------

def test_control_layout_matches_what_the_model_was_trained_on():
    """The keyboard has to emit the same 14 numbers, in the same order, that
    MotionPrep extracts from recordings — otherwise the model is being fed
    something it has never seen."""
    from src.gpsm.control.control_features import DEFAULT_HORIZONS_SECONDS, extract_control
    from src.gpsm.control.root_trajectory import RootTrajectory

    driver = ControlDriver()
    straight = RootTrajectory(position=np.zeros((20, 2)), heading=np.zeros(20), fps=30.0)
    from_recording = extract_control(straight, horizons_seconds=DEFAULT_HORIZONS_SECONDS)

    assert driver.names == from_recording.names
    assert driver.control_dim == from_recording.control_dim


def test_holding_forward_ramps_up_to_walking_speed():
    driver = ControlDriver()
    for _ in range(60):                       # 2 seconds at 30fps
        control = driver.update(PlayerIntent(forward=1.0), 1 / 30)

    assert driver.speed == pytest.approx(WALK_SPEED, abs=0.05)
    assert column(driver, control, "vel_fwd") == pytest.approx(WALK_SPEED, abs=0.05)
    # Walking straight: half a second ahead is speed * 0.5 metres forward,
    # nothing sideways, and no turn.
    assert column(driver, control, "pos_fwd_500ms") == pytest.approx(WALK_SPEED * 0.5, abs=0.05)
    assert column(driver, control, "pos_left_500ms") == pytest.approx(0.0, abs=1e-6)
    assert column(driver, control, "turn_sin_500ms") == pytest.approx(0.0, abs=1e-6)


def test_speed_ramps_rather_than_jumping():
    """A key press should not teleport the character from standing to full
    speed in one frame — the training data contains no such acceleration."""
    driver = ControlDriver()
    first = driver.update(PlayerIntent(forward=1.0), 1 / 30)
    assert 0 < column(driver, first, "vel_fwd") < WALK_SPEED * 0.5


def test_holding_back_goes_backwards():
    driver = ControlDriver()
    for _ in range(60):
        control = driver.update(PlayerIntent(forward=-1.0), 1 / 30)

    assert column(driver, control, "vel_fwd") == pytest.approx(BACK_SPEED, abs=0.05)
    assert column(driver, control, "pos_fwd_500ms") < 0, "backwards must mean a negative forward offset"


def test_left_turns_left_and_right_turns_right():
    """Sign convention, which is easy to get backwards and produces a
    character that steers the wrong way: turn_sin > 0 is a left turn, and
    the root should end up to the character's left."""
    left, right = ControlDriver(), ControlDriver()
    for _ in range(60):
        left_control = left.update(PlayerIntent(forward=1.0, turn=1.0), 1 / 30)
        right_control = right.update(PlayerIntent(forward=1.0, turn=-1.0), 1 / 30)

    assert left.turn_rate == pytest.approx(TURN_RATE, abs=0.05)
    assert column(left, left_control, "turn_sin_500ms") > 0
    assert column(left, left_control, "pos_left_500ms") > 0

    assert column(right, right_control, "turn_sin_500ms") < 0
    assert column(right, right_control, "pos_left_500ms") < 0


def test_releasing_the_keys_eases_back_to_standing():
    driver = ControlDriver()
    for _ in range(60):
        driver.update(PlayerIntent(forward=1.0, turn=1.0), 1 / 30)
    for _ in range(90):
        control = driver.update(PlayerIntent(), 1 / 30)

    assert driver.speed == pytest.approx(0.0, abs=0.01)
    assert driver.turn_rate == pytest.approx(0.0, abs=0.01)
    assert column(driver, control, "vel_fwd") == pytest.approx(0.0, abs=0.01)


def test_easing_does_not_depend_on_the_frame_rate():
    """A slow frame must cover proportionally more ground, or the character
    would accelerate differently whenever the renderer hitches."""
    fast, slow = ControlDriver(), ControlDriver()
    for _ in range(60):
        fast.update(PlayerIntent(forward=1.0), 1 / 60)
    for _ in range(30):
        slow.update(PlayerIntent(forward=1.0), 1 / 30)

    assert fast.speed == pytest.approx(slow.speed, abs=0.02)


def test_turning_on_the_spot_goes_nowhere():
    """Turning with no forward speed should not translate the root."""
    driver = ControlDriver()
    for _ in range(60):
        control = driver.update(PlayerIntent(forward=0.0, turn=1.0), 1 / 30)

    assert column(driver, control, "pos_fwd_500ms") == pytest.approx(0.0, abs=1e-6)
    assert column(driver, control, "pos_left_500ms") == pytest.approx(0.0, abs=1e-6)
    assert column(driver, control, "turn_sin_500ms") > 0, "but it should still be turning"


# ---------------------------------------------------------------------------
# The live loop, end to end
# ---------------------------------------------------------------------------

needs_trained_model = pytest.mark.skipif(
    not (CHECKPOINT.with_suffix(".safetensors").exists() and CLIPS_DIR.exists()
         and list(CLIPS_DIR.glob("*.npz"))),
    reason="needs a control-trained checkpoint and MotionPrep output",
)


@pytest.fixture(scope="module")
def runner():
    from src.gpsm.smplx.config import find_smplx_model
    from simulator.play_model import CharacterRunner, default_seed_clip

    model_path = find_smplx_model()
    if model_path is None:
        pytest.skip("no SMPL-X body model found")
    return CharacterRunner(str(CHECKPOINT), default_seed_clip(str(CLIPS_DIR)), str(model_path))


@needs_trained_model
def test_one_frame_produces_a_body_we_can_draw(runner):
    vertices, pose = runner.advance(PlayerIntent(forward=1.0), 1 / 30)

    assert pose.shape == (runner.session.pose_dim,)
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert np.isfinite(vertices).all() and np.isfinite(pose).all()
    # A human body is roughly 1.5-2m tall; anything wildly off means the
    # pose was fed in normalized, or in the wrong units.
    height = vertices[:, 2].max() - vertices[:, 2].min()
    assert 1.0 < height < 2.5, f"body is {height:.2f}m tall"


@needs_trained_model
def test_the_character_keeps_running_without_blowing_up(runner):
    """A long session must stay finite — autoregressive loops can diverge,
    and a NaN halfway through would end the game."""
    for _ in range(120):                      # 4 seconds
        vertices, pose = runner.advance(PlayerIntent(forward=1.0), 1 / 30)
    assert np.isfinite(vertices).all() and np.isfinite(pose).all()


@needs_trained_model
def test_the_window_slides_instead_of_growing(runner):
    """The model is fed a fixed-size window; if it grew every frame the
    simulator would get slower and slower until it stalled."""
    before = runner.session._window.shape
    for _ in range(20):
        runner.advance(PlayerIntent(forward=1.0), 1 / 30)
    assert runner.session._window.shape == before


@needs_trained_model
def test_different_keys_produce_different_motion(runner):
    """The whole point of the harness: what the player presses has to change
    what the character does."""
    from src.gpsm.smplx.config import find_smplx_model
    from simulator.play_model import CharacterRunner, default_seed_clip

    def path_for(intent):
        fresh = CharacterRunner(str(CHECKPOINT), default_seed_clip(str(CLIPS_DIR)),
                                 str(find_smplx_model()))
        return np.array([fresh.advance(intent, 1 / 30)[1][:2] for _ in range(60)])

    forward = path_for(PlayerIntent(forward=1.0))
    turning = path_for(PlayerIntent(forward=1.0, turn=1.0))

    assert np.linalg.norm(forward - turning, axis=1).mean() > 0.05, (
        "holding 'turn left' produced the same path as walking straight — "
        "the control is not reaching the model"
    )


@needs_trained_model
def test_a_checkpoint_without_control_is_rejected():
    """Driving a model that was never given control would silently ignore
    every key press, so it is refused up front with an explanation."""
    import torch

    from simulator.live_model import LiveMotionSession
    from src.gpsm.ssm.motion_mamba import MotionMamba
    from simulator.play_model import default_seed_clip

    no_control = MotionMamba({
        "state_dim": 69, "n_embd": 16, "n_layers": 1, "dropout": 0.0, "learning_rate": 1e-3,
        "d_state": 8, "d_conv": 3, "expand": 2, "dt_rank": 2,   # no control_dim -> 0
    })
    with pytest.raises(ValueError, match="control"):
        LiveMotionSession(no_control, default_seed_clip(str(CLIPS_DIR)))


@needs_trained_model
def test_headless_run_renders_real_frames(tmp_path, runner):
    """End to end: the same loop the window uses, rendered offscreen. Guards
    against a scene that silently renders nothing."""
    from simulator.play_model import run_headless

    summary = run_headless(runner, ["forward", "left"], seconds_each=0.4,
                            out_path=tmp_path / "run.gif", viewport=(320, 240))

    assert summary["frames"] == 24            # 2 intents * 0.4s * 30fps
    assert (tmp_path / "run.gif").exists()
    assert summary["distance_m"] > 0


# ---------------------------------------------------------------------------
# Third-person controls: WASD moves the character, the mouse steers it
# ---------------------------------------------------------------------------

def test_strafing_moves_sideways_without_turning():
    """A and D step left/right. That is a different thing from steering:
    the character slides sideways but keeps facing the same way."""
    driver = ControlDriver()
    for _ in range(60):
        control = driver.update(PlayerIntent(strafe=1.0), 1 / 30)

    assert column(driver, control, "vel_left") > 0, "strafe left must be positive sideways speed"
    assert column(driver, control, "pos_left_500ms") > 0
    assert column(driver, control, "turn_sin_500ms") == pytest.approx(0.0, abs=1e-6), (
        "strafing is not turning — the facing must not change"
    )

    right = ControlDriver()
    for _ in range(60):
        right_control = right.update(PlayerIntent(strafe=-1.0), 1 / 30)
    assert column(right, right_control, "vel_left") < 0


def test_mouse_turn_is_continuous_not_just_on_off():
    """Steering comes from the mouse, so a small movement should ask for a
    small turn — otherwise the character can only snap between hard left,
    straight, and hard right."""
    gentle, hard = ControlDriver(), ControlDriver()
    for _ in range(30):
        gentle_control = gentle.update(PlayerIntent(turn=0.2), 1 / 30)
        hard_control = hard.update(PlayerIntent(turn=1.0), 1 / 30)

    assert 0 < gentle.turn_rate < hard.turn_rate
    assert abs(column(gentle, gentle_control, "turn_sin_500ms")) < abs(
        column(hard, hard_control, "turn_sin_500ms")
    )


def test_turn_request_is_cleared_once_read():
    """The viewer hands its accumulated mouse movement over once and then
    forgets it, so a mouse that stops moving stops steering. Without this
    the character would keep spinning after the player let go."""
    from simulator.play_model import PlayerViewer

    viewer = PlayerViewer.__new__(PlayerViewer)   # no window needed for this
    viewer._turn_request = 0.0

    PlayerViewer.on_mouse_motion(viewer, 0, 0, dx=-25, dy=0)
    first = viewer.take_turn_request()
    second = viewer.take_turn_request()

    assert first > 0, "moving the mouse left should ask for a left turn"
    assert second == 0.0, "a second read with no new movement must ask for nothing"


def test_mouse_turn_request_is_clamped():
    """A fast flick must not ask for a physically absurd turn rate."""
    from simulator.play_model import PlayerViewer

    viewer = PlayerViewer.__new__(PlayerViewer)
    viewer._turn_request = 0.0
    PlayerViewer.on_mouse_motion(viewer, 0, 0, dx=-5000, dy=0)

    assert viewer.take_turn_request() == pytest.approx(1.0)
