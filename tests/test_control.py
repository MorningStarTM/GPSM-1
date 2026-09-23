"""Tests for control extraction (src/gpsm/control/).

Two kinds of test:

* **Synthetic** trajectories whose correct answer is known exactly (a straight
  walk, a circle). These pin down the maths and the sign conventions.
* **Real data** from ``data/``, skipped if the file is not there. These check
  the things synthetic data cannot: that the heading read out of real mocap
  agrees with the direction the character actually moves, that the labelled
  "Walk turn left 90 / 135" clips come out as ~90 / ~135 degrees, and that no
  marker file produces a physically impossible spin (a bug that once happened
  — see ``test_no_impossible_spins_in_marker_files``).

Run with:
    pytest tests/test_control.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from src.gpsm.control.control_features import extract_control
from src.gpsm.control.inspect_control import turn_while_moving_degrees
from src.gpsm.control.root_trajectory import (
    RootTrajectory,
    _fill_gaps,
    heading_from_left_vector,
    load_root_trajectory,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def column(control, name):
    """One named column of a ControlFeatures, for readable assertions."""
    return control.values[:, control.names.index(name)]


def straight_walk(speed=1.5, fps=100.0, seconds=4.0, heading=0.0):
    """A character walking in a straight line, facing where it walks."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    direction = np.array([np.cos(heading), np.sin(heading)])
    return RootTrajectory(
        position=speed * t[:, None] * direction, heading=np.full(n, heading), fps=fps
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_heading_convention():
    """Left = +Y means facing +X (heading 0); left = -X means facing +Y
    (heading +90 degrees, i.e. a left turn is a positive heading change)."""
    heading = heading_from_left_vector(np.array([[0.0, 1.0], [-1.0, 0.0]]))
    assert heading[0] == pytest.approx(0.0)
    assert heading[1] == pytest.approx(np.pi / 2)


def test_fill_gaps_interpolates_and_holds_the_edges():
    filled = _fill_gaps(np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan]))
    assert filled.tolist() == pytest.approx([1.0, 1.0, 2.0, 3.0, 4.0, 4.0])


def test_fill_gaps_with_no_valid_sample_raises():
    with pytest.raises(ValueError):
        _fill_gaps(np.array([np.nan, np.nan]))


# ---------------------------------------------------------------------------
# Synthetic trajectories with known answers
# ---------------------------------------------------------------------------

def test_straight_walk_gives_pure_forward_control():
    """Walking straight at 1.5 m/s: velocity is all forward, the root will be
    1.5 * t ahead after t seconds, dead ahead, with no turning."""
    control = extract_control(straight_walk(speed=1.5))
    valid = control.valid

    assert column(control, "vel_fwd")[valid] == pytest.approx(1.5, abs=1e-6)
    assert column(control, "vel_left")[valid] == pytest.approx(0.0, abs=1e-6)
    assert column(control, "pos_fwd_500ms")[valid] == pytest.approx(0.75, abs=1e-6)
    assert column(control, "pos_left_500ms")[valid] == pytest.approx(0.0, abs=1e-6)
    assert column(control, "turn_cos_500ms")[valid] == pytest.approx(1.0, abs=1e-6)
    assert column(control, "turn_sin_500ms")[valid] == pytest.approx(0.0, abs=1e-6)


def test_counter_clockwise_circle_is_a_left_turn():
    """Walking counter-clockwise on a circle (radius 2 m, 1 m/s) turns left at
    0.5 rad/s, so over 0.5 s the heading changes by 0.25 rad, and the root
    ends up slightly to the LEFT (positive) of straight ahead."""
    fps, radius, speed = 100.0, 2.0, 1.0
    omega = speed / radius
    t = np.arange(int(6 * fps)) / fps
    angle = omega * t
    trajectory = RootTrajectory(
        position=radius * np.stack([np.sin(angle), 1 - np.cos(angle)], axis=1),
        heading=angle,  # facing along the direction of travel
        fps=fps,
    )

    control = extract_control(trajectory)
    valid = control.valid

    assert np.arcsin(column(control, "turn_sin_500ms")[valid]) == pytest.approx(0.25, abs=1e-6)
    assert (column(control, "pos_left_500ms")[valid] > 0).all()
    assert column(control, "vel_fwd")[valid] == pytest.approx(1.0, abs=1e-3)


def test_control_does_not_depend_on_where_or_which_way_the_clip_was_recorded():
    """The point of expressing everything in the character's own frame: the
    same motion, recorded at a different place and facing a different
    direction in the world, must give identical control values. (Otherwise a
    model would learn the room, not the motion.)"""
    rng = np.random.default_rng(0)
    fps = 100.0
    n = 400
    heading = np.cumsum(rng.normal(0, 0.02, n))  # a wandering path
    step = 0.015 * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    original = RootTrajectory(position=np.cumsum(step, axis=0), heading=heading, fps=fps)

    angle, shift = 1.234, np.array([7.0, -3.0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = RootTrajectory(
        position=original.position @ rotation.T + shift, heading=heading + angle, fps=fps
    )

    np.testing.assert_allclose(
        extract_control(original).values, extract_control(moved).values, atol=1e-9
    )


def test_control_is_the_same_at_different_frame_rates():
    """Look-ahead is defined in seconds, so the same physical motion recorded
    at 60 fps and at 120 fps must give the same control values."""
    slow, fast = straight_walk(fps=60.0), straight_walk(fps=120.0)
    a, b = extract_control(slow), extract_control(fast)
    assert a.names == b.names
    np.testing.assert_allclose(a.values[a.valid][0], b.values[b.valid][0], atol=1e-6)


def test_last_frames_are_marked_invalid():
    """The final frames have no full 0.5 s of future, so they must be flagged
    rather than silently given a made-up one: exactly 0.5 s * fps of them."""
    fps = 100.0
    control = extract_control(straight_walk(fps=fps, seconds=3.0))
    assert (~control.valid).sum() == 50
    assert not control.valid[-1] and control.valid[0]


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def _first_match(pattern):
    """Find a sample file anywhere under data/.

    Searched recursively (``rglob``) on purpose: data/ has been organised
    into per-format subfolders (data/c3d/, data/npz/) at least once already,
    and a flat glob silently turned these real-data tests into skips when
    that happened — coverage quietly disappearing is worse than a failure.
    """
    matches = sorted(DATA_DIR.rglob(pattern))
    if not matches:
        pytest.skip(f"no file matching {pattern} under data/")
    return str(matches[0])


@pytest.mark.parametrize("pattern, labelled_degrees", [
    ("B9*.c3d", 90.0),    # "Walk turn left 90"
    ("B11*.c3d", 135.0),  # "Walk turn left 135"
])
def test_labelled_turn_clips_come_out_near_their_labels(pattern, labelled_degrees):
    """Ground truth from the file names: the clip labelled "turn left N"
    should show a left (positive) turn of about N degrees while walking.
    Tolerance is 15 degrees: a person does not turn exactly N degrees."""
    trajectory = load_root_trajectory(_first_match(pattern))
    assert turn_while_moving_degrees(trajectory) == pytest.approx(labelled_degrees, abs=15.0)


@pytest.mark.parametrize("name", [
    "12_L_2_stageii.npz", "SW_B_3_stageii.npz", "run_stand_stageii.npz",
    "turntwist_walk_stageii.npz", "0013_knocking1_poses.npz",
])
def test_npz_heading_matches_direction_of_travel(name):
    """The heading is read from the root *rotation*; the direction of travel
    comes from the root *position*. They are independent measurements of the
    same thing while walking forward, so they must agree. Covers both npz
    layouts in data/ (SMPL-X 'stageii' and SMPL-H)."""
    trajectory = load_root_trajectory(_first_match(name))
    half_window = max(1, int(round(0.05 * trajectory.fps)))
    position = trajectory.position
    velocity = (position[2 * half_window:] - position[:-2 * half_window]) / (2 * half_window / trajectory.fps)
    moving = np.linalg.norm(velocity, axis=1) > 0.8
    if not moving.any():
        pytest.skip("character never moves fast enough in this clip")

    travel_direction = np.arctan2(velocity[:, 1], velocity[:, 0])
    difference = np.angle(np.exp(1j * (travel_direction - trajectory.heading[half_window:-half_window])))
    assert np.abs(np.median(difference[moving])) < np.radians(20)


def test_no_impossible_spins_in_marker_files():
    """Regression test. An earlier version averaged 'whichever waist markers
    are visible this frame'; when markers dropped out, the estimate jumped and
    one file (B17) showed a 99th-percentile turn rate of ~3000 deg/s — about
    8 full spins per second. No person does that, so any value that high means
    the heading estimate is broken, not that the character spun."""
    files = sorted(DATA_DIR.rglob("*.c3d"))
    if not files:
        pytest.skip("no .c3d files in data/")
    for path in files:
        trajectory = load_root_trajectory(str(path))
        turn_rate = np.degrees(np.abs(np.diff(trajectory.heading))) * trajectory.fps
        assert np.percentile(turn_rate, 99) < 500, f"{path.name}: implausible turn rate"
