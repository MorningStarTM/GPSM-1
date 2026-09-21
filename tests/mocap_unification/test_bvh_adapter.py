import os

import numpy as np
import pytest

from src.gpsm.smplx.adapters.bvh import (
    _guess_units_to_meters, load_bvh, match_bvh_layout,
)
from src.gpsm.smplx.fitting.solver import fit_observation
from src.gpsm.smplx.schema import Observation

SAMPLE_BVH = "C:/Users/Ernest/Downloads/MoCap/Male1_bvh/Male1_A10_LieToCrouch.bvh"
TRAVELING_BVH = "C:/Users/Ernest/Downloads/MoCap/Male1_bvh/Male1_A13_Skipping.bvh"
requires_sample_bvh = pytest.mark.skipif(not os.path.isfile(SAMPLE_BVH), reason="sample BVH not found")
requires_traveling_bvh = pytest.mark.skipif(not os.path.isfile(TRAVELING_BVH), reason="sample BVH not found")


@requires_sample_bvh
def test_load_real_bvh_file():
    cap = load_bvh(SAMPLE_BVH)
    assert cap.points_m.ndim == 3 and cap.points_m.shape[2] == 3
    assert cap.points_m.shape[1] == len(cap.joint_names)
    assert cap.fps > 0
    assert not np.isnan(cap.points_m).any()
    assert cap.parents[0] == -1  # root has no parent
    assert all(p < i for i, p in enumerate(cap.parents) if p >= 0)  # parent-before-child ordering


@requires_sample_bvh
def test_match_bvh_layout_finds_cmu_style_names():
    cap = load_bvh(SAMPLE_BVH)
    layout, cols = match_bvh_layout(cap.joint_names)
    assert layout.num_points == len(cols)
    assert layout.num_points >= 15  # this rig has 22 joints, ~21 should match
    assert not layout.is_joint_only or layout.is_joint_only  # sanity: property doesn't raise
    assert set(cols).issubset(set(range(len(cap.joint_names))))


def test_match_bvh_layout_raises_on_unrecognized_names():
    with pytest.raises(ValueError):
        match_bvh_layout(["bone_a", "bone_b", "bone_c"])


@requires_sample_bvh
def test_fit_real_bvh_end_to_end(smplx_body):
    cap = load_bvh(SAMPLE_BVH)
    layout, cols = match_bvh_layout(cap.joint_names)
    t = 10
    points = cap.points_m[:t][:, cols, :]
    conf = np.ones((t, len(cols)))
    obs = Observation(points=points, conf=conf, joint_names=layout.point_names,
                       space="world3d", fps=cap.fps, up_axis=cap.up_axis)
    motion, quality = fit_observation(obs, smplx_body, layout)
    assert not np.isnan(motion.to_array()).any()
    assert quality.verdict in ("GOOD", "DEGRADED", "REJECT")  # always a real verdict, never a crash


@requires_traveling_bvh
def test_units_heuristic_is_not_confused_by_locomotion_travel():
    """Regression test for the actual bug found: a traveling clip's overall
    XYZ bounding-box diagonal is dominated by how far the character moves
    across the room, not by body size — using that diagonal for the units
    heuristic silently picked the wrong scale (a ~0.19m first attempt
    trying to read a value that should be ~1.9m). The fix uses only the
    per-frame span along the up-axis, which travel does not affect."""
    cap = load_bvh(TRAVELING_BVH)
    # A real per-frame body height must land in a plausible human range,
    # not be swamped by the multi-meter travel distance also present in
    # this clip's horizontal axes.
    up_idx = {"x": 0, "y": 1, "z": 2}[cap.up_axis]
    per_frame_height = cap.points_m[:, :, up_idx].max(axis=1) - cap.points_m[:, :, up_idx].min(axis=1)
    assert 0.5 < per_frame_height.max() < 2.2

    # And the horizontal travel distance is legitimately large and must
    # NOT have been mistaken for body size (this is what the old,
    # norm-based heuristic got wrong).
    horiz_idx = [i for i in range(3) if i != up_idx]
    horiz_extent = (cap.points_m[:, :, horiz_idx].max(axis=(0, 1))
                    - cap.points_m[:, :, horiz_idx].min(axis=(0, 1)))
    assert horiz_extent.max() > 2.0


def test_guess_units_uses_up_axis_only_not_full_diagonal():
    """Direct unit test of the fixed heuristic: a synthetic clip with a
    small vertical span but a huge horizontal travel must still be
    recognized as already-correct-scale (not rescaled down) because only
    the vertical axis is consulted."""
    t = 5
    positions = np.zeros((t, 3, 3))
    # Vertical (axis 1) span: 1.7m tall figure, per frame.
    positions[:, 0, 1] = 0.0
    positions[:, 1, 1] = 1.7
    # Horizontal (axis 0) travel: character moves 50m across frames.
    for f in range(t):
        positions[f, :, 0] = f * 10.0
    scale, guessed = _guess_units_to_meters(positions, up_axis_index=1)
    assert scale == 1.0  # already in meters; must not be misled by the 50m horizontal spread
    assert not guessed


def test_normalize_bvh_name_strips_mixamo_prefix():
    from src.gpsm.smplx.adapters.bvh import _normalize_bvh_name
    assert _normalize_bvh_name("mixamorig:Hips") == "hips"
    assert _normalize_bvh_name("mixamorig_LeftArm") == "leftarm"
    assert _normalize_bvh_name("Hips") == "hips"
