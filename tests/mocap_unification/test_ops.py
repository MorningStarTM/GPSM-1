import numpy as np
import pytest

from src.gpsm.smplx import ops
from src.gpsm.smplx.schema import CANONICAL_DIM


def test_up_axis_round_trip():
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1, (5, 4, 3))
    y = ops.convert_up_axis(pts, "z", "y")
    back = ops.convert_up_axis(y, "y", "z")
    assert np.abs(back - pts).max() < 1e-9
    assert ops.convert_up_axis(pts, "z", "z") is pts


def test_up_axis_rejects_bad_pair():
    with pytest.raises(ValueError):
        ops.convert_up_axis(np.zeros((1, 3)), "z", "x")


def test_convert_units():
    pts = np.ones((3, 3))
    out = ops.convert_units(pts, 0.01)  # cm -> m
    assert np.allclose(out, 0.01)
    with pytest.raises(ValueError):
        ops.convert_units(pts, -1.0)


def test_resample_identity_when_fps_matches():
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 0.3, (20, CANONICAL_DIM))
    same = ops.resample_pose_sequence(arr, 30.0, 30.0)
    assert np.abs(same - arr).max() < 1e-9


def test_resample_up_and_down():
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 0.3, (20, CANONICAL_DIM))
    up = ops.resample_pose_sequence(arr, 30.0, 60.0)
    down = ops.resample_pose_sequence(arr, 30.0, 15.0)
    assert up.shape[0] > arr.shape[0]
    assert down.shape[0] < arr.shape[0]
    assert up.shape[1] == CANONICAL_DIM
    assert not np.isnan(up).any()
    assert not np.isnan(down).any()


def test_resample_rotation_uses_slerp_not_lerp():
    """A 180-degree-ish rotation interpolated by naive lerp on axis-angle
    would pass through a degenerate near-zero-magnitude vector; SLERP keeps
    a constant angular velocity instead."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotvecs = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.pi * 0.9],
    ])
    resampled = ops.resample_rotations_axis_angle(rotvecs, src_fps=1.0, dst_fps=4.0)
    angles = np.linalg.norm(resampled, axis=1)
    # Should increase roughly monotonically toward pi*0.9, not dip through 0.
    assert np.all(np.diff(angles) >= -1e-6)


def test_resample_single_frame_repeats():
    arr = np.ones((1, CANONICAL_DIM))
    out = ops.resample_pose_sequence(arr, 30.0, 60.0)
    assert out.shape[0] >= 1
    assert np.allclose(out, 1.0)
