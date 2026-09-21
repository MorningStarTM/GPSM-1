import numpy as np
import pytest

from src.gpsm.smplx.schema import (
    CANONICAL_DIM, NUM_PARTS, CanonicalMotion, FitQuality, frozen_part_mask,
)


def _make_motion(t=10):
    rng = np.random.default_rng(0)
    return CanonicalMotion(
        trans=rng.normal(0, 1, (t, 3)),
        global_orient=rng.normal(0, 1, (t, 3)),
        body_pose=rng.normal(0, 1, (t, 63)),
        left_hand_pose=rng.normal(0, 1, (t, 45)),
        right_hand_pose=rng.normal(0, 1, (t, 45)),
        betas=rng.normal(0, 1, (10,)),
        part_mask=np.ones((t, NUM_PARTS)),
        fps=30.0,
        meta={"k": "v"},
    )


def test_array_round_trip():
    m = _make_motion()
    arr = m.to_array()
    assert arr.shape == (10, CANONICAL_DIM)
    m2 = CanonicalMotion.from_array(arr, m.betas, m.part_mask, fps=30.0)
    assert np.allclose(m2.to_array(), arr)


def test_npz_round_trip():
    m = _make_motion()
    d = m.to_npz_dict()
    m2 = CanonicalMotion.from_npz_dict(d)
    assert np.allclose(m2.to_array(), m.to_array())
    assert m2.meta == {"k": "v"}
    assert np.isclose(m2.fps, 30.0)


@pytest.mark.parametrize("field,bad_shape", [
    ("trans", (10, 2)),
    ("body_pose", (10, 62)),
    ("left_hand_pose", (9, 45)),
])
def test_shape_validation_rejects_bad_arrays(field, bad_shape):
    m = _make_motion()
    kwargs = dict(
        trans=m.trans, global_orient=m.global_orient, body_pose=m.body_pose,
        left_hand_pose=m.left_hand_pose, right_hand_pose=m.right_hand_pose,
        betas=m.betas, part_mask=m.part_mask, fps=30.0,
    )
    kwargs[field] = np.zeros(bad_shape)
    with pytest.raises(ValueError):
        CanonicalMotion(**kwargs)


def test_frozen_part_mask_detects_constant_and_not_varying():
    constant = np.tile(np.array([0.1, -0.4, 0.2]), (100, 1))
    varying = constant + np.random.default_rng(1).normal(0, 0.5, constant.shape)
    assert frozen_part_mask(constant) is True
    assert frozen_part_mask(varying) is False


def test_fit_quality_verdicts():
    assert FitQuality(joint_rmse_mm=10, observed_fraction=1.0).finalize().verdict == "GOOD"
    assert FitQuality(joint_rmse_mm=45, observed_fraction=1.0).finalize().verdict == "DEGRADED"
    assert FitQuality(joint_rmse_mm=10, observed_fraction=0.3).finalize().verdict == "DEGRADED"
    assert FitQuality(joint_rmse_mm=90, observed_fraction=1.0).finalize().verdict == "REJECT"
    assert FitQuality(joint_rmse_mm=10, observed_fraction=0.05).finalize().verdict == "REJECT"
