import numpy as np

from src.gpsm.smplx.quality import (
    detect_acceleration_spikes, observed_fraction_from_mask, score_canonical_motion,
)
from src.gpsm.smplx.schema import CanonicalMotion, NUM_PARTS


def _motion(t=10, part_mask=None):
    rng = np.random.default_rng(0)
    # Smooth (small constant-velocity) translation, matching what a
    # converged fit actually looks like. iid per-frame jitter would double-
    # differentiate into enormous spurious acceleration regardless of
    # quality (1cm iid noise at 30fps aliases to ~tens of m/s^2) — that is
    # a real property of finite differencing, not something to hide from
    # the detector, so the *test data* is smooth here rather than the
    # detector being loosened.
    trans = np.cumsum(np.full((t, 3), 0.01), axis=0) + rng.normal(0, 0.0005, (t, 3))
    return CanonicalMotion(
        trans=trans,
        global_orient=rng.normal(0, 0.1, (t, 3)),
        body_pose=rng.normal(0, 0.1, (t, 63)),
        left_hand_pose=np.zeros((t, 45)),
        right_hand_pose=np.zeros((t, 45)),
        betas=np.zeros(10),
        part_mask=part_mask if part_mask is not None else np.ones((t, NUM_PARTS)),
        fps=30.0,
    )


def test_observed_fraction_ignores_face():
    mask = np.ones((5, NUM_PARTS))
    mask[:, 4] = 0.0  # face always unobserved in this pipeline
    assert observed_fraction_from_mask(mask) == 1.0  # face excluded from the score


def test_observed_fraction_reflects_missing_body_parts():
    mask = np.ones((5, NUM_PARTS))
    mask[:, 2] = 0.0  # left hand unobserved
    mask[:, 3] = 0.0  # right hand unobserved
    frac = observed_fraction_from_mask(mask)
    assert 0.4 < frac < 0.6  # 2 of 4 non-face parts unobserved


def test_acceleration_spike_detection():
    t = 10
    smooth = np.cumsum(np.full((t, 3), 0.01), axis=0)
    assert detect_acceleration_spikes(smooth, fps=30.0) == 0

    jumpy = smooth.copy()
    jumpy[5] += 10.0  # a teleport
    assert detect_acceleration_spikes(jumpy, fps=30.0) > 0


def test_score_canonical_motion_end_to_end():
    motion = _motion()
    q = score_canonical_motion(motion, joint_rmse_mm=12.0)
    assert q.verdict == "GOOD"
    mask = np.ones((10, NUM_PARTS))
    mask[:, 2:4] = 0.0
    motion2 = _motion(part_mask=mask)
    q2 = score_canonical_motion(motion2, joint_rmse_mm=12.0)
    assert q2.observed_fraction < q.observed_fraction
