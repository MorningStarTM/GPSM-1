import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.gpsm.smplx.fitting.align import umeyama_alignment


def test_exact_recovery_no_noise():
    rng = np.random.default_rng(3)
    src = rng.normal(0, 1, (12, 3))
    R_true = Rotation.random(random_state=rng).as_matrix()
    s_true, t_true = 1.7, rng.normal(0, 5, 3)
    tgt = s_true * (src @ R_true.T) + t_true

    result = umeyama_alignment(src, tgt)
    assert np.abs(result.R - R_true).max() < 1e-8
    assert abs(result.s - s_true) < 1e-8
    assert np.abs(result.t - t_true).max() < 1e-6
    assert np.abs(result.apply(src) - tgt).max() < 1e-6


def test_missing_points_are_excluded_not_zero_filled():
    rng = np.random.default_rng(4)
    src = rng.normal(0, 1, (10, 3))
    R_true = Rotation.random(random_state=rng).as_matrix()
    s_true, t_true = 0.8, rng.normal(0, 3, 3)
    tgt = s_true * (src @ R_true.T) + t_true

    weights = np.ones(10)
    drop = [2, 5, 7]
    weights[drop] = 0.0
    tgt_corrupted = tgt.copy()
    tgt_corrupted[drop] = 9999.0  # garbage that would wreck the fit if included

    result = umeyama_alignment(src, tgt_corrupted, weights=weights)
    assert np.abs(result.R - R_true).max() < 1e-6
    assert abs(result.s - s_true) < 1e-6


def test_raises_below_minimum_points():
    with pytest.raises(ValueError):
        umeyama_alignment(np.zeros((2, 3)), np.zeros((2, 3)))


def test_raises_on_degenerate_source():
    src = np.zeros((5, 3))  # all points identical -> zero variance
    tgt = np.random.default_rng(0).normal(0, 1, (5, 3))
    with pytest.raises(ValueError):
        umeyama_alignment(src, tgt)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        umeyama_alignment(np.zeros((5, 3)), np.zeros((4, 3)))
