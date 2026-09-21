import numpy as np

from src.gpsm.smplx.features import (
    MODEL_FEATURE_DIM, canonical_to_model_features, model_features_to_canonical,
)
from src.gpsm.smplx.schema import CANONICAL_DIM


def test_round_trip_random():
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 0.4, (15, CANONICAL_DIM))
    feats = canonical_to_model_features(arr)
    assert feats.shape == (15, MODEL_FEATURE_DIM)
    back = model_features_to_canonical(feats, initial_trans=arr[0, :3])
    assert np.abs(back - arr).max() < 1e-6


def test_round_trip_near_singular_rotations():
    """Rotations near the axis-angle singularities (pi, 2*pi) must still
    round-trip through the 6D representation without blowing up."""
    rng = np.random.default_rng(1)
    t = 8
    arr = np.zeros((t, CANONICAL_DIM))
    arr[:, 0:3] = rng.normal(0, 1, (t, 3))  # trans
    # push several rotation triplets close to pi magnitude
    for j in range(52):
        axis = rng.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        arr[:, 3 + 3 * j: 6 + 3 * j] = axis * (np.pi - 1e-3)
    feats = canonical_to_model_features(arr)
    back = model_features_to_canonical(feats, initial_trans=arr[0, :3])
    # Compare via rotation matrices (axis-angle itself is not unique at the
    # +/-pi boundary — e.g. +pi*axis and -pi*(-axis) are the same rotation).
    from scipy.spatial.transform import Rotation
    r1 = Rotation.from_rotvec(arr[:, 3:159].reshape(-1, 3)).as_matrix()
    r2 = Rotation.from_rotvec(back[:, 3:159].reshape(-1, 3)).as_matrix()
    assert np.abs(r1 - r2).max() < 1e-4
    assert np.abs(back[:, :3] - arr[:, :3]).max() < 1e-6


def test_root_velocity_integration():
    """A constant per-frame displacement should integrate back to the
    correct absolute trajectory."""
    t = 6
    arr = np.zeros((t, CANONICAL_DIM))
    step = np.array([0.1, 0.0, -0.05])
    for i in range(t):
        arr[i, 0:3] = step * i
    feats = canonical_to_model_features(arr)
    assert np.allclose(feats[1:, 0:3], step)
    back = model_features_to_canonical(feats, initial_trans=arr[0, :3])
    assert np.allclose(back[:, 0:3], arr[:, 0:3])
