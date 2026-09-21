"""Invertible conversion between the canonical 159-d axis-angle pose and the
315-d model feature (6D rotations + root linear velocity).

See doc/Mocap_Unification_Design.docx section 2.2. Axis-angle is
discontinuous (the +/-pi antipodal ambiguity, the 2*pi wraparound) and makes
a poor training target; 6D rotation (Zhou et al., 2019) is continuous and
recovered exactly via Gram-Schmidt.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.gpsm.smplx.schema import CANONICAL_DIM

NUM_JOINTS_TOTAL = 1 + 21 + 15 + 15  # global_orient + body + left_hand + right_hand = 52
MODEL_FEATURE_DIM = 3 + NUM_JOINTS_TOTAL * 6  # root linear velocity + 52 * 6D rotations
assert MODEL_FEATURE_DIM == 315


def _axis_angle_to_6d(rotvecs: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 6) 6D rotation (first two matrix columns)."""
    flat = rotvecs.reshape(-1, 3)
    mats = Rotation.from_rotvec(flat).as_matrix()  # (N, 3, 3)
    # First two matrix *columns*, laid out as [col0_xyz, col1_xyz]. mats[:, :, :2]
    # has shape (N, 3, 2) — row-major reshape would interleave the two columns
    # element-by-element instead, so transpose to (N, 2, 3) before flattening.
    six_d = mats[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return six_d.reshape(*rotvecs.shape[:-1], 6)


def _sixd_to_axis_angle(six_d: np.ndarray) -> np.ndarray:
    """(..., 6) 6D rotation -> (..., 3) axis-angle, via Gram-Schmidt."""
    flat = six_d.reshape(-1, 6)
    a1 = flat[:, 0:3]
    a2 = flat[:, 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8, None)
    proj = np.sum(b1 * a2, axis=1, keepdims=True)
    b2_raw = a2 - proj * b1
    b2 = b2_raw / np.clip(np.linalg.norm(b2_raw, axis=1, keepdims=True), 1e-8, None)
    b3 = np.cross(b1, b2)
    mats = np.stack([b1, b2, b3], axis=-1)  # columns b1,b2,b3 -> (N,3,3)
    rotvecs = Rotation.from_matrix(mats).as_rotvec()
    return rotvecs.reshape(*six_d.shape[:-1], 3)


def canonical_to_model_features(canonical_array: np.ndarray) -> np.ndarray:
    """(T, 159) axis-angle -> (T, 315) model features.

    Root translation is expressed as *velocity* (finite difference; the
    first frame's velocity is defined as zero) rather than absolute
    position, and every rotation block becomes a continuous 6D rotation.
    """
    if canonical_array.shape[1] != CANONICAL_DIM:
        raise ValueError(f"Expected (T, {CANONICAL_DIM}), got {canonical_array.shape}")
    t = canonical_array.shape[0]

    trans = canonical_array[:, 0:3]
    root_vel = np.zeros_like(trans)
    root_vel[1:] = trans[1:] - trans[:-1]

    rot_aa = canonical_array[:, 3:159].reshape(t, NUM_JOINTS_TOTAL, 3)
    rot_6d = _axis_angle_to_6d(rot_aa).reshape(t, NUM_JOINTS_TOTAL * 6)

    return np.concatenate([root_vel, rot_6d], axis=1).astype(np.float64)


def model_features_to_canonical(
    features: np.ndarray, initial_trans: np.ndarray
) -> np.ndarray:
    """(T, 315) model features -> (T, 159) axis-angle canonical array.

    `initial_trans` (3,) supplies the absolute root position for frame 0,
    since root position is stored only as a velocity — it must be
    integrated back to an absolute trajectory.
    """
    if features.shape[1] != MODEL_FEATURE_DIM:
        raise ValueError(f"Expected (T, {MODEL_FEATURE_DIM}), got {features.shape}")
    t = features.shape[0]

    root_vel = features[:, 0:3]
    trans = np.cumsum(root_vel, axis=0)
    trans[0] = 0.0
    trans = trans + np.asarray(initial_trans).reshape(1, 3)

    rot_6d = features[:, 3:].reshape(t, NUM_JOINTS_TOTAL, 6)
    rot_aa = _sixd_to_axis_angle(rot_6d).reshape(t, NUM_JOINTS_TOTAL * 3)

    return np.concatenate([trans, rot_aa], axis=1).astype(np.float64)
