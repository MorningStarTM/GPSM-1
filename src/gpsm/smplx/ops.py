"""Coordinate/rate normalization: fps resampling, up-axis, units.

These are always applied, never optional — see doc/Mocap_Unification_Design
section 6.4: the real sample data already spans 100/120 fps and AMASS's
Z-up convention differs from HumanML3D's Y-up.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# Z-up (AMASS/SMPL-X) -> Y-up (HumanML3D-style): rotate -90 degrees about X,
# so old-Z becomes new-Y and old-Y becomes new-(-Z).
_Z_UP_TO_Y_UP = Rotation.from_euler("x", -90, degrees=True)


def convert_up_axis(points: np.ndarray, from_axis: str, to_axis: str) -> np.ndarray:
    """Rotate 3D points between a Z-up and a Y-up world convention.

    `points` has shape (..., 3). No-op if `from_axis == to_axis`.
    """
    if from_axis == to_axis:
        return points
    if {from_axis, to_axis} != {"y", "z"}:
        raise ValueError(f"Unsupported axis pair: {from_axis} -> {to_axis}")
    shape = points.shape
    flat = points.reshape(-1, 3)
    if from_axis == "z" and to_axis == "y":
        rotated = _Z_UP_TO_Y_UP.apply(flat)
    else:  # y -> z
        rotated = _Z_UP_TO_Y_UP.inv().apply(flat)
    return rotated.reshape(shape)


def convert_units(points: np.ndarray, scale_to_meters: float) -> np.ndarray:
    """Rescale point coordinates into meters given a to-meters scale factor."""
    if scale_to_meters <= 0:
        raise ValueError(f"scale_to_meters must be positive, got {scale_to_meters}")
    return points * scale_to_meters


def guess_up_axis(points_m: np.ndarray, point_names, head_keywords=("head", "hd"),
                   foot_keywords=("ankle", "ank", "toe", "heel", "foot")):
    """Heuristic: the axis on which head-ish points sit consistently above
    foot-ish points is "up". Shared by every raw-mocap adapter that has no
    reliable declared world convention (C3D marker labels, BVH joint
    names) — neither format carries a standard up-axis, so this is
    genuinely necessary, not a shortcut.

    Only y/z are representable by this pipeline's up_axis convention; an
    x-up result (rare) falls back to z with `guessed=True` so the caller
    can surface that reduced confidence.

    Returns (up_axis: str, guessed: bool).
    """
    upper = [str(n).upper() for n in point_names]
    head_idx = [i for i, n in enumerate(upper) if any(k.upper() in n for k in head_keywords)]
    foot_idx = [i for i, n in enumerate(upper) if any(k.upper() in n for k in foot_keywords)]
    if not head_idx or not foot_idx:
        return "z", True
    diff = points_m[:, head_idx, :].mean(axis=(0, 1)) - points_m[:, foot_idx, :].mean(axis=(0, 1))
    axis = int(np.argmax(diff))
    if axis == 2:
        return "z", False
    if axis == 1:
        return "y", False
    return "z", True  # x-up is not representable; fall back with a flag


def resample_translation(trans: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly resample a (T, 3) translation signal to a new frame rate."""
    return _resample_linear(trans, src_fps, dst_fps)


def resample_rotations_axis_angle(rotvecs: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample a (T, 3) axis-angle rotation signal via SLERP.

    Never use linear interpolation on axis-angle values directly — the
    discontinuity at +/-pi and the 2*pi wraparound make naive lerp produce
    physically wrong intermediate rotations. SLERP interpolates on the
    rotation manifold itself.
    """
    src_times, dst_times = _resample_time_grids(rotvecs.shape[0], src_fps, dst_fps)
    if rotvecs.shape[0] == 1:
        return np.repeat(rotvecs, len(dst_times), axis=0)
    rotations = Rotation.from_rotvec(rotvecs)
    slerp = Slerp(src_times, rotations)
    dst_times_clamped = np.clip(dst_times, src_times[0], src_times[-1])
    return slerp(dst_times_clamped).as_rotvec()


def resample_pose_sequence(canonical_array: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample a full (T, 159) canonical pose array to `dst_fps`.

    Translation is interpolated linearly; every rotation block (global
    orient, body pose joints, hand joints) is interpolated independently
    via SLERP, each joint's axis-angle triplet treated as one rotation.
    """
    from src.gpsm.smplx.schema import (
        SLICE_TRANS, SLICE_GLOBAL_ORIENT, SLICE_BODY_POSE,
        SLICE_LEFT_HAND, SLICE_RIGHT_HAND, CANONICAL_DIM,
    )

    if canonical_array.shape[1] != CANONICAL_DIM:
        raise ValueError(f"Expected (T, {CANONICAL_DIM}) array, got {canonical_array.shape}")

    t_dst = _num_dst_frames(canonical_array.shape[0], src_fps, dst_fps)
    out = np.empty((t_dst, CANONICAL_DIM), dtype=canonical_array.dtype)

    out[:, SLICE_TRANS] = resample_translation(canonical_array[:, SLICE_TRANS], src_fps, dst_fps)

    rotation_slices = [SLICE_GLOBAL_ORIENT] + [
        slice(SLICE_BODY_POSE.start + 3 * j, SLICE_BODY_POSE.start + 3 * j + 3) for j in range(21)
    ] + [
        slice(SLICE_LEFT_HAND.start + 3 * j, SLICE_LEFT_HAND.start + 3 * j + 3) for j in range(15)
    ] + [
        slice(SLICE_RIGHT_HAND.start + 3 * j, SLICE_RIGHT_HAND.start + 3 * j + 3) for j in range(15)
    ]
    for sl in rotation_slices:
        out[:, sl] = resample_rotations_axis_angle(canonical_array[:, sl], src_fps, dst_fps)

    return out


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _num_dst_frames(t_src: int, src_fps: float, dst_fps: float) -> int:
    duration = (t_src - 1) / src_fps if t_src > 1 else 0.0
    return max(1, int(round(duration * dst_fps)) + 1)


def _resample_time_grids(t_src: int, src_fps: float, dst_fps: float):
    src_times = np.arange(t_src) / src_fps
    t_dst = _num_dst_frames(t_src, src_fps, dst_fps)
    dst_times = np.arange(t_dst) / dst_fps
    return src_times, dst_times


def _resample_linear(signal: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    src_times, dst_times = _resample_time_grids(signal.shape[0], src_fps, dst_fps)
    if signal.shape[0] == 1:
        return np.repeat(signal, len(dst_times), axis=0)
    dst_times_clamped = np.clip(dst_times, src_times[0], src_times[-1])
    out = np.empty((len(dst_times), signal.shape[1]), dtype=signal.dtype)
    for d in range(signal.shape[1]):
        out[:, d] = np.interp(dst_times_clamped, src_times, signal[:, d])
    return out
