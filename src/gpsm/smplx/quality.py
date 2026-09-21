"""Fit-quality scoring, usable independently of which adapter produced a
CanonicalMotion. See doc/Mocap_Unification_Design.docx section 8.

`fitting/solver.py` already computes and attaches a `FitQuality` for fits it
produces; this module exists so Tier-0 (pure parameter conversion, no
fitting error to speak of) and any future adapter can score results the
same way, and so the verdict thresholds live in one place.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.gpsm.smplx.schema import CanonicalMotion, FitQuality, PART_NAMES


def observed_fraction_from_mask(part_mask: np.ndarray) -> float:
    """Fraction of (frame, part) cells actually observed, ignoring FACE —
    face is never solved anywhere in this package (see solver.py), so
    including it would make every single fit look artificially incomplete."""
    body_cols = [i for i, name in enumerate(PART_NAMES) if name != "face"]
    return float(part_mask[:, body_cols].mean())


def detect_acceleration_spikes(trans: np.ndarray, fps: float, threshold_m_s2: float = 15.0) -> int:
    """Count frames whose translational acceleration exceeds a physically
    implausible threshold (default 15 m/s^2, well above normal human sprint
    acceleration) — a cheap proxy for "the fit jumped/teleported"."""
    if trans.shape[0] < 3:
        return 0
    dt = 1.0 / fps
    vel = np.diff(trans, axis=0) / dt
    accel = np.diff(vel, axis=0) / dt
    accel_mag = np.linalg.norm(accel, axis=1)
    return int((accel_mag > threshold_m_s2).sum())


def foot_skate_cm_s(joint_positions: np.ndarray, foot_indices, fps: float,
                     contact_height_m: float = 0.05) -> Optional[float]:
    """Mean horizontal foot speed while a foot is judged to be in ground
    contact (height below `contact_height_m`). Returns None if
    `joint_positions` is not supplied (this is an optional, best-effort
    signal, not always available from every adapter)."""
    if joint_positions is None:
        return None
    dt = 1.0 / fps
    speeds = []
    for idx in foot_indices:
        pos = joint_positions[:, idx, :]  # (T, 3), assumed Z-up
        contact = pos[:-1, 2] < contact_height_m
        horiz_vel = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / dt
        if contact.any():
            speeds.append(horiz_vel[contact])
    if not speeds:
        return 0.0
    return float(np.concatenate(speeds).mean() * 100.0)  # m/s -> cm/s


def score_canonical_motion(
    motion: CanonicalMotion,
    joint_rmse_mm: float = float("nan"),
    joint_positions: Optional[np.ndarray] = None,
    foot_indices: Optional[list] = None,
) -> FitQuality:
    """Assemble a FitQuality for any CanonicalMotion, fitted or converted."""
    observed_fraction = observed_fraction_from_mask(motion.part_mask)
    accel_spikes = detect_acceleration_spikes(motion.trans, motion.fps)
    skate = (
        foot_skate_cm_s(joint_positions, foot_indices, motion.fps)
        if (joint_positions is not None and foot_indices)
        else float("nan")
    )
    quality = FitQuality(
        joint_rmse_mm=joint_rmse_mm,
        observed_fraction=observed_fraction,
        foot_skate_cm_s=skate if skate is not None else float("nan"),
        accel_spike_count=accel_spikes,
    )
    return quality.finalize()
