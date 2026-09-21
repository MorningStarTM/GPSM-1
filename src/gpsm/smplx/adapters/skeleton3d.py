"""Tier 1: 3D joint arrays with a known layout (BVH, Mixamo, FBX, OptiTrack,
generic keypoints) -> CanonicalMotion, via the staged-optimization solver.

This is the thin, user-facing entry point; the actual fitting machinery
lives in fitting/solver.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.gpsm.smplx.fitting.body import SMPLXBody
from src.gpsm.smplx.fitting.priors import PosePrior
from src.gpsm.smplx.fitting.solver import SolverConfig, fit_observation
from src.gpsm.smplx.layouts import MotionLayout, detect_layout, get_layout
from src.gpsm.smplx.schema import CanonicalMotion, FitQuality, Observation


def ingest_skeleton3d(
    points: np.ndarray,
    fps: float,
    body: SMPLXBody,
    layout: Optional[str] = None,
    conf: Optional[np.ndarray] = None,
    space: str = "world3d",
    units_to_meters: float = 1.0,
    up_axis: str = "z",
    prior: Optional[PosePrior] = None,
    config: Optional[SolverConfig] = None,
) -> "tuple[CanonicalMotion, FitQuality]":
    """Fit a (T, K, 3) 3D joint array to SMPL-X.

    `layout`: a registered layout name (e.g. "coco_17"). If omitted, the
    layout is auto-detected from K — this raises if K is ambiguous between
    multiple registered layouts (e.g. 17, shared by coco_17 and h36m_17).
    `conf`: (T, K) confidence, defaulting to all-ones (fully observed) if
    omitted — pass real per-point confidence whenever the source can
    provide it; never silently encode "unobserved" as a zero point.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"Expected points of shape (T, K, 3), got {points.shape}")

    resolved_layout: MotionLayout = get_layout(layout) if layout else detect_layout(points.shape[1])

    if conf is None:
        conf = np.ones(points.shape[:2], dtype=np.float64)
    else:
        conf = np.asarray(conf, dtype=np.float64)

    observation = Observation(
        points=points, conf=conf, joint_names=resolved_layout.point_names,
        space=space, fps=fps, units_to_meters=units_to_meters, up_axis=up_axis,
    )
    return fit_observation(observation, body, resolved_layout, prior=prior, config=config)
