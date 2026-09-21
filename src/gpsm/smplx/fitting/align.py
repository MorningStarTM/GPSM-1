"""Similarity alignment (Umeyama / Kabsch with scale).

See doc/SMPLX_Fitting_Methods.docx section 1. Closed-form solve for the
rotation, uniform scale, and translation that best align a source point set
onto a target point set, weighted by confidence. Used as the initialization
for Method 2's Stage 1 (global-only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimilarityTransform:
    R: np.ndarray  # (3, 3) rotation
    s: float       # uniform scale
    t: np.ndarray  # (3,) translation

    def apply(self, points: np.ndarray) -> np.ndarray:
        """points: (..., 3) -> (..., 3), transformed as s * R @ p + t."""
        shape = points.shape
        flat = points.reshape(-1, 3)
        out = self.s * (flat @ self.R.T) + self.t
        return out.reshape(shape)


def umeyama_alignment(source: np.ndarray, target: np.ndarray,
                       weights: np.ndarray = None) -> SimilarityTransform:
    """Solve for (R, s, t) minimizing sum_i w_i * || s*R*source_i + t - target_i ||^2.

    `source`, `target`: (N, 3). `weights`: (N,) confidence/structural weight,
    defaults to uniform. Points with weight <= 0 are excluded entirely.
    Requires at least 3 non-collinear, positively-weighted correspondences.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[1] != 3:
        raise ValueError(f"source/target must both be (N,3); got {source.shape} vs {target.shape}")
    n = source.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = weights > 0
    n_valid = int(valid.sum())
    if n_valid < 3:
        raise ValueError(
            f"umeyama_alignment needs >= 3 positively-weighted correspondences, got {n_valid}"
        )

    w = weights[valid]
    src = source[valid]
    tgt = target[valid]
    w_sum = w.sum()
    w_norm = (w / w_sum).reshape(-1, 1)

    mu_src = (w_norm * src).sum(axis=0)
    mu_tgt = (w_norm * tgt).sum(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    # weighted cross-covariance
    H = (src_c * w_norm).T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    var_src = float((w_norm[:, 0] * np.sum(src_c ** 2, axis=1)).sum())
    if var_src < 1e-12:
        raise ValueError("Source points are degenerate (near-zero variance); cannot solve scale.")

    # A degenerate *target* (all confident points coincide, e.g. a tracker
    # reporting the same coordinate for a lost frame) makes the
    # cross-covariance H identically zero, so its SVD gives S = [0, 0, 0]
    # and the scale formula below would silently evaluate to exactly 0
    # rather than failing — which then poisons every downstream use of this
    # transform with a confusing, misattributed "source is degenerate"
    # error. Check target variance explicitly and fail here instead, with
    # a message that names the actual cause.
    var_tgt = float((w_norm[:, 0] * np.sum(tgt_c ** 2, axis=1)).sum())
    if var_tgt < 1e-12:
        raise ValueError("Target points are degenerate (near-zero variance); cannot solve rotation/scale.")

    scale = float(np.sum(S * np.diag(D)) / var_src)

    t = mu_tgt - scale * (R @ mu_src)
    return SimilarityTransform(R=R, s=scale, t=t)
