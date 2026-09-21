"""Loss terms for the staged optimizer. See doc/SMPLX_Fitting_Methods.docx
section 2.2.

All functions are pure torch so they compose inside a single autograd graph
optimized by LBFGS in solver.py.
"""
from __future__ import annotations

import torch


def geman_mcclure(residual_sq: torch.Tensor, sigma: float = 100.0) -> torch.Tensor:
    """Robust kernel: rho(x^2) = x^2 / (x^2 + sigma^2).

    Bounded above by 1 as the residual grows, so a single wildly-wrong
    observation (e.g. a MediaPipe misdetection) cannot dominate the sum the
    way it would under plain squared error.
    """
    return residual_sq / (residual_sq + sigma ** 2)


def robust_data_term(pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor,
                      sigma: float = 0.1) -> torch.Tensor:
    """Confidence-weighted robust distance between predicted and observed points.

    pred, target: (..., K, 3). conf: (..., K), 0 at unobserved points.
    `sigma` is in the same units as the points (meters) — it sets the
    residual magnitude at which the robust kernel starts saturating.
    """
    residual_sq = ((pred - target) ** 2).sum(dim=-1)  # (..., K)
    robust = geman_mcclure(residual_sq, sigma=sigma)
    weighted = conf * robust
    denom = conf.sum().clamp_min(1e-8)
    return weighted.sum() / denom


def geodesic_so3_distance(aa_a: torch.Tensor, aa_b: torch.Tensor) -> torch.Tensor:
    """Geodesic (great-circle) distance between two batches of rotations,
    given as axis-angle (..., 3) each. Never use plain L2 on axis-angle —
    it misjudges distance across the +/-pi antipodal ambiguity and the
    2*pi wraparound.

    Returns the rotation angle (radians) of R_a^T @ R_b, per element,
    shape (...).
    """
    from smplx.lbs import batch_rodrigues

    shape = aa_a.shape[:-1]
    Ra = batch_rodrigues(aa_a.reshape(-1, 3)).reshape(*shape, 3, 3)
    Rb = batch_rodrigues(aa_b.reshape(-1, 3)).reshape(*shape, 3, 3)
    R_rel = torch.matmul(Ra.transpose(-1, -2), Rb)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    # acos'(x) -> -inf as x -> 1, so the clamp epsilon must be tight enough
    # that it doesn't itself become the dominant source of error near
    # identical rotations (float64 precision near trace=3 is ~1e-15; a
    # looser epsilon such as 1e-7 would inflate a true-zero angle to
    # sqrt(2e-7) ~ 4.5e-4, which is what a first version of this function did).
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-12, 1.0 - 1e-12)
    return torch.acos(cos_angle)


def temporal_smoothness(pose_seq: torch.Tensor) -> torch.Tensor:
    """Mean squared geodesic distance between consecutive frames.

    pose_seq: (T, J, 3) axis-angle per joint. Requires T >= 2.
    """
    t = pose_seq.shape[0]
    if t < 2:
        return torch.zeros((), dtype=pose_seq.dtype, device=pose_seq.device)
    dist = geodesic_so3_distance(pose_seq[:-1], pose_seq[1:])  # (T-1, J)
    return (dist ** 2).mean()


def translation_acceleration(trans_seq: torch.Tensor) -> torch.Tensor:
    """Mean squared second-difference of a (T, 3) translation signal — an
    acceleration/jerk penalty discouraging physically implausible motion."""
    t = trans_seq.shape[0]
    if t < 3:
        return torch.zeros((), dtype=trans_seq.dtype, device=trans_seq.device)
    accel = trans_seq[2:] - 2 * trans_seq[1:-1] + trans_seq[:-2]
    return (accel ** 2).mean()


def joint_limit_penalty(pose: torch.Tensor, limit: float) -> torch.Tensor:
    """Soft anatomical limit: penalize |angle| exceeding `limit` (radians)
    per axis-angle component. pose: (..., 3) or (..., J, 3)."""
    excess = torch.clamp(pose.abs() - limit, min=0.0)
    return (excess ** 2).mean()


def shape_regularization(betas: torch.Tensor) -> torch.Tensor:
    return (betas ** 2).mean()
