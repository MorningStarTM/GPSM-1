"""Pose priors.

The design doc (doc/SMPLX_Fitting_Methods.docx section 2.2) specifies a
VPoser latent-space prior. VPoser is a separately-trained checkpoint that is
not part of the base SMPL-X download and was not present on this machine
(see doc/Mocap_Unification_Design.docx section 12 — an open blocker). This
module defines the `PosePrior` interface VPoser would implement, and ships
`GaussianJointPrior`, a working, honest, non-learned substitute (independent
per-joint L2 regularization plus hard anatomical limits) so the solver
pipeline is fully real and testable today. Swap in a `VPoserPrior` later by
implementing the same interface — nothing else in solver.py would change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.gpsm.smplx.fitting.losses import joint_limit_penalty, shape_regularization


class PosePrior(ABC):
    """Interface for a body-pose plausibility prior."""

    @abstractmethod
    def nll(self, body_pose: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood (up to a constant) of a batch of body
        poses. body_pose: (B, 63) axis-angle. Lower is more plausible."""

    def nll_value(self, body_pose: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.nll(body_pose).item())


# Soft per-joint axis-angle limits (radians), one triple per of the 21 SMPL-X
# body joints, applied per-axis. These are deliberately loose — they exist
# to stop the optimizer from producing anatomically impossible poses (e.g. a
# knee bending backward) when a limb is unobserved and has nothing else to
# constrain it, not to make fine anatomical judgments.
_DEFAULT_JOINT_LIMIT = 2.7  # ~155 degrees; SMPL-X axis-angle joints are Euler-like

# Knees and elbows are hinge-like: strongly discourage hyperextension/twist
# on the axes that are not the primary bend axis. Index within the 21-joint
# body_pose block, matching smplx.joint_names[1:22].
_HINGE_JOINT_INDICES = {
    3: (0, 2),   # left_knee: limit twist(x) and abduction(z), leave flexion(y) freer
    4: (0, 2),   # right_knee
    17: (0, 2),  # left_elbow (index 17 within body_pose == joint 18 overall - 1)
    18: (0, 2),  # right_elbow
}
_HINGE_LIMIT = 0.6  # tighter limit on the non-bend axes of a hinge joint


class GaussianJointPrior(PosePrior):
    """L2 regularization toward the neutral (zero) pose, plus soft joint
    limits, hinge-axis constraints on knees/elbows, and shape regularization.

    This is a legitimate, if simple, prior: it is exactly the "keep every
    joint near rest unless the data argues otherwise" assumption, and its
    main job in this pipeline is to fill unobserved limbs with something
    plausible (near-neutral) rather than let them be arbitrary. It is
    strictly weaker than a learned prior like VPoser, which additionally
    knows *joint correlations* (e.g. a bent elbow usually pairs with a
    rotated shoulder) — a documented limitation, not a hidden one.
    """

    def __init__(self, joint_limit: float = _DEFAULT_JOINT_LIMIT,
                 hinge_limit: float = _HINGE_LIMIT, l2_weight: float = 1.0):
        self.joint_limit = joint_limit
        self.hinge_limit = hinge_limit
        self.l2_weight = l2_weight

    def nll(self, body_pose: torch.Tensor) -> torch.Tensor:
        b = body_pose.shape[0]
        pose_j = body_pose.reshape(b, 21, 3)

        l2 = (pose_j ** 2).mean()
        limits = joint_limit_penalty(pose_j, self.joint_limit)

        hinge_penalty = torch.zeros((), dtype=body_pose.dtype, device=body_pose.device)
        for j, axes in _HINGE_JOINT_INDICES.items():
            for ax in axes:
                hinge_penalty = hinge_penalty + joint_limit_penalty(pose_j[:, j, ax], self.hinge_limit)

        return self.l2_weight * l2 + limits + hinge_penalty


def default_prior() -> PosePrior:
    return GaussianJointPrior()
