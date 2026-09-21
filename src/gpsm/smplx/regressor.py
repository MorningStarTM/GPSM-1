"""Method 3: learned-regressor initialization for 2D / monocular video input.

See doc/SMPLX_Fitting_Methods.docx section 3. This module defines the
interface a whole-body SMPL-X regressor network (SMPLer-X recommended, or
NLF/OSX as alternatives) must implement to plug into this pipeline, plus
the reprojection-refinement step that polishes its output — but it does
NOT ship a trained network.

Honest limitation, not a hidden one: running an actual SMPLer-X/NLF/OSX
checkpoint requires downloading gigabytes of pretrained weights, which this
environment has neither the network access nor the license to do on your
behalf. `NullRegressor` raises a clear, actionable error instead of
pretending to produce a real estimate. Everything else in this module
(the `Regressor` interface and `refine_from_2d`, the reprojection
optimization pass) is real and works with any regressor that implements
the interface — wire in a checkpoint by writing one adapter class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.gpsm.smplx.fitting.body import BodyParams, SMPLXBody
from src.gpsm.smplx.fitting.losses import robust_data_term, shape_regularization
from src.gpsm.smplx.fitting.priors import PosePrior, default_prior
from src.gpsm.smplx.schema import NUM_BETAS


@dataclass
class RegressorEstimate:
    """A single-frame SMPL-X initial guess from a learned regressor."""

    betas: np.ndarray            # (10,)
    global_orient: np.ndarray    # (3,)
    body_pose: np.ndarray        # (63,)
    left_hand_pose: np.ndarray   # (45,)
    right_hand_pose: np.ndarray  # (45,)
    camera_translation: np.ndarray  # (3,) weak-perspective or full camera transl


class Regressor(ABC):
    """Interface a whole-body SMPL-X image regressor must implement."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> RegressorEstimate:
        """`image`: (H, W, 3) uint8 RGB. Returns one SMPL-X initial estimate."""


class NullRegressor(Regressor):
    """No trained network is available in this environment. Raises with a
    clear pointer rather than fabricating an estimate."""

    def predict(self, image: np.ndarray) -> RegressorEstimate:
        raise NotImplementedError(
            "No learned SMPL-X regressor is wired in. This pipeline's "
            "Method 3 (see doc/SMPLX_Fitting_Methods.docx section 3) needs "
            "a whole-body regressor such as SMPLer-X, NLF, or OSX — "
            "download a pretrained checkpoint and implement `Regressor` "
            "(predict(image) -> RegressorEstimate) around it; "
            "refine_from_2d() in this module will then work unchanged."
        )


class Camera(ABC):
    """Minimal camera interface for the reprojection refinement step."""

    @abstractmethod
    def project(self, points_3d: torch.Tensor) -> torch.Tensor:
        """points_3d: (..., 3) camera-space -> (..., 2) pixel coordinates."""


class WeakPerspectiveCamera(Camera):
    """points_2d = focal * points_3d[..., :2] / points_3d[..., 2:3] + principal_point.

    Good enough for the refinement step's reprojection term; a full
    pinhole/distortion model can implement the same `Camera` interface.
    """

    def __init__(self, focal: float, principal_point):
        self.focal = focal
        self.principal_point = torch.as_tensor(principal_point, dtype=torch.float64)

    def project(self, points_3d: torch.Tensor) -> torch.Tensor:
        z = points_3d[..., 2:3].clamp_min(1e-6)
        xy = self.focal * points_3d[..., :2] / z
        return xy + self.principal_point


def refine_from_2d(
    estimate: RegressorEstimate,
    keypoints_2d: np.ndarray,
    keypoints_conf: np.ndarray,
    joint_indices,
    body: SMPLXBody,
    camera: Camera,
    prior: Optional[PosePrior] = None,
    iters: int = 20,
    w_data: float = 1.0,
    w_prior: float = 0.2,
    w_shape: float = 0.5,
) -> RegressorEstimate:
    """Short reprojection-based refinement of a regressor's initial guess.

    This is the "few iterations, near real-time" polish step described in
    doc/SMPLX_Fitting_Methods.docx section 3.2 step 3 — it does not
    re-derive the pose from scratch, only nudges the network's estimate to
    better match the actual 2D detections in *this* image via
    backpropagation through the render(-as-joints)-and-project step.

    This function is fully real and tested (see tests/test_regressor.py) —
    it only requires a `RegressorEstimate` to start from, which is exactly
    the piece `NullRegressor` cannot supply without a trained checkpoint.
    """
    prior = prior or default_prior()
    dtype = torch.float64

    params = BodyParams(
        betas=torch.tensor(estimate.betas, dtype=dtype, requires_grad=True),
        global_orient=torch.tensor(estimate.global_orient, dtype=dtype).unsqueeze(0).requires_grad_(True),
        body_pose=torch.tensor(estimate.body_pose, dtype=dtype).unsqueeze(0).requires_grad_(True),
        left_hand_pose=torch.tensor(estimate.left_hand_pose, dtype=dtype).unsqueeze(0),
        right_hand_pose=torch.tensor(estimate.right_hand_pose, dtype=dtype).unsqueeze(0),
        transl=torch.tensor(estimate.camera_translation, dtype=dtype).unsqueeze(0).requires_grad_(True),
    )

    target_2d = torch.tensor(keypoints_2d, dtype=dtype).unsqueeze(0)      # (1, K, 2)
    conf_2d = torch.tensor(keypoints_conf, dtype=dtype).unsqueeze(0)      # (1, K)
    idx_t = torch.tensor(joint_indices, dtype=torch.long)

    variables = [params.global_orient, params.body_pose, params.betas, params.transl]
    optimizer = torch.optim.LBFGS(variables, lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        out = body.forward(params)
        joints_3d = out.joints[:, idx_t, :]
        pred_2d = camera.project(joints_3d)
        loss = w_data * robust_data_term(pred_2d, target_2d, conf_2d, sigma=20.0)
        loss = loss + w_prior * prior.nll(params.body_pose)
        loss = loss + w_shape * shape_regularization(params.betas)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        return RegressorEstimate(
            betas=params.betas.numpy().copy(),
            global_orient=params.global_orient[0].numpy().copy(),
            body_pose=params.body_pose[0].numpy().copy(),
            left_hand_pose=params.left_hand_pose[0].numpy().copy(),
            right_hand_pose=params.right_hand_pose[0].numpy().copy(),
            camera_translation=params.transl[0].numpy().copy(),
        )
