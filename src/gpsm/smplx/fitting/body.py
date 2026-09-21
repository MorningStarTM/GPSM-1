"""Differentiable SMPL-X body model wrapper, batched, for use by the solver.

Wraps the real `smplx` pip package around whatever body-model .npz file
`config.find_smplx_model()` locates. This is the actual SMPL-X mesh/joint
forward kinematics — not a placeholder skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.gpsm.smplx.config import SMPLXModelNotFoundError, find_smplx_model, gender_from_path
from src.gpsm.smplx.schema import NUM_BETAS


@dataclass
class BodyParams:
    """Pose/shape parameters for a batch of frames, all torch tensors.

    Shapes: betas (10,), everything else (B, ...) with B = batch size
    (typically the number of frames in a clip, fit jointly).
    """

    betas: torch.Tensor           # (10,)
    global_orient: torch.Tensor   # (B, 3)
    body_pose: torch.Tensor       # (B, 63)
    left_hand_pose: torch.Tensor  # (B, 45)
    right_hand_pose: torch.Tensor  # (B, 45)
    transl: torch.Tensor          # (B, 3)

    @property
    def batch_size(self) -> int:
        return self.global_orient.shape[0]

    @staticmethod
    def zeros(batch_size: int, device: str = "cpu", dtype=torch.float64) -> "BodyParams":
        z = lambda d: torch.zeros(batch_size, d, device=device, dtype=dtype)
        return BodyParams(
            betas=torch.zeros(NUM_BETAS, device=device, dtype=dtype),
            global_orient=z(3),
            body_pose=z(63),
            left_hand_pose=z(45),
            right_hand_pose=z(45),
            transl=z(3),
        )

    def requires_grad_(self, flags: dict) -> "BodyParams":
        for name, flag in flags.items():
            getattr(self, name).requires_grad_(flag)
        return self

    def detach_numpy(self) -> dict:
        return {
            "betas": self.betas.detach().cpu().numpy(),
            "global_orient": self.global_orient.detach().cpu().numpy(),
            "body_pose": self.body_pose.detach().cpu().numpy(),
            "left_hand_pose": self.left_hand_pose.detach().cpu().numpy(),
            "right_hand_pose": self.right_hand_pose.detach().cpu().numpy(),
            "transl": self.transl.detach().cpu().numpy(),
        }


class SMPLXBody:
    """Thin, batched wrapper around `smplx.SMPLXLayer`.

    Raises `SMPLXModelNotFoundError` at construction time if no model file
    is available, rather than failing confusingly deep inside a solver.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 dtype=torch.float64, num_betas: int = NUM_BETAS):
        import smplx as smplx_pkg  # the pip package, distinct from this src.gpsm.smplx package

        resolved = model_path or find_smplx_model()
        if resolved is None:
            raise SMPLXModelNotFoundError()
        resolved = str(resolved)

        self.device = device
        self.dtype = dtype
        self.num_betas = num_betas
        self.gender = gender_from_path(__import__("pathlib").Path(resolved))
        self.layer = smplx_pkg.SMPLXLayer(
            model_path=resolved,
            gender=self.gender,
            num_betas=num_betas,
            use_pca=False,
            flat_hand_mean=True,
            dtype=dtype,
        ).to(device)
        self.layer.eval()

        from smplx.joint_names import JOINT_NAMES
        self.joint_names = JOINT_NAMES

    def forward(self, params: BodyParams, return_verts: bool = False):
        """Batched forward pass. betas is broadcast to the batch size.

        `SMPLXLayer` (unlike the stateful `SMPLX` module) expects every pose
        as a rotation *matrix*, not axis-angle, so axis-angle inputs are
        converted here via `smplx.lbs.batch_rodrigues` — a torch operation,
        so gradients flow through it back to the axis-angle parameters the
        solver actually optimizes.
        """
        from smplx.lbs import batch_rodrigues

        b = params.batch_size
        betas = params.betas.unsqueeze(0).expand(b, -1)

        def to_mats(aa: torch.Tensor, num_joints: int) -> torch.Tensor:
            flat = aa.reshape(b * num_joints, 3)
            mats = batch_rodrigues(flat)
            return mats.reshape(b, num_joints, 3, 3)

        out = self.layer(
            betas=betas,
            global_orient=to_mats(params.global_orient, 1),
            body_pose=to_mats(params.body_pose, 21),
            left_hand_pose=to_mats(params.left_hand_pose, 15),
            right_hand_pose=to_mats(params.right_hand_pose, 15),
            transl=params.transl,
            return_verts=return_verts,
        )
        return out

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)
