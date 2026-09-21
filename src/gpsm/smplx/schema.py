"""The canonical mocap contract: CanonicalMotion, Observation, part masks.

Every adapter in this package (src/smplx) — whatever the input format — ends
by producing a `CanonicalMotion`. Every fitting method reads its raw input
as an `Observation`. Nothing downstream of this module needs to know where
the data originally came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# The 159-dimensional canonical pose layout (see doc/Mocap_Unification_Design.docx
# section 2.1). Confirmed against real AMASS SMPL+H (156-d) and SMPL-X
# (165-d) `poses` arrays.
# ---------------------------------------------------------------------------
TRANS_DIM = 3
GLOBAL_ORIENT_DIM = 3
BODY_POSE_DIM = 63
HAND_POSE_DIM = 45
CANONICAL_DIM = TRANS_DIM + GLOBAL_ORIENT_DIM + BODY_POSE_DIM + 2 * HAND_POSE_DIM
assert CANONICAL_DIM == 159

# Slices into a (T, 159) canonical array.
SLICE_TRANS = slice(0, 3)
SLICE_GLOBAL_ORIENT = slice(3, 6)
SLICE_BODY_POSE = slice(6, 69)
SLICE_LEFT_HAND = slice(69, 114)
SLICE_RIGHT_HAND = slice(114, 159)

NUM_BODY_JOINTS = 21
NUM_HAND_JOINTS = 15
NUM_BETAS = 10


class BodyPart(IntEnum):
    """Index into a `part_mask` column. Order matches PART_NAMES below."""

    TRANS = 0
    BODY = 1
    LEFT_HAND = 2
    RIGHT_HAND = 3
    FACE = 4


PART_NAMES = ["trans", "body", "left_hand", "right_hand", "face"]
NUM_PARTS = len(PART_NAMES)

# Std-dev threshold below which a part is considered "frozen" (i.e. not
# actually observed, just a constant rest pose baked into the file) rather
# than genuinely captured motion. Calibrated against real AMASS data where a
# byte-identical MANO mean hand pose produced std ~1e-14.
FROZEN_STD_THRESHOLD = 1e-6


def frozen_part_mask(pose_block: np.ndarray, threshold: float = FROZEN_STD_THRESHOLD) -> bool:
    """True if `pose_block` (T, D) is constant across time (T>1) — i.e. a
    baked-in rest pose rather than observed motion."""
    if pose_block.shape[0] < 2:
        return False
    return bool(np.std(pose_block, axis=0).max() < threshold)


@dataclass
class FitQuality:
    """Scorecard for a fitted (or converted) motion clip.

    See doc/Mocap_Unification_Design.docx section 8. Fields default to
    "unknown" (NaN / -1) rather than a false 0.0 or 1.0 when a tier-0
    conversion has nothing to score (e.g. slicing SMPL-X params directly
    involves no fitting error at all).
    """

    joint_rmse_mm: float = float("nan")
    reproj_px: float = float("nan")
    observed_fraction: float = 1.0
    prior_nll: float = float("nan")
    foot_skate_cm_s: float = float("nan")
    accel_spike_count: int = 0
    verdict: str = "GOOD"

    _THRESHOLDS = {
        "joint_rmse_mm": (30.0, 60.0),   # <30 GOOD, <60 DEGRADED, else REJECT
        "observed_fraction": (0.5, 0.15),  # >=0.5 GOOD, >=0.15 DEGRADED (inverted)
    }

    def compute_verdict(self) -> str:
        rmse = self.joint_rmse_mm
        frac = self.observed_fraction
        if frac < 0.15:
            return "REJECT"
        if (not np.isnan(rmse)) and rmse >= 60.0:
            return "REJECT"
        if frac < 0.5 or (not np.isnan(rmse) and rmse >= 30.0) or self.accel_spike_count > 0:
            return "DEGRADED"
        return "GOOD"

    def finalize(self) -> "FitQuality":
        self.verdict = self.compute_verdict()
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_rmse_mm": self.joint_rmse_mm,
            "reproj_px": self.reproj_px,
            "observed_fraction": self.observed_fraction,
            "prior_nll": self.prior_nll,
            "foot_skate_cm_s": self.foot_skate_cm_s,
            "accel_spike_count": self.accel_spike_count,
            "verdict": self.verdict,
        }


@dataclass
class CanonicalMotion:
    """A single motion clip in the unified SMPL-X representation.

    All pose fields are axis-angle, in radians, SMPL-X joint ordering.
    `part_mask[t, part]` is 1.0 where that body part was actually observed
    at frame t, 0.0 where it was filled in by a prior / left at rest.
    """

    trans: np.ndarray            # (T, 3)
    global_orient: np.ndarray    # (T, 3)
    body_pose: np.ndarray        # (T, 63)
    left_hand_pose: np.ndarray   # (T, 45)
    right_hand_pose: np.ndarray  # (T, 45)
    betas: np.ndarray            # (10,)
    part_mask: np.ndarray        # (T, 5) — see BodyPart
    fps: float
    meta: Dict[str, Any] = field(default_factory=dict)
    quality: FitQuality = field(default_factory=FitQuality)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        t = self.trans.shape[0]
        for name, arr, dim in [
            ("trans", self.trans, TRANS_DIM),
            ("global_orient", self.global_orient, GLOBAL_ORIENT_DIM),
            ("body_pose", self.body_pose, BODY_POSE_DIM),
            ("left_hand_pose", self.left_hand_pose, HAND_POSE_DIM),
            ("right_hand_pose", self.right_hand_pose, HAND_POSE_DIM),
        ]:
            if arr.shape != (t, dim):
                raise ValueError(
                    f"CanonicalMotion.{name} has shape {arr.shape}, expected ({t}, {dim})"
                )
        if self.betas.shape != (NUM_BETAS,):
            raise ValueError(
                f"CanonicalMotion.betas has shape {self.betas.shape}, expected ({NUM_BETAS},)"
            )
        if self.part_mask.shape != (t, NUM_PARTS):
            raise ValueError(
                f"CanonicalMotion.part_mask has shape {self.part_mask.shape}, "
                f"expected ({t}, {NUM_PARTS})"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    @property
    def num_frames(self) -> int:
        return self.trans.shape[0]

    # ------------------------------------------------------------------
    def to_array(self) -> np.ndarray:
        """Pack into a single (T, 159) canonical axis-angle array."""
        return np.concatenate(
            [
                self.trans,
                self.global_orient,
                self.body_pose,
                self.left_hand_pose,
                self.right_hand_pose,
            ],
            axis=1,
        ).astype(np.float64)

    @staticmethod
    def from_array(
        arr: np.ndarray,
        betas: np.ndarray,
        part_mask: np.ndarray,
        fps: float,
        meta: Optional[Dict[str, Any]] = None,
        quality: Optional[FitQuality] = None,
    ) -> "CanonicalMotion":
        if arr.shape[1] != CANONICAL_DIM:
            raise ValueError(f"Expected (T, {CANONICAL_DIM}), got {arr.shape}")
        return CanonicalMotion(
            trans=arr[:, SLICE_TRANS].copy(),
            global_orient=arr[:, SLICE_GLOBAL_ORIENT].copy(),
            body_pose=arr[:, SLICE_BODY_POSE].copy(),
            left_hand_pose=arr[:, SLICE_LEFT_HAND].copy(),
            right_hand_pose=arr[:, SLICE_RIGHT_HAND].copy(),
            betas=betas,
            part_mask=part_mask,
            fps=fps,
            meta=meta or {},
            quality=quality or FitQuality(),
        )

    def to_npz_dict(self) -> Dict[str, Any]:
        """Serialize to a dict suitable for `np.savez`."""
        return {
            "trans": self.trans,
            "global_orient": self.global_orient,
            "body_pose": self.body_pose,
            "left_hand_pose": self.left_hand_pose,
            "right_hand_pose": self.right_hand_pose,
            "betas": self.betas,
            "part_mask": self.part_mask,
            "fps": np.float64(self.fps),
            "meta_json": _dict_to_json(self.meta),
            "quality_json": _dict_to_json(self.quality.to_dict()),
        }

    @staticmethod
    def from_npz_dict(d: Dict[str, Any]) -> "CanonicalMotion":
        import json

        meta = json.loads(str(d["meta_json"])) if "meta_json" in d else {}
        quality_dict = json.loads(str(d["quality_json"])) if "quality_json" in d else {}
        return CanonicalMotion(
            trans=np.asarray(d["trans"]),
            global_orient=np.asarray(d["global_orient"]),
            body_pose=np.asarray(d["body_pose"]),
            left_hand_pose=np.asarray(d["left_hand_pose"]),
            right_hand_pose=np.asarray(d["right_hand_pose"]),
            betas=np.asarray(d["betas"]),
            part_mask=np.asarray(d["part_mask"]),
            fps=float(d["fps"]),
            meta=meta,
            quality=FitQuality(**quality_dict) if quality_dict else FitQuality(),
        )


def _dict_to_json(d: Dict[str, Any]) -> str:
    import json

    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return json.dumps(d, default=default)


@dataclass
class Observation:
    """Arbitrary mocap input, reduced to points + confidence + correspondence.

    This is the single intermediate every fitting method (Similarity
    Alignment, Staged Optimization) consumes, regardless of whether the
    original source was a marker rig, a skeleton file, or keypoints.
    """

    points: np.ndarray           # (T, K, 2 or 3)
    conf: np.ndarray             # (T, K), 0.0 = unobserved
    joint_names: list            # length K, names resolved via a layout's correspondence
    space: str                   # "world3d" | "camera3d" | "rootrel3d" | "image2d"
    fps: float
    units_to_meters: float = 1.0
    up_axis: str = "z"           # "z" or "y"

    def __post_init__(self) -> None:
        t, k = self.points.shape[0], self.points.shape[1]
        if self.conf.shape != (t, k):
            raise ValueError(f"conf shape {self.conf.shape} must match points (T,K)=({t},{k})")
        if len(self.joint_names) != k:
            raise ValueError(f"joint_names has {len(self.joint_names)} entries, expected {k}")
        if self.space not in ("world3d", "camera3d", "rootrel3d", "image2d"):
            raise ValueError(f"Unknown observation space: {self.space}")
        if self.up_axis not in ("y", "z"):
            raise ValueError(f"up_axis must be 'y' or 'z', got {self.up_axis}")

    @property
    def num_frames(self) -> int:
        return self.points.shape[0]

    @property
    def num_points(self) -> int:
        return self.points.shape[1]

    @property
    def is_3d(self) -> bool:
        return self.points.shape[2] == 3
