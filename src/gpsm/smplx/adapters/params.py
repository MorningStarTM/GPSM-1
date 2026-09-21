"""Tier 0: SMPL / SMPL-H / SMPL-X parameter dicts -> CanonicalMotion.

No fitting is involved here — every supported format already parameterizes
the body the same way SMPL-X does (axis-angle joint rotations), so
conversion is pure slicing and reordering. See
doc/Mocap_Unification_Design.docx sections 2.1 and 6.

Verified against real AMASS sample data (both the SMPL+H `*_poses.npz` and
the SMPL-X `*_stageii.npz` release formats) in tests/test_params_adapter.py.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.gpsm.smplx.schema import (
    BodyPart,
    CanonicalMotion,
    FitQuality,
    NUM_BETAS,
    NUM_PARTS,
    frozen_part_mask,
)

# SMPL+H `poses` (T, 156) layout: root_orient(3) + body(63) + hands(90)
_SMPLH_ROOT = slice(0, 3)
_SMPLH_BODY = slice(3, 66)
_SMPLH_LHAND = slice(66, 111)
_SMPLH_RHAND = slice(111, 156)

# SMPL-X `poses` (T, 165) layout, verified byte-exact against explicit keys:
# root_orient(3) + body(63) + jaw(3) + eyes(6) + hands(90)
_SMPLX_ROOT = slice(0, 3)
_SMPLX_BODY = slice(3, 66)
_SMPLX_JAW = slice(66, 69)
_SMPLX_EYES = slice(69, 75)
_SMPLX_HANDS = slice(75, 165)


class UnsupportedFormatError(ValueError):
    pass


def detect_param_format(data: Dict[str, Any]) -> str:
    """Identify whether a raw npz dict is SMPL+H (156), SMPL-X (165), or
    already-canonical (159), from its `poses` (or explicit) array width."""
    if "poses" in data:
        width = np.asarray(data["poses"]).shape[-1]
        if width == 156:
            return "smplh"
        if width == 165:
            return "smplx165"
        if width == 159:
            return "canonical159"
        raise UnsupportedFormatError(
            f"`poses` has width {width}; expected 156 (SMPL+H), 165 (SMPL-X), "
            "or 159 (already canonical)."
        )
    has_smplx_keys = {"pose_body", "pose_hand"}.issubset(data)
    if has_smplx_keys:
        return "smplx_split"
    raise UnsupportedFormatError(
        f"Could not detect a supported parameter format from keys: {sorted(data)}"
    )


def _get_trans(data: Dict[str, Any], t: int) -> np.ndarray:
    if "trans" in data:
        trans = np.asarray(data["trans"], dtype=np.float64)
        if trans.shape != (t, 3):
            raise ValueError(f"`trans` has shape {trans.shape}, expected ({t}, 3)")
        return trans
    return np.zeros((t, 3), dtype=np.float64)


def _get_betas(data: Dict[str, Any]) -> np.ndarray:
    if "betas" not in data:
        return np.zeros(NUM_BETAS, dtype=np.float64)
    betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)
    if betas.shape[0] < NUM_BETAS:
        return np.pad(betas, (0, NUM_BETAS - betas.shape[0]))
    return betas[:NUM_BETAS]


def _part_mask_from_blocks(body: np.ndarray, lhand: np.ndarray, rhand: np.ndarray, t: int) -> np.ndarray:
    mask = np.ones((t, NUM_PARTS), dtype=np.float64)
    if frozen_part_mask(lhand):
        mask[:, BodyPart.LEFT_HAND] = 0.0
    if frozen_part_mask(rhand):
        mask[:, BodyPart.RIGHT_HAND] = 0.0
    # trans/body/face masks default to observed=1; face has no signal in
    # this tier at all (SMPL+H has no face params, SMPL-X jaw/eyes are
    # dropped from the 159-d layout per the design doc), so mark it
    # explicitly unobserved.
    mask[:, BodyPart.FACE] = 0.0
    return mask


def convert_smplh(data: Dict[str, Any], fps: float, source_name: str = "") -> CanonicalMotion:
    """SMPL+H `poses` (T, 156) + `trans` (T, 3) -> CanonicalMotion."""
    poses = np.asarray(data["poses"], dtype=np.float64)
    t = poses.shape[0]
    body = poses[:, _SMPLH_BODY]
    lhand = poses[:, _SMPLH_LHAND]
    rhand = poses[:, _SMPLH_RHAND]
    return CanonicalMotion(
        trans=_get_trans(data, t),
        global_orient=poses[:, _SMPLH_ROOT].copy(),
        body_pose=body.copy(),
        left_hand_pose=lhand.copy(),
        right_hand_pose=rhand.copy(),
        betas=_get_betas(data),
        part_mask=_part_mask_from_blocks(body, lhand, rhand, t),
        fps=fps,
        meta={"source_format": "smplh156", "source_name": source_name,
              "gender": str(data.get("gender", "unknown"))},
    )


def convert_smplx165(data: Dict[str, Any], fps: float, source_name: str = "") -> CanonicalMotion:
    """SMPL-X `poses` (T, 165) -> CanonicalMotion (jaw/eyes dropped, per the
    159-d layout)."""
    poses = np.asarray(data["poses"], dtype=np.float64)
    t = poses.shape[0]
    body = poses[:, _SMPLX_BODY]
    hands = poses[:, _SMPLX_HANDS]
    lhand, rhand = hands[:, :45], hands[:, 45:]
    mask = _part_mask_from_blocks(body, lhand, rhand, t)
    jaw = poses[:, _SMPLX_JAW]
    eyes = poses[:, _SMPLX_EYES]
    face_observed = not (frozen_part_mask(jaw) and frozen_part_mask(eyes))
    mask[:, BodyPart.FACE] = 1.0 if face_observed else 0.0
    return CanonicalMotion(
        trans=_get_trans(data, t),
        global_orient=poses[:, _SMPLX_ROOT].copy(),
        body_pose=body.copy(),
        left_hand_pose=lhand.copy(),
        right_hand_pose=rhand.copy(),
        betas=_get_betas(data),
        part_mask=mask,
        fps=fps,
        meta={"source_format": "smplx165", "source_name": source_name,
              "gender": str(data.get("gender", "unknown")),
              "dropped_face_dims": 9},
    )


def convert_smplx_split(data: Dict[str, Any], fps: float, source_name: str = "") -> CanonicalMotion:
    """SMPL-X stored as separate keys (root_orient/pose_body/pose_hand/...)
    rather than one concatenated `poses` array."""
    body = np.asarray(data["pose_body"], dtype=np.float64)
    t = body.shape[0]
    hands = np.asarray(data["pose_hand"], dtype=np.float64)
    lhand, rhand = hands[:, :45], hands[:, 45:]
    root = np.asarray(data.get("root_orient", np.zeros((t, 3))), dtype=np.float64)
    mask = _part_mask_from_blocks(body, lhand, rhand, t)
    if "pose_jaw" in data or "pose_eye" in data:
        jaw = np.asarray(data.get("pose_jaw", np.zeros((t, 3))))
        eyes = np.asarray(data.get("pose_eye", np.zeros((t, 6))))
        face_observed = not (frozen_part_mask(jaw) and frozen_part_mask(eyes))
        mask[:, BodyPart.FACE] = 1.0 if face_observed else 0.0
    return CanonicalMotion(
        trans=_get_trans(data, t),
        global_orient=root.copy(),
        body_pose=body.copy(),
        left_hand_pose=lhand.copy(),
        right_hand_pose=rhand.copy(),
        betas=_get_betas(data),
        part_mask=mask,
        fps=fps,
        meta={"source_format": "smplx_split", "source_name": source_name,
              "gender": str(data.get("gender", "unknown"))},
    )


def convert_canonical159(data: Dict[str, Any], fps: float, source_name: str = "") -> CanonicalMotion:
    """Already a 159-d canonical `poses` array (e.g. re-ingesting our own
    exported data)."""
    poses = np.asarray(data["poses"], dtype=np.float64)
    t = poses.shape[0]
    body = poses[:, 6:69]
    lhand, rhand = poses[:, 69:114], poses[:, 114:159]
    return CanonicalMotion(
        trans=poses[:, 0:3].copy() if "trans" not in data else _get_trans(data, t),
        global_orient=poses[:, 3:6].copy(),
        body_pose=body.copy(),
        left_hand_pose=lhand.copy(),
        right_hand_pose=rhand.copy(),
        betas=_get_betas(data),
        part_mask=_part_mask_from_blocks(body, lhand, rhand, t),
        fps=fps,
        meta={"source_format": "canonical159", "source_name": source_name},
    )


_CONVERTERS = {
    "smplh": convert_smplh,
    "smplx165": convert_smplx165,
    "smplx_split": convert_smplx_split,
    "canonical159": convert_canonical159,
}


def ingest_params(data: Dict[str, Any], fps: float = None, source_name: str = "") -> CanonicalMotion:
    """Auto-detect a Tier-0 parameter format and convert it to CanonicalMotion.

    `fps` overrides any rate stored in `data`; if omitted, looks for
    `mocap_framerate` / `mocap_frame_rate` in the input, and raises if
    neither is available.
    """
    fmt = detect_param_format(data)
    resolved_fps = fps
    if resolved_fps is None:
        for key in ("mocap_framerate", "mocap_frame_rate"):
            if key in data:
                resolved_fps = float(np.asarray(data[key]).reshape(-1)[0])
                break
    if resolved_fps is None:
        raise ValueError(
            "No fps provided and none found in the input under "
            "'mocap_framerate' / 'mocap_frame_rate'."
        )
    return _CONVERTERS[fmt](data, resolved_fps, source_name)
