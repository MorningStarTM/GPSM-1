"""Tier 1: BVH skeleton animation -> CanonicalMotion.

Parses hierarchy + per-frame Euler channels with the `bvh` package, does
forward kinematics ourselves (the package only exposes raw channels, not
world positions), then matches whatever joint names the file uses against
a broad alias table for the common CMU/Mixamo-style naming convention.
Matched joint positions become a JointCorr Observation, fit via the same
staged solver used for MediaPipe/marker input (fitting/solver.py).

Verified against real files in a local BVH corpus (Male1_bvh/*.bvh,
CMU-style naming: Hips, Spine, Spine1, Neck, Head, Left/RightShoulder,
Left/RightArm, Left/RightForeArm, Left/RightHand, Left/RightUpLeg,
Left/RightLeg, Left/RightFoot, Left/RightToeBase) — see
tests/test_bvh_adapter.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, List, Union

import numpy as np
from scipy.spatial.transform import Rotation

from src.gpsm.smplx.layouts.correspondence import JointCorr, MotionLayout
from src.gpsm.smplx.ops import guess_up_axis

MIN_MATCHED_BVH_JOINTS = 6

# Alias table: canonical alias (matched case-insensitively, after stripping
# a "namespace:" or "namespace_" prefix such as Mixamo's "mixamorig:") ->
# SMPL-X joint name. A BVH joint's *position* is its pivot point, which is
# exactly what each of these SMPL-X joints also represents (e.g. BVH
# "LeftArm"'s position is the shoulder pivot -> SMPL-X "left_shoulder"),
# so this is a position correspondence, not a bone/segment correspondence.
_BVH_ALIASES = {
    "hips": "pelvis", "hip": "pelvis", "pelvis": "pelvis", "root": "pelvis",
    "spine": "spine1", "spine1": "spine2", "spine2": "spine3", "chest": "spine2",
    "neck": "neck", "neck1": "neck",
    "head": "head",
    "leftshoulder": "left_collar", "leftcollar": "left_collar",
    "leftarm": "left_shoulder", "leftupperarm": "left_shoulder",
    "leftforearm": "left_elbow", "leftlowarm": "left_elbow",
    "lefthand": "left_wrist",
    "rightshoulder": "right_collar", "rightcollar": "right_collar",
    "rightarm": "right_shoulder", "rightupperarm": "right_shoulder",
    "rightforearm": "right_elbow", "rightlowarm": "right_elbow",
    "righthand": "right_wrist",
    "leftupleg": "left_hip", "leftthigh": "left_hip", "leftuplegroll": "left_hip",
    "leftleg": "left_knee", "leftshin": "left_knee", "leftcalf": "left_knee",
    "leftfoot": "left_ankle",
    "lefttoebase": "left_foot", "lefttoe": "left_foot",
    "rightupleg": "right_hip", "rightthigh": "right_hip", "rightuplegroll": "right_hip",
    "rightleg": "right_knee", "rightshin": "right_knee", "rightcalf": "right_knee",
    "rightfoot": "right_ankle",
    "righttoebase": "right_foot", "righttoe": "right_foot",
}


@dataclass
class BvhCapture:
    points_m: np.ndarray     # (T, J, 3), meters
    joint_names: List[str]
    parents: List[int]       # BVH's own hierarchy, for raw-tab bone drawing
    fps: float
    up_axis: str
    up_axis_guessed: bool
    units_guessed: bool


def _channel_rotation_matrix(channel_name: str, angle_deg: float) -> np.ndarray:
    axis = channel_name[0].lower()  # "Xrotation" -> "x", etc.
    return Rotation.from_euler(axis, angle_deg, degrees=True).as_matrix()


def _forward_kinematics(mocap) -> np.ndarray:
    """Returns (T, J, 3) world-space joint positions via standard BVH FK:
    world_pos[child] = world_pos[parent] + world_rot[parent] @ offset[child];
    world_rot[child] = world_rot[parent] @ local_rot[child] (root has no
    parent — its position channels ARE its absolute world position)."""
    joints = mocap.get_joints()
    names = [j.name for j in joints]
    parents = [mocap.joint_parent_index(n) for n in names]
    offsets = [np.array(mocap.joint_offset(n), dtype=np.float64) for n in names]
    channels = [mocap.joint_channels(n) for n in names]

    t = mocap.nframes
    positions = np.zeros((t, len(names), 3), dtype=np.float64)

    for f in range(t):
        world_pos = [None] * len(names)
        world_rot = [None] * len(names)
        for j, name in enumerate(names):
            ch = channels[j]
            values = mocap.frame_joint_channels(f, name, ch)
            rot_mat = np.eye(3)
            for ch_name, val in zip(ch, values):
                if ch_name.endswith("rotation"):
                    rot_mat = rot_mat @ _channel_rotation_matrix(ch_name, val)

            if parents[j] == -1:
                pos_channels = {c: v for c, v in zip(ch, values) if c.endswith("position")}
                local_pos = np.array([
                    pos_channels.get("Xposition", 0.0),
                    pos_channels.get("Yposition", 0.0),
                    pos_channels.get("Zposition", 0.0),
                ])
                world_pos[j] = local_pos
                world_rot[j] = rot_mat
            else:
                p = parents[j]
                world_pos[j] = world_pos[p] + world_rot[p] @ offsets[j]
                world_rot[j] = world_rot[p] @ rot_mat

            positions[f, j] = world_pos[j]

    return positions, names, parents


def _guess_units_to_meters(positions: np.ndarray, up_axis_index: int) -> "tuple[float, bool]":
    """BVH has no standard unit — every studio/tool picks its own (cm, mm,
    inches...). Heuristic: pick the scale factor that brings a reference
    "body height" into a plausible standing/crouching human range
    (0.5m-2.2m before scaling).

    That reference height is deliberately NOT the overall bounding-box
    diagonal across all frames: a traveling clip (e.g. a skip or a walk
    that crosses the room) can rack up an arbitrarily large *horizontal*
    bounding box that has nothing to do with the person's actual size,
    which silently produced a wrong scale before this fix (a real "Skipping"
    clip's combined XYZ diagonal was ~910 raw units purely from travel
    distance, versus its real ~190-unit vertical extent). Using the
    per-frame span *along the up axis only*, maximized over frames,
    is translation-invariant (a constant per-frame offset cancels out of a
    within-frame max-min) and immune to horizontal travel entirely.
    """
    up_coord = positions[:, :, up_axis_index]              # (T, J)
    per_frame_height = up_coord.max(axis=1) - up_coord.min(axis=1)  # (T,)
    reference_height = float(per_frame_height.max())
    if reference_height <= 0:
        return 1.0, True
    for scale, guessed in [(1.0, False), (0.01, True), (0.001, True), (0.0254, True)]:
        if 0.5 <= reference_height * scale <= 2.2:
            return scale, guessed
    return 0.01, True  # default guess: centimeters, the most common BVH convention


def load_bvh(file_like: Union[str, BinaryIO]) -> BvhCapture:
    import bvh as bvh_pkg

    if isinstance(file_like, str):
        with open(file_like, "r") as f:
            text = f.read()
    else:
        raw = file_like.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    mocap = bvh_pkg.Bvh(text)
    positions_raw, names, parents = _forward_kinematics(mocap)

    # up-axis detection is scale-invariant (it only compares which axis
    # separates head-ish from foot-ish joints), so it can run on the raw,
    # not-yet-rescaled positions — and must, since the unit heuristic below
    # needs to already know which axis is "up".
    up_axis, up_guessed = guess_up_axis(positions_raw, names)
    up_axis_index = {"x": 0, "y": 1, "z": 2}[up_axis]

    scale, units_guessed = _guess_units_to_meters(positions_raw, up_axis_index)
    positions_m = positions_raw * scale

    fps = 1.0 / mocap.frame_time if mocap.frame_time else 30.0

    return BvhCapture(
        points_m=positions_m, joint_names=names, parents=parents, fps=fps,
        up_axis=up_axis, up_axis_guessed=up_guessed, units_guessed=units_guessed,
    )


def _normalize_bvh_name(name: str) -> str:
    for sep in (":", "_"):
        if sep in name:
            candidate = name.rsplit(sep, 1)[-1]
            if candidate.lower() in _BVH_ALIASES:
                name = candidate
                break
    return name.strip().lower()


def match_bvh_layout(joint_names: List[str]) -> "tuple[MotionLayout, List[int]]":
    """Intersect a BVH file's joint names with the alias table above.

    Returns (layout restricted to matched joints, column indices into the
    original (T, J, 3) array). Raises ValueError if fewer than
    `MIN_MATCHED_BVH_JOINTS` names are recognized.
    """
    matched_cols, matched_names, correspondences = [], [], []
    for i, name in enumerate(joint_names):
        key = _normalize_bvh_name(name)
        if key in _BVH_ALIASES:
            matched_cols.append(i)
            matched_names.append(name)
            correspondences.append(JointCorr(_BVH_ALIASES[key]))

    if len(matched_cols) < MIN_MATCHED_BVH_JOINTS:
        raise ValueError(
            f"Only {len(matched_cols)} of this file's {len(joint_names)} joints "
            f"match a known naming convention (need >= {MIN_MATCHED_BVH_JOINTS}); "
            f"joint names found: {joint_names}"
        )

    layout = MotionLayout(
        name="bvh_matched",
        point_names=matched_names,
        correspondences=correspondences,
        description=f"{len(matched_names)}/{len(joint_names)} BVH joints matched by name "
                    "(CMU/Mixamo-style naming convention).",
    )
    return layout, matched_cols
