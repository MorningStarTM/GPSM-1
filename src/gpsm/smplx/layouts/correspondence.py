"""Correspondence types linking an input point to a location on the SMPL-X body.

See doc/Mocap_Unification_Design.docx section 4.2.

`JointCorr` maps a point to one of the SMPL-X model's named output joints.
`VertexCorr` maps a point to a specific vertex on the 10,475-vertex SMPL-X
mesh — used for marker-based fitting (e.g. C3D optical mocap), where a
marker sits on the body *surface*, not at a joint center. The
`layouts/builtin.py::VICON_50` layout uses real vertex ids taken directly
from a real AMASS/SOMA marker-to-vertex correspondence file (see that
module's docstring), not placeholders.
`RegressorCorr` (a point as a learned linear combination of vertices) is
not implemented — a documented extension point for future work.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

# The `smplx` package's own canonical joint ordering. Index 0..54 are the
# kinematic-tree joints (pelvis .. right hand fingers); index 55+ are extra
# landmark joints (face, toes, heels, fingertips) produced via the model's
# built-in vertex-to-joint regressors. We import defensively so this module
# still loads (with an informative error only if actually used) even in an
# environment where the `smplx` pip package is not installed.
try:
    from smplx.joint_names import JOINT_NAMES as _SMPLX_JOINT_NAMES
except ImportError:  # pragma: no cover - exercised only without the smplx pip package
    _SMPLX_JOINT_NAMES = []

SMPLX_JOINT_NAME_TO_INDEX = {name: i for i, name in enumerate(_SMPLX_JOINT_NAMES)}
SMPLX_NUM_VERTICES = 10475


def smplx_joint_index(name: str) -> int:
    """Resolve an SMPL-X joint name to its output-joint index.

    Raises KeyError with the available names on typo, rather than silently
    returning a wrong joint.
    """
    if not SMPLX_JOINT_NAME_TO_INDEX:
        raise RuntimeError(
            "The `smplx` pip package is not installed; joint-name resolution "
            "is unavailable. Install it with `pip install smplx`."
        )
    try:
        return SMPLX_JOINT_NAME_TO_INDEX[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown SMPL-X joint name: {name!r}. "
            f"Available: {sorted(SMPLX_JOINT_NAME_TO_INDEX)}"
        ) from e


@dataclass(frozen=True)
class JointCorr:
    """One input point maps to a single named SMPL-X joint.

    `weight` lets a correspondence be down-weighted in the fitting objective
    without removing it entirely (distinct from per-frame confidence, which
    lives on the Observation) — e.g. a MediaPipe "shoulder" landmark sits
    near, but not exactly on, the SMPL-X shoulder joint center, so it is
    given a slightly lower structural weight than a marker known to be
    placed precisely.
    """

    smplx_joint: str
    weight: float = 1.0
    kind = "joint"

    @property
    def smplx_index(self) -> int:
        return smplx_joint_index(self.smplx_joint)

    @property
    def target_index(self) -> int:
        return self.smplx_index


@dataclass(frozen=True)
class VertexCorr:
    """One input point (typically an optical mocap marker) maps to a fixed
    vertex on the SMPL-X mesh surface."""

    vertex_id: int
    weight: float = 1.0
    kind = "vertex"

    def __post_init__(self) -> None:
        if not (0 <= self.vertex_id < SMPLX_NUM_VERTICES):
            raise ValueError(
                f"vertex_id {self.vertex_id} out of range [0, {SMPLX_NUM_VERTICES})"
            )

    @property
    def target_index(self) -> int:
        return self.vertex_id


Corr = Union[JointCorr, VertexCorr]


@dataclass(frozen=True)
class MotionLayout:
    """The full correspondence table for one input skeleton/keypoint format.

    `correspondences` may freely mix `JointCorr` and `VertexCorr` entries —
    the solver (fitting/solver.py) gathers predicted positions from the
    body model's joints or vertices per-entry accordingly.
    """

    name: str
    point_names: List[str]     # names as used by the *source* format
    correspondences: List[Corr]  # same length/order as point_names
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.point_names) != len(self.correspondences):
            raise ValueError(
                f"Layout {self.name!r}: {len(self.point_names)} point_names but "
                f"{len(self.correspondences)} correspondences"
            )

    @property
    def num_points(self) -> int:
        return len(self.point_names)

    @property
    def is_joint_only(self) -> bool:
        return all(c.kind == "joint" for c in self.correspondences)

    def smplx_indices(self) -> List[int]:
        """Joint indices — only valid for a joint-only layout. Kept for
        backward compatibility with skeleton-format layouts; mixed or
        vertex-only layouts must use `target_indices`/`target_kinds`."""
        if not self.is_joint_only:
            raise ValueError(
                f"Layout {self.name!r} is not joint-only; use target_indices()/"
                "target_kinds() instead of smplx_indices()."
            )
        return [c.target_index for c in self.correspondences]

    def target_indices(self) -> List[int]:
        return [c.target_index for c in self.correspondences]

    def target_kinds(self) -> List[str]:
        return [c.kind for c in self.correspondences]

    def weights(self) -> List[float]:
        return [c.weight for c in self.correspondences]
