"""Work out what kind of mocap file we are looking at.

MotionPrep handles two broadly different kinds of source:

* **Parameter files** (``.npz`` holding SMPL / SMPL-H / SMPL-X pose
  parameters). These already describe the body the same way SMPL-X does, so
  converting them is pure slicing — cheap, and exact.
* **Point files** (``.c3d`` optical markers, ``.bvh`` skeletons). These are
  just 3D points in space with no body model behind them, so a body has to
  be *fitted* to them — slow, and approximate.

That distinction matters a lot downstream (see ``unify.py``), which is why
it is spelled out here rather than left implicit in a chain of ``if``s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Source kinds this module can name.
NPZ_PARAMS = "npz_params"   # SMPL/SMPL-H/SMPL-X parameters -> convert (cheap)
C3D = "c3d"                 # optical markers              -> fit (slow)
BVH = "bvh"                 # skeleton joints              -> fit (slow)

#: Kinds that need a body model fitted to them rather than a direct conversion.
NEEDS_FITTING = (C3D, BVH)


class UnknownFormatError(ValueError):
    """Raised for a file MotionPrep does not know how to read.

    Deliberately an error rather than a silent skip: a whole folder quietly
    producing nothing is much harder to debug than one clear message naming
    the file and what was wrong with it.
    """


def detect_source_format(path: str) -> str:
    """Return the source kind of one mocap file (one of the constants above).

    Args:
        path: Path to the file.

    Returns:
        ``NPZ_PARAMS``, ``C3D``, or ``BVH``.

    Raises:
        UnknownFormatError: For an unrecognized extension, or for a ``.npz``
            that does not actually contain pose parameters (e.g. one holding
            only raw joint positions — readable, but not something the
            parameter adapter can convert).
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".c3d":
        return C3D
    if suffix == ".bvh":
        return BVH
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            keys = set(data.files)
        # The parameter adapter recognizes either a combined `poses` array or
        # the split `pose_body`/`pose_hand` keys; anything else is some other
        # kind of .npz that happens to share the extension.
        if "poses" in keys or {"pose_body", "pose_hand"}.issubset(keys):
            return NPZ_PARAMS
        raise UnknownFormatError(
            f"{Path(path).name}: .npz without pose parameters "
            f"(expected a 'poses' array or 'pose_body'+'pose_hand'; found: {sorted(keys)})"
        )

    raise UnknownFormatError(f"{Path(path).name}: unsupported file type {suffix!r}")
