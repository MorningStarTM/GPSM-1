"""Locate the SMPL-X body-model asset on disk.

The official SMPL-X model files (SMPLX_{FEMALE,MALE,NEUTRAL}.npz) are
registration-gated at https://smpl-x.is.tue.mpg.de and are therefore never
committed to this repository. This module only *finds* a model file already
present on the machine; it never downloads or ships one.

Resolution order:
    1. The `GPSM_SMPLX_MODEL` environment variable, if set — either a
       path directly to a ``SMPLX_<GENDER>.npz`` file, or a directory
       containing one or more of them. (`LEONIDAS_SMPLX_MODEL`, the name this
       module used in the project it was copied from, is still accepted.)
    2. `<repo root>/model/` — the in-repo location (kept out of git; see
       .gitignore) where GPSM already keeps ``SMPLX_FEMALE.npz``.

If nothing is found, `find_smplx_model` returns `None` and callers must
raise a clear, actionable error rather than silently falling back to a
different representation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_VAR = "GPSM_SMPLX_MODEL"
_LEGACY_ENV_VAR = "LEONIDAS_SMPLX_MODEL"


def _first_npz_in_dir(directory: Path) -> Optional[Path]:
    if not directory.is_dir():
        return None
    for gender in ("NEUTRAL", "FEMALE", "MALE"):
        candidate = directory / f"SMPLX_{gender}.npz"
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob("SMPLX_*.npz"))
    return matches[0] if matches else None


def find_smplx_model() -> Optional[Path]:
    """Return a path to a usable ``SMPLX_*.npz`` model file, or None."""
    env_value = os.environ.get(_ENV_VAR) or os.environ.get(_LEGACY_ENV_VAR)
    if env_value:
        p = Path(env_value)
        if p.is_file():
            return p
        found = _first_npz_in_dir(p)
        if found is not None:
            return found

    repo_root = Path(__file__).resolve().parents[3]  # src/gpsm/smplx/config.py -> repo root
    return _first_npz_in_dir(repo_root / "model")


def gender_from_path(path: Path) -> str:
    """Infer the gender token (neutral/female/male) from a model filename."""
    name = path.stem.upper()
    for gender in ("NEUTRAL", "FEMALE", "MALE"):
        if gender in name:
            return gender.lower()
    return "neutral"


class SMPLXModelNotFoundError(RuntimeError):
    """Raised when fitting code needs the SMPL-X asset and none is found."""

    def __init__(self) -> None:
        super().__init__(
            "No SMPL-X model file found. Fitting (Method 2/3) requires the "
            "official body model, which is registration-gated and cannot be "
            "shipped with this repository. Download it from "
            "https://smpl-x.is.tue.mpg.de and either set the "
            f"{_ENV_VAR} environment variable to the .npz file (or its "
            "containing directory), or place it in the repo's model/ folder."
        )
