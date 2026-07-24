"""
Turns a single mocap .npz file into the same (D,) feature frame(s) the
trained StateMachineGPT model expects — the identical feature construction
and per-file z-score normalization used by MultiSMPLXNPZNextPoseDataset
(see src/gpsm/utils/dataset.py) — so a frame returned here is numerically
what the model actually saw during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from src.gpsm.utils.dataset import build_smplx_features, forward_fill_nan_2d
from src.gpsm.tests.visualize_npz import load_npz


def load_sequence_features(
    npz_path: Union[str, Path],
    feature_set: str = "poses+trans",
    include_betas: bool = False,
    include_expression: bool = False,
    include_face: bool = False,
    prefer_joints_if_available: bool = False,
    fill_nan: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build the full (T, D) feature matrix for one npz file.

    Returns:
        X    : (T, D) float32 — z-score normalized per-file if normalize=True
               (matching training preprocessing), else raw feature values.
        mean : (D,) per-file mean, or None if normalize=False.
        std  : (D,) per-file std (+eps), or None if normalize=False.

    mean/std are needed to map model input/output back to real SMPL-X
    pose/trans values (e.g. before feeding predictions through the SMPL-X
    body model for visualization).
    """
    npz = load_npz(Path(npz_path))
    X, _info = build_smplx_features(
        npz,
        feature_set=feature_set,
        include_betas=include_betas,
        include_expression=include_expression,
        include_face=include_face,
        prefer_joints_if_available=prefer_joints_if_available,
    )
    if fill_nan:
        X = forward_fill_nan_2d(X, fill_value=0.0)

    if not normalize:
        return X.astype(np.float32), None, None

    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0, keepdims=True) + 1e-6
    Xn   = (X - mean) / std
    return Xn.astype(np.float32), mean.reshape(-1), std.reshape(-1)


def get_first_frame(npz_path: Union[str, Path], **kwargs) -> np.ndarray:
    """Return just the first frame (D,) of a given npz — same preprocessing as
    load_sequence_features(). This is what gets fed to the model as the
    autoregressive rollout seed."""
    X, _mean, _std = load_sequence_features(npz_path, **kwargs)
    return X[0]
