from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# NaN-filling utilities
# ---------------------------------------------------------------------------

def forward_fill_nan(points_t_m_3: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Forward-fill (then back-fill for leading NaNs) per (marker, coord).
    Any all-NaN columns are set to fill_value.

    points_t_m_3 : (T, M, 3)
    """
    out = points_t_m_3.copy()
    T, M, C = out.shape

    for m in range(M):
        for c in range(C):
            col  = out[:, m, c]
            isn  = np.isnan(col)
            if not isn.any():
                continue
            valid = ~isn
            if not valid.any():
                out[:, m, c] = fill_value
                continue
            idx = np.where(valid, np.arange(T), 0)
            np.maximum.accumulate(idx, out=idx)
            col_ffill = col[idx]
            col_ffill[: np.argmax(valid)] = col[np.argmax(valid)]
            out[:, m, c] = col_ffill

    return out


def forward_fill_nan_2d(x_t_d: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Forward-fill per feature dimension.

    x_t_d : (T, D)
    """
    out = x_t_d.copy()
    T, D = out.shape

    for d in range(D):
        col  = out[:, d]
        isn  = np.isnan(col)
        if not isn.any():
            continue
        valid = ~isn
        if not valid.any():
            out[:, d] = fill_value
            continue
        idx = np.where(valid, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        col_ffill = col[idx]
        col_ffill[: int(np.argmax(valid))] = col[np.argmax(valid)]
        out[:, d] = col_ffill

    return out


# ---------------------------------------------------------------------------
# C3D helpers
# ---------------------------------------------------------------------------

def drop_bad_markers(points_t_m_3: np.ndarray, labels, nan_thresh: float = 0.20):
    nan_rate = np.isnan(points_t_m_3).any(axis=2).mean(axis=0)
    keep     = np.where(nan_rate <= nan_thresh)[0]
    return points_t_m_3[:, keep, :], [labels[i] for i in keep]


def load_c3d(c3d_path: Union[str, Path]):
    import ezc3d
    c3d         = ezc3d.c3d(str(c3d_path))
    pts_4_m_t   = c3d["data"]["points"]
    points      = np.transpose(pts_4_m_t[:3, :, :], (2, 1, 0)).astype(np.float32)
    try:
        labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    except Exception:
        labels = [f"marker_{i}" for i in range(points.shape[1])]
    return points, labels


# ---------------------------------------------------------------------------
# C3D multi-file dataset
# ---------------------------------------------------------------------------

class MultiC3DNextPoseDataset(Dataset):
    """
    One dataset over many .c3d files.

    FIX 3 — vestigial pos tensor removed:
      The old version returned (x, pos, y) where pos = torch.arange(L). The
      model (StateMachineGPT) generates its own positional indices internally
      and never consumed this tensor. It is now dropped from __getitem__ so
      each sample is simply (x, y). The trainer is updated accordingly.

    Each sample:
      x : (k+1, D)  — history window (k frames + current frame)
      y : (D,)      — next frame (target)
    """

    def __init__(
        self,
        c3d_paths,
        k: int = 10,
        nan_thresh: float = 0.20,
        normalize: bool = True,
        strict_dim: bool = False,
        preload: bool = False,
    ):
        super().__init__()
        self.k          = k
        self.L          = k + 1
        self.nan_thresh = nan_thresh
        self.normalize  = normalize
        self.strict_dim = strict_dim
        self.preload    = preload

        if isinstance(c3d_paths, (str, Path)):
            c3d_paths = [c3d_paths]
        self.c3d_paths = [Path(p) for p in c3d_paths]

        self._files: List[Dict] = []
        self._index: List[Tuple[int, int]] = []
        self.D: Optional[int] = None

        for p in self.c3d_paths:
            try:
                points, labels = load_c3d(p)
                points, labels = drop_bad_markers(points, labels, nan_thresh=self.nan_thresh)
                points         = forward_fill_nan(points)

                T, M, _  = points.shape
                D        = M * 3
                if T < (self.k + 2):
                    continue

                X = points.reshape(T, D).astype(np.float32)
                if self.normalize:
                    mean = X.mean(axis=0, keepdims=True)
                    std  = X.std(axis=0, keepdims=True) + 1e-6
                    Xn   = (X - mean) / std
                else:
                    mean = std = None
                    Xn = X

                if self.D is None:
                    self.D = D
                elif D != self.D:
                    if self.strict_dim:
                        raise ValueError(f"[dim mismatch] {p}: D={D}, expected D={self.D}")
                    continue

                valid_t  = list(range(self.k, T - 1))
                file_i   = len(self._files)
                self._files.append({
                    "path":    p,
                    "T":       T,
                    "D":       D,
                    "valid_t": valid_t,
                    "mean":    mean,
                    "std":     std,
                    "X_np":    Xn if not self.preload else None,
                    "X":       torch.from_numpy(Xn).float() if self.preload else None,
                })
                for local_i in range(len(valid_t)):
                    self._index.append((file_i, local_i))

            except Exception:
                continue

        if not self._files:
            raise RuntimeError("No valid .c3d files loaded.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        """
        Returns:
            x : (L, D) — history window
            y : (D,)   — next frame target
        """
        file_i, local_i = self._index[idx]
        rec = self._files[file_i]
        t   = rec["valid_t"][local_i]

        X   = rec["X"] if self.preload else torch.from_numpy(rec["X_np"]).float()
        x   = X[t - self.k : t + 1]   # (L, D)
        y   = X[t + 1]                 # (D,)
        return x, y                    # FIX 3: pos tensor removed

    def file_stats(self) -> Dict:
        return {
            "num_files":   len(self._files),
            "num_samples": len(self._index),
            "D":           self.D,
            "files":       [(str(r["path"]), r["T"], len(r["valid_t"])) for r in self._files],
        }


# ---------------------------------------------------------------------------
# SMPL-X / AMASS NPZ helpers
# ---------------------------------------------------------------------------

def _as_path_list(
    paths_or_dir: Union[str, Path, Sequence[Union[str, Path]]],
    suffix: str = ".npz",
) -> List[Path]:
    if isinstance(paths_or_dir, (str, Path)):
        p = Path(paths_or_dir)
        if p.is_dir():
            return sorted(p.rglob(f"*{suffix}"))
        return [p]
    return [Path(x) for x in paths_or_dir]


def _safe_npz_load(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def build_smplx_features(
    npz: Dict[str, np.ndarray],
    feature_set: str = "poses+trans",
    *,
    include_betas: bool = False,
    include_dmpls: bool = False,
    include_expression: bool = False,
    include_face: bool = False,
    prefer_joints_if_available: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build a (T, D) float32 feature matrix from an AMASS/SMPL-X npz dict.

    Returns:
        X    : (T, D) float32
        info : dict mapping component name -> its feature width
    """
    info: Dict[str, int] = {}

    def has(*names: str) -> Optional[str]:
        for n in names:
            if n in npz:
                return n
        return None

    if prefer_joints_if_available:
        jkey = has("joints3d", "joints_3d", "joints", "Jtr", "joints_xyz")
        if jkey is not None:
            j = np.asarray(npz[jkey])
            if j.ndim != 3 or j.shape[-1] != 3:
                raise ValueError(f"Key '{jkey}' is not (T,J,3). Got {j.shape}")
            T, J, _ = j.shape
            X = j.reshape(T, J * 3).astype(np.float32)
            info["joints3d"] = X.shape[1]
            return X, info

    parts: List[np.ndarray] = []
    T: Optional[int] = None

    def add_part(name: str, arr: np.ndarray):
        nonlocal T
        a = np.asarray(arr)
        if a.ndim == 1:
            if T is None:
                raise ValueError(f"Cannot broadcast static '{name}' — T unknown.")
            a = np.broadcast_to(a.reshape(1, -1), (T, a.size))
        elif a.ndim == 2:
            pass
        elif a.ndim == 3 and a.shape[-1] == 3:
            a = a.reshape(a.shape[0], -1)
        else:
            raise ValueError(f"Unsupported shape for '{name}': {a.shape}")

        if T is None:
            T = a.shape[0]
        elif a.shape[0] != T:
            raise ValueError(f"Time length mismatch for '{name}': {a.shape[0]} vs T={T}")

        a = a.astype(np.float32)
        parts.append(a)
        info[name] = a.shape[1]

    pose_key  = has("poses", "pose", "pose_body")
    trans_key = has("trans", "translations", "root_trans")

    if feature_set in ("poses+trans", "poses_trans", "pt"):
        if pose_key is None:
            raise KeyError("feature_set='poses+trans': no 'poses' key found.")
        add_part("poses", npz[pose_key])
        if trans_key is not None:
            add_part("trans", npz[trans_key])

    elif feature_set in ("poses_only", "poses"):
        if pose_key is None:
            raise KeyError("feature_set='poses_only': no 'poses' key found.")
        add_part("poses", npz[pose_key])

    elif feature_set in ("trans_only", "trans"):
        if trans_key is None:
            raise KeyError("feature_set='trans_only': no 'trans' key found.")
        add_part("trans", npz[trans_key])

    elif feature_set in ("joints3d", "joints"):
        jkey = has("joints3d", "joints_3d", "joints", "Jtr", "joints_xyz")
        if jkey is None:
            raise KeyError("feature_set='joints3d': no joints key found.")
        add_part("joints3d", npz[jkey])

    else:
        raise ValueError(f"Unknown feature_set='{feature_set}'")

    if include_betas:
        bkey = has("betas", "shape", "beta")
        if bkey is not None:
            b = np.asarray(npz[bkey]).reshape(-1).astype(np.float32)
            add_part("betas", b)

    if include_dmpls:
        dkey = has("dmpls", "dmpl", "dmp")
        if dkey is not None:
            add_part("dmpls", npz[dkey])

    if include_expression:
        ekey = has("expression", "expr", "exp")
        if ekey is not None:
            add_part("expression", npz[ekey])

    if include_face:
        for name, keys in [
            ("jaw_pose",  ("jaw_pose", "jaw")),
            ("leye_pose", ("leye_pose", "left_eye_pose", "leye")),
            ("reye_pose", ("reye_pose", "right_eye_pose", "reye")),
        ]:
            k = has(*keys)
            if k is not None:
                add_part(name, npz[k])

    if T is None or not parts:
        raise RuntimeError("Failed to build features — no valid parts found.")

    return np.concatenate(parts, axis=1).astype(np.float32), info


# ---------------------------------------------------------------------------
# SMPL-X multi-file dataset
# ---------------------------------------------------------------------------

class MultiSMPLXNPZNextPoseDataset(Dataset):
    """
    One dataset over many SMPL-X / AMASS-style .npz files.

    FIX 3 — vestigial pos tensor removed:
      The old version returned (x, pos, y). StateMachineGPT never used the
      pos tensor — it generates torch.arange(T) internally. It is now
      dropped, making each sample simply (x, y) to avoid confusion and
      wasted DataLoader bandwidth.

    Each sample:
      x : (k+1, D)  — history window
      y : (D,)      — next frame (target)
    """

    def __init__(
        self,
        npz_paths_or_dir: Union[str, Path, Sequence[Union[str, Path]]],
        k: int = 10,
        *,
        feature_set: str = "poses+trans",
        prefer_joints_if_available: bool = False,
        include_betas: bool = False,
        include_dmpls: bool = False,
        include_expression: bool = False,
        include_face: bool = False,
        fill_nan: bool = True,
        normalize: bool = True,
        strict_dim: bool = False,
        preload: bool = False,
        min_frames: Optional[int] = None,
        file_range: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.k                          = int(k)
        self.L                          = self.k + 1
        self.feature_set                = feature_set
        self.prefer_joints_if_available = prefer_joints_if_available
        self.include_betas              = include_betas
        self.include_dmpls              = include_dmpls
        self.include_expression         = include_expression
        self.include_face               = include_face
        self.fill_nan                   = fill_nan
        self.normalize                  = normalize
        self.strict_dim                 = strict_dim
        self.preload                    = preload

        self.npz_paths = _as_path_list(npz_paths_or_dir, suffix=".npz")
        if not self.npz_paths:
            raise RuntimeError("No .npz files found.")

        # file_range lets you train on a slice of the sorted file list
        # (e.g. file_range=(0, 500), then (500, 1000), ...) instead of the
        # whole dataset at once, so one epoch stays a manageable size.
        if file_range is not None:
            start, end = file_range
            self.npz_paths = self.npz_paths[start:end]
            if not self.npz_paths:
                raise RuntimeError(
                    f"file_range={file_range} selected no files out of "
                    f"{len(_as_path_list(npz_paths_or_dir, suffix='.npz'))} total."
                )

        self._files: List[Dict]            = []
        self._index: List[Tuple[int, int]] = []
        self.D: Optional[int]              = None

        min_needed = max(self.k + 2, int(min_frames) if min_frames else self.k + 2)

        for p in self.npz_paths:
            try:
                npz  = _safe_npz_load(p)
                X, info = build_smplx_features(
                    npz,
                    feature_set=self.feature_set,
                    include_betas=self.include_betas,
                    include_dmpls=self.include_dmpls,
                    include_expression=self.include_expression,
                    include_face=self.include_face,
                    prefer_joints_if_available=self.prefer_joints_if_available,
                )

                if self.fill_nan:
                    X = forward_fill_nan_2d(X, fill_value=0.0)

                T, D = X.shape
                if T < min_needed:
                    continue

                if self.normalize:
                    mean = X.mean(axis=0, keepdims=True)
                    std  = X.std(axis=0, keepdims=True) + 1e-6
                    Xn   = (X - mean) / std
                else:
                    mean = std = None
                    Xn = X

                if self.D is None:
                    self.D = D
                elif D != self.D:
                    if self.strict_dim:
                        raise ValueError(f"[dim mismatch] {p}: D={D}, expected D={self.D}")
                    continue

                valid_t  = list(range(self.k, T - 1))
                file_i   = len(self._files)
                self._files.append({
                    "path":         p,
                    "T":            T,
                    "D":            D,
                    "valid_t":      valid_t,
                    "mean":         mean,
                    "std":          std,
                    "feature_info": info,
                    "X_np":         Xn if not self.preload else None,
                    "X":            torch.from_numpy(Xn).float() if self.preload else None,
                })
                for local_i in range(len(valid_t)):
                    self._index.append((file_i, local_i))

            except Exception:
                continue

        if not self._files:
            raise RuntimeError(
                "No valid .npz files loaded. "
                "Common causes: missing 'poses'/'trans' keys, sequences too short, "
                "or feature-dimension mismatches across files."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        """
        Returns:
            x : (L, D)  — history window  (L = k + 1)
            y : (D,)    — next frame target

        FIX 3: the vestigial pos tensor (torch.arange(L)) has been removed.
        StateMachineGPT.forward() creates its own positional indices with
        torch.arange(T) internally — no external pos is needed.
        """
        file_i, local_i = self._index[idx]
        rec = self._files[file_i]
        t   = rec["valid_t"][local_i]

        X   = rec["X"] if self.preload else torch.from_numpy(rec["X_np"]).float()
        x   = X[t - self.k : t + 1]   # (L, D)
        y   = X[t + 1]                 # (D,)
        return x, y                    # FIX 3: pos removed

    def file_stats(self) -> Dict:
        return {
            "num_files":   len(self._files),
            "num_samples": len(self._index),
            "D":           self.D,
            "files": [
                (str(r["path"]), r["T"], len(r["valid_t"]), r["feature_info"])
                for r in self._files
            ],
        }