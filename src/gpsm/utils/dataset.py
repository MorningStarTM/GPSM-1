from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence, Tuple, Union


# -------------------- utils (same as yours) --------------------
def forward_fill_nan(points_t_m_3: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    ffill + bfill per (marker, coord). Any remaining all-NaN columns -> fill_value.
    """
    out = points_t_m_3.copy()
    T, M, C = out.shape

    for m in range(M):
        for c in range(C):
            col = out[:, m, c]  # (T,)
            isn = np.isnan(col)
            if not isn.any():
                continue

            valid = ~isn
            if not valid.any():
                # Entire column is NaN
                out[:, m, c] = fill_value
                continue

            # forward fill
            idx = np.where(valid, np.arange(T), 0)
            np.maximum.accumulate(idx, out=idx)
            col_ffill = col[idx]

            # backward fill for leading NaNs (use first valid value)
            first_valid_idx = np.argmax(valid)  # first True
            col_ffill[:first_valid_idx] = col[first_valid_idx]

            out[:, m, c] = col_ffill

    return out


def drop_bad_markers(points_t_m_3: np.ndarray, labels, nan_thresh=0.20):
    nan_mask = np.isnan(points_t_m_3).any(axis=2)  # (T, M)
    nan_rate = nan_mask.mean(axis=0)               # (M,)
    keep = np.where(nan_rate <= nan_thresh)[0]
    points = points_t_m_3[:, keep, :]
    labels = [labels[i] for i in keep]
    return points, labels

def load_c3d(c3d_path: str | Path):
    import ezc3d
    c3d = ezc3d.c3d(str(c3d_path))
    pts_4_m_t = c3d["data"]["points"]  # (4, M, T)
    points = np.transpose(pts_4_m_t[:3, :, :], (2, 1, 0)).astype(np.float32)  # (T, M, 3)
    try:
        labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    except Exception:
        labels = [f"marker_{i}" for i in range(points.shape[1])]
    return points, labels

# -------------------- multi-file dataset --------------------
class MultiC3DNextPoseDataset(Dataset):
    """
    Build one dataset over MANY .c3d files.

    Each sample is: history window from a specific file + target next frame

    Returns:
      x   : (L, D)
      pos : (L,)
      y   : (D,)

    Notes:
      - For simplicity, we require all files end up with the SAME D after dropping markers.
        (This is usually true for CMU MoCap when marker set is consistent.)
      - If a file produces different D, we skip it by default (or you can set strict=True to error).
    """
    def __init__(
        self,
        c3d_paths,
        k=10,
        nan_thresh=0.20,
        normalize=True,
        strict_dim=False,
        preload=False,   # preload all tensors (fast iteration, more RAM)
    ):
        super().__init__()
        self.k = k
        self.L = k + 1
        self.nan_thresh = nan_thresh
        self.normalize = normalize
        self.strict_dim = strict_dim
        self.preload = preload

        # normalize input to list[Path]
        if isinstance(c3d_paths, (str, Path)):
            c3d_paths = [c3d_paths]
        self.c3d_paths = [Path(p) for p in c3d_paths]

        # fixed pos for all samples
        self.pos = torch.arange(self.L, dtype=torch.long)

        # per-file storage / metadata
        self._files = []   # each: dict(path, X, T, D, valid_t, mean, std)
        self._index = []   # global index: list of (file_i, local_i)

        self.D = None

        for p in self.c3d_paths:
            try:
                points, labels = load_c3d(p)
                points, labels = drop_bad_markers(points, labels, nan_thresh=self.nan_thresh)
                points = forward_fill_nan(points)

                T, M, _ = points.shape
                D = M * 3
                if T < (self.k + 2):  # need at least (k+1) history + 1 target
                    continue

                X = points.reshape(T, D).astype(np.float32)

                if self.normalize:
                    mean = X.mean(axis=0, keepdims=True)
                    std = X.std(axis=0, keepdims=True) + 1e-6
                    Xn = (X - mean) / std
                else:
                    mean, std = None, None
                    Xn = X

                # enforce consistent D across files
                if self.D is None:
                    self.D = D
                elif D != self.D:
                    if self.strict_dim:
                        raise ValueError(f"[dim mismatch] {p} produced D={D}, expected D={self.D}")
                    else:
                        # skip inconsistent files
                        continue

                X_tensor = torch.from_numpy(Xn).float() if self.preload else None

                valid_t = list(range(self.k, T - 1))  # t in [k .. T-2]

                file_rec = {
                    "path": p,
                    "T": T,
                    "D": D,
                    "valid_t": valid_t,
                    "mean": mean,
                    "std": std,
                    "X_np": Xn if not self.preload else None,
                    "X": X_tensor,
                }
                file_i = len(self._files)
                self._files.append(file_rec)

                # build global index map
                for local_i in range(len(valid_t)):
                    self._index.append((file_i, local_i))

            except Exception as e:
                # skip broken files
                # (you can print/log e if you want)
                continue

        if len(self._files) == 0:
            raise RuntimeError("No valid .c3d files found/loaded (after filtering and length checks).")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_i, local_i = self._index[idx]
        rec = self._files[file_i]
        t = rec["valid_t"][local_i]  # end-of-history index

        # load X either from preloaded torch tensor or stored numpy
        if self.preload:
            X = rec["X"]  # (T,D) torch
        else:
            X = torch.from_numpy(rec["X_np"]).float()

        x = X[t - self.k : t + 1]  # (L,D)
        y = X[t + 1]               # (D,)
        return x, self.pos, y

    def file_stats(self):
        """Optional: inspect what got loaded."""
        return {
            "num_files": len(self._files),
            "num_samples": len(self._index),
            "D": self.D,
            "files": [(str(r["path"]), r["T"], len(r["valid_t"])) for r in self._files],
        }




def forward_fill_nan_2d(x_t_d: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    ffill + bfill per feature-dimension. Any all-NaN columns -> fill_value.
    x_t_d: (T, D)
    """
    out = x_t_d.copy()
    T, D = out.shape

    for d in range(D):
        col = out[:, d]
        isn = np.isnan(col)
        if not isn.any():
            continue

        valid = ~isn
        if not valid.any():
            out[:, d] = fill_value
            continue

        # forward fill
        idx = np.where(valid, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        col_ffill = col[idx]

        # backward fill for leading NaNs
        first_valid_idx = int(np.argmax(valid))
        col_ffill[:first_valid_idx] = col[first_valid_idx]

        out[:, d] = col_ffill

    return out


def _as_path_list(paths_or_dir: Union[str, Path, Sequence[Union[str, Path]]],
                  suffix: str = ".npz") -> List[Path]:
    if isinstance(paths_or_dir, (str, Path)):
        p = Path(paths_or_dir)
        if p.is_dir():
            return sorted(p.rglob(f"*{suffix}"))
        return [p]
    return [Path(x) for x in paths_or_dir]


def _safe_npz_load(npz_path: Path) -> Dict[str, np.ndarray]:
    # AMASS-style npz often uses allow_pickle; keep it on.
    with np.load(str(npz_path), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# -------------------- SMPL-X npz feature builder --------------------
def build_smplx_features(
    npz: Dict[str, np.ndarray],
    feature_set: str = "poses+trans",
    *,
    include_betas: bool = False,
    include_dmpls: bool = False,
    include_expression: bool = False,
    include_face: bool = False,     # jaw/eyes if present
    prefer_joints_if_available: bool = False,  # if npz already has joints/joints3d, use them
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Returns:
      X: (T, D) float32
      info: dict with dimensions per component

    Common SMPL-X/AMASS keys you may see:
      - poses: (T, P) axis-angle for all joints (SMPL/SMPL-X layout depends on exporter)
      - trans: (T, 3)
      - betas: (B,) or (1,B)
      - dmpls: (T, 8) (if SMPL+D)
      - expression: (T, E)
      - jaw_pose / leye_pose / reye_pose: (T, 3) each (sometimes)
      - joints / joints3d / joints_3d: (T, J, 3) (sometimes already computed)
    """
    info: Dict[str, int] = {}

    def has(*names: str) -> Optional[str]:
        for n in names:
            if n in npz:
                return n
        return None

    # Optionally use precomputed joints if present (no SMPL-X layer needed)
    if prefer_joints_if_available:
        jkey = has("joints3d", "joints_3d", "joints", "Jtr", "joints_xyz")
        if jkey is not None:
            j = np.asarray(npz[jkey])
            if j.ndim != 3 or j.shape[-1] != 3:
                raise ValueError(f"Key '{jkey}' exists but is not (T,J,3). Got {j.shape}")
            T, J, _ = j.shape
            X = j.reshape(T, J * 3).astype(np.float32)
            info["joints3d"] = X.shape[1]
            # You can still append trans/betas/etc if you want
            # (kept minimal here)
            return X, info

    parts: List[np.ndarray] = []
    T: Optional[int] = None

    def add_part(name: str, arr: np.ndarray):
        nonlocal T
        a = np.asarray(arr)
        if a.ndim == 1:
            # static vector -> broadcast over time
            if T is None:
                raise ValueError(f"Cannot broadcast static '{name}' because T is unknown yet.")
            a = np.broadcast_to(a.reshape(1, -1), (T, a.size))
        elif a.ndim == 2:
            pass
        elif a.ndim == 3 and a.shape[-1] == 3:
            # (T,J,3) -> flatten
            a = a.reshape(a.shape[0], -1)
        else:
            raise ValueError(f"Unsupported shape for '{name}': {a.shape}")

        if T is None:
            T = a.shape[0]
        else:
            if a.shape[0] != T:
                raise ValueError(f"Time length mismatch for '{name}': {a.shape[0]} vs T={T}")

        a = a.astype(np.float32)
        parts.append(a)
        info[name] = a.shape[1]

    # --- main feature set ---
    # default "poses+trans" is the most common for AMASS/SMPL(-X) motion.
    pose_key = has("poses", "pose", "pose_body")  # different exports
    trans_key = has("trans", "translations", "root_trans")

    if feature_set in ("poses+trans", "poses_trans", "pt"):
        if pose_key is None:
            raise KeyError("feature_set='poses+trans' but no 'poses' key found in npz.")
        add_part("poses", npz[pose_key])

        # after poses, T is known
        if trans_key is not None:
            add_part("trans", npz[trans_key])

    elif feature_set in ("poses_only", "poses"):
        if pose_key is None:
            raise KeyError("feature_set='poses_only' but no 'poses' key found in npz.")
        add_part("poses", npz[pose_key])

    elif feature_set in ("trans_only", "trans"):
        if trans_key is None:
            raise KeyError("feature_set='trans_only' but no 'trans' key found in npz.")
        add_part("trans", npz[trans_key])

    elif feature_set in ("joints3d", "joints"):
        jkey = has("joints3d", "joints_3d", "joints", "Jtr", "joints_xyz")
        if jkey is None:
            raise KeyError("feature_set='joints3d' but no joints key found (joints/joints3d/Jtr/...).")
        add_part("joints3d", npz[jkey])

    else:
        raise ValueError(f"Unknown feature_set='{feature_set}'")

    # --- optional additions ---
    if include_betas:
        bkey = has("betas", "shape", "beta")
        if bkey is not None:
            # betas often (B,) or (1,B)
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
        # some SMPL-X exports store these separately; others bake into poses
        for name, keys in [
            ("jaw_pose", ("jaw_pose", "jaw")),
            ("leye_pose", ("leye_pose", "left_eye_pose", "leye")),
            ("reye_pose", ("reye_pose", "right_eye_pose", "reye")),
        ]:
            k = has(*keys)
            if k is not None:
                add_part(name, npz[k])

    if T is None or len(parts) == 0:
        raise RuntimeError("Failed to build features (no valid parts found).")

    X = np.concatenate(parts, axis=1).astype(np.float32)  # (T,D)
    return X, info


# -------------------- multi-file dataset (SMPL-X npz) --------------------
class MultiSMPLXNPZNextPoseDataset(Dataset):
    """
    Build one dataset over MANY SMPL-X/AMASS-like .npz files.

    Each sample:
      x   : (L, D) history window
      pos : (L,)
      y   : (D,) next frame

    Notes:
      - D must be consistent across files (or you can set strict_dim=True to error,
        otherwise mismatching files are skipped).
      - By default we normalize PER-FILE (like your C3D version).
        If you want global normalization, see `compute_global_norm()` below.
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
        preload: bool = False,   # preload all tensors (fast iteration, more RAM)
        min_frames: Optional[int] = None,  # override minimum length if you want
    ):
        super().__init__()
        self.k = int(k)
        self.L = self.k + 1
        self.feature_set = feature_set
        self.prefer_joints_if_available = prefer_joints_if_available
        self.include_betas = include_betas
        self.include_dmpls = include_dmpls
        self.include_expression = include_expression
        self.include_face = include_face
        self.fill_nan = fill_nan
        self.normalize = normalize
        self.strict_dim = strict_dim
        self.preload = preload

        self.npz_paths = _as_path_list(npz_paths_or_dir, suffix=".npz")
        if len(self.npz_paths) == 0:
            raise RuntimeError("No .npz files found.")

        self.pos = torch.arange(self.L, dtype=torch.long)

        self._files: List[Dict] = []
        self._index: List[Tuple[int, int]] = []
        self.D: Optional[int] = None

        # Need at least (k+1) history + 1 target => k+2 frames
        min_needed = (self.k + 2) if min_frames is None else int(min_frames)
        min_needed = max(min_needed, self.k + 2)

        for p in self.npz_paths:
            try:
                npz = _safe_npz_load(p)

                X, info = build_smplx_features(
                    npz,
                    feature_set=self.feature_set,
                    include_betas=self.include_betas,
                    include_dmpls=self.include_dmpls,
                    include_expression=self.include_expression,
                    include_face=self.include_face,
                    prefer_joints_if_available=self.prefer_joints_if_available,
                )  # (T,D)

                if self.fill_nan:
                    X = forward_fill_nan_2d(X, fill_value=0.0)

                T, D = X.shape
                if T < min_needed:
                    continue

                if self.normalize:
                    mean = X.mean(axis=0, keepdims=True)
                    std = X.std(axis=0, keepdims=True) + 1e-6
                    Xn = (X - mean) / std
                else:
                    mean, std = None, None
                    Xn = X

                # enforce consistent D across files
                if self.D is None:
                    self.D = D
                elif D != self.D:
                    if self.strict_dim:
                        raise ValueError(f"[dim mismatch] {p} produced D={D}, expected D={self.D}")
                    else:
                        continue

                X_tensor = torch.from_numpy(Xn).float() if self.preload else None
                valid_t = list(range(self.k, T - 1))  # t in [k .. T-2]

                file_rec = {
                    "path": p,
                    "T": T,
                    "D": D,
                    "valid_t": valid_t,
                    "mean": mean,
                    "std": std,
                    "feature_info": info,  # debugging
                    "X_np": Xn if not self.preload else None,
                    "X": X_tensor,
                }
                file_i = len(self._files)
                self._files.append(file_rec)

                for local_i in range(len(valid_t)):
                    self._index.append((file_i, local_i))

            except Exception:
                # skip broken/unknown npz files
                continue

        if len(self._files) == 0:
            raise RuntimeError(
                "No valid .npz files loaded. "
                "Common reasons: missing keys (poses/trans), too short sequences, or dim mismatches."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        file_i, local_i = self._index[idx]
        rec = self._files[file_i]
        t = rec["valid_t"][local_i]

        if self.preload:
            X = rec["X"]  # (T,D) torch
        else:
            X = torch.from_numpy(rec["X_np"]).float()

        x = X[t - self.k : t + 1]  # (L,D)
        y = X[t + 1]               # (D,)
        return x, self.pos, y

    def file_stats(self) -> Dict:
        return {
            "num_files": len(self._files),
            "num_samples": len(self._index),
            "D": self.D,
            "files": [
                (str(r["path"]), r["T"], len(r["valid_t"]), r["feature_info"])
                for r in self._files
            ],
        }
