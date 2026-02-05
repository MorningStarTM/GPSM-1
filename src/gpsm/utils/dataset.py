from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

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
