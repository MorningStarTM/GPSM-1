"""Dataset wrapping a train.pt / val.pt written by
``scripts/preprocess_motion_mamba_cache.py``.

Windowing is done lazily here (slicing, near-free) -- the cache itself only
holds each file's already-built, already-normalized array; see that
script's docstring for why nothing more expensive happens at load time.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class CachedWindowDataset(Dataset):
    """Each sample: ``x`` (k+1, D) history window, ``y`` (D,) next frame.

    Same (x, y) shape contract as ``MultiSMPLXNPZNextPoseDataset``, so it
    drops into the same training code unchanged.
    """

    def __init__(self, cache_path: str, file_range: Optional[Tuple[int, int]] = None):
        """
        Args:
            cache_path: Path to a train.pt or val.pt written by
                        preprocess_motion_mamba_cache.py.
            file_range: Optional ``(start, end)`` to use only a slice of the
                        cached files -- e.g. for splitting a DDP run's data
                        across GPUs, one disjoint file range per rank (same
                        idea as ``MultiSMPLXNPZNextPoseDataset``'s
                        ``file_range``, applied to the already-cached file
                        list instead of re-scanning disk).
        """
        cache = torch.load(cache_path)
        self.k = cache["meta"]["k"]
        self.state_dim = cache["meta"]["state_dim"]
        self.meta = cache["meta"]

        files = cache["files"]
        if file_range is not None:
            start, end = file_range
            files = files[start:end]
            if not files:
                raise RuntimeError(
                    f"file_range={file_range} selected no files out of "
                    f"{len(cache['files'])} total in {cache_path}."
                )
        self._files = files
        self._index = [(fi, t) for fi, rec in enumerate(self._files) for t in rec["valid_t"]]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        file_i, t = self._index[i]
        X = self._files[file_i]["X"]
        return X[t - self.k : t + 1], X[t + 1]

    def file_stats(self) -> dict:
        return {"num_files": len(self._files), "num_samples": len(self._index)}


class CachedSFTDataset(Dataset):
    """SFT counterpart to CachedWindowDataset -- same train.pt/val.pt cache,
    different sample shape: a single seed frame plus the real multi-frame
    continuation that followed it, for free-running rollout supervision
    instead of single-step teacher forcing (see
    ``MotionMambaSFTTrainer``'s docstring for why).

    Same ``(seed, target)`` contract as ``MultiSMPLXNPZSFTDataset``, so it
    drops into the same training code unchanged -- this just reads the
    already-built, already-normalized cache instead of rescanning .npz.

    Each sample: ``seed`` (D,) frame to start generating from, ``target``
    (n_frames, D) the real frames that actually followed it.
    """

    def __init__(self, cache_path: str, n_frames: int, file_range: Optional[Tuple[int, int]] = None):
        """
        Args:
            cache_path: Path to a train.pt or val.pt written by
                        preprocess_motion_mamba_cache.py.
            n_frames:   Rollout length supervised per sample. The cache's
                        own ``k`` (its next-pose window length) is unrelated
                        to this -- the cache stores each file's full
                        sequence, not pre-windowed, so it works for either
                        windowing convention.
            file_range: Optional ``(start, end)`` file slice, same as
                        ``CachedWindowDataset``.
        """
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        self.n_frames = int(n_frames)

        cache = torch.load(cache_path)
        self.state_dim = cache["meta"]["state_dim"]
        self.meta = cache["meta"]

        files = cache["files"]
        if file_range is not None:
            start, end = file_range
            files = files[start:end]
            if not files:
                raise RuntimeError(
                    f"file_range={file_range} selected no files out of "
                    f"{len(cache['files'])} total in {cache_path}."
                )
        self._files = files

        # Valid seed positions t: seed=X[t], target=X[t+1 : t+1+n_frames],
        # both need to stay in bounds, i.e. t + n_frames <= T - 1 -- same
        # rule as MultiSMPLXNPZSFTDataset, computed fresh here since the
        # cache's own valid_t was built for the next-pose (k-history)
        # convention, not this one.
        self._index = []
        for file_i, rec in enumerate(self._files):
            T = rec["X"].shape[0]
            for t in range(0, T - self.n_frames):
                self._index.append((file_i, t))
        if not self._index:
            raise RuntimeError(
                f"No file in {cache_path} has >= n_frames+1={self.n_frames + 1} frames."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        file_i, t = self._index[i]
        X = self._files[file_i]["X"]
        seed = X[t]
        target = X[t + 1 : t + 1 + self.n_frames]
        return seed, target

    def file_stats(self) -> dict:
        return {"num_files": len(self._files), "num_samples": len(self._index)}
