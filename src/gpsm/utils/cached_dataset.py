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
