"""Dataset wrapping a train.pt / val.pt written by
``scripts/preprocess_motion_mamba_cache.py``.

Windowing is done lazily here (slicing, near-free) -- the cache itself only
holds each file's already-built, already-normalized array; see that
script's docstring for why nothing more expensive happens at load time.

CONTROL
-------
If the cache was built with control (the default since
preprocess_motion_mamba_cache.py started extracting it), each record also
carries ``control``/``control_valid``. Both dataset classes below detect
this from ``meta["control_dim"]`` and, when present, concatenate
``control`` onto every pose frame they hand to the model --
``concat(pose, control)``, matching ``MotionMamba``'s ``input_dim =
state_dim + control_dim`` and the exact alignment rule
``src/gpsm/utils/control_dataset.py`` documents ("control sits at the same
index as the pose it belongs to"). Loading a control-free cache (built with
--no-control) falls back to the old pose-only behaviour automatically.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class CachedWindowDataset(Dataset):
    """Each sample: ``x`` (k+1, D) history window, ``y`` (D,) next frame.

    ``D`` is ``state_dim`` for a pose-only cache, or ``state_dim +
    control_dim`` (``x`` = concat(pose, control) per frame) for a cache
    built with control -- see module docstring. ``y`` is always pose-only
    (``state_dim``): the model never predicts control.
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
        self.control_dim = cache["meta"].get("control_dim") or 0
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

        # With control, a window's LAST frame (t) must have valid control --
        # that's the frame whose control value is actually fed to the model.
        # control_valid is only ever False for a suffix at the end of a
        # clip (see extract_control's docstring), and t-k < t, so checking
        # just t is enough: if t is valid, every earlier frame in [t-k, t]
        # is too.
        if self.control_dim > 0:
            self._index = [
                (fi, t) for fi, rec in enumerate(self._files) for t in rec["valid_t"]
                if rec["control_valid"][t]
            ]
        else:
            self._index = [(fi, t) for fi, rec in enumerate(self._files) for t in rec["valid_t"]]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        file_i, t = self._index[i]
        rec = self._files[file_i]
        X = rec["X"]
        y = X[t + 1]
        if self.control_dim > 0:
            x = torch.cat([X[t - self.k : t + 1], rec["control"][t - self.k : t + 1]], dim=-1)
        else:
            x = X[t - self.k : t + 1]
        return x, y

    def file_stats(self) -> dict:
        return {"num_files": len(self._files), "num_samples": len(self._index)}


class CachedSFTDataset(Dataset):
    """SFT counterpart to CachedWindowDataset -- same train.pt/val.pt cache,
    different sample shape: a single seed frame plus the real multi-frame
    continuation that followed it, for free-running rollout supervision
    instead of single-step teacher forcing (see
    ``MotionMambaSFTTrainer``'s docstring for why).

    Each sample: ``seed`` (D,) frame to start generating from, ``target``
    (n_frames, state_dim) the real poses that actually followed it, and
    ``control_future`` (n_frames, control_dim) the real control at each of
    those same frames (empty, width 0, for a pose-only cache) -- see module
    docstring. ``MotionMambaSFTTrainer._rollout()`` teacher-forces this
    control while letting the model free-run its own pose predictions: at
    step ``i`` it feeds back ``concat(predicted_pose_i, control_future[i])``,
    exactly mirroring ``MotionMamba.rollout()``'s own
    "``control[i]`` belongs to the newly generated frame ``i``" convention.
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
        self.control_dim = cache["meta"].get("control_dim") or 0
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
        # convention, not this one. With control, the seed's own control
        # (control[t]) and every control the rollout will be fed
        # (control[t+1 : t+1+n_frames]) must also be real -- i.e.
        # control_valid[t : t+n_frames+1] all True, checked via its last
        # (most restrictive, since invalidity is a trailing suffix) index.
        self._index = []
        for file_i, rec in enumerate(self._files):
            T = rec["X"].shape[0]
            control_valid = rec.get("control_valid") if self.control_dim > 0 else None
            for t in range(0, T - self.n_frames):
                if control_valid is not None and not control_valid[t + self.n_frames]:
                    continue
                self._index.append((file_i, t))
        if not self._index:
            raise RuntimeError(
                f"No file in {cache_path} has >= n_frames+1={self.n_frames + 1} frames "
                f"with valid control throughout."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        file_i, t = self._index[i]
        rec = self._files[file_i]
        X = rec["X"]
        seed_pose = X[t]
        target = X[t + 1 : t + 1 + self.n_frames]
        if self.control_dim > 0:
            control = rec["control"]
            seed = torch.cat([seed_pose, control[t]], dim=-1)
            control_future = control[t + 1 : t + 1 + self.n_frames]
        else:
            seed = seed_pose
            control_future = torch.empty(self.n_frames, 0, dtype=X.dtype)
        return seed, target, control_future

    def file_stats(self) -> dict:
        return {"num_files": len(self._files), "num_samples": len(self._index)}
