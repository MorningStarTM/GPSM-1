"""Training windows built from MotionPrep's output.

Reads ``processed/with_control/*.npz`` (MotionPrep Stage 2) and serves the
pairs the control-conditioned model trains on:

    x = concat(pose, control) for frames [t-k .. t]     (k+1, pose_dim + control_dim)
    y = pose at frame t+1                               (pose_dim,)

The alignment rule, which is the thing most worth getting right: ``control``
sits at the **same index** as the pose it belongs to, and the target is
pose-only. Because the target is frame ``t+1`` and the window ends at frame
``t``, the newest control the model ever sees is ``control[t]`` — never
``control[t+1]``, which in a real game would not exist yet at that moment.

Which pose numbers are used
---------------------------
MotionPrep stores the full 159-number canonical pose, but the hand poses in
this data never move: every source either has no hand markers at all (C3D,
BVH) or carries a frozen rest pose (AMASS parameters), and ``part_mask``
marks them unobserved. Training on them would hand the model 90 constant
numbers to "predict", which costs capacity and quietly flatters the loss.
So the default is the 69 numbers that actually move — root translation, root
rotation, and body joints — with the field list left configurable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

#: The pose fields that carry real motion in MotionPrep's output.
#: trans (3) + global_orient (3) + body_pose (63) = 69 numbers.
BODY_POSE_FIELDS = ("trans", "global_orient", "body_pose")

#: Every canonical pose field, hands included (159 numbers). Only useful on
#: data where the hands were actually observed.
ALL_POSE_FIELDS = ("trans", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose")


class ControlWindowDataset(Dataset):
    """Windows of (pose + control) -> next pose, from MotionPrep clips.

    Args:
        clips_dir:   Folder of MotionPrep Stage 2 ``.npz`` files.
        k:           How many past frames precede the current one, so each
                     window is ``k + 1`` frames long.
        pose_fields: Which canonical pose fields to train on (see the module
                     docstring for why hands are excluded by default).
        normalize:   Z-score the pose per clip, matching the convention the
                     existing datasets in ``dataset.py`` use. Control is left
                     in its own physical units — its zero points are
                     meaningful (``turn_sin`` of 0 really means "going
                     straight"), and per-clip rescaling would destroy that.
        use_control: Include control in the input. Setting this False gives
                     the plain next-pose task, which is useful as a baseline
                     to compare against.

    Raises:
        RuntimeError: If no clip in the folder is long enough to make even
            one window.
    """

    def __init__(
        self,
        clips_dir: str,
        k: int = 8,
        pose_fields: Sequence[str] = BODY_POSE_FIELDS,
        normalize: bool = True,
        use_control: bool = True,
    ):
        self.k = int(k)
        self.pose_fields = tuple(pose_fields)
        self.normalize = normalize
        self.use_control = use_control

        self._clips: List[Dict] = []
        self._index: List[Tuple[int, int]] = []   # (clip number, window start)
        self.pose_dim: int = 0
        self.control_dim: int = 0

        for path in sorted(Path(clips_dir).glob("*.npz")):
            clip = self._load_clip(path)
            if clip is None:
                continue
            clip_number = len(self._clips)
            self._clips.append(clip)
            # A window needs k+1 frames of history plus one target frame.
            for start in range(clip["n_frames"] - self.k - 1):
                self._index.append((clip_number, start))

        if not self._index:
            raise RuntimeError(
                f"No usable clips in {clips_dir} — every clip needs more than "
                f"{self.k + 1} frames to make a single window."
            )

    def _load_clip(self, path: Path):
        """Read one clip, or return None if it is too short to use."""
        with np.load(path, allow_pickle=True) as data:
            pose = np.concatenate([data[f] for f in self.pose_fields], axis=1).astype(np.float64)
            control = data["control"].astype(np.float64) if "control" in data.files else None
            # Frames near the end have no full look-ahead, so their control
            # is not a real measurement; those windows are dropped below.
            control_valid = data["control_valid"] if "control_valid" in data.files else None
            fps = float(data["fps"])

        n_frames = pose.shape[0]
        if self.use_control:
            if control is None:
                raise KeyError(f"{path.name}: no 'control' — was this file produced by MotionPrep Stage 2?")
            # Keep only the stretch where control is trustworthy.
            if control_valid is not None and not control_valid.all():
                n_frames = int(control_valid.sum())
                pose, control = pose[:n_frames], control[:n_frames]

        if n_frames < self.k + 2:
            return None

        if self.pose_dim == 0:
            self.pose_dim = pose.shape[1]
            self.control_dim = control.shape[1] if (self.use_control and control is not None) else 0

        mean = pose.mean(axis=0, keepdims=True) if self.normalize else np.zeros((1, pose.shape[1]))
        # The +1e-6 keeps a constant channel from dividing by zero; such a
        # channel has pose - mean == 0 anyway, so it normalizes to 0.
        std = pose.std(axis=0, keepdims=True) + 1e-6 if self.normalize else np.ones((1, pose.shape[1]))

        return {
            "name": path.name,
            "pose": ((pose - mean) / std).astype(np.float32),
            "control": control.astype(np.float32) if control is not None else None,
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "fps": fps,
            "n_frames": n_frames,
        }

    @property
    def input_dim(self) -> int:
        """Width of one input frame: pose, plus control if it is in use."""
        return self.pose_dim + (self.control_dim if self.use_control else 0)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        """Returns ``(x, y)``: ``x`` is ``(k+1, input_dim)``, ``y`` is ``(pose_dim,)``."""
        clip_number, start = self._index[i]
        clip = self._clips[clip_number]
        end = start + self.k + 1              # window covers frames [start, end)

        pose_window = clip["pose"][start:end]
        if self.use_control:
            x = np.concatenate([pose_window, clip["control"][start:end]], axis=1)
        else:
            x = pose_window
        y = clip["pose"][end]                  # the next pose — pose only, never control

        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

    def summary(self) -> str:
        return (
            f"{len(self._clips)} clips, {len(self)} windows, "
            f"pose_dim={self.pose_dim}, control_dim={self.control_dim if self.use_control else 0}, "
            f"input_dim={self.input_dim}, k={self.k}"
        )
