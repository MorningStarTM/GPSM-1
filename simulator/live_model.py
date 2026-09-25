"""Drive a trained MotionMamba live, one frame at a time.

This is the run-time counterpart to training: instead of reading a recorded
clip, it keeps a rolling window of recent frames, is handed a control vector
each frame (from the player), asks the model for the next pose, and turns
that pose into a posed SMPL-X body ready to draw.

    control (from keys) ─┐
                          ├─> model.predict_next(window) ─> pose ─> SMPL-X ─> vertices
    window of past frames ┘            │
                                        └─ appended to the window, so the next
                                           frame continues from this one

Normalization: the model is trained on pose that was z-scored per clip, so
the numbers it emits are normalized too. To turn those back into a real body
we need the mean/std they were scaled by — so a session is *seeded* from a
real clip and reuses that clip's statistics. That is a real limitation of
this harness (a shipped game would train with fixed, global statistics), and
it is why the simulator asks for a seed clip rather than starting from
nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.gpsm.ssm.motion_mamba import MotionMamba

#: Layout of the 69-number pose the model predicts (see
#: src/gpsm/utils/control_dataset.py: trans + global_orient + body_pose).
TRANS = slice(0, 3)
GLOBAL_ORIENT = slice(3, 6)
BODY_POSE = slice(6, 69)


def load_model(checkpoint: str) -> MotionMamba:
    """Rebuild a trained MotionMamba from its checkpoint and its saved config."""
    base = Path(checkpoint).with_suffix("")
    with open(base.with_suffix(".json"), "r", encoding="utf-8") as handle:
        config = json.load(handle)["config"]

    model = MotionMamba(config)
    model.to(model.device)
    model.load_safetensors(str(base), device=model.device, load_optimizer=False)
    model.eval()
    return model


class SmplxPoser:
    """Turns one 69-number pose into SMPL-X body vertices.

    Built once and reused every frame: creating the body model is slow, and
    a live loop cannot afford to redo it. ``batch_size=1`` because frames
    arrive one at a time.
    """

    def __init__(self, model_path: str, betas: Optional[np.ndarray] = None):
        import smplx

        name = Path(model_path).stem.upper()
        gender = next((g for g in ("NEUTRAL", "FEMALE", "MALE") if g in name), "NEUTRAL").lower()

        self.body = smplx.create(
            str(model_path), model_type="smplx", gender=gender,
            use_pca=False, flat_hand_mean=True, num_betas=10, ext="npz", batch_size=1,
        )
        self.faces = np.asarray(self.body.faces, dtype=np.int64)
        self._betas = None
        if betas is not None:
            self._betas = torch.tensor(np.asarray(betas, dtype=np.float32)[:10]).unsqueeze(0)

    @torch.no_grad()
    def vertices(self, pose: np.ndarray) -> np.ndarray:
        """Args: pose ``(69,)`` in real (de-normalized) units.
        Returns: ``(V, 3)`` body vertices in metres."""
        as_tensor = lambda a: torch.tensor(np.asarray(a, dtype=np.float32)).unsqueeze(0)
        output = self.body(
            transl=as_tensor(pose[TRANS]),
            global_orient=as_tensor(pose[GLOBAL_ORIENT]),
            body_pose=as_tensor(pose[BODY_POSE]),
            betas=self._betas,
            return_verts=True,
        )
        return output.vertices[0].numpy()


class LiveMotionSession:
    """A running character: rolling window in, next pose out.

    Args:
        model:      A trained MotionMamba (control-conditioned).
        seed_clip:  A MotionPrep Stage 2 ``.npz``, used for the opening
                    frames and for the normalization statistics.
        window:     How many past frames the model is shown each step.

    Raises:
        ValueError: If the checkpoint was trained without control (there
            would be nothing for the player to do), or if the seed clip is
            too short to fill the window.
    """

    def __init__(self, model: MotionMamba, seed_clip: str, window: int = 9):
        if model.config.get("control_dim", 0) <= 0:
            raise ValueError(
                "This checkpoint has control_dim=0, so the character cannot be "
                "steered. Train with control (main_motion_mamba_control.py) to "
                "drive it from the keyboard."
            )

        self.model = model
        self.window = int(window)
        self.pose_dim = model.config["state_dim"]
        self.control_dim = model.config["control_dim"]

        with np.load(seed_clip, allow_pickle=True) as data:
            pose = np.concatenate(
                [data["trans"], data["global_orient"], data["body_pose"]], axis=1
            ).astype(np.float64)
            self.betas = data["betas"] if "betas" in data.files else None
            self.fps = float(data["fps"])

        if pose.shape[0] < self.window:
            raise ValueError(
                f"{Path(seed_clip).name} has {pose.shape[0]} frames, too few to fill "
                f"a {self.window}-frame window."
            )
        if pose.shape[1] != self.pose_dim:
            raise ValueError(
                f"{Path(seed_clip).name} has {pose.shape[1]} pose numbers but the model "
                f"predicts {self.pose_dim}."
            )

        # The statistics the model's outputs are expressed in — see the
        # module docstring for why these come from the seed clip.
        self.mean = pose.mean(axis=0, keepdims=True)
        self.std = pose.std(axis=0, keepdims=True) + 1e-6
        normalized = (pose - self.mean) / self.std

        # The window starts as real motion, with zero control: the character
        # is simply continuing what the seed clip was doing until the player
        # presses something.
        opening = np.concatenate(
            [normalized[:self.window], np.zeros((self.window, self.control_dim))], axis=1
        )
        self._window = torch.tensor(opening, dtype=torch.float32, device=model.device)
        self.frame_count = 0

    @torch.no_grad()
    def step(self, control: np.ndarray) -> np.ndarray:
        """Advance one frame under the given control.

        Args:
            control: ``(control_dim,)`` — what the player is asking for now.

        Returns:
            ``(pose_dim,)`` the next pose, de-normalized into real units and
            ready for :class:`SmplxPoser`.
        """
        control_t = torch.tensor(
            np.asarray(control, dtype=np.float32), device=self.model.device
        )

        next_pose = self.model.predict_next(self._window)          # normalized

        # Slide the window forward: the new frame is the predicted pose
        # paired with the control that belongs to it (the same alignment
        # training used — see control_dataset.py).
        new_row = torch.cat([next_pose, control_t]).unsqueeze(0)
        self._window = torch.cat([self._window[1:], new_row], dim=0)
        self.frame_count += 1

        return next_pose.cpu().numpy() * self.std[0] + self.mean[0]

    @property
    def root_position(self) -> np.ndarray:
        """Where the character currently is in the world, ``(3,)`` metres."""
        latest = self._window[-1, :self.pose_dim].cpu().numpy()
        return (latest * self.std[0] + self.mean[0])[TRANS]
