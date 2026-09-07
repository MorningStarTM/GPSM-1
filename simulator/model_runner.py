"""Turns a trained StateMachineGPT checkpoint (or real recorded mocap) into a
posed SMPL-X vertex sequence ready for rendering.

Deliberately reuses the exact rollout + denormalization + SMPL-X
forward-kinematics logic already implemented and used to produce this
project's GIF demos, instead of reimplementing it:
    - src/gpsm/tests/rollout_inference.rollout_from_first_frame  (model rollout)
    - src/gpsm/tests/simulate_smplx.build_forward_kwargs         (npz -> SMPL-X forward() kwargs)
    - src/gpsm/tests/simulate_smplx.resolve_gender

This module only adds the "give me plain numpy vertices/joints/faces, ready
to hand to a renderer" layer on top, instead of writing out a GIF.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.gpsm.tests.rollout_inference import rollout_from_first_frame
from src.gpsm.tests.simulate_smplx import build_forward_kwargs, resolve_gender
from src.gpsm.tests.visualize_npz import load_npz


class MotionSequence:
    """A posed SMPL-X sequence ready for rendering."""

    def __init__(self, vertices: np.ndarray, joints: np.ndarray, faces: np.ndarray, fps: int):
        if vertices.ndim != 3 or vertices.shape[2] != 3:
            raise ValueError(f"vertices must be (T, V, 3), got {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError("MotionSequence: vertices contain non-finite (NaN/Inf) values.")
        self.vertices = vertices  # (T, V, 3)
        self.joints = joints      # (T, J, 3)
        self.faces = faces        # (F, 3)
        self.fps = fps

    @property
    def n_frames(self) -> int:
        return self.vertices.shape[0]


def _resolve_model_path(model_folder: str, resolved_gender: str) -> "tuple[Path, str]":
    """Mirrors src/gpsm/tests/simulate_smplx.simulate()'s path/gender resolution
    exactly: a direct model FILE's filename is the source of truth for gender
    (overrides whatever the npz metadata said), since that's the body actually
    being loaded; a model FOLDER instead uses the already-resolved gender to
    pick which file inside it to load."""
    model_folder_p = Path(model_folder)
    if model_folder_p.is_dir():
        resolved_model_path = model_folder_p / "smplx" / f"SMPLX_{resolved_gender.upper()}.npz"
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"Expected SMPL-X model file not found: {resolved_model_path}")
        return resolved_model_path, resolved_gender
    if model_folder_p.is_file():
        name = model_folder_p.stem.upper()
        for g in ("NEUTRAL", "FEMALE", "MALE"):  # check FEMALE before MALE ("MALE" is a substring of "FEMALE")
            if g in name:
                resolved_gender = g.lower()
                break
        return model_folder_p, resolved_gender
    raise FileNotFoundError(f"--model-folder does not exist: {model_folder_p}")


def _run_smplx(fake_npz: dict, num_frames: int, model_folder: str, gender: Optional[str],
                num_betas: Optional[int], use_pca: bool, flat_hand_mean: bool,
                keep_root_motion: bool, fps: int) -> MotionSequence:
    import smplx

    resolved_gender = resolve_gender(fake_npz, gender)
    resolved_model_path, resolved_gender = _resolve_model_path(model_folder, resolved_gender)
    resolved_num_betas = num_betas or (
        int(np.asarray(fake_npz["betas"]).reshape(-1).shape[0]) if "betas" in fake_npz else 10
    )

    fwd_kwargs = build_forward_kwargs(fake_npz, num_frames=num_frames,
                                       num_betas=resolved_num_betas, device="cpu")

    body_model = smplx.create(
        str(resolved_model_path),
        model_type="smplx",
        gender=resolved_gender,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        num_betas=resolved_num_betas,
        ext="npz",
        batch_size=num_frames,
    )

    with torch.no_grad():
        output = body_model(return_verts=True, **fwd_kwargs)

    vertices = output.vertices.detach().cpu().numpy()
    joints = output.joints.detach().cpu().numpy()
    faces = np.asarray(body_model.faces, dtype=np.int64)

    if not keep_root_motion:
        root = joints[:, [0], :]
        vertices = vertices - root
        joints = joints - root

    return MotionSequence(vertices=vertices, joints=joints, faces=faces, fps=fps)


def generate_rollout_sequence(
    npz_path: str,
    checkpoint_path: str,
    model_folder: str,
    n_steps: int = 60,
    device: Optional[str] = None,
    gender: Optional[str] = None,
    num_betas: Optional[int] = None,
    use_pca: bool = False,
    flat_hand_mean: bool = True,
    keep_root_motion: bool = False,
    fps: int = 24,
) -> MotionSequence:
    """Seed the trained model with the first real frame of `npz_path`, roll
    out `n_steps` autoregressive predictions, and pass every frame (seed +
    predictions) through the real SMPL-X body model to get posed vertices."""
    result = rollout_from_first_frame(npz_path, checkpoint_path, n_steps=n_steps, device=device)
    sequence = result["sequence_denorm"]  # (n_steps+1, D) = poses + trans

    if sequence.shape[1] < 3:
        raise ValueError(
            f"Expected poses+trans features (>=3 dims, trans alone is 3-dim), got D={sequence.shape[1]}."
        )

    poses = sequence[:, :-3]
    trans = sequence[:, -3:]

    npz_orig = load_npz(Path(npz_path))
    fake_npz = {"poses": poses, "trans": trans}
    if "betas" in npz_orig:
        fake_npz["betas"] = npz_orig["betas"]
    if "gender" in npz_orig:
        fake_npz["gender"] = npz_orig["gender"]

    return _run_smplx(
        fake_npz, num_frames=sequence.shape[0], model_folder=model_folder, gender=gender,
        num_betas=num_betas, use_pca=use_pca, flat_hand_mean=flat_hand_mean,
        keep_root_motion=keep_root_motion, fps=fps,
    )


def generate_ground_truth_sequence(
    npz_path: str,
    model_folder: str,
    n_frames: int = 60,
    start: int = 0,
    gender: Optional[str] = None,
    num_betas: Optional[int] = None,
    use_pca: bool = False,
    flat_hand_mean: bool = True,
    keep_root_motion: bool = False,
    fps: int = 24,
) -> MotionSequence:
    """Same output shape as generate_rollout_sequence, but plays back the REAL
    recorded frames from npz_path (no model involved) — useful as a
    ground-truth comparison in the viewer."""
    npz_orig = load_npz(Path(npz_path))
    total = int(np.asarray(npz_orig.get("trans", npz_orig.get("poses"))).shape[0])
    if start < 0 or start >= total:
        raise ValueError(f"--start={start} out of range for a {total}-frame sequence.")
    end = min(start + n_frames, total)

    npz_sub = {
        k: (v[start:end] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == total else v)
        for k, v in npz_orig.items()
    }

    return _run_smplx(
        npz_sub, num_frames=end - start, model_folder=model_folder, gender=gender,
        num_betas=num_betas, use_pca=use_pca, flat_hand_mean=flat_hand_mean,
        keep_root_motion=keep_root_motion, fps=fps,
    )
