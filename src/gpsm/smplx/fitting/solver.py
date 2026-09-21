"""Staged optimization (SMPLify-X style, coarse to fine).

See doc/SMPLX_Fitting_Methods.docx section 2. Fits a full SMPL-X sequence
(shared betas, per-frame pose/translation) to a 3D `Observation`, using
`umeyama_alignment` for initialization and a 4-stage annealed LBFGS
schedule for the pose fit itself.

Scope note (honest, not hidden): the built-in layouts (coco_17, openpose_25,
mediapipe_33, h36m_17) provide at most a few sparse fingertip landmarks per
hand — nowhere near enough independent constraints to identify a 45-dof
hand pose per hand. Rather than let 1-3 points silently overfit 45
parameters into noise, this solver leaves hand_pose at the neutral rest
pose and marks the hands unobserved in `part_mask`, exactly the same
"absent, not measured" convention used everywhere else in this package.
Layouts with genuine per-finger-joint correspondences can extend
`fit_observation` to add a hand stage; the staging machinery already
supports it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.gpsm.smplx.fitting.align import umeyama_alignment
from src.gpsm.smplx.fitting.body import BodyParams, SMPLXBody
from src.gpsm.smplx.fitting.losses import (
    robust_data_term,
    shape_regularization,
    temporal_smoothness,
    translation_acceleration,
)
from src.gpsm.smplx.fitting.priors import PosePrior, default_prior
from src.gpsm.smplx.layouts.correspondence import MotionLayout
from src.gpsm.smplx.schema import BodyPart, CanonicalMotion, FitQuality, NUM_BETAS, NUM_PARTS
from src.gpsm.smplx import ops as motion_ops

_TORSO_JOINTS = {
    "pelvis", "left_hip", "right_hip", "left_shoulder", "right_shoulder",
    "neck", "spine1", "spine2", "spine3",
}


@dataclass
class StageSpec:
    name: str
    optimize: List[str]           # subset of {"global","betas","body_pose"}
    iters: int
    w_data: float
    w_prior: float
    w_limits: float
    w_shape: float
    w_smooth: float = 0.0
    w_accel: float = 0.0
    torso_only: bool = False


def scale_temporal_weights_for_subsampling(stages: List[StageSpec], step: int) -> List[StageSpec]:
    """Return a copy of `stages` with temporal-smoothness/acceleration
    weights scaled down for evenly-subsampled input (see
    `app/pipeline.py::_fit_frame_selection` — a clip longer than the fit
    frame budget is sampled every `step` real frames, not truncated).

    `temporal_smoothness`/`translation_acceleration` (losses.py) compute
    plain frame-to-frame differences with no notion of how much real time
    separates two samples. When `step > 1`, "consecutive" samples are
    `step` real frames apart, so genuine motion between them is naturally
    `step` times larger per first-difference term (velocity-like) and
    roughly `step**2` times larger per second-difference term
    (acceleration-like, from naive discrete differencing). Left
    unscaled, the same fixed weight then over-penalizes real fast motion
    in a heavily-subsampled clip as if it were implausible jitter —
    confirmed directly: fitting a real gym squat video's full range of
    motion (subsampled 6x) measured *worse* (234.9mm RMSE) than fitting a
    narrower, less-subsampled window (43-50mm), even though the former
    used more, not less, of the source data. Scaling these two weights by
    1/step and 1/step**2 respectively restores a roughly constant
    per-unit-of-real-time penalty regardless of the subsampling rate.
    """
    if step <= 1:
        return list(stages)
    scaled = []
    for s in stages:
        scaled.append(StageSpec(
            name=s.name, optimize=s.optimize, iters=s.iters, w_data=s.w_data,
            w_prior=s.w_prior, w_limits=s.w_limits, w_shape=s.w_shape,
            w_smooth=s.w_smooth / step, w_accel=s.w_accel / (step ** 2),
            torso_only=s.torso_only,
        ))
    return scaled


DEFAULT_STAGES: List[StageSpec] = [
    StageSpec("global_only", ["global"], iters=50, w_data=1.0, w_prior=0.05, w_limits=0.0, w_shape=0.0, torso_only=True),
    StageSpec("shape", ["global", "betas"], iters=50, w_data=1.0, w_prior=0.3, w_limits=0.1, w_shape=1.0, torso_only=True),
    StageSpec("body_pose", ["global", "betas", "body_pose"], iters=120, w_data=2.0, w_prior=0.1, w_limits=0.3, w_shape=0.5),
    StageSpec("temporal_polish", ["global", "betas", "body_pose"], iters=60, w_data=1.5, w_prior=0.08, w_limits=0.3,
              w_shape=0.3, w_smooth=2.0, w_accel=0.5),
]
# Tuned against real C3D marker data (tests/test_markers_adapter.py) as well
# as synthetic recovery tests; roughly doubles iteration counts relative to
# an earlier, faster-but-less-accurate schedule, trading solve time for a
# meaningfully lower converged RMSE on noisy real-world input.


@dataclass
class SolverConfig:
    stages: List[StageSpec] = field(default_factory=lambda: list(DEFAULT_STAGES))
    sigma_data_m: float = 0.15
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    lbfgs_lr: float = 0.5


def _prepare_observation_np(observation) -> np.ndarray:
    pts = np.asarray(observation.points, dtype=np.float64)
    if observation.up_axis != "z":
        pts = motion_ops.convert_up_axis(pts, observation.up_axis, "z")
    if observation.units_to_meters != 1.0:
        pts = motion_ops.convert_units(pts, observation.units_to_meters)
    return pts


def gather_targets(out, indices_t: torch.Tensor, kinds: List[str]) -> torch.Tensor:
    """Predicted (T, K, 3) positions from a body-model output, gathered per
    correspondence: joint centers from `out.joints`, marker/surface points
    from `out.vertices` — a layout may freely mix both (see VICON_50)."""
    joint_mask = [k == "joint" for k in kinds]
    if all(joint_mask):
        return out.joints[:, indices_t, :]
    if not any(joint_mask):
        return out.vertices[:, indices_t, :]
    # Mixed layout: build per-source predictions and interleave.
    t = out.joints.shape[0]
    k = indices_t.shape[0]
    pred = torch.empty(t, k, 3, dtype=out.joints.dtype, device=out.joints.device)
    joint_pos = [i for i, is_j in enumerate(joint_mask) if is_j]
    vertex_pos = [i for i, is_j in enumerate(joint_mask) if not is_j]
    if joint_pos:
        pred[:, joint_pos, :] = out.joints[:, indices_t[joint_pos], :]
    if vertex_pos:
        pred[:, vertex_pos, :] = out.vertices[:, indices_t[vertex_pos], :]
    return pred


def _neutral_source_points(neutral_out, indices: List[int], kinds: List[str]) -> np.ndarray:
    """Same gathering as `gather_targets`, for the (unbatched) neutral pose,
    returned as a plain (K, 3) numpy array for the numpy-side alignment init."""
    joints = neutral_out.joints[0].cpu().numpy()
    out = np.empty((len(indices), 3), dtype=np.float64)
    has_verts = neutral_out.vertices is not None
    verts = neutral_out.vertices[0].cpu().numpy() if has_verts else None
    for i, (idx, kind) in enumerate(zip(indices, kinds)):
        out[i] = joints[idx] if kind == "joint" else verts[idx]
    return out


def _init_global_pose(neutral_points: np.ndarray, points: np.ndarray, conf: np.ndarray,
                       corr_weights: List[float]):
    """Two-pass per-frame Umeyama init: pass 1 estimates one global scale
    from the best-observed frame; pass 2 re-aligns every frame at that
    fixed scale to get per-frame rotation + translation.

    `neutral_points` (K, 3): the template position (joint center or mesh
    vertex, per-correspondence) in the body model's own neutral pose.
    """
    t = points.shape[0]
    src = neutral_points  # (K, 3)
    base_w = np.asarray(corr_weights, dtype=np.float64)

    total_conf_per_frame = (conf * base_w[None, :]).sum(axis=1)
    # Try frames in descending order of confidence until one yields a
    # non-degenerate scale estimate — a single bad/degenerate frame (e.g.
    # a tracker reporting a collapsed point) should not derail scale
    # estimation for the whole clip when better frames exist.
    scale = 1.0
    for best_frame in np.argsort(-total_conf_per_frame):
        w0 = conf[best_frame] * base_w
        if (w0 > 0).sum() < 3:
            continue
        try:
            tf0 = umeyama_alignment(src, points[best_frame], weights=w0)
            scale = tf0.s
            break
        except ValueError:
            continue  # degenerate frame (e.g. zero-variance target); try the next

    src_scaled = src * scale
    global_orient = np.zeros((t, 3))
    transl = np.zeros((t, 3))
    last_valid = None
    for frame in range(t):
        w = conf[frame] * base_w
        degenerate = False
        if (w > 0).sum() < 3:
            degenerate = True
        else:
            try:
                tf = umeyama_alignment(src_scaled, points[frame], weights=w)
            except ValueError:
                degenerate = True
        if degenerate:
            if last_valid is not None:
                global_orient[frame] = global_orient[last_valid]
                transl[frame] = transl[last_valid]
            continue
        global_orient[frame] = Rotation.from_matrix(tf.R).as_rotvec()
        transl[frame] = tf.t
        last_valid = frame

    return global_orient, transl, scale


def fit_observation(
    observation,
    body: SMPLXBody,
    layout: MotionLayout,
    prior: Optional[PosePrior] = None,
    config: Optional[SolverConfig] = None,
) -> "tuple[CanonicalMotion, FitQuality]":
    """Fit a 3D Observation to SMPL-X via staged optimization.

    Returns (CanonicalMotion, FitQuality). Raises ValueError if the
    observation is not 3D — 2D/video input goes through Method 3
    (src/smplx/regressor.py) first.
    """
    if not observation.is_3d:
        raise ValueError(
            "fit_observation requires 3D points; 2D observations must first "
            "go through the learned-regressor pipeline (see regressor.py)."
        )
    if layout.num_points != observation.num_points:
        raise ValueError(
            f"Layout {layout.name!r} has {layout.num_points} points but the "
            f"observation has {observation.num_points}"
        )

    cfg = config or SolverConfig()
    prior = prior or default_prior()
    device, dtype = cfg.device, cfg.dtype

    points_np = _prepare_observation_np(observation)
    conf_np = np.asarray(observation.conf, dtype=np.float64)
    t = points_np.shape[0]

    corr_indices = layout.target_indices()
    corr_kinds = layout.target_kinds()
    corr_weights = layout.weights()
    needs_verts = any(k == "vertex" for k in corr_kinds)
    torso_mask_np = np.array(
        [getattr(c, "smplx_joint", None) in _TORSO_JOINTS for c in layout.correspondences],
        dtype=np.float64,
    )
    if torso_mask_np.sum() == 0:
        torso_mask_np[:] = 1.0  # no torso joints in this layout; fall back to all points

    with torch.no_grad():
        neutral_out = body.forward(BodyParams.zeros(1, device=device, dtype=dtype), return_verts=needs_verts)
        neutral_points = _neutral_source_points(neutral_out, corr_indices, corr_kinds)

    init_orient, init_transl, _scale = _init_global_pose(
        neutral_points, points_np, conf_np, corr_weights
    )

    params = BodyParams(
        betas=torch.zeros(NUM_BETAS, device=device, dtype=dtype),
        global_orient=torch.tensor(init_orient, device=device, dtype=dtype),
        body_pose=torch.zeros(t, 63, device=device, dtype=dtype),
        left_hand_pose=torch.zeros(t, 45, device=device, dtype=dtype),
        right_hand_pose=torch.zeros(t, 45, device=device, dtype=dtype),
        transl=torch.tensor(init_transl, device=device, dtype=dtype),
    )

    target = torch.tensor(points_np, device=device, dtype=dtype)          # (T, K, 3)
    conf = torch.tensor(conf_np, device=device, dtype=dtype)              # (T, K)
    torso_mask = torch.tensor(torso_mask_np, device=device, dtype=dtype)  # (K,)
    corr_idx_t = torch.tensor(corr_indices, dtype=torch.long, device=device)
    struct_weight = torch.tensor(corr_weights, device=device, dtype=dtype)

    all_params = {
        "global": [params.global_orient, params.transl],
        "betas": [params.betas],
        "body_pose": [params.body_pose],
    }
    for plist in all_params.values():
        for p in plist:
            p.requires_grad_(False)

    for stage in cfg.stages:
        variables = [p for key in stage.optimize for p in all_params[key]]
        for p in variables:
            p.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            variables, lr=cfg.lbfgs_lr, max_iter=stage.iters,
            line_search_fn="strong_wolfe",
        )
        frame_weight = torso_mask if stage.torso_only else torch.ones_like(torso_mask)
        effective_conf = conf * (frame_weight * struct_weight)[None, :]

        def closure():
            optimizer.zero_grad()
            out = body.forward(params, return_verts=needs_verts)
            pred = gather_targets(out, corr_idx_t, corr_kinds)  # (T, K, 3)
            loss = stage.w_data * robust_data_term(pred, target, effective_conf, sigma=cfg.sigma_data_m)
            loss = loss + stage.w_prior * prior.nll(params.body_pose)
            loss = loss + stage.w_shape * shape_regularization(params.betas)
            if stage.w_smooth > 0:
                pose_seq = torch.cat(
                    [params.global_orient.unsqueeze(1), params.body_pose.reshape(t, 21, 3)], dim=1
                )
                loss = loss + stage.w_smooth * temporal_smoothness(pose_seq)
            if stage.w_accel > 0:
                loss = loss + stage.w_accel * translation_acceleration(params.transl)
            loss.backward()
            return loss

        optimizer.step(closure)

        for p in variables:
            p.requires_grad_(False)

    with torch.no_grad():
        final_out = body.forward(params, return_verts=needs_verts)
        pred_final = gather_targets(final_out, corr_idx_t, corr_kinds)
        residual = (pred_final - target).norm(dim=-1)  # (T, K) meters
        has_obs = conf_np > 0
        rmse_mm = (
            float(residual[torch.tensor(has_obs)].pow(2).mean().sqrt().item()) * 1000.0
            if has_obs.any() else float("nan")
        )
        observed_fraction = float(has_obs.mean())
        accel = translation_acceleration(params.transl).item()
        accel_spikes = int(accel > 4.0)  # >~2 m/s^2 average squared jerk proxy

    quality = FitQuality(
        joint_rmse_mm=rmse_mm,
        observed_fraction=observed_fraction,
        accel_spike_count=accel_spikes,
    ).finalize()

    part_mask = np.zeros((t, NUM_PARTS), dtype=np.float64)
    body_conf_per_frame = (conf_np * (1 - torso_mask_np[None, :]) * np.array(corr_weights)[None, :]).sum(axis=1) \
        + (conf_np * torso_mask_np[None, :] * np.array(corr_weights)[None, :]).sum(axis=1)
    part_mask[:, BodyPart.BODY] = (body_conf_per_frame > 0).astype(np.float64)
    part_mask[:, BodyPart.TRANS] = part_mask[:, BodyPart.BODY]
    # Hands and face are never solved by this tier-1 solver (see module
    # docstring) — always mark unobserved rather than implying a real fit.

    with torch.no_grad():
        betas_np = params.betas.cpu().numpy().copy()
        global_orient_np = params.global_orient.cpu().numpy().copy()
        body_pose_np = params.body_pose.cpu().numpy().copy()
        transl_np = params.transl.cpu().numpy().copy()

    motion = CanonicalMotion(
        trans=transl_np,
        global_orient=global_orient_np,
        body_pose=body_pose_np,
        left_hand_pose=np.zeros((t, 45)),
        right_hand_pose=np.zeros((t, 45)),
        betas=betas_np,
        part_mask=part_mask,
        fps=observation.fps,
        meta={"source_format": f"fitted:{layout.name}", "solver": "staged_lbfgs"},
        quality=quality,
    )
    return motion, quality
