"""
Simulate a mocap .npz sequence through the real SMPL-X body model
(https://github.com/vchoutas/smplx), instead of just plotting raw pose curves
(see visualize_npz.py for that lighter-weight, model-free option).

This runs actual forward kinematics: SMPL-X pose parameters -> posed 3D body
joints (and optionally the full mesh), which is what "simulating" the data
through the body model actually means (raw `poses` values are just axis-angle
rotations — you can't tell what a body looks like from them without a model).

--------------------------------------------------------------------------
IMPORTANT — model weights are NOT downloadable by this script.
--------------------------------------------------------------------------
The vchoutas/smplx GitHub repo only ships the *code*. The SMPL-X model
weights are gated behind a license you must accept by hand:

    1. Register + accept the license at https://smpl-x.is.tue.mpg.de
    2. Download the model archive from the same site (choose the .npz
       models — this script uses `ext="npz"` — .pkl also works if you flip
       --ext).
    3. Unzip so you end up with this layout:

           <model-folder>/
               smplx/
                   SMPLX_NEUTRAL.npz
                   SMPLX_MALE.npz
                   SMPLX_FEMALE.npz

    4. Pass that folder as --model-folder below.

Also requires the `smplx` pip package: `pip install smplx`

Usage:
    python -m src.gpsm.tests.simulate_smplx data/12_L_2_stageii.npz \
        --model-folder /path/to/model-folder \
        --out src/gpsm/tests/output --fps 24

    # also render the full body mesh (slower — keep --max-frames small)
    python -m src.gpsm.tests.simulate_smplx data/12_L_2_stageii.npz \
        --model-folder /path/to/model-folder --mesh --max-frames 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless-safe (Kaggle/CI) — no GUI backend required
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.gpsm.tests.visualize_npz import animate_joints, load_npz, summarize


# ---------------------------------------------------------------------------
# npz -> SMPL-X forward() kwargs
# ---------------------------------------------------------------------------

# AMASS/SMPL-X "stageii" npz key -> smplx forward() argument name.
# Split fields (pose_hand -> left/right, pose_eye -> leye/reye) are handled
# separately below since one npz array maps to two forward() args.
DIRECT_KEY_MAP = {
    "root_orient": "global_orient",
    "pose_body":   "body_pose",
    "pose_jaw":    "jaw_pose",
    "trans":       "transl",
}

# Fallback slice offsets into the combined `poses` (T, 165) array, following
# the standard AMASS SMPL-X layout: [root_orient(3), pose_body(63),
# pose_jaw(3), pose_eye(6), pose_hand(90)]. Only used when the individual
# fields above are missing from the npz.
POSES_FALLBACK_SLICES = {
    "global_orient": slice(0, 3),
    "body_pose":      slice(3, 66),
    "jaw_pose":       slice(66, 69),
    "pose_eye":       slice(69, 75),   # further split into leye/reye below
    "pose_hand":      slice(75, 165),  # further split into left/right below
}


def build_forward_kwargs(npz: dict, num_frames: int, num_betas: int, device: str) -> dict:
    """Map whatever pose fields are present in the npz onto SMPL-X forward() kwargs."""
    kwargs: dict = {}

    def as_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr, dtype=np.float32), device=device)

    # `trans` deliberately excluded from this check — a dict with only
    # `poses` + `trans` (e.g. a model rollout prediction) must still take
    # the `poses`-slicing fallback below, not be misread as "direct fields
    # present" just because `trans` happens to also be a DIRECT_KEY_MAP key.
    pose_direct_keys = ("root_orient", "pose_body", "pose_jaw")
    have_direct = any(k in npz for k in pose_direct_keys) or "pose_hand" in npz or "pose_eye" in npz

    if have_direct:
        for npz_key, arg_name in DIRECT_KEY_MAP.items():
            if npz_key in npz:
                kwargs[arg_name] = as_tensor(npz[npz_key])
        if "pose_hand" in npz:
            pose_hand = np.asarray(npz["pose_hand"], dtype=np.float32)
            if pose_hand.shape[-1] == 90:
                kwargs["left_hand_pose"]  = as_tensor(pose_hand[:, :45])
                kwargs["right_hand_pose"] = as_tensor(pose_hand[:, 45:])
        if "pose_eye" in npz:
            pose_eye = np.asarray(npz["pose_eye"], dtype=np.float32)
            if pose_eye.shape[-1] == 6:
                kwargs["leye_pose"] = as_tensor(pose_eye[:, :3])
                kwargs["reye_pose"] = as_tensor(pose_eye[:, 3:])
    elif "poses" in npz:
        poses = np.asarray(npz["poses"], dtype=np.float32)
        kwargs["global_orient"] = as_tensor(poses[:, POSES_FALLBACK_SLICES["global_orient"]])
        kwargs["body_pose"]     = as_tensor(poses[:, POSES_FALLBACK_SLICES["body_pose"]])
        if poses.shape[-1] >= 165:
            kwargs["jaw_pose"] = as_tensor(poses[:, POSES_FALLBACK_SLICES["jaw_pose"]])
            eye = poses[:, POSES_FALLBACK_SLICES["pose_eye"]]
            kwargs["leye_pose"], kwargs["reye_pose"] = as_tensor(eye[:, :3]), as_tensor(eye[:, 3:])
            hand = poses[:, POSES_FALLBACK_SLICES["pose_hand"]]
            kwargs["left_hand_pose"], kwargs["right_hand_pose"] = as_tensor(hand[:, :45]), as_tensor(hand[:, 45:])
        if "trans" in npz:
            kwargs["transl"] = as_tensor(npz["trans"])
    else:
        raise KeyError(
            "Could not find pose data in this .npz — expected either "
            "'root_orient'/'pose_body'/... fields or a combined 'poses' array."
        )

    if "betas" in npz:
        betas = np.asarray(npz["betas"], dtype=np.float32).reshape(-1)
        fitted = np.zeros(num_betas, dtype=np.float32)
        n = min(num_betas, betas.shape[0])
        fitted[:n] = betas[:n]
        kwargs["betas"] = as_tensor(np.tile(fitted[None, :], (num_frames, 1)))

    return kwargs


def subsample_indices(total_frames: int, max_frames: int, stride: Optional[int]) -> np.ndarray:
    if stride is not None and stride > 0:
        return np.arange(0, total_frames, stride)
    if total_frames <= max_frames:
        return np.arange(total_frames)
    return np.linspace(0, total_frames - 1, num=max_frames).round().astype(int)


def resolve_gender(npz: dict, override: Optional[str]) -> str:
    if override:
        return override
    if "gender" in npz:
        g = str(np.asarray(npz["gender"]).item()).lower()
        if g in ("male", "female", "neutral"):
            return g
    return "neutral"


# ---------------------------------------------------------------------------
# Mesh animation (optional — heavier than the joints-only animation)
# ---------------------------------------------------------------------------

def animate_mesh(vertices: np.ndarray, faces: np.ndarray, out_path: Path,
                  fps: int = 15, title: str = "") -> None:
    """vertices: (T, V, 3), faces: (F, 3) — animate the posed body surface."""
    T = vertices.shape[0]
    lo = vertices.reshape(-1, 3).min(axis=0)
    hi = vertices.reshape(-1, 3).max(axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2, 1e-3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(vertices[0][faces], facecolor="lightsteelblue",
                             edgecolor="none", alpha=0.95)
    ax.add_collection3d(mesh)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title)
    ax.set_axis_off()

    def update(frame_idx):
        mesh.set_verts(vertices[frame_idx][faces])
        ax.set_xlabel(f"frame {frame_idx}/{T - 1}")
        return (mesh,)

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved mesh animation    -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def simulate(
    npz_path: str,
    model_folder: str,
    model_type: str = "smplx",
    ext: str = "npz",
    gender: Optional[str] = None,
    num_betas: Optional[int] = None,
    use_pca: bool = False,
    flat_hand_mean: bool = True,
    max_frames: int = 150,
    stride: Optional[int] = None,
    fps: int = 24,
    out_dir: str = "src/gpsm/tests/output",
    device: str = "cpu",
    with_mesh: bool = False,
    keep_root_motion: bool = False,
) -> None:
    try:
        import smplx
    except ImportError as e:
        raise ImportError(
            "The `smplx` package is not installed. Run: pip install smplx"
        ) from e

    npz_path_p = Path(npz_path)
    out_dir_p  = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    npz = load_npz(npz_path_p)
    summarize(npz, npz_path_p)

    resolved_gender = resolve_gender(npz, gender)
    resolved_num_betas = num_betas or (
        int(np.asarray(npz["betas"]).reshape(-1).shape[0]) if "betas" in npz else 10
    )

    # --model-folder may be either the root layout smplx.create() expects
    # (<folder>/<model_type>/<MODEL_TYPE>_<GENDER>.<ext>), or a direct path to
    # one specific model file (e.g. .../SMPLX_FEMALE.npz) — smplx.create()
    # supports both, so mirror that here instead of forcing the folder layout.
    # Resolved before the summary print below so the logged gender always
    # matches the model actually loaded.
    model_folder_p = Path(model_folder)
    if model_folder_p.is_dir():
        resolved_model_path = model_folder_p / model_type / f"{model_type.upper()}_{resolved_gender.upper()}.{ext}"
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"Expected SMPL-X model file not found: {resolved_model_path}\n"
                "Download it (after registering) from https://smpl-x.is.tue.mpg.de "
                f"and place it at that path (model-folder/{model_type}/...), "
                "or pass --model-folder pointing directly at a specific model file."
            )
    elif model_folder_p.is_file():
        resolved_model_path = model_folder_p
        # gender is only used for file *selection*; when a direct file is
        # given, relabel from its filename instead (cosmetic — for titles).
        # check FEMALE before MALE — "MALE" is a substring of "FEMALE"
        name = model_folder_p.stem.upper()
        for g in ("NEUTRAL", "FEMALE", "MALE"):
            if g in name:
                resolved_gender = g.lower()
                break
    else:
        raise FileNotFoundError(f"--model-folder does not exist: {model_folder_p}")

    total_frames = int(np.asarray(npz.get("trans", npz.get("poses"))).shape[0])
    idx = subsample_indices(total_frames, max_frames, stride)
    print(f"Simulating {len(idx)}/{total_frames} frames "
          f"(gender={resolved_gender}, num_betas={resolved_num_betas})")

    npz_sub = {k: (v[idx] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == total_frames else v)
               for k, v in npz.items()}

    fwd_kwargs = build_forward_kwargs(npz_sub, num_frames=len(idx),
                                       num_betas=resolved_num_betas, device=device)

    body_model = smplx.create(
        str(resolved_model_path),
        model_type=model_type,
        gender=resolved_gender,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        num_betas=resolved_num_betas,
        ext=ext,
        batch_size=len(idx),
    ).to(device)

    with torch.no_grad():
        output = body_model(return_verts=with_mesh, **fwd_kwargs)

    joints = output.joints.detach().cpu().numpy()
    stem = npz_path_p.stem

    if not keep_root_motion:
        # Root (pelvis) translation drift across a sequence is often many
        # metres (e.g. walking across the capture volume) while the body
        # itself is ~1.6m tall — animating with fixed axes sized to the
        # global bounding box then squeezes the body into a tiny corner of
        # the frame. Recenter every frame on the pelvis so the "camera"
        # follows the character instead, which is what you actually want
        # for eyeballing the pose/motion.
        root = joints[:, [0], :]
        joints = joints - root

    animate_joints(joints, out_dir_p / f"{stem}_smplx_joints.gif", fps=fps,
                    title=f"{stem} — SMPL-X joints ({resolved_gender})")

    if with_mesh:
        vertices = output.vertices.detach().cpu().numpy()
        if not keep_root_motion:
            vertices = vertices - root
        faces = body_model.faces
        animate_mesh(vertices, faces, out_dir_p / f"{stem}_smplx_mesh.gif", fps=fps,
                     title=f"{stem} — SMPL-X mesh ({resolved_gender})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a mocap .npz sequence through the real SMPL-X body model and animate the result."
    )
    parser.add_argument("npz_path", help="Path to a single motion .npz file")
    parser.add_argument("--model-folder", required=True,
                         help="Folder containing SMPL-X model files (see module docstring for how to obtain these)")
    parser.add_argument("--model-type", default="smplx", choices=["smplx"])
    parser.add_argument("--ext", default="npz", choices=["npz", "pkl"])
    parser.add_argument("--gender", default=None, choices=[None, "neutral", "male", "female"])
    parser.add_argument("--num-betas", type=int, default=None)
    parser.add_argument("--use-pca", action="store_true",
                         help="Use PCA-reduced hand pose instead of full axis-angle (AMASS data is full, so leave off)")
    parser.add_argument("--no-flat-hand-mean", dest="flat_hand_mean", action="store_false")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument("--stride", type=int, default=None, help="Explicit frame stride (overrides --max-frames)")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--out", default="src/gpsm/tests/output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mesh", action="store_true",
                         help="Also render the full body mesh (slower — use a small --max-frames)")
    parser.add_argument("--keep-root-motion", action="store_true",
                         help="Don't recenter on the pelvis each frame — shows true world-space translation, "
                              "but the body will look tiny if the sequence covers much distance")
    args = parser.parse_args()

    simulate(
        args.npz_path,
        model_folder=args.model_folder,
        model_type=args.model_type,
        ext=args.ext,
        gender=args.gender,
        num_betas=args.num_betas,
        use_pca=args.use_pca,
        flat_hand_mean=args.flat_hand_mean,
        max_frames=args.max_frames,
        stride=args.stride,
        fps=args.fps,
        out_dir=args.out,
        device=args.device,
        with_mesh=args.mesh,
        keep_root_motion=args.keep_root_motion,
    )


if __name__ == "__main__":
    main()
