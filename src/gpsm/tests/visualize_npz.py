"""
Quick visual sanity-check for a single SMPL-X / AMASS-style .npz motion file.

Loads whatever a given .npz contains and "simulates" (animates/plots) its data
points, without depending on an SMPL-X body model:

  - per-frame joint positions (T, J, 3)  -> animated 3D point-cloud (gif)
  - root translation          (T, 3)     -> animated moving point tracing the
                                             root motion path through space (gif)
  - pose parameters           (T, D)     -> heatmap of values over time (png)

Only the arrays that are actually present in the file are visualized.

Usage:
    python -m src.gpsm.tests.visualize_npz data/12_L_2_stageii.npz
    python -m src.gpsm.tests.visualize_npz data/12_L_2_stageii.npz --out out_dir --fps 24
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe (Kaggle/CI) — no GUI backend required
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)


JOINT_KEYS = ("joints3d", "joints_3d", "joints", "Jtr", "joints_xyz")
POSE_KEYS  = ("poses", "pose_body", "pose")
TRANS_KEYS = ("trans", "translations", "root_trans")


def load_npz(path: Path) -> dict:
    with np.load(str(path), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def summarize(npz: dict, path: Path) -> None:
    print(f"\n{path.name}")
    print("-" * len(path.name))
    for k, v in npz.items():
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", type(v))
        print(f"  {k:20s} shape={shape} dtype={dtype}")


def _first_key(npz: dict, candidates) -> Optional[str]:
    for k in candidates:
        if k in npz:
            return k
    return None


def animate_joints(joints: np.ndarray, out_path: Path, fps: int = 30, title: str = "") -> None:
    """joints: (T, J, 3) — animate the full point cloud frame by frame."""
    T, J, _ = joints.shape
    lo = joints.reshape(-1, 3).min(axis=0)
    hi = joints.reshape(-1, 3).max(axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2, 1e-3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter([], [], [], s=15, c="tab:blue")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title)

    def update(frame_idx):
        pts = joints[frame_idx]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_xlabel(f"frame {frame_idx}/{T - 1}")
        return (scat,)

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved joint animation   -> {out_path}")


def animate_trans(trans: np.ndarray, out_path: Path, fps: int = 30, title: str = "") -> None:
    """trans: (T, 3) — animate a moving point tracing the root translation path."""
    T = trans.shape[0]
    lo, hi = trans.min(axis=0), trans.max(axis=0)
    center = (lo + hi) / 2
    radius = max(float((hi - lo).max()) / 2, 1e-3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(trans[:, 0], trans[:, 1], trans[:, 2], color="lightgray", lw=1, label="full path")
    point = ax.scatter([], [], [], s=40, c="tab:red", label="current frame")
    trail, = ax.plot([], [], [], color="tab:red", lw=1.5, alpha=0.6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(title)
    ax.legend(loc="upper left")

    def update(frame_idx):
        p = trans[frame_idx]
        point._offsets3d = ([p[0]], [p[1]], [p[2]])
        trail.set_data(trans[: frame_idx + 1, 0], trans[: frame_idx + 1, 1])
        trail.set_3d_properties(trans[: frame_idx + 1, 2])
        return point, trail

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved trans animation   -> {out_path}")


def plot_pose_curves(pose: np.ndarray, out_path: Path, title: str = "", max_dims: int = 12) -> None:
    """pose: (T, D) — heatmap of pose values over time, one row per dimension."""
    dims = min(pose.shape[1], max_dims)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pose[:, :dims].T, aspect="auto", cmap="coolwarm")
    ax.set_xlabel("frame")
    ax.set_ylabel("pose dim")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved pose curve plot   -> {out_path}")


def visualize(npz_path: str, out_dir: str = "src/gpsm/tests/output", fps: int = 30) -> None:
    npz_path = Path(npz_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = load_npz(npz_path)
    summarize(npz, npz_path)

    stem      = npz_path.stem
    made_any  = False

    joint_key = _first_key(npz, JOINT_KEYS)
    if joint_key is not None:
        joints = np.asarray(npz[joint_key])
        if joints.ndim == 3 and joints.shape[-1] == 3:
            animate_joints(joints, out_dir / f"{stem}_joints.gif", fps=fps,
                            title=f"{stem} — {joint_key}")
            made_any = True

    trans_key = _first_key(npz, TRANS_KEYS)
    if trans_key is not None:
        trans = np.asarray(npz[trans_key])
        if trans.ndim == 2 and trans.shape[-1] == 3:
            animate_trans(trans, out_dir / f"{stem}_trans.gif", fps=fps,
                          title=f"{stem} — {trans_key} (root path)")
            made_any = True

    pose_key = _first_key(npz, POSE_KEYS)
    if pose_key is not None:
        pose = np.asarray(npz[pose_key])
        if pose.ndim == 2:
            plot_pose_curves(pose, out_dir / f"{stem}_pose.png",
                              title=f"{stem} — {pose_key}")
            made_any = True

    if not made_any:
        print("No recognized joints/trans/pose arrays found in this .npz — nothing to visualize.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate/visualize the data points inside a SMPL-X/AMASS .npz motion file."
    )
    parser.add_argument("npz_path", help="Path to a single .npz motion file")
    parser.add_argument("--out", default="src/gpsm/tests/output", help="Output directory for plots/animations")
    parser.add_argument("--fps", type=int, default=30, help="Playback fps for animations")
    args = parser.parse_args()
    visualize(args.npz_path, out_dir=args.out, fps=args.fps)


if __name__ == "__main__":
    main()
