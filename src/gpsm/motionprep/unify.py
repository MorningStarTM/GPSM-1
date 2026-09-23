"""MotionPrep Stage 1: any mocap file -> one canonical pose format.

Takes a folder of mixed mocap (``.npz`` parameters, ``.c3d`` markers,
``.bvh`` skeletons, at whatever frame rate each was recorded) and writes one
``.npz`` per clip, all in the same shape: a 159-number SMPL-X pose per
frame, every clip at the same frame rate, each with a quality verdict.

Two very different paths, depending on the source (see ``formats.py``):

* **Parameter files** already describe the body the SMPL-X way, so they are
  converted directly. Fast (well under a second per clip) and exact.
* **Point files** (markers/skeletons) have no body model behind them, so one
  has to be *fitted* to the points by optimization. This is slow — measured
  at roughly 1.3 s per frame on CPU for a 30-marker C3D clip — which is why
  such files are resampled down to the target frame rate *before* fitting
  rather than after. Fitting a 120 fps clip and then throwing away 3 of
  every 4 fitted frames would cost four times as much for the same result.

Run it:
    python -m src.gpsm.motionprep.unify data --out processed/unified
    python -m src.gpsm.motionprep.unify data --out processed/unified --fps 30 --limit 3
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.gpsm.motionprep.formats import (
    BVH,
    C3D,
    NEEDS_FITTING,
    NPZ_PARAMS,
    UnknownFormatError,
    detect_source_format,
)
from src.gpsm.motionprep.manifest import ManifestWriter, already_processed
from src.gpsm.smplx.adapters.params import ingest_params
from src.gpsm.smplx.ops import resample_pose_sequence
from src.gpsm.smplx.quality import score_canonical_motion
from src.gpsm.smplx.schema import CanonicalMotion, Observation

DEFAULT_TARGET_FPS = 30.0

#: Verdicts worth keeping. REJECT clips are recorded in the manifest with
#: their score, but not written out — see the module docstring of
#: ``src/gpsm/smplx/quality.py`` for what drives the verdict.
KEEP_VERDICTS = ("GOOD", "DEGRADED")

MANIFEST_COLUMNS = [
    "source_path", "source_format", "status", "reason",
    "n_frames_raw", "fps_raw", "n_frames_out", "fps_out",
    "verdict", "joint_rmse_mm", "observed_fraction", "seconds", "output_path",
]


# ---------------------------------------------------------------------------
# Resampling raw points (the fitted path)
# ---------------------------------------------------------------------------

def resample_points(
    points: np.ndarray, valid: np.ndarray, src_fps: float, dst_fps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a (T, K, 3) point cloud to a new frame rate.

    Straight linear interpolation over time, per point per axis. That is the
    right choice here because these are *positions* — unlike joint rotations,
    which need SLERP (see ``ops.resample_pose_sequence``).

    A resampled sample is marked valid only if both of the original samples
    it was interpolated between were valid, so an occluded marker never gets
    laundered into a confident-looking measurement.

    Args:
        points:  ``(T, K, 3)`` positions.
        valid:   ``(T, K)`` bool, False where that point was not observed.
        src_fps: Frame rate of the input.
        dst_fps: Desired frame rate.

    Returns:
        ``(points, valid)`` at the new rate.
    """
    n_src = points.shape[0]
    if n_src < 2 or abs(src_fps - dst_fps) < 1e-9:
        return points, valid

    duration = (n_src - 1) / src_fps
    n_dst = max(1, int(round(duration * dst_fps)) + 1)
    src_times = np.arange(n_src) / src_fps
    dst_times = np.minimum(np.arange(n_dst) / dst_fps, src_times[-1])

    out = np.empty((n_dst, points.shape[1], 3), dtype=np.float64)
    for k in range(points.shape[1]):
        for axis in range(3):
            out[:, k, axis] = np.interp(dst_times, src_times, points[:, k, axis])

    # Validity: both bracketing source samples must be valid. Where a target
    # time lands exactly on a source frame, that one frame's validity is
    # what counts (no interpolation actually happened there).
    right = np.searchsorted(src_times, dst_times, side="left").clip(1, n_src - 1)
    left = right - 1
    out_valid = valid[left] & valid[right]                     # (n_dst, K)
    landed_on_source_frame = np.isclose(dst_times, src_times[left])[:, None]  # (n_dst, 1)
    out_valid = np.where(landed_on_source_frame, valid[left], out_valid)
    return out, out_valid


# ---------------------------------------------------------------------------
# The two ingest paths
# ---------------------------------------------------------------------------

def unify_params(path: str, target_fps: float) -> CanonicalMotion:
    """Convert a parameter ``.npz`` (SMPL / SMPL-H / SMPL-X) to canonical form.

    No fitting involved — the conversion is exact, so the only work is
    slicing the pose apart and resampling it to the target frame rate.
    """
    with np.load(path, allow_pickle=True) as raw:
        data = {key: raw[key] for key in raw.files}

    motion = ingest_params(data, source_name=Path(path).name)
    return resample_motion(motion, target_fps)


def unify_points(path: str, source_format: str, target_fps: float, body) -> CanonicalMotion:
    """Fit a body to a point-based file (``.c3d`` markers or ``.bvh`` joints).

    The points are resampled to ``target_fps`` *first*, then fitted — see the
    module docstring for why that ordering matters so much for runtime.

    Args:
        path:          The mocap file.
        source_format: ``C3D`` or ``BVH``.
        target_fps:    Frame rate to fit at.
        body:          A loaded ``SMPLXBody`` (expensive to construct, so it
                       is created once by the caller and reused).

    Raises:
        ValueError: If too few of the file's points match a known layout to
            fit safely — raised by the layout matchers, and worth surfacing
            as-is since the message names the labels it did find.
    """
    if source_format == C3D:
        from src.gpsm.smplx.adapters.markers import build_matched_layout, load_c3d

        capture = load_c3d(path)
        layout, columns = build_matched_layout(capture.labels)
        points, valid = capture.points_m[:, columns, :], capture.valid[:, columns]
        source_fps, up_axis = capture.fps, capture.up_axis
    else:
        from src.gpsm.smplx.adapters.bvh import load_bvh, match_bvh_layout

        capture = load_bvh(path)
        layout, columns = match_bvh_layout(capture.joint_names)
        points = capture.points_m[:, columns, :]
        # A BVH file is a solved skeleton: every joint has a position at
        # every frame, so there is nothing to mark unobserved.
        valid = np.ones(points.shape[:2], dtype=bool)
        source_fps, up_axis = capture.fps, capture.up_axis

    points, valid = resample_points(points, valid, source_fps, target_fps)

    from src.gpsm.smplx.fitting.solver import fit_observation

    observation = Observation(
        points=points,
        # `conf` is how the solver is told which samples are real. Occluded
        # markers sit at (0,0,0) in the file; passing them through as if
        # they were measurements pulls the fit badly (measured on a real
        # clip: 639mm joint error versus 66mm once masked out).
        conf=valid.astype(np.float64),
        joint_names=layout.point_names,
        space="world3d",
        fps=target_fps,
        up_axis=up_axis,
    )
    motion, quality = fit_observation(observation, body, layout)
    motion.quality = quality
    motion.meta.setdefault("source_format", source_format)
    motion.meta.setdefault("source_name", Path(path).name)
    motion.meta["matched_points"] = f"{layout.num_points}/{len(capture.labels) if source_format == C3D else len(capture.joint_names)}"
    return motion


def resample_motion(motion: CanonicalMotion, target_fps: float) -> CanonicalMotion:
    """Resample an already-canonical motion to ``target_fps``.

    Rotations are interpolated with SLERP rather than linearly (see
    ``ops.resample_pose_sequence``) — averaging two rotations component-wise
    does not give the rotation halfway between them.
    """
    if abs(motion.fps - target_fps) < 1e-9:
        return motion

    resampled = resample_pose_sequence(motion.to_array(), motion.fps, target_fps)
    n_out = resampled.shape[0]

    # part_mask is a per-frame flag, not a continuous signal, so it is
    # carried over by nearest source frame rather than interpolated.
    src_index = np.minimum(
        (np.arange(n_out) * motion.fps / target_fps).round().astype(int),
        motion.part_mask.shape[0] - 1,
    )
    return CanonicalMotion.from_array(
        resampled,
        betas=motion.betas,
        part_mask=motion.part_mask[src_index],
        fps=target_fps,
        meta={**motion.meta, "resampled_from_fps": motion.fps},
        quality=motion.quality,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def output_name(source_path: Path, input_root: Path) -> str:
    """A flat, unique output filename for a source file.

    Input folders are often nested (``data/c3d/...``, ``data/npz/...``) and
    may repeat filenames across subfolders, so the relative path is folded
    into the name rather than just taking the stem.
    """
    relative = source_path.relative_to(input_root) if input_root in source_path.parents else Path(source_path.name)
    flat = "_".join(relative.with_suffix("").parts)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in flat) + ".npz"


def save_unified(motion: CanonicalMotion, out_path: Path, source_path: str, source_format: str) -> None:
    """Write one unified clip to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        trans=motion.trans.astype(np.float32),
        global_orient=motion.global_orient.astype(np.float32),
        body_pose=motion.body_pose.astype(np.float32),
        left_hand_pose=motion.left_hand_pose.astype(np.float32),
        right_hand_pose=motion.right_hand_pose.astype(np.float32),
        betas=motion.betas.astype(np.float32),
        part_mask=motion.part_mask.astype(np.float32),
        fps=float(motion.fps),
        verdict=motion.quality.verdict,
        joint_rmse_mm=float(motion.quality.joint_rmse_mm),
        observed_fraction=float(motion.quality.observed_fraction),
        source_path=str(source_path),
        source_format=str(source_format),
    )


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def find_input_files(input_dir: Path) -> List[Path]:
    """Every file under ``input_dir`` (recursively) that looks like mocap."""
    known = {".npz", ".c3d", ".bvh"}
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in known)


def run(
    input_dir: str,
    out_dir: str,
    target_fps: float = DEFAULT_TARGET_FPS,
    limit: Optional[int] = None,
    keep_rejects: bool = False,
) -> None:
    """Run Stage 1 over a folder of mocap files.

    One bad file never stops the batch: every file is handled inside a
    try/except and gets a manifest row either way. Files already recorded in
    the manifest are skipped, so an interrupted run can simply be restarted.

    Args:
        input_dir:    Folder of raw mocap (searched recursively).
        out_dir:      Where to write unified clips + ``manifest.csv``.
        target_fps:   Frame rate every clip is converted to.
        limit:        Process at most this many files (for quick trials).
        keep_rejects: Also write out clips whose fit scored REJECT. Off by
                      default; the manifest records them regardless.
    """
    input_root = Path(input_dir)
    out_root = Path(out_dir)
    manifest_path = out_root / "manifest.csv"

    files = find_input_files(input_root)
    done = already_processed(manifest_path)
    todo = [p for p in files if str(p) not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"{len(files)} mocap files found, {len(done)} already done, {len(todo)} to process.")
    if not todo:
        return

    # The body model is only needed for point-based sources, and loading it
    # is slow, so it is loaded once and only if something actually needs it.
    body = None
    counts = {"ok": 0, "skipped": 0, "failed": 0}

    with ManifestWriter(manifest_path, MANIFEST_COLUMNS) as manifest:
        for i, path in enumerate(todo, 1):
            row = {column: "" for column in MANIFEST_COLUMNS}
            row["source_path"] = str(path)
            started = time.time()
            try:
                source_format = detect_source_format(str(path))
                row["source_format"] = source_format

                if source_format in NEEDS_FITTING and body is None:
                    from src.gpsm.smplx.fitting.body import SMPLXBody
                    print("  loading SMPL-X body model (needed for marker/skeleton fitting)...")
                    body = SMPLXBody()

                if source_format == NPZ_PARAMS:
                    motion = unify_params(str(path), target_fps)
                else:
                    motion = unify_points(str(path), source_format, target_fps, body)

                if np.isnan(motion.quality.joint_rmse_mm) and motion.quality.verdict == "GOOD":
                    # Tier-0 conversions involve no fitting, so there is no
                    # fit error to score; still run the shared scorer so
                    # every clip's verdict comes from the same place.
                    motion.quality = score_canonical_motion(motion, joint_rmse_mm=float("nan"))

                row["n_frames_out"] = motion.num_frames
                row["fps_out"] = motion.fps
                row["fps_raw"] = motion.meta.get("resampled_from_fps", motion.fps)
                row["verdict"] = motion.quality.verdict
                row["joint_rmse_mm"] = round(float(motion.quality.joint_rmse_mm), 2)
                row["observed_fraction"] = round(float(motion.quality.observed_fraction), 3)

                if motion.quality.verdict in KEEP_VERDICTS or keep_rejects:
                    out_path = out_root / output_name(path, input_root)
                    save_unified(motion, out_path, str(path), source_format)
                    row["status"] = "ok"
                    row["output_path"] = str(out_path)
                    counts["ok"] += 1
                else:
                    row["status"] = "skipped"
                    row["reason"] = f"verdict {motion.quality.verdict}"
                    counts["skipped"] += 1

            except UnknownFormatError as error:
                row["status"] = "skipped"
                row["reason"] = str(error)
                counts["skipped"] += 1
            except Exception as error:  # noqa: BLE001 - one bad file must not stop the batch
                row["status"] = "failed"
                row["reason"] = f"{type(error).__name__}: {error}"
                counts["failed"] += 1
                traceback.print_exc()

            row["seconds"] = round(time.time() - started, 1)
            manifest.append(row)
            print(f"[{i}/{len(todo)}] {path.name}: {row['status']}"
                  f"{' (' + str(row['reason'])[:70] + ')' if row['reason'] else ''}"
                  f"  {row['seconds']}s")

    print(f"\nDone. ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionPrep Stage 1: unify mocap into one canonical pose format.")
    parser.add_argument("input_dir", help="Folder of raw mocap files (searched recursively)")
    parser.add_argument("--out", default="processed/unified", help="Output folder")
    parser.add_argument("--fps", type=float, default=DEFAULT_TARGET_FPS, help="Target frame rate for every clip")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N files (for a quick trial)")
    parser.add_argument("--keep-rejects", action="store_true",
                         help="Also write clips whose fit scored REJECT (recorded in the manifest either way)")
    args = parser.parse_args()

    run(args.input_dir, args.out, target_fps=args.fps, limit=args.limit, keep_rejects=args.keep_rejects)


if __name__ == "__main__":
    main()
