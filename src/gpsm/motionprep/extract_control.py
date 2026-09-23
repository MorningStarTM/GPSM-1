"""MotionPrep Stage 2: work out the control signal for every unified clip.

Raw mocap records what a body *did*, but nothing about what a player would
have been *asking* for. Stage 2 recovers that intent from the recording
itself: for each frame, where the root actually goes over the next fraction
of a second, and how much it turns. At training time that stands in for
"what the player wanted"; at play time the same 14 numbers come from a
joystick instead.

This stage is cheap (plain NumPy, no fitting, well under a second per clip)
and reads only Stage 1's output — which is exactly why the two stages are
kept apart. Retuning control (different look-ahead times, say) means re-running
only this, never the slow body-fitting in Stage 1.

Alignment matters here and is easy to get subtly wrong, so it is worth
stating plainly: ``control[t]`` is saved at the **same index** as ``pose[t]``,
and describes where the body goes *from* frame ``t``. Training then predicts
``pose[t+1]`` from ``(pose[t], control[t])`` — never from ``control[t+1]``,
which in a real game would not exist yet at that moment.

Run it:
    python -m src.gpsm.motionprep.extract_control processed/unified --out processed/with_control
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.gpsm.control.control_features import DEFAULT_HORIZONS_SECONDS, extract_control
from src.gpsm.control.root_trajectory import root_trajectory_from_canonical
from src.gpsm.motionprep.manifest import ManifestWriter, already_processed

MANIFEST_COLUMNS = [
    "source_path", "status", "reason", "n_frames",
    "n_valid_control_frames", "control_dim", "seconds", "output_path",
]

#: Fields copied straight through from Stage 1's file, untouched.
_PASSTHROUGH_FIELDS = (
    "trans", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
    "betas", "part_mask", "fps", "verdict", "joint_rmse_mm", "observed_fraction",
    "source_path", "source_format",
)


def add_control_to_clip(unified_path: Path, out_path: Path, horizons=DEFAULT_HORIZONS_SECONDS) -> dict:
    """Read one unified clip, compute its control signal, save both together.

    Args:
        unified_path: A ``.npz`` written by Stage 1.
        out_path:     Where to write the clip plus its control signal.
        horizons:     Look-ahead times in seconds.

    Returns:
        A small summary dict (frame counts) for the manifest.
    """
    with np.load(unified_path, allow_pickle=True) as data:
        clip = {key: data[key] for key in data.files}

    trajectory = root_trajectory_from_canonical(
        trans=clip["trans"],
        global_orient=clip["global_orient"],
        fps=float(clip["fps"]),
        source=unified_path.name,
    )
    control = extract_control(trajectory, horizons_seconds=horizons)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **{key: clip[key] for key in _PASSTHROUGH_FIELDS if key in clip},
        control=control.values.astype(np.float32),
        control_valid=control.valid,
        control_names=np.array(control.names),
    )
    return {
        "n_frames": int(control.values.shape[0]),
        "n_valid_control_frames": int(control.valid.sum()),
        "control_dim": int(control.values.shape[1]),
    }


def run(input_dir: str, out_dir: str, limit: Optional[int] = None) -> None:
    """Run Stage 2 over a folder of Stage 1 output.

    Same batch behaviour as Stage 1: per-file try/except so one bad clip
    cannot stop the run, a manifest row for every clip either way, and
    already-finished clips skipped so an interrupted run can be restarted.
    """
    input_root = Path(input_dir)
    out_root = Path(out_dir)
    manifest_path = out_root / "manifest.csv"

    clips: List[Path] = sorted(p for p in input_root.glob("*.npz"))
    done = already_processed(manifest_path)
    todo = [p for p in clips if str(p) not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"{len(clips)} unified clips found, {len(done)} already done, {len(todo)} to process.")
    if not todo:
        return

    counts = {"ok": 0, "failed": 0}
    with ManifestWriter(manifest_path, MANIFEST_COLUMNS) as manifest:
        for i, path in enumerate(todo, 1):
            row = {column: "" for column in MANIFEST_COLUMNS}
            row["source_path"] = str(path)
            started = time.time()
            try:
                out_path = out_root / path.name
                summary = add_control_to_clip(path, out_path)
                row.update(summary)
                row["status"] = "ok"
                row["output_path"] = str(out_path)
                counts["ok"] += 1
            except Exception as error:  # noqa: BLE001 - one bad clip must not stop the batch
                row["status"] = "failed"
                row["reason"] = f"{type(error).__name__}: {error}"
                counts["failed"] += 1
                traceback.print_exc()

            row["seconds"] = round(time.time() - started, 2)
            manifest.append(row)
            print(f"[{i}/{len(todo)}] {path.name}: {row['status']}"
                  f"  frames={row['n_frames']} valid_control={row['n_valid_control_frames']}")

    print(f"\nDone. ok={counts['ok']} failed={counts['failed']}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MotionPrep Stage 2: add the control signal to unified clips."
    )
    parser.add_argument("input_dir", help="Folder of Stage 1 output (processed/unified)")
    parser.add_argument("--out", default="processed/with_control", help="Output folder")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N clips")
    args = parser.parse_args()

    run(args.input_dir, args.out, limit=args.limit)


if __name__ == "__main__":
    main()
