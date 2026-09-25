"""One-call MotionPrep runner, meant for a Kaggle notebook.

MotionPrep is two stages (unify, then extract control) and normally you run
them as two command lines. In a notebook that is awkward, so this module
wraps both in a single ``prepare()`` call and prints a readable summary of
what came out.

Typical Kaggle cell::

    !git clone -q https://github.com/MorningStarTM/GPSM-1.git
    %cd GPSM-1
    from src.gpsm.motionprep.kaggle import prepare
    prepare("/kaggle/input/<dataset-slug>")

Dependencies: for parameter files (AMASS-style ``.npz``) this needs only
numpy and scipy, both of which Kaggle already has. Marker files (``.c3d``)
additionally need torch and the SMPL-X body model, because a body has to be
fitted to the points — that path is far slower and is not what this helper
is designed for. See ``doc/MotionPrep.docx``.
"""
from __future__ import annotations

import csv
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from src.gpsm.motionprep import extract_control, unify

#: Where Kaggle lets you write. Anything under /kaggle/input is read-only.
DEFAULT_OUT_DIR = "/kaggle/working/processed"


def prepare(
    input_dir: str,
    out_dir: str = DEFAULT_OUT_DIR,
    target_fps: float = 30.0,
    limit: Optional[int] = None,
    keep_rejects: bool = False,
) -> str:
    """Run both MotionPrep stages over a folder of raw mocap.

    Safe to re-run: both stages skip files already recorded in their
    manifest, so if the notebook times out you can simply call this again
    and it picks up where it stopped.

    Args:
        input_dir:    Folder of raw mocap, searched recursively. On Kaggle
                      this is ``/kaggle/input/<dataset-slug>``.
        out_dir:      Where to write. Two subfolders are created: ``unified``
                      (Stage 1) and ``with_control`` (Stage 2).
        target_fps:   Frame rate every clip is resampled to.
        limit:        Process at most this many *unprocessed* files, not this
                      many in total. Calling again therefore advances through
                      the rest rather than repeating, which is what you want
                      when a session times out. Use a small number for a first
                      trial before committing the notebook.
        keep_rejects: Also keep clips whose fit scored REJECT. Only affects
                      marker sources; parameter files are never rejected.

    Returns:
        The path to the Stage 2 output folder, which is what the training
        code should be pointed at.
    """
    out_root = Path(out_dir)
    unified_dir = out_root / "unified"
    control_dir = out_root / "with_control"

    started = time.time()

    print("=" * 70)
    print("MotionPrep  Stage 1/2 - unify to canonical SMPL-X")
    print("=" * 70)
    unify.run(
        input_dir=str(input_dir),
        out_dir=str(unified_dir),
        target_fps=target_fps,
        limit=limit,
        keep_rejects=keep_rejects,
    )

    print()
    print("=" * 70)
    print("MotionPrep  Stage 2/2 - extract the control signal")
    print("=" * 70)
    extract_control.run(input_dir=str(unified_dir), out_dir=str(control_dir))

    print()
    summarize(out_dir)
    print("\nTotal wall time: %s" % _human_time(time.time() - started))
    print("Training data is in: %s" % control_dir)

    return str(control_dir)


def summarize(out_dir: str = DEFAULT_OUT_DIR) -> None:
    """Print what the two manifests say, without re-reading the clips.

    Useful on its own after an interrupted run, to see how far it got.
    """
    out_root = Path(out_dir)
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    _summarize_stage(
        "Stage 1  unify",
        out_root / "unified" / "manifest.csv",
        frames_column="n_frames_out",
        extra_column="verdict",
    )
    _summarize_stage(
        "Stage 2  control",
        out_root / "with_control" / "manifest.csv",
        frames_column="n_frames",
    )


def _summarize_stage(
    title: str, manifest_path: Path, frames_column: str, extra_column: str = ""
) -> None:
    """Print one manifest's status counts, frame total and failure reasons."""
    print("\n%s" % title)
    if not manifest_path.exists():
        print("  no manifest yet (%s)" % manifest_path)
        return

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print("  manifest is empty")
        return

    statuses = Counter(row.get("status", "") for row in rows)
    frames = sum(_as_int(row.get(frames_column)) for row in rows)
    fps = _first_float(rows, "fps_out") or 30.0

    print("  files      : %d" % len(rows))
    print("  status     : %s" % ", ".join(
        "%s=%d" % (name or "?", count) for name, count in statuses.most_common()))
    print("  frames kept: %s  (%s at %.0f fps)" % (
        format(frames, ","), _human_time(frames / fps), fps))

    if extra_column:
        verdicts = Counter(row.get(extra_column, "") for row in rows if row.get(extra_column))
        if verdicts:
            print("  fit quality: %s" % ", ".join(
                "%s=%d" % (name, count) for name, count in verdicts.most_common()))

    # Failures are the thing you actually need to see, so show why.
    reasons = Counter(
        row.get("reason", "").split("\n")[0][:60]
        for row in rows
        if row.get("status") not in ("ok", "skipped") and row.get("reason")
    )
    for reason, count in reasons.most_common(5):
        print("  ! %3d x %s" % (count, reason))


def _as_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _first_float(rows, column: str) -> Optional[float]:
    for row in rows:
        try:
            return float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _human_time(seconds: float) -> str:
    """Seconds as the largest sensible unit, for log lines."""
    if seconds < 90:
        return "%.0f s" % seconds
    if seconds < 5400:
        return "%.1f min" % (seconds / 60)
    return "%.1f h" % (seconds / 3600)
