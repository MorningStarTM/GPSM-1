"""Inspect the control signal extracted from every mocap file in a folder.

For each ``.npz`` / ``.c3d`` file this:

1. prints one summary line (frame rate, length, distance walked, how much the
   character turned, ...),
2. saves a plot (``<name>_control.png``): the root path seen from above with
   arrows showing which way the character faces, next to the control signals
   over time,
3. saves the extracted control arrays (``<name>_control.npz``) — the actual
   product of this module, ready to be fed to a model later.

Use it to *look at* the data before trusting it: does the path look like what
the file name says? Do turns show up as turn signals?

Usage:
    python -m src.gpsm.control.inspect_control data
    python -m src.gpsm.control.inspect_control "data/B9 -  Walk turn left 90.c3d"
    python -m src.gpsm.control.inspect_control data --out some/other/folder --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # no window needed: plots are saved to files
import matplotlib.pyplot as plt

from src.gpsm.control.control_features import ControlFeatures, extract_control
from src.gpsm.control.root_trajectory import RootTrajectory, load_root_trajectory

SUPPORTED_SUFFIXES = (".npz", ".c3d")

# A frame counts as "walking" if the root moves faster than this (m/s).
MOVING_SPEED = 0.6


def turn_while_moving_degrees(trajectory: RootTrajectory) -> float:
    """How much the character turned *while it was actually moving*.

    Measured between the first fifth and the last fifth of the frames where
    the root is moving. This is a better "did it turn?" number than simply
    last-frame minus first-frame heading, because clips usually begin and end
    with the character standing still, and the standing pose adds an
    unrelated twist to the heading at the very ends. On the two labelled
    clips in ``data/`` it gives +87 deg for "Walk turn left 90" and +142 deg
    for "Walk turn left 135".

    Returns:
        Degrees (left turn positive), or NaN if the character never moves.
    """
    fps = trajectory.fps
    half_window = max(1, int(round(0.05 * fps)))
    position = trajectory.position
    velocity = (position[2 * half_window:] - position[:-2 * half_window]) / (2 * half_window / fps)
    speed = np.linalg.norm(velocity, axis=1)
    heading = trajectory.heading[half_window:-half_window]

    moving = np.where(speed > MOVING_SPEED)[0]
    if len(moving) < 5:
        return float("nan")
    fifth = max(1, len(moving) // 5)
    change = np.median(heading[moving[-fifth:]]) - np.median(heading[moving[:fifth]])
    return float(np.degrees(change))


def summarize(trajectory: RootTrajectory, control: ControlFeatures) -> dict:
    """One row of numbers describing a recording."""
    step = np.linalg.norm(np.diff(trajectory.position, axis=0), axis=1)
    turn_rate = np.abs(np.diff(trajectory.heading)) * trajectory.fps  # rad/s
    return {
        "name": trajectory.source,
        "fps": trajectory.fps,
        "frames": trajectory.n_frames,
        "seconds": trajectory.n_frames / trajectory.fps,
        "distance_m": float(step.sum()),
        "turn_moving_deg": turn_while_moving_degrees(trajectory),
        "peak_turn_deg_s": float(np.degrees(np.percentile(turn_rate, 99))),
        "valid_frac": float(control.valid.mean()),
    }


def plot_control(trajectory: RootTrajectory, control: ControlFeatures, out_path: Path) -> None:
    """Save the two-panel plot described in the module docstring."""
    fps = trajectory.fps
    seconds = np.arange(trajectory.n_frames) / fps
    column = {name: control.values[:, i] for i, name in enumerate(control.names)}

    figure, (ax_path, ax_signals) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: the path from above, coloured by time, with facing arrows.
    ax_path.scatter(*trajectory.position.T, c=seconds, s=4, cmap="viridis")
    every = max(1, int(round(0.5 * fps)))
    arrow_at = np.arange(0, trajectory.n_frames, every)
    ax_path.quiver(
        trajectory.position[arrow_at, 0], trajectory.position[arrow_at, 1],
        np.cos(trajectory.heading[arrow_at]), np.sin(trajectory.heading[arrow_at]),
        color="crimson", width=0.004, scale=25,
    )
    ax_path.set_aspect("equal")
    ax_path.set_xlabel("world X (m)")
    ax_path.set_ylabel("world Y (m)")
    ax_path.set_title("Root path from above (colour = time, arrows = facing)")

    # Right: control signals over time. Grey = tail frames with no full look-ahead.
    ax_signals.plot(seconds, column["vel_fwd"], label="velocity forward (m/s)")
    ax_signals.plot(seconds, column["vel_left"], label="velocity left (m/s)")
    ax_signals.set_xlabel("time (s)")
    ax_signals.set_ylabel("velocity (m/s)")
    if not control.valid.all():
        first_invalid = seconds[~control.valid][0]
        ax_signals.axvspan(first_invalid, seconds[-1], color="grey", alpha=0.15, label="no full look-ahead")

    ax_turn = ax_signals.twinx()
    longest = control.names[-1].split("_")[-1]  # e.g. "500ms"
    turn_deg = np.degrees(np.arctan2(column[f"turn_sin_{longest}"], column[f"turn_cos_{longest}"]))
    ax_turn.plot(seconds, turn_deg, color="crimson", alpha=0.8, label=f"turn in next {longest} (deg, left +)")
    ax_turn.set_ylabel("turn (deg)")

    lines = ax_signals.get_legend_handles_labels()
    lines_turn = ax_turn.get_legend_handles_labels()
    ax_signals.legend(lines[0] + lines_turn[0], lines[1] + lines_turn[1], loc="upper right", fontsize=8)
    ax_signals.set_title("Control signals")

    figure.suptitle(trajectory.source)
    figure.tight_layout()
    figure.savefig(out_path, dpi=110)
    plt.close(figure)


def find_files(path: Path) -> List[Path]:
    """A single file, or every supported file directly inside a folder."""
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect control signals extracted from mocap files.")
    parser.add_argument("path", help="A .npz/.c3d file, or a folder containing them (e.g. data)")
    parser.add_argument("--out", default="src/gpsm/control/output", help="Where to write plots and control arrays")
    parser.add_argument("--no-plots", action="store_true", help="Only print the summary table")
    args = parser.parse_args()

    files = find_files(Path(args.path))
    if not files:
        raise SystemExit(f"No .npz/.c3d files found in {args.path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'file':44s} {'fps':>4s} {'sec':>6s} {'dist_m':>7s} {'turn_moving':>12s} {'peak_turn':>10s} {'valid':>6s}")
    failures = 0
    for path in files:
        try:
            trajectory = load_root_trajectory(str(path))
            control = extract_control(trajectory)
        except Exception as error:  # keep going: one odd file should not hide the rest
            failures += 1
            print(f"{path.name:44s} FAILED: {type(error).__name__}: {error}")
            continue

        row = summarize(trajectory, control)
        print(
            f"{row['name']:44s} {row['fps']:4.0f} {row['seconds']:6.1f} {row['distance_m']:7.2f} "
            f"{row['turn_moving_deg']:+11.1f}d {row['peak_turn_deg_s']:8.0f}/s {row['valid_frac']:6.2f}"
        )

        stem = path.stem.replace(" ", "_")
        np.savez(
            out_dir / f"{stem}_control.npz",
            values=control.values, valid=control.valid, names=np.array(control.names), fps=control.fps,
        )
        if not args.no_plots:
            plot_control(trajectory, control, out_dir / f"{stem}_control.png")

    print(f"\nOutputs written to {out_dir}  ({len(files) - failures} ok, {failures} failed)")


if __name__ == "__main__":
    main()
