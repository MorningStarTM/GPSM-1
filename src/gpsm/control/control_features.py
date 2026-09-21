"""Turn a root trajectory into a per-frame *control* vector.

The idea (see the discussion that led to this module): the training data has
no record of what a player wanted, but every recording does contain what the
character *actually did next*. So for each frame we describe "where the root
goes over the next fraction of a second, and how it turns" — and treat that
as the control input for that frame. At play time, the player's stick/keys
are converted into the same kind of vector, and the model is steered by it.

What is in the vector
---------------------
For frame ``t``, everything is expressed in the character's *own* frame at
``t`` (origin at its root, x axis = where it faces, y axis = its left).
That matters: raw mocap has arbitrary world positions and headings, and
without this step a model would learn *where in the room* a clip was recorded
instead of "turn left".

* ``vel_fwd``, ``vel_left``          current velocity (m/s).
* for each look-ahead time (default 0.1 s, 0.25 s, 0.5 s):
    * ``pos_fwd_<ms>``, ``pos_left_<ms>``   where the root will be (m).
    * ``turn_cos_<ms>``, ``turn_sin_<ms>``  how much it will have turned:
      the heading change encoded as (cos, sin), so there is no jump when the
      angle wraps around +/-180 degrees. ``turn_sin > 0`` means a left turn.

Look-ahead times are in *seconds*, converted to frames using each file's own
frame rate, so files recorded at 100 fps and 120 fps give comparable vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from src.gpsm.control.root_trajectory import RootTrajectory

DEFAULT_HORIZONS_SECONDS = (0.1, 0.25, 0.5)

# Velocity is a central difference over +/- this many seconds: short enough to
# follow real changes, long enough not to amplify mocap noise.
_VELOCITY_HALF_WINDOW_SECONDS = 0.05


@dataclass
class ControlFeatures:
    """The control vectors for one recording.

    Attributes:
        values: ``(T, D)`` control vector per frame.
        valid:  ``(T,)`` bool. ``False`` for the last frames of the clip,
            where the longest look-ahead runs past the end of the recording.
            Those frames have no real "future", so training should skip them
            rather than learn from a made-up one.
        names:  ``D`` column names, in order (e.g. ``"pos_fwd_500ms"``).
        fps:    Frame rate the look-ahead times were converted with.
    """

    values: np.ndarray
    valid: np.ndarray
    names: List[str]
    fps: float

    @property
    def control_dim(self) -> int:
        return self.values.shape[1]


def _to_character_frame(offset_xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Re-express world-frame ground offsets in each frame's own
    (forward, left) axes.

    Args:
        offset_xy: ``(T, 2)`` world-frame offsets.
        heading:   ``(T,)`` the character's heading at each frame.

    Returns:
        ``(T, 2)`` columns are (forward, left).
    """
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    forward = offset_xy[:, 0] * cos_h + offset_xy[:, 1] * sin_h
    left = -offset_xy[:, 0] * sin_h + offset_xy[:, 1] * cos_h
    return np.stack([forward, left], axis=1)


def extract_control(
    trajectory: RootTrajectory,
    horizons_seconds: Sequence[float] = DEFAULT_HORIZONS_SECONDS,
) -> ControlFeatures:
    """Compute the per-frame control vector for a recording.

    Args:
        trajectory:       The root trajectory (see ``root_trajectory.py``).
        horizons_seconds: Look-ahead times, in seconds.

    Returns:
        :class:`ControlFeatures` with ``2 + 4 * len(horizons_seconds)``
        columns.

    Raises:
        ValueError: If the recording has fewer than 2 frames.
    """
    n_frames = trajectory.n_frames
    fps = trajectory.fps
    if n_frames < 2:
        raise ValueError("Need at least 2 frames to compute a velocity.")

    position, heading = trajectory.position, trajectory.heading
    frame_index = np.arange(n_frames)

    # --- current velocity: central difference, edges use what exists ---
    half_window = max(1, int(round(_VELOCITY_HALF_WINDOW_SECONDS * fps)))
    before = np.clip(frame_index - half_window, 0, n_frames - 1)
    after = np.clip(frame_index + half_window, 0, n_frames - 1)
    velocity_world = (position[after] - position[before]) / ((after - before) / fps)[:, None]
    velocity_local = _to_character_frame(velocity_world, heading)

    columns = [velocity_local[:, 0], velocity_local[:, 1]]
    names = ["vel_fwd", "vel_left"]

    # --- where the root will be, and how much it will have turned ---
    longest_lookahead = 0
    for horizon in horizons_seconds:
        lookahead = max(1, int(round(horizon * fps)))
        longest_lookahead = max(longest_lookahead, lookahead)

        future = np.minimum(frame_index + lookahead, n_frames - 1)
        offset_local = _to_character_frame(position[future] - position, heading)
        turn = heading[future] - heading

        milliseconds = int(round(horizon * 1000))
        columns += [offset_local[:, 0], offset_local[:, 1], np.cos(turn), np.sin(turn)]
        names += [
            f"pos_fwd_{milliseconds}ms",
            f"pos_left_{milliseconds}ms",
            f"turn_cos_{milliseconds}ms",
            f"turn_sin_{milliseconds}ms",
        ]

    valid = frame_index + longest_lookahead <= n_frames - 1
    return ControlFeatures(values=np.stack(columns, axis=1), valid=valid, names=names, fps=fps)
