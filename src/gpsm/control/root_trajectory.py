"""Extract a character's *root trajectory* from raw mocap files.

Why this exists
---------------
GPSM's training data is raw mocap: it has poses, but no record of what a
player "wanted" the character to do. To train a model that can be steered
(press left -> turn left) we need a control signal, and we get it from the
recording itself: where the character's root actually went, and which way it
faced. This module produces that raw ingredient, in one common format, from
two very different kinds of file:

* ``.npz`` (AMASS / SMPL-H / SMPL-X): the root position is ``trans`` and the
  root orientation is the first 3 values of ``poses`` (axis-angle).
* ``.c3d`` (raw optical markers): there are no joint rotations at all, so the
  root is *estimated* from the four waist markers (``LFWT``, ``RFWT``,
  ``LBWT``, ``RBWT``).

Both come out as a :class:`RootTrajectory`, so everything downstream
(``control_features.py``) does not care which kind of file it came from.

Conventions (all verified against the real files in ``data/``)
-----------------------------------------------------------------
* **Z is up.** The ground plane is world X/Y. All ``.npz`` and ``.c3d`` files
  here are Z-up.
* **Units are metres.** ``.c3d`` files store millimetres; they are converted.
* **Heading** is the angle of the character's facing direction in the ground
  plane, in radians: ``0`` faces world ``+X``, and it *increases* when the
  character turns **left** (counter-clockwise seen from above). It is
  unwrapped (no jump at +/-pi), so a full left turn is ``+2*pi``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class RootTrajectory:
    """A character's root motion on the ground plane.

    Attributes:
        position: ``(T, 2)`` root position in metres (world X, Y).
        heading:  ``(T,)`` facing direction in radians, unwrapped. See the
            module docstring for the sign convention.
        fps:      Frames per second of the source recording.
        source:   File name this came from (for messages/plots).
    """

    position: np.ndarray
    heading: np.ndarray
    fps: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.position.ndim != 2 or self.position.shape[1] != 2:
            raise ValueError(f"position must be (T, 2), got {self.position.shape}")
        if self.heading.shape != (self.position.shape[0],):
            raise ValueError(
                f"heading must be (T,) matching position, got {self.heading.shape} "
                f"vs position {self.position.shape}"
            )
        if not (np.isfinite(self.position).all() and np.isfinite(self.heading).all()):
            raise ValueError("RootTrajectory contains non-finite values.")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    @property
    def n_frames(self) -> int:
        return self.position.shape[0]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def heading_from_left_vector(left_xy: np.ndarray) -> np.ndarray:
    """Turn "which way is the character's left" into a facing direction.

    We measure the character's *left* (the direction pointing out of its
    left hip) rather than its forward, because leaning forward or back
    rotates the body *around* the left-right axis and so never changes it —
    the left vector stays put while someone bends over, and only really
    changes when the character turns. That makes the heading stable during
    bending, lifting and crouching, where a direct "forward" vector would
    wobble.

    With Z up, forward = left x Z, i.e. ``(left_y, -left_x)``.

    Args:
        left_xy: ``(T, 2)`` ground-plane direction of the character's left.

    Returns:
        ``(T,)`` unwrapped heading in radians (0 = +X, left turn = positive).
    """
    forward_x = left_xy[:, 1]
    forward_y = -left_xy[:, 0]
    return np.unwrap(np.arctan2(forward_y, forward_x))


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert ``(T, 3)`` axis-angle rotations to ``(T, 3, 3)`` matrices
    (Rodrigues' formula). Written in plain NumPy so this module needs no
    extra dependency."""
    theta = np.linalg.norm(axis_angle, axis=1, keepdims=True)
    k = axis_angle / np.maximum(theta, 1e-8)

    cross = np.zeros((len(axis_angle), 3, 3))
    cross[:, 0, 1], cross[:, 0, 2] = -k[:, 2], k[:, 1]
    cross[:, 1, 0], cross[:, 1, 2] = k[:, 2], -k[:, 0]
    cross[:, 2, 0], cross[:, 2, 1] = -k[:, 1], k[:, 0]

    sin = np.sin(theta)[..., None]
    cos = np.cos(theta)[..., None]
    return np.eye(3) + sin * cross + (1.0 - cos) * (cross @ cross)


def _nanmean_quiet(values: np.ndarray, axis: int) -> np.ndarray:
    """``np.nanmean`` that stays quiet when a slice is entirely NaN.

    In a marker file, a frame where *every* waist marker is hidden is
    expected; the mean there is simply NaN, and :func:`_fill_gaps` fills it
    in afterwards. NumPy would print a warning for each such case, which is
    noise here, so it is silenced.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _fill_gaps(values: np.ndarray) -> np.ndarray:
    """Fill missing (NaN) samples by linear interpolation over time.

    Optical mocap loses markers now and then (occlusion). Leading or trailing
    gaps are filled by holding the nearest valid value.

    Args:
        values: ``(T,)`` or ``(T, K)`` array with NaNs marking missing samples.

    Returns:
        A copy with no NaNs, same shape.

    Raises:
        ValueError: If a column has no valid sample at all.
    """
    filled = np.array(values, dtype=float, copy=True)
    squeeze = filled.ndim == 1
    if squeeze:
        filled = filled[:, None]

    time = np.arange(len(filled))
    for col in range(filled.shape[1]):
        missing = np.isnan(filled[:, col])
        if missing.all():
            raise ValueError("Cannot fill gaps: no valid samples at all.")
        if missing.any():
            filled[missing, col] = np.interp(time[missing], time[~missing], filled[~missing, col])

    return filled[:, 0] if squeeze else filled


# ---------------------------------------------------------------------------
# .npz  (AMASS / SMPL-H / SMPL-X)
# ---------------------------------------------------------------------------

def root_trajectory_from_canonical(
    trans: np.ndarray, global_orient: np.ndarray, fps: float, source: str = ""
) -> RootTrajectory:
    """Build a root trajectory from canonical SMPL-X fields.

    This is the one entry point that matters once data has been through
    MotionPrep: every source format (parameters, markers, BVH) comes out of
    unification with a real ``global_orient``, so heading is read the same
    way for all of them, with no per-format special cases.

    Args:
        trans:         ``(T, 3)`` root translation, metres.
        global_orient: ``(T, 3)`` root rotation, axis-angle.
        fps:           Frame rate.
        source:        Name to carry along for messages/plots.

    Returns:
        The :class:`RootTrajectory`.
    """
    rotation_matrices = _axis_angle_to_matrix(np.asarray(global_orient, dtype=np.float64))
    left_in_world = rotation_matrices @ np.array([1.0, 0.0, 0.0])  # SMPL body-space left = +X
    heading = heading_from_left_vector(left_in_world[:, :2])
    return RootTrajectory(
        position=np.asarray(trans, dtype=np.float64)[:, :2],
        heading=heading,
        fps=float(fps),
        source=source,
    )


def root_trajectory_from_npz(path: str) -> RootTrajectory:
    """Read the root trajectory from an AMASS-style ``.npz`` file.

    Position is ``trans`` (ground-plane part). Heading comes from the root
    orientation, the first 3 numbers of ``poses``: rotate the body's left
    axis (SMPL's ``+X``) into the world and look at where it points on the
    ground (see :func:`heading_from_left_vector` for why left, not forward).

    Both AMASS layouts in ``data/`` work: the SMPL-H files
    (``mocap_framerate`` key, 156-wide poses) and the SMPL-X "stageii" files
    (``mocap_frame_rate`` key, 165-wide poses) — the first 3 values of
    ``poses`` mean the same in both.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        The :class:`RootTrajectory`.

    Raises:
        KeyError: If the file has no ``poses``/``trans`` or no frame rate.
    """
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        for needed in ("poses", "trans"):
            if needed not in keys:
                raise KeyError(f"{Path(path).name}: missing '{needed}' (found keys: {sorted(keys)})")

        fps_key = "mocap_frame_rate" if "mocap_frame_rate" in keys else "mocap_framerate"
        if fps_key not in keys:
            raise KeyError(f"{Path(path).name}: no frame-rate key ('mocap_frame_rate'/'mocap_framerate')")

        fps = float(data[fps_key])
        root_rotation = data["poses"][:, :3].astype(np.float64)
        trans = data["trans"].astype(np.float64)

    # The first 3 numbers of `poses` are exactly the canonical
    # `global_orient`, so this is the canonical path with a reader in front.
    return root_trajectory_from_canonical(trans, root_rotation, fps, source=Path(path).name)


# ---------------------------------------------------------------------------
# .c3d  (raw optical markers)
# ---------------------------------------------------------------------------

# Waist markers (CMU marker set): Left/Right x Front/Back.
_LEFT_WAIST = ("LFWT", "LBWT")
_RIGHT_WAIST = ("RFWT", "RBWT")

_UNIT_TO_METRES = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}


def load_c3d_markers(path: str) -> Tuple[np.ndarray, List[str], float]:
    """Read marker positions from a ``.c3d`` file, in metres.

    Uses the pure-Python ``c3d`` package (``pip install c3d``). The repo's
    existing ``.c3d`` loader (``dataset.py``) uses ``ezc3d`` instead, which
    has no wheel for every platform (none for Python 3.8 on Windows), so this
    module reads markers itself rather than depend on it.

    Notes on the files in ``data/``:

    * Missing markers are stored as ``(0, 0, 0)`` with a negative residual.
      Those samples are returned as NaN, *not* as zeros — averaging in a
      fake marker at the origin would drag every estimate toward it.
    * Marker labels may carry a subject prefix (``male2:LFWT``); it is removed.
    * These files have no analog channels but a garbage analog rate, which
      makes the ``c3d`` package refuse to read them. Analog data is not needed
      here, so that check is skipped (see ``_MarkerOnlyReader``).

    Args:
        path: Path to the ``.c3d`` file.

    Returns:
        ``(points, labels, fps)``: ``points`` is ``(T, M, 3)`` in metres with
        NaN for missing samples, ``labels`` the ``M`` marker names, ``fps``
        the marker frame rate.

    Raises:
        ImportError: If the ``c3d`` package is not installed.
        ValueError: If the file's length unit is not mm/cm/m.
    """
    try:
        import c3d
    except ImportError as error:
        raise ImportError("Reading .c3d files needs the 'c3d' package: pip install c3d") from error

    class _MarkerOnlyReader(c3d.Reader):
        """``c3d.Reader`` that ignores analog channels (see docstring above)."""

        def _check_metadata(self):
            pass

        def get_analog_transform(self):
            return np.zeros((0, 0)), np.zeros((0, 0))

    with open(path, "rb") as handle:
        reader = _MarkerOnlyReader(handle)
        labels = [name.strip().split(":")[-1] for name in reader.point_labels]
        frames = [points[:, :4] for _, points, _ in reader.read_frames()]  # x, y, z, residual
        fps = float(reader.point_rate)
        unit_param = reader.get("POINT:UNITS")
        unit = unit_param.string_value.strip().lower() if unit_param else "mm"

    if unit not in _UNIT_TO_METRES:
        raise ValueError(f"{Path(path).name}: unsupported length unit {unit!r}")

    stacked = np.stack(frames)
    xyz = stacked[..., :3] * _UNIT_TO_METRES[unit]
    xyz[stacked[..., 3] < 0] = np.nan  # negative residual = marker not seen
    return xyz, labels, fps


def root_trajectory_from_c3d(path: str) -> RootTrajectory:
    """Estimate the root trajectory from a marker (``.c3d``) file.

    There are no joint rotations in a marker file, so the root is estimated
    from the four waist markers:

    * **Position** is the average of the four waist markers (the pelvis
      centre).
    * **Heading** comes from the left-waist vs right-waist markers: the
      direction from the right pair to the left pair is the character's left
      (see :func:`heading_from_left_vector`).

    Each marker's hidden stretches are first filled by interpolating over
    time (:func:`_fill_gaps`), and only then are the four combined — see the
    comment in the body for why that order matters. Limitation: interpolation
    is only a guess for a long occlusion; a marker hidden for most of a clip
    will make the estimate there rough.

    Args:
        path: Path to the ``.c3d`` file.

    Returns:
        The :class:`RootTrajectory`.

    Raises:
        ValueError: If the waist markers are missing from the file, or the
            file does not look Z-up.
    """
    points, labels, fps = load_c3d_markers(path)

    missing = [name for name in _LEFT_WAIST + _RIGHT_WAIST if name not in labels]
    if missing:
        raise ValueError(f"{Path(path).name}: missing waist markers {missing} (has {labels})")

    def filled_marker(name: str) -> np.ndarray:
        """One marker's (T, 3) path with its hidden stretches interpolated."""
        try:
            return _fill_gaps(points[:, labels.index(name)])
        except ValueError:
            raise ValueError(f"{Path(path).name}: waist marker {name} is never visible") from None

    # Fill each marker's gaps *before* combining them. The alternative —
    # averaging "whichever markers are visible this frame" — makes the
    # estimate jump whenever the visible set changes (e.g. from all four
    # markers to two on opposite corners), and a jump in the left-right
    # direction reads as an impossible spin (thousands of degrees/second).
    left = np.stack([filled_marker(n) for n in _LEFT_WAIST], axis=1)    # (T, 2, 3)
    right = np.stack([filled_marker(n) for n in _RIGHT_WAIST], axis=1)  # (T, 2, 3)

    pelvis_centre = np.concatenate([left, right], axis=1).mean(axis=1)  # (T, 3)
    left_vector = left.mean(axis=1) - right.mean(axis=1)                # (T, 3)

    _check_z_up(points, labels, pelvis_centre)

    position = pelvis_centre[:, :2]
    heading = heading_from_left_vector(left_vector[:, :2])

    return RootTrajectory(position=position, heading=heading, fps=fps, source=Path(path).name)


def _check_z_up(points: np.ndarray, labels: List[str], pelvis_centre: np.ndarray) -> None:
    """Fail loudly if the file is not Z-up, instead of quietly producing a
    meaningless ground-plane trajectory. The head should sit well above the
    pelvis along Z; if it cannot be checked (head markers absent), skip."""
    head_names = [n for n in ("LFHD", "RFHD", "LBHD", "RBHD") if n in labels]
    if not head_names:
        return
    head = points[:, [labels.index(n) for n in head_names]]
    head_centre = _nanmean_quiet(head, axis=1)
    height = np.nanmedian((head_centre - pelvis_centre)[:, 2])
    if not height > 0.3:
        raise ValueError(
            f"Expected a Z-up file (head ~0.6 m above pelvis along Z) but the median "
            f"head-minus-pelvis Z is {height:.2f} m."
        )


# ---------------------------------------------------------------------------
# One entry point for both
# ---------------------------------------------------------------------------

def load_root_trajectory(path: str) -> RootTrajectory:
    """Load a root trajectory from a ``.npz`` or ``.c3d`` file, chosen by
    file extension.

    Raises:
        ValueError: For any other extension.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return root_trajectory_from_npz(path)
    if suffix == ".c3d":
        return root_trajectory_from_c3d(path)
    raise ValueError(f"Unsupported file type {suffix!r} (expected .npz or .c3d): {path}")
