"""Tier 2: raw C3D optical-mocap marker clouds -> CanonicalMotion.

Uses the real VICON_50 marker->vertex correspondence table
(layouts/builtin.py, extracted from an actual AMASS/SOMA capture file) and
fits whichever of its 50 markers are actually present in the uploaded file
by label match — real C3D exports rarely carry the exact same 50 markers,
so this degrades gracefully to however many match rather than requiring an
exact set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, List, Optional, Union

import numpy as np

from src.gpsm.smplx.layouts.builtin import VICON_50_VERTEX_IDS
from src.gpsm.smplx.layouts.correspondence import MotionLayout, VertexCorr
from src.gpsm.smplx.ops import guess_up_axis

MIN_MATCHED_MARKERS = 6  # well above the umeyama minimum of 3; keeps fits sane


class LenientC3DReader:
    """Wraps the `c3d` package's Reader, tolerating malformed metadata that
    real-world C3D exports routinely carry — we only ever read point
    (marker) data, never analog channels, so none of the following is
    actually relevant to correctness here, only to whether the `c3d`
    package's own parsing crashes before we get to the points:

    - `_check_metadata`: enforces the header and parameter blocks agree on
      the analog sample rate; a mismatch there (verified against the real
      01_01.c3d sample in this repo's referenced dataset) is a harmless
      inconsistency for point-only reading.
    - `analog_per_frame`: the `c3d` package computes this as
      `int(ANALOG:RATE / POINT:RATE)` unconditionally, even when
      `ANALOG:USED == 0` (no analog channels at all) — and real files with
      no analog channels routinely leave `ANALOG:RATE` at whatever
      uninitialized sentinel their recording software happened to write
      (verified against a real file: `ANALOG:USED=0` but
      `ANALOG:RATE=-1.7e+38`, an obvious garbage/uninitialized value). That
      feeds into `np.broadcast_to(..., (analog_used, analog_per_frame))`
      inside `read_frames()`, which raises `ValueError: all elements of
      broadcast shape must be non-negative` on the resulting huge negative
      int — happening unconditionally, regardless of the `analog_transform`
      flag `read_frames()` exposes (a no-op for this specific call, which
      is itself an upstream oversight). Since a file with zero analog
      channels needs analog_per_frame to be exactly 0 regardless of what
      ANALOG:RATE says, that is what we clamp it to.
    """

    def __init__(self, handle: BinaryIO):
        import c3d as c3d_pkg

        class _Reader(c3d_pkg.Reader):
            def _check_metadata(self) -> None:  # noqa: D401 - intentional no-op
                pass

            @property
            def analog_per_frame(self) -> int:
                if self.analog_used <= 0:
                    return 0
                try:
                    value = int(c3d_pkg.Reader.analog_per_frame.fget(self))
                except (ValueError, OverflowError):
                    return 0
                return value if value >= 0 else 0

        self._reader = _Reader(handle)

    @property
    def labels(self) -> List[str]:
        return [l.strip() for l in self._reader.point_labels]

    @property
    def point_rate(self) -> float:
        return float(self._reader.point_rate)

    def units_to_meters(self) -> float:
        try:
            param = self._reader.get("POINT:UNITS")
            unit = param.string_value.strip().lower() if param else "mm"
        except Exception:
            unit = "mm"
        return {"mm": 0.001, "cm": 0.01, "m": 1.0}.get(unit, 0.001)

    def read_points(self) -> np.ndarray:
        """Returns (T, K, 3) marker positions, in the file's native units
        (NOT yet converted to meters — see `units_to_meters`)."""
        frames = []
        for _frame_no, points, _analog in self._reader.read_frames():
            frames.append(points[:, :3])
        return np.stack(frames, axis=0)


@dataclass
class RawMarkerCapture:
    points_m: np.ndarray   # (T, K, 3), meters
    labels: List[str]
    fps: float
    up_axis: str
    up_axis_guessed: bool


def load_c3d(file_like: Union[str, BinaryIO]) -> RawMarkerCapture:
    """Parse a C3D file into meters + labels + fps, with a best-effort
    up-axis guess (overridable by the caller/UI)."""
    if isinstance(file_like, str):
        handle = open(file_like, "rb")
        owns_handle = True
    else:
        handle = file_like
        owns_handle = False
    try:
        reader = LenientC3DReader(handle)
        labels = reader.labels
        raw_points = reader.read_points()
        points_m = raw_points * reader.units_to_meters()
        up_axis, guessed = guess_up_axis(points_m, labels)
        return RawMarkerCapture(
            points_m=points_m, labels=labels, fps=reader.point_rate,
            up_axis=up_axis, up_axis_guessed=guessed,
        )
    finally:
        if owns_handle:
            handle.close()


def _strip_subject_prefix(label: str) -> str:
    """Multi-actor Vicon/Qualisys exports commonly namespace every marker
    with the subject name, e.g. "male2:LFHD" for plain "LFHD" (verified
    against a real file: MartialArtsWalksTurns_c3d's "E10 - side step
    right.c3d" prefixes all 41 markers with "male2:"). Only the segment
    after the last separator is the actual marker name."""
    label = label.strip()
    for sep in (":", "."):
        if sep in label:
            label = label.rsplit(sep, 1)[-1]
    return label


def build_matched_layout(labels: List[str]) -> "tuple[MotionLayout, List[int]]":
    """Intersect uploaded marker labels with the known VICON_50 table.

    Returns (layout restricted to the matched markers, column indices into
    the original (T, K, 3) array to select). Raises ValueError if fewer
    than `MIN_MATCHED_MARKERS` markers match — too few correspondences to
    fit safely.
    """
    upper_lookup = {name.upper(): name for name in VICON_50_VERTEX_IDS}
    matched_cols, matched_names = [], []
    for i, label in enumerate(labels):
        key = _strip_subject_prefix(label).upper()
        if key in upper_lookup:
            matched_cols.append(i)
            matched_names.append(upper_lookup[key])

    if len(matched_cols) < MIN_MATCHED_MARKERS:
        raise ValueError(
            f"Only {len(matched_cols)} of this file's markers match the known "
            f"VICON_50 protocol (need >= {MIN_MATCHED_MARKERS}); labels found: {labels}"
        )

    correspondences = [VertexCorr(VICON_50_VERTEX_IDS[name]) for name in matched_names]
    layout = MotionLayout(
        name="vicon_50_matched",
        point_names=matched_names,
        correspondences=correspondences,
        description=f"{len(matched_names)}/{len(VICON_50_VERTEX_IDS)} VICON_50 markers matched by label.",
    )
    return layout, matched_cols
