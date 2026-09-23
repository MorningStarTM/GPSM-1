import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.gpsm.smplx.adapters.markers import build_matched_layout, load_c3d
from src.gpsm.smplx.fitting.solver import fit_observation
from src.gpsm.smplx.schema import Observation

# Searched recursively, and relative to the repo, so that reorganising
# data/ into per-format subfolders (as has already happened once) does not
# silently turn these real-data tests into skips.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_c3d_samples = sorted((_REPO_ROOT / "data").rglob("*.c3d"))
SAMPLE_C3D = str(_c3d_samples[0]) if _c3d_samples else ""
requires_sample_c3d = pytest.mark.skipif(
    not SAMPLE_C3D, reason="no .c3d sample found under data/"
)

# The exact file reported to crash `load_c3d` — ships inside a zip archive
# rather than as a standalone file, so it's extracted into memory for the
# duration of the test rather than left unpacked on disk.
_ANALOG_BUG_ZIP = "C:/Users/Ernest/Downloads/MoCap/MartialArtsWalksTurns_c3d.zip"
_ANALOG_BUG_MEMBER = "E10 - side step right.c3d"
requires_analog_bug_sample = pytest.mark.skipif(
    not os.path.isfile(_ANALOG_BUG_ZIP), reason="reported sample archive not found"
)


def _read_analog_bug_sample() -> bytes:
    with zipfile.ZipFile(_ANALOG_BUG_ZIP) as zf:
        return zf.read(_ANALOG_BUG_MEMBER)


@requires_sample_c3d
def test_load_real_c3d_file():
    cap = load_c3d(SAMPLE_C3D)
    assert cap.points_m.ndim == 3 and cap.points_m.shape[2] == 3
    assert cap.fps > 0
    assert cap.up_axis in ("y", "z")
    assert not cap.up_axis_guessed  # head/foot markers present -> confident guess
    assert not np.isnan(cap.points_m).any()
    # Sanity: marker coordinates converted to meters, not left in mm.
    assert np.abs(cap.points_m).max() < 50.0


@requires_sample_c3d
def test_build_matched_layout_finds_real_overlap():
    cap = load_c3d(SAMPLE_C3D)
    layout, cols = build_matched_layout(cap.labels)
    assert layout.num_points == len(cols)
    assert layout.num_points >= 6
    assert not layout.is_joint_only
    assert set(layout.point_names).issubset(set(cap.labels))


def test_build_matched_layout_raises_on_insufficient_overlap():
    with pytest.raises(ValueError):
        build_matched_layout(["totally_unknown_marker_a", "totally_unknown_marker_b"])


@requires_sample_c3d
def test_fit_real_c3d_markers_end_to_end(smplx_body):
    cap = load_c3d(SAMPLE_C3D)
    layout, cols = build_matched_layout(cap.labels)
    t = 15
    points = cap.points_m[:t][:, cols, :]
    conf = np.ones((t, len(cols)))
    obs = Observation(points=points, conf=conf, joint_names=layout.point_names,
                       space="world3d", fps=cap.fps, up_axis=cap.up_axis)

    motion, quality = fit_observation(obs, smplx_body, layout)
    assert not np.isnan(motion.to_array()).any()
    assert quality.observed_fraction == 1.0
    assert quality.verdict in ("GOOD", "DEGRADED", "REJECT")  # always a real verdict, never a crash
    # Trans should land in a plausible standing-human range, not at the origin.
    assert 0.3 < np.linalg.norm(motion.trans[0]) < 5.0


@requires_analog_bug_sample
def test_load_c3d_survives_garbage_analog_rate_metadata():
    """Regression test for the reported crash: 'Could not read ... as a
    .c3d file: all elements of broadcast shape must be non-negative'.

    Root cause (confirmed by direct inspection of this exact file): its
    ANALOG:USED parameter is 0 (no analog channels at all) but its
    ANALOG:RATE parameter holds an uninitialized sentinel value
    (-1.7e+38). The `c3d` package computes `analog_per_frame =
    int(ANALOG:RATE / POINT:RATE)` unconditionally, even when there are no
    analog channels to need it, producing a huge negative int that then
    blows up `np.broadcast_to` inside `read_frames()`. `LenientC3DReader`
    clamps `analog_per_frame` to 0 whenever `analog_used <= 0`.
    """
    data = _read_analog_bug_sample()
    import io
    cap = load_c3d(io.BytesIO(data))
    assert cap.points_m.ndim == 3 and cap.points_m.shape[2] == 3
    assert not np.isnan(cap.points_m).any()
    assert cap.fps > 0


@requires_analog_bug_sample
def test_build_matched_layout_strips_subject_name_prefix():
    """This same file's 41 markers are all namespaced as 'male2:LFHD' etc.
    (a common multi-actor Vicon/Qualisys export convention) rather than
    bare 'LFHD' — build_matched_layout must recognize these as the same
    markers, not silently match zero of them."""
    import io
    cap = load_c3d(io.BytesIO(_read_analog_bug_sample()))
    assert any(":" in label for label in cap.labels)  # confirms the premise
    layout, cols = build_matched_layout(cap.labels)
    assert layout.num_points >= 15
    assert "LFHD" in layout.point_names  # normalized, not "male2:LFHD"


@requires_analog_bug_sample
def test_fit_reported_file_end_to_end(smplx_body):
    """The exact file from the bug report, all the way through a real fit
    — must never crash, and must produce a real (even if imperfect)
    quality verdict rather than silently failing."""
    import io
    cap = load_c3d(io.BytesIO(_read_analog_bug_sample()))
    layout, cols = build_matched_layout(cap.labels)
    t = 10
    points = cap.points_m[:t][:, cols, :]
    conf = np.ones((t, len(cols)))
    obs = Observation(points=points, conf=conf, joint_names=layout.point_names,
                       space="world3d", fps=cap.fps, up_axis=cap.up_axis)
    motion, quality = fit_observation(obs, smplx_body, layout)
    assert not np.isnan(motion.to_array()).any()
    assert quality.verdict in ("GOOD", "DEGRADED", "REJECT")


@requires_sample_c3d
def test_occluded_markers_are_reported_not_silently_zeroed():
    """C3D stores an occluded marker as literally (0, 0, 0) with a negative
    residual. Those samples must be reported as unobserved via `valid`, so
    callers can pass them to the solver as zero-confidence.

    Regression test for a measured bug: feeding them through as if they were
    real measurements put phantom markers at the world origin and dragged a
    real clip's fit to 639mm joint RMSE (REJECT); masking them out brought
    the same clip to 66mm.
    """
    cap = load_c3d(SAMPLE_C3D)

    assert cap.valid.shape == cap.points_m.shape[:2]
    assert cap.valid.dtype == bool
    assert cap.valid.any(), "a usable file cannot be entirely occluded"

    # Wherever a sample is marked invalid, it must be the (0,0,0) sentinel —
    # i.e. `valid` really is tracking occlusion, not something unrelated.
    if not cap.valid.all():
        occluded = cap.points_m[~cap.valid]
        assert np.allclose(occluded, 0.0)
