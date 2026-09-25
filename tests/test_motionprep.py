"""Tests for the MotionPrep pipeline (src/gpsm/motionprep/).

Covers the pieces where a mistake would be silent rather than loud:

* format detection, including the files it should refuse,
* the manifest's resumability rules (what a re-run skips, and what it retries),
* resampling raw points, especially how occlusion is carried across,
* both stages end to end on a real clip, and
* the index alignment between pose and control, which training depends on.

Tests needing real mocap are skipped if data/ is not present.

Run with:
    pytest tests/test_motionprep.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from src.gpsm.motionprep.formats import C3D, NPZ_PARAMS, UnknownFormatError, detect_source_format
from src.gpsm.motionprep.manifest import ManifestWriter, already_processed
from src.gpsm.motionprep.unify import output_name, resample_points, unify_params

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def a_real_npz() -> str:
    matches = sorted(DATA_DIR.glob("npz/*.npz")) or sorted(DATA_DIR.glob("*.npz"))
    if not matches:
        pytest.skip("no parameter .npz files in data/")
    return str(matches[0])


def a_real_c3d() -> str:
    matches = sorted(DATA_DIR.glob("c3d/*.c3d")) or sorted(DATA_DIR.glob("*.c3d"))
    if not matches:
        pytest.skip("no .c3d files in data/")
    return str(matches[0])


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def test_detects_real_files():
    assert detect_source_format(a_real_npz()) == NPZ_PARAMS
    assert detect_source_format(a_real_c3d()) == C3D


def test_unknown_extension_is_an_error_not_a_silent_skip(tmp_path):
    """A whole folder quietly producing nothing is far harder to debug than
    one clear message, so unreadable input must raise."""
    stray = tmp_path / "notes.txt"
    stray.write_text("not mocap")
    with pytest.raises(UnknownFormatError):
        detect_source_format(str(stray))


def test_npz_without_pose_parameters_is_rejected(tmp_path):
    """A .npz holding something else entirely (here, plain joint positions)
    is readable but not convertible — it must not be mistaken for a
    parameter file."""
    path = tmp_path / "joints.npz"
    np.savez(path, joints=np.zeros((10, 21, 3)))
    with pytest.raises(UnknownFormatError):
        detect_source_format(str(path))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_appends_and_survives_reopening(tmp_path):
    path = tmp_path / "manifest.csv"
    columns = ["source_path", "status"]
    with ManifestWriter(path, columns) as writer:
        writer.append({"source_path": "a.npz", "status": "ok"})
    with ManifestWriter(path, columns) as writer:  # reopened, must not truncate
        writer.append({"source_path": "b.npz", "status": "ok"})

    assert already_processed(path) == {"a.npz", "b.npz"}
    assert path.read_text().count("source_path") == 1, "header should be written once"


def test_manifest_rejects_a_row_with_the_wrong_columns(tmp_path):
    """Catching a typo at write time beats finding a half-empty column in a
    manifest after a multi-hour run."""
    with ManifestWriter(tmp_path / "m.csv", ["source_path", "status"]) as writer:
        with pytest.raises(ValueError):
            writer.append({"source_path": "a.npz", "typo_here": "ok"})


def test_rerun_skips_finished_work_but_retries_failures(tmp_path):
    """Resumability rule: 'ok' and 'skipped' are final decisions, but
    'failed' usually means a crash or a bug (possibly one since fixed), so
    those files must be picked up again rather than written off."""
    path = tmp_path / "manifest.csv"
    with ManifestWriter(path, ["source_path", "status"]) as writer:
        writer.append({"source_path": "done.npz", "status": "ok"})
        writer.append({"source_path": "rejected.npz", "status": "skipped"})
        writer.append({"source_path": "crashed.npz", "status": "failed"})

    resumable = already_processed(path)
    assert resumable == {"done.npz", "rejected.npz"}
    assert "crashed.npz" not in resumable


# ---------------------------------------------------------------------------
# Resampling raw points (the fitted path)
# ---------------------------------------------------------------------------

def test_resample_points_keeps_positions_and_timing():
    """A point moving at a constant 1 m/s must still be at 1 m after one
    second, whatever the frame rate it is resampled to."""
    n_src = 121  # exactly 1.0s at 120 fps
    points = np.zeros((n_src, 2, 3))
    points[:, :, 0] = (np.arange(n_src) / 120.0)[:, None]
    valid = np.ones((n_src, 2), dtype=bool)

    out, out_valid = resample_points(points, valid, 120.0, 30.0)

    assert out.shape == (31, 2, 3) and out_valid.shape == (31, 2)
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[-1, 0, 0] == pytest.approx(1.0)


def test_resample_points_does_not_launder_occlusion_into_real_data():
    """An occluded marker must stay marked unobserved after resampling — a
    phantom position at the world origin, presented as a real measurement,
    wrecks a fit (measured on a real clip: 639mm joint error vs 66mm)."""
    n_src = 121
    points = np.zeros((n_src, 2, 3))
    valid = np.ones((n_src, 2), dtype=bool)
    valid[40:44, 1] = False  # marker 1 hidden for a few frames

    _, out_valid = resample_points(points, valid, 120.0, 30.0)

    assert out_valid[:, 0].all(), "the fully-observed marker must stay valid"
    hidden = np.where(~out_valid[:, 1])[0]
    assert len(hidden) > 0, "the occluded stretch must survive resampling"
    # 40/120s .. 43/120s maps onto 30fps frames 10-11.
    assert set(hidden) <= {9, 10, 11}, f"occlusion landed on unexpected frames: {hidden}"


def test_resample_points_is_a_noop_at_the_same_rate():
    points = np.zeros((5, 2, 3))
    valid = np.ones((5, 2), dtype=bool)
    out, out_valid = resample_points(points, valid, 30.0, 30.0)
    assert out is points and out_valid is valid


def test_output_name_keeps_subfolder_files_distinct(tmp_path):
    """data/c3d/walk.c3d and data/npz/walk.npz must not collide in one flat
    output folder."""
    root = Path("data")
    assert output_name(root / "c3d" / "walk.c3d", root) != output_name(root / "npz" / "walk.npz", root)
    assert output_name(root / "npz" / "walk.npz", root).endswith(".npz")


# ---------------------------------------------------------------------------
# Stage 1 (parameter path — no fitting, so fast enough for a test)
# ---------------------------------------------------------------------------

def test_stage1_converts_and_resamples_a_real_clip():
    source = a_real_npz()
    motion = unify_params(source, target_fps=30.0)

    assert motion.fps == pytest.approx(30.0)
    t = motion.num_frames
    assert motion.trans.shape == (t, 3)
    assert motion.global_orient.shape == (t, 3)
    assert motion.body_pose.shape == (t, 63)
    assert motion.to_array().shape == (t, 159), "canonical pose must be 159 numbers wide"
    assert np.isfinite(motion.to_array()).all()


def test_stage1_resampling_preserves_clip_duration():
    """Resampling changes how many frames a clip has, not how long it lasts."""
    source = a_real_npz()
    with np.load(source, allow_pickle=True) as raw:
        source_fps = float(raw["mocap_frame_rate" if "mocap_frame_rate" in raw.files else "mocap_framerate"])
        source_frames = raw["poses"].shape[0]

    motion = unify_params(source, target_fps=30.0)

    assert motion.num_frames / 30.0 == pytest.approx(source_frames / source_fps, abs=0.05)


# ---------------------------------------------------------------------------
# Stage 2, and the alignment training depends on
# ---------------------------------------------------------------------------

def test_stage2_control_is_index_aligned_with_pose(tmp_path):
    """The rule training relies on: control[t] sits at the same index as
    pose[t]. If these ever drift apart, the model would be conditioned on
    the wrong instant and nothing would visibly fail."""
    from src.gpsm.motionprep.extract_control import add_control_to_clip
    from src.gpsm.motionprep.unify import save_unified

    motion = unify_params(a_real_npz(), target_fps=30.0)
    unified = tmp_path / "clip.npz"
    save_unified(motion, unified, source_path="test", source_format=NPZ_PARAMS)

    out = tmp_path / "clip_with_control.npz"
    summary = add_control_to_clip(unified, out)

    with np.load(out, allow_pickle=True) as data:
        assert data["control"].shape[0] == data["trans"].shape[0], "one control row per pose frame"
        assert data["control"].shape[1] == 14
        assert data["control_valid"].shape == (data["trans"].shape[0],)
        # Stage 1's fields must survive untouched.
        assert data["body_pose"].shape[1] == 63
        assert float(data["fps"]) == pytest.approx(30.0)

    assert summary["n_valid_control_frames"] < summary["n_frames"], (
        "the tail of a clip has no full look-ahead and must be marked invalid"
    )


def test_stage2_marks_exactly_the_unpredictable_tail_invalid(tmp_path):
    """The last 0.5s (the longest look-ahead) has no future to measure, so
    at 30 fps exactly 15 frames should be flagged."""
    from src.gpsm.motionprep.extract_control import add_control_to_clip
    from src.gpsm.motionprep.unify import save_unified

    motion = unify_params(a_real_npz(), target_fps=30.0)
    unified = tmp_path / "clip.npz"
    save_unified(motion, unified, source_path="test", source_format=NPZ_PARAMS)
    summary = add_control_to_clip(unified, tmp_path / "out.npz")

    assert summary["n_frames"] - summary["n_valid_control_frames"] == 15


def test_pipeline_preserves_the_motion_it_started_with(tmp_path):
    """End-to-end sanity: a clip's turn, measured from the raw file, should
    survive unification and resampling. This is the check that would catch a
    broken rotation conversion or a resampling bug, which unit tests on
    shapes alone would not."""
    from src.gpsm.control.inspect_control import turn_while_moving_degrees
    from src.gpsm.control.root_trajectory import load_root_trajectory, root_trajectory_from_canonical

    source = a_real_npz()
    raw_turn = turn_while_moving_degrees(load_root_trajectory(source))
    if not np.isfinite(raw_turn):
        pytest.skip("this clip never moves fast enough to measure a turn")

    motion = unify_params(source, target_fps=30.0)
    prepped_turn = turn_while_moving_degrees(
        root_trajectory_from_canonical(motion.trans, motion.global_orient, motion.fps)
    )

    assert prepped_turn == pytest.approx(raw_turn, abs=5.0)


# ---------------------------------------------------------------------------
# The one-call Kaggle runner
# ---------------------------------------------------------------------------

def test_prepare_runs_both_stages_and_returns_the_training_folder(tmp_path):
    """The notebook calls one function and trains on what it returns, so the
    returned path must actually hold Stage 2 output, not Stage 1's."""
    from src.gpsm.motionprep.kaggle import prepare

    out = prepare(str(DATA_DIR / "npz"), out_dir=str(tmp_path), limit=2)

    assert Path(out) == tmp_path / "with_control"
    clips = sorted(Path(out).glob("*.npz"))
    assert clips, "Stage 2 produced nothing"

    clip = np.load(clips[0], allow_pickle=True)
    assert "control" in clip and "control_valid" in clip
    assert clip["control"].shape[0] == clip["trans"].shape[0], (
        "control must be index-aligned with pose"
    )


def test_prepare_resumes_instead_of_redoing_work(tmp_path):
    """Kaggle sessions time out, so the run has to be restartable: a second
    call must not redo finished files."""
    from src.gpsm.motionprep.kaggle import prepare

    prepare(str(DATA_DIR / "npz"), out_dir=str(tmp_path))
    first = sorted(p.name for p in (tmp_path / "with_control").glob("*.npz"))
    stamps = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "with_control").glob("*.npz")}

    prepare(str(DATA_DIR / "npz"), out_dir=str(tmp_path))
    second = sorted(p.name for p in (tmp_path / "with_control").glob("*.npz"))

    assert first == second, "a complete re-run must not add or lose clips"
    after = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "with_control").glob("*.npz")}
    assert after == stamps, "finished clips must not be rewritten on a re-run"


def test_limit_processes_the_next_files_not_the_same_ones(tmp_path):
    """`limit` caps the work *remaining*, not the total. That is what makes it
    usable on Kaggle: call it again after a timeout and it chips away at the
    rest instead of grinding over what is already done."""
    from src.gpsm.motionprep.kaggle import prepare

    prepare(str(DATA_DIR / "npz"), out_dir=str(tmp_path), limit=2)
    first = sorted(p.name for p in (tmp_path / "with_control").glob("*.npz"))

    prepare(str(DATA_DIR / "npz"), out_dir=str(tmp_path), limit=2)
    second = sorted(p.name for p in (tmp_path / "with_control").glob("*.npz"))

    assert len(first) == 2
    assert len(second) == 4, "a second call should advance, not repeat"
    assert set(first) < set(second), "and it must keep what was already done"


def test_summarize_survives_a_run_that_never_started(tmp_path, capsys):
    """Called on its own after a crash, it should report that there is
    nothing yet rather than raise."""
    from src.gpsm.motionprep.kaggle import summarize

    summarize(str(tmp_path))
    assert "no manifest yet" in capsys.readouterr().out
