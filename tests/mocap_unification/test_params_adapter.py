import glob
import os

import numpy as np
import pytest

from src.gpsm.smplx.adapters.params import (
    convert_smplh, convert_smplx165, detect_param_format, ingest_params,
    UnsupportedFormatError,
)
from src.gpsm.smplx.schema import CANONICAL_DIM

REAL_DATA_DIR = "E:/github_clone/GPSM-1/data/"
_real_files = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "*.npz"))) if os.path.isdir(REAL_DATA_DIR) else []

requires_real_data = pytest.mark.skipif(
    not _real_files, reason=f"Real sample data not found at {REAL_DATA_DIR}"
)


@requires_real_data
@pytest.mark.parametrize("path", _real_files, ids=[os.path.basename(p) for p in _real_files])
def test_real_sample_files_convert_without_error(path):
    with np.load(path, allow_pickle=True) as d:
        data = dict(d)
    motion = ingest_params(data, source_name=os.path.basename(path))
    arr = motion.to_array()
    assert arr.shape[1] == CANONICAL_DIM
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()
    assert motion.fps > 0
    assert motion.num_frames == arr.shape[0]


@requires_real_data
def test_frozen_hand_pose_detected_in_every_real_file():
    """Regression test for the specific finding in
    doc/Mocap_Unification_Design.docx section 6.3: every real sample file's
    hand pose is a frozen MANO rest pose, not observed motion, and the
    adapter must mark it unobserved rather than treating it as real data."""
    for path in _real_files:
        with np.load(path, allow_pickle=True) as d:
            data = dict(d)
        motion = ingest_params(data, source_name=os.path.basename(path))
        assert motion.part_mask[:, 2].max() == 0.0, f"{path}: left hand should be masked unobserved"
        assert motion.part_mask[:, 3].max() == 0.0, f"{path}: right hand should be masked unobserved"


def test_detect_param_format_widths():
    t = 5
    assert detect_param_format({"poses": np.zeros((t, 156))}) == "smplh"
    assert detect_param_format({"poses": np.zeros((t, 165))}) == "smplx165"
    assert detect_param_format({"poses": np.zeros((t, 159))}) == "canonical159"
    with pytest.raises(UnsupportedFormatError):
        detect_param_format({"poses": np.zeros((t, 999))})
    with pytest.raises(UnsupportedFormatError):
        detect_param_format({"nonsense": 1})


def test_smplh_slicing_matches_known_layout():
    t = 4
    poses = np.arange(t * 156, dtype=np.float64).reshape(t, 156)
    data = {"poses": poses, "trans": np.zeros((t, 3)), "mocap_framerate": 120.0}
    motion = convert_smplh(data, fps=120.0)
    assert np.array_equal(motion.global_orient, poses[:, 0:3])
    assert np.array_equal(motion.body_pose, poses[:, 3:66])
    assert np.array_equal(motion.left_hand_pose, poses[:, 66:111])
    assert np.array_equal(motion.right_hand_pose, poses[:, 111:156])


def test_smplx165_drops_jaw_and_eyes():
    t = 4
    poses = np.arange(t * 165, dtype=np.float64).reshape(t, 165)
    data = {"poses": poses, "mocap_frame_rate": 100.0}
    motion = convert_smplx165(data, fps=100.0)
    assert np.array_equal(motion.body_pose, poses[:, 3:66])
    hands = poses[:, 75:165]
    assert np.array_equal(motion.left_hand_pose, hands[:, :45])
    assert np.array_equal(motion.right_hand_pose, hands[:, 45:])
    assert motion.meta["dropped_face_dims"] == 9


def test_ingest_requires_fps_when_absent():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        ingest_params({"poses": np.zeros((3, 156))})


def test_betas_padded_and_truncated():
    from src.gpsm.smplx.adapters.params import _get_betas
    short = _get_betas({"betas": np.ones(4)})
    assert short.shape == (10,)
    assert np.array_equal(short[:4], np.ones(4)) and np.array_equal(short[4:], np.zeros(6))

    long = _get_betas({"betas": np.arange(16)})
    assert long.shape == (10,)
    assert np.array_equal(long, np.arange(10))
