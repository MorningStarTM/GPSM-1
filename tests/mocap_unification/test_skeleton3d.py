import numpy as np
import pytest

from src.gpsm.smplx.adapters.skeleton3d import ingest_skeleton3d
from src.gpsm.smplx.fitting.body import BodyParams
from src.gpsm.smplx.layouts import get_layout


def test_ingest_skeleton3d_auto_detects_unambiguous_layout(smplx_body):
    rng = np.random.default_rng(5)
    layout = get_layout("openpose_25")  # 25 points is unambiguous
    import torch
    params = BodyParams.zeros(batch_size=4)
    params.body_pose = torch.tensor(rng.normal(0, 0.2, (4, 63)))
    with torch.no_grad():
        joints = smplx_body.forward(params).joints.numpy()
    points = joints[:, layout.smplx_indices(), :]

    motion, quality = ingest_skeleton3d(points, fps=30.0, body=smplx_body)  # no layout= given
    assert motion.num_frames == 4
    assert quality.verdict in ("GOOD", "DEGRADED")


def test_ingest_skeleton3d_ambiguous_count_requires_explicit_layout(smplx_body):
    points = np.zeros((3, 17, 3))
    with pytest.raises(ValueError):
        ingest_skeleton3d(points, fps=30.0, body=smplx_body)  # 17 is ambiguous
    # explicit layout resolves it
    motion, _ = ingest_skeleton3d(points, fps=30.0, body=smplx_body, layout="coco_17")
    assert motion.num_frames == 3


def test_ingest_skeleton3d_rejects_bad_shape(smplx_body):
    with pytest.raises(ValueError):
        ingest_skeleton3d(np.zeros((5, 17)), fps=30.0, body=smplx_body)


def test_ingest_skeleton3d_default_confidence_is_all_observed(smplx_body):
    points = np.random.default_rng(0).normal(0, 0.3, (3, 25, 3))
    motion, quality = ingest_skeleton3d(points, fps=30.0, body=smplx_body, layout="openpose_25")
    assert quality.observed_fraction > 0.0
