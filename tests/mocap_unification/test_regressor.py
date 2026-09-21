import numpy as np
import pytest
import torch

from src.gpsm.smplx.fitting.body import BodyParams
from src.gpsm.smplx.layouts import get_layout
from src.gpsm.smplx.regressor import (
    NullRegressor, RegressorEstimate, WeakPerspectiveCamera, refine_from_2d,
)


def test_null_regressor_raises_informatively():
    with pytest.raises(NotImplementedError, match="SMPLer-X"):
        NullRegressor().predict(np.zeros((4, 4, 3), dtype=np.uint8))


def test_weak_perspective_projection_basic():
    cam = WeakPerspectiveCamera(focal=1000.0, principal_point=[500.0, 500.0])
    pts = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=torch.float64)
    proj = cam.project(pts)
    assert torch.allclose(proj[0], torch.tensor([500.0, 500.0], dtype=torch.float64))
    assert proj[1, 0] > proj[0, 0]  # point offset in +x projects to the right


def test_refine_from_2d_improves_on_bad_initial_guess(smplx_body):
    layout = get_layout("coco_17")
    idx = layout.smplx_indices()
    rng = np.random.default_rng(1)

    gt_pose = rng.normal(0, 0.2, 63)
    gt_orient = rng.normal(0, 0.1, 3)
    gt_transl = np.array([0.0, 0.0, 3.0])
    gt_betas = rng.normal(0, 0.3, 10)

    with torch.no_grad():
        p = BodyParams(
            betas=torch.tensor(gt_betas), global_orient=torch.tensor(gt_orient).unsqueeze(0),
            body_pose=torch.tensor(gt_pose).unsqueeze(0),
            left_hand_pose=torch.zeros(1, 45, dtype=torch.float64),
            right_hand_pose=torch.zeros(1, 45, dtype=torch.float64),
            transl=torch.tensor(gt_transl).unsqueeze(0),
        )
        joints3d = smplx_body.forward(p).joints[0, idx, :]

    cam = WeakPerspectiveCamera(focal=1000.0, principal_point=[500.0, 500.0])
    kp2d = cam.project(joints3d).numpy()
    conf = np.ones(len(idx))

    bad_estimate = RegressorEstimate(
        betas=np.zeros(10), global_orient=np.zeros(3), body_pose=np.zeros(63),
        left_hand_pose=np.zeros(45), right_hand_pose=np.zeros(45),
        camera_translation=np.array([0.0, 0.0, 2.5]),
    )
    refined = refine_from_2d(bad_estimate, kp2d, conf, idx, smplx_body, cam, iters=40)

    err_before = np.abs(bad_estimate.body_pose - gt_pose).mean()
    err_after = np.abs(refined.body_pose - gt_pose).mean()
    assert err_after <= err_before
    assert np.isfinite(refined.body_pose).all()
    assert np.isfinite(refined.global_orient).all()
