import torch

from src.gpsm.smplx.fitting.body import BodyParams


def test_body_forward_shapes(smplx_body):
    params = BodyParams.zeros(batch_size=3)
    out = smplx_body.forward(params, return_verts=True)
    assert out.joints.shape[0] == 3
    assert out.joints.shape[2] == 3
    assert out.vertices.shape == (3, 10475, 3)


def test_body_gradients_flow(smplx_body):
    params = BodyParams.zeros(batch_size=2)
    params.body_pose.requires_grad_(True)
    out = smplx_body.forward(params)
    loss = out.joints.sum()
    loss.backward()
    assert params.body_pose.grad is not None
    assert torch.isfinite(params.body_pose.grad).all()
    assert params.body_pose.grad.abs().sum() > 0


def test_named_joint_lookup(smplx_body):
    idx = smplx_body.joint_index("pelvis")
    assert isinstance(idx, int)
    params = BodyParams.zeros(batch_size=1)
    out = smplx_body.forward(params)
    # SMPL-X's own *unposed* rest frame (global_orient=0) is Y-up locally —
    # Z-up only applies to the *world* frame once a real AMASS-style
    # global_orient rotates the body upright (see doc/Mocap_Unification
    # section 6.4, which is about posed world `trans`, not this local rest
    # frame). So compare Y here, not Z.
    pelvis_y = out.joints[0, idx, 1].item()
    head_y = out.joints[0, smplx_body.joint_index("head"), 1].item()
    ankle_y = out.joints[0, smplx_body.joint_index("left_ankle"), 1].item()
    assert ankle_y < pelvis_y < head_y


def test_betas_change_body_size(smplx_body):
    params_a = BodyParams.zeros(batch_size=1)
    params_b = BodyParams.zeros(batch_size=1)
    params_b.betas = torch.tensor([2.0] + [0.0] * 9, dtype=torch.float64)
    out_a = smplx_body.forward(params_a)
    out_b = smplx_body.forward(params_b)
    head_a = out_a.joints[0, smplx_body.joint_index("head"), 2].item()
    head_b = out_b.joints[0, smplx_body.joint_index("head"), 2].item()
    assert abs(head_a - head_b) > 1e-4
