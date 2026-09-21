import torch

from src.gpsm.smplx.fitting.priors import GaussianJointPrior


def test_neutral_pose_has_lower_nll_than_extreme_pose():
    prior = GaussianJointPrior()
    neutral = torch.zeros(1, 63, dtype=torch.float64)
    extreme = torch.full((1, 63), 3.0, dtype=torch.float64)
    assert prior.nll(neutral).item() < prior.nll(extreme).item()


def test_hinge_joints_penalized_more_on_non_bend_axes():
    prior = GaussianJointPrior()
    pose_a = torch.zeros(1, 63, dtype=torch.float64)
    pose_b = pose_a.clone()
    knee_joint_idx = 3  # left_knee within the 21-joint body_pose block
    pose_b[0, knee_joint_idx * 3 + 0] = 0.8  # twist axis, tightly limited for a hinge
    assert prior.nll(pose_b).item() > prior.nll(pose_a).item()


def test_nll_is_batched_and_differentiable():
    prior = GaussianJointPrior()
    pose = torch.zeros(4, 63, dtype=torch.float64, requires_grad=True)
    loss = prior.nll(pose)
    assert loss.dim() == 0
    loss.backward()
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()
