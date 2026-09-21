import math

import torch

from src.gpsm.smplx.fitting.losses import (
    geman_mcclure, geodesic_so3_distance, joint_limit_penalty, robust_data_term,
    shape_regularization, temporal_smoothness, translation_acceleration,
)


def test_geman_mcclure_bounded():
    small = geman_mcclure(torch.tensor(0.0001))
    huge = geman_mcclure(torch.tensor(1e9))
    assert small.item() < 0.01
    assert huge.item() <= 1.0 and huge.item() > 0.99


def test_geodesic_distance_known_angles():
    zero = torch.zeros(1, 3, dtype=torch.float64)
    quarter_turn = torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=torch.float64)
    d0 = geodesic_so3_distance(zero, zero)
    d90 = geodesic_so3_distance(zero, quarter_turn)
    assert d0.item() < 1e-5
    assert abs(d90.item() - math.pi / 2) < 1e-6


def test_geodesic_distance_handles_wraparound():
    zero = torch.zeros(1, 3, dtype=torch.float64)
    near_2pi = torch.tensor([[0.0, 0.0, 2 * math.pi - 0.01]], dtype=torch.float64)
    d = geodesic_so3_distance(zero, near_2pi)
    assert d.item() < 0.02  # true angular distance is ~0.01, not ~2*pi


def test_robust_data_term_downweights_outliers():
    pred = torch.zeros(1, 1, 3, dtype=torch.float64)
    close = torch.tensor([[[0.01, 0.0, 0.0]]], dtype=torch.float64)
    far = torch.tensor([[[500.0, 0.0, 0.0]]], dtype=torch.float64)
    conf = torch.ones(1, 1, dtype=torch.float64)
    l_close = robust_data_term(pred, close, conf)
    l_far = robust_data_term(pred, far, conf)
    assert l_close.item() < l_far.item()
    assert l_far.item() < 1.01  # bounded, doesn't blow up


def test_robust_data_term_ignores_zero_confidence():
    pred = torch.zeros(1, 2, 3, dtype=torch.float64)
    target = torch.tensor([[[0.0, 0.0, 0.0], [999.0, 999.0, 999.0]]], dtype=torch.float64)
    conf = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    loss = robust_data_term(pred, target, conf)
    assert loss.item() < 1e-6


def test_temporal_smoothness_zero_for_constant_and_positive_otherwise():
    constant = torch.zeros(5, 21, 3, dtype=torch.float64)
    assert temporal_smoothness(constant).item() < 1e-8

    varying = torch.zeros(5, 21, 3, dtype=torch.float64)
    varying[:, 0, 2] = torch.linspace(0, 1, 5, dtype=torch.float64)
    assert temporal_smoothness(varying).item() > 0


def test_translation_acceleration_zero_for_constant_velocity():
    t = torch.arange(6, dtype=torch.float64).unsqueeze(1).repeat(1, 3)  # constant velocity
    assert translation_acceleration(t).item() < 1e-10


def test_joint_limit_penalty_only_penalizes_excess():
    within = joint_limit_penalty(torch.tensor([0.5, -0.5], dtype=torch.float64), limit=1.0)
    exceeding = joint_limit_penalty(torch.tensor([2.0, -2.0], dtype=torch.float64), limit=1.0)
    assert within.item() == 0.0
    assert exceeding.item() > 0.0


def test_shape_regularization():
    assert shape_regularization(torch.zeros(10, dtype=torch.float64)).item() == 0.0
    assert shape_regularization(torch.ones(10, dtype=torch.float64)).item() == 1.0


def test_gradients_flow_through_geodesic_distance():
    pose = torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    loss = geodesic_so3_distance(pose, target).sum()
    loss.backward()
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()
