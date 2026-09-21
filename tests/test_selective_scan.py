"""Unit tests for ``SelectiveScan`` (src/gpsm/ssm/motion_mamba.py).

This is the single highest-risk piece of the whole Mamba-for-GPSM project
(see the "Risks" section of doc/Mamba_Implementation_Task_Breakdown.docx):
the discretization formulas are the two easiest lines to get subtly wrong,
and a subtle sign or ordering error would quietly corrupt every downstream
result without ever raising an exception. So this file checks the scan
against numbers computed by hand, independently of the implementation —
not just "does it run".

Run with:
    pytest tests/test_selective_scan.py -v
"""

import math

import torch

from src.gpsm.ssm.motion_mamba import SelectiveScan


def test_output_shape_and_finite():
    """A random, moderately sized problem should run end to end and
    produce a finite output of the expected shape."""
    torch.manual_seed(0)
    batch, seq_len, d_inner, d_state = 3, 9, 5, 4

    u = torch.randn(batch, seq_len, d_inner)
    delta = torch.rand(batch, seq_len, d_inner) + 0.01  # delta must stay positive
    A = -torch.rand(d_inner, d_state) - 0.1  # A must stay negative for a stable/decaying state
    B = torch.randn(batch, seq_len, d_state)
    C = torch.randn(batch, seq_len, d_state)
    D = torch.randn(d_inner)

    y = SelectiveScan()(u, delta, A, B, C, D)

    assert y.shape == (batch, seq_len, d_inner)
    assert torch.isfinite(y).all()


def test_gradients_flow_without_custom_backward():
    """No custom backward() exists for SelectiveScan — this confirms
    PyTorch's autograd differentiates straight through the hand-written
    Python loop on its own, the same way it would for any other
    nn.Module, and that every input actually receives a gradient (i.e.
    every input genuinely participates in the computation, nothing is
    silently disconnected from the output)."""
    torch.manual_seed(0)
    batch, seq_len, d_inner, d_state = 2, 6, 3, 4

    u = torch.randn(batch, seq_len, d_inner, requires_grad=True)
    delta = (torch.rand(batch, seq_len, d_inner) + 0.01).requires_grad_()
    A = (-torch.rand(d_inner, d_state) - 0.1).requires_grad_()
    B = torch.randn(batch, seq_len, d_state, requires_grad=True)
    C = torch.randn(batch, seq_len, d_state, requires_grad=True)
    D = torch.randn(d_inner, requires_grad=True)

    y = SelectiveScan()(u, delta, A, B, C, D)
    y.pow(2).mean().backward()

    for name, tensor in [("u", u), ("delta", delta), ("A", A), ("B", B), ("C", C), ("D", D)]:
        assert tensor.grad is not None, f"{name} received no gradient at all"
        assert torch.isfinite(tensor.grad).all(), f"{name}'s gradient has non-finite values"
        assert (tensor.grad != 0).any(), f"{name}'s gradient is all zero — looks disconnected"


def test_discretization_matches_hand_computed_values():
    """The core correctness check: a toy problem (batch=1, d_inner=1,
    d_state=1, 2 timesteps) small enough to compute by hand, checked
    against the implementation's output.

    By hand, with A=-0.5, delta=[1.0, 2.0], B=[1.0, 1.0], C=[1.0, 1.0],
    u=[1.0, 1.0], D=0 (skip connection isolated out — see the separate
    test below):

        A_bar(1) = exp(delta(1) * A) = exp(1.0 * -0.5) = exp(-0.5)
        h(1)     = A_bar(1) * 0 + delta(1) * B(1) * u(1) = 1.0 * 1.0 * 1.0 = 1.0
        y(1)     = C(1) * h(1) = 1.0

        A_bar(2) = exp(delta(2) * A) = exp(2.0 * -0.5) = exp(-1.0)
        h(2)     = A_bar(2) * h(1) + delta(2) * B(2) * u(2)
                 = exp(-1.0) * 1.0 + 2.0 * 1.0 * 1.0
        y(2)     = C(2) * h(2)
    """
    A_value = -0.5
    delta_values = [1.0, 2.0]
    B_values = [1.0, 1.0]
    C_values = [1.0, 1.0]
    u_values = [1.0, 1.0]

    a_bar_1 = math.exp(delta_values[0] * A_value)
    h_1 = a_bar_1 * 0.0 + delta_values[0] * B_values[0] * u_values[0]
    y_1_expected = C_values[0] * h_1

    a_bar_2 = math.exp(delta_values[1] * A_value)
    h_2 = a_bar_2 * h_1 + delta_values[1] * B_values[1] * u_values[1]
    y_2_expected = C_values[1] * h_2

    u = torch.tensor(u_values).view(1, 2, 1)
    delta = torch.tensor(delta_values).view(1, 2, 1)
    A = torch.tensor([[A_value]])
    B = torch.tensor(B_values).view(1, 2, 1)
    C = torch.tensor(C_values).view(1, 2, 1)
    D = torch.zeros(1)

    y = SelectiveScan()(u, delta, A, B, C, D)

    expected = torch.tensor([y_1_expected, y_2_expected]).view(1, 2, 1)
    assert torch.allclose(y, expected, atol=1e-6), (
        f"Selective scan does not match the hand-computed values.\n"
        f"got:      {y.flatten().tolist()}\n"
        f"expected: {expected.flatten().tolist()}"
    )


def test_skip_connection_is_added_correctly():
    """D is a direct, learned input-to-output skip that bypasses the
    state entirely — this isolates it by running the same toy problem
    twice (D=0 vs. D=0.5) and checking the difference is exactly D * u,
    elementwise, with nothing else changed."""
    u = torch.tensor([1.0, 1.0]).view(1, 2, 1)
    delta = torch.tensor([1.0, 2.0]).view(1, 2, 1)
    A = torch.tensor([[-0.5]])
    B = torch.tensor([1.0, 1.0]).view(1, 2, 1)
    C = torch.tensor([1.0, 1.0]).view(1, 2, 1)

    scan = SelectiveScan()
    y_no_skip = scan(u, delta, A, B, C, D=torch.zeros(1))
    D_value = 0.5
    y_with_skip = scan(u, delta, A, B, C, D=torch.tensor([D_value]))

    expected_difference = D_value * u
    assert torch.allclose(y_with_skip - y_no_skip, expected_difference, atol=1e-6)
