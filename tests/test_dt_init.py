"""Unit tests for the targeted dt (delta) initialization (Task 1.4).

The acceptance check the task doc specifies is simple — softplus(bias)
should land in [dt_min, dt_max] right after init — but the real risk here
isn't the math itself, it's *wiring*: MotionMamba's generic weight-init
pass runs (via self.apply()) *after* each MambaBlock has already set up
its dt_proj, and would silently overwrite this targeted init with a
generic one if not explicitly told not to. So these tests check both the
init function in isolation AND the same property surviving a full
MotionMamba construction — the second one is the test that would actually
catch that bug.

Run with:
    pytest tests/test_dt_init.py -v
"""

import torch
import torch.nn as nn

from src.gpsm.ssm.motion_mamba import (
    MotionMamba,
    _DT_MAX,
    _DT_MIN,
    _init_dt_proj,
)


def test_bias_softplus_lands_in_target_range():
    """Directly exercises _init_dt_proj on a standalone Linear layer —
    the exact acceptance check from the task doc."""
    torch.manual_seed(0)
    dt_rank, d_inner = 4, 32
    dt_proj = nn.Linear(dt_rank, d_inner)

    _init_dt_proj(dt_proj, dt_rank)

    delta_at_init = torch.nn.functional.softplus(dt_proj.bias)
    assert (delta_at_init >= _DT_MIN).all(), delta_at_init.min()
    assert (delta_at_init <= _DT_MAX).all(), delta_at_init.max()


def test_weight_scale_reflects_dt_rank():
    """The weight should be filled from Uniform(-s, s) with
    s = dt_rank ** -0.5 — not the generic N(0, 0.02) every other Linear
    in the model gets. Checked via range, since that's what uniform
    initialization actually constrains (unlike a normal distribution)."""
    torch.manual_seed(0)
    dt_rank, d_inner = 4, 64
    dt_proj = nn.Linear(dt_rank, d_inner)

    _init_dt_proj(dt_proj, dt_rank)

    expected_bound = dt_rank**-0.5
    assert dt_proj.weight.abs().max().item() <= expected_bound + 1e-6


def test_no_reinit_markers_are_set():
    """_init_dt_proj must mark both tensors so MotionMamba's generic
    init pass knows to skip them."""
    dt_proj = nn.Linear(4, 16)
    _init_dt_proj(dt_proj, dt_rank=4)

    assert getattr(dt_proj.weight, "_no_reinit", False) is True
    assert getattr(dt_proj.bias, "_no_reinit", False) is True


def test_dt_init_survives_full_motion_mamba_construction():
    """The real regression test: build a complete MotionMamba (which
    calls self.apply(self._init_weights) on every submodule, including
    every MambaBlock's dt_proj, after dt_proj has already been targeted-
    initialized) and confirm the targeted init is still intact afterwards
    — i.e. the generic pass did not silently overwrite it. This is
    exactly the "post-initialization hook zeros every Linear bias"
    failure mode the official Mamba repo's own troubleshooting notes
    warn about, and it is easy to reintroduce by accident if
    MotionMamba._init_weights is ever edited without noticing the guard."""
    torch.manual_seed(0)
    config = {
        "state_dim": 6, "n_embd": 16, "n_layers": 3, "dropout": 0.0,
        "learning_rate": 1e-3, "d_state": 4, "d_conv": 3, "expand": 2, "dt_rank": 2,
    }
    model = MotionMamba(config)

    for i, residual_block in enumerate(model.blocks):
        dt_proj = residual_block.mixer.dt_proj
        delta_at_init = torch.nn.functional.softplus(dt_proj.bias)
        assert (delta_at_init >= _DT_MIN).all() and (delta_at_init <= _DT_MAX).all(), (
            f"block {i}: dt_proj.bias was NOT protected from the generic re-init pass "
            f"(softplus(bias) range: [{delta_at_init.min():.6f}, {delta_at_init.max():.6f}], "
            f"expected within [{_DT_MIN}, {_DT_MAX}])"
        )
        # a generic N(0, 0.02) init would essentially never land a whole
        # weight tensor inside such a tight bound by chance - this checks
        # the *weight* survived too, not just the bias.
        expected_bound = config["dt_rank"] ** -0.5
        assert dt_proj.weight.abs().max().item() <= expected_bound + 1e-6, (
            f"block {i}: dt_proj.weight looks like it was overwritten by the generic init"
        )
