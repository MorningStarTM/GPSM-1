"""Tests for control-conditioned MotionMamba (MP-6).

The wiring being checked: each input frame is ``concat(pose, control)``, the
output stays pose-only, and rollout has to be *told* the control for every
frame it generates rather than inventing it.

Run with:
    pytest tests/test_motion_mamba_control.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.gpsm.ssm.motion_mamba import MotionMamba

CLIPS_DIR = Path(__file__).resolve().parent.parent / "processed" / "with_control"

BASE_CONFIG = {
    "state_dim": 12, "n_embd": 16, "n_layers": 2, "dropout": 0.0,
    "learning_rate": 1e-3, "d_state": 8, "d_conv": 3, "expand": 2, "dt_rank": 2,
}


def make_model(control_dim: int) -> MotionMamba:
    model = MotionMamba({**BASE_CONFIG, "control_dim": control_dim})
    model.to(model.device)
    return model


def test_control_dim_defaults_to_zero_so_old_configs_still_work():
    """A config written before control existed must keep working unchanged:
    input width equals state_dim, and rollout free-runs on n_steps."""
    model = MotionMamba(BASE_CONFIG)   # note: no control_dim key at all
    model.to(model.device)

    assert model.config["control_dim"] == 0
    assert model.config["input_dim"] == BASE_CONFIG["state_dim"]

    out = model.forward(torch.randn(2, 5, 12).to(model.device))
    assert out.shape == (2, 5, 12)


def test_input_grows_by_control_but_output_stays_pose_only():
    """The model consumes pose+control and predicts pose. It never predicts
    control — that comes from the player."""
    model = make_model(control_dim=14)

    assert model.config["input_dim"] == 12 + 14

    out = model.forward(torch.randn(2, 5, 26).to(model.device))
    assert out.shape == (2, 5, 12), "output must be pose-width, not input-width"


def test_wrong_input_width_is_rejected_with_a_useful_message():
    model = make_model(control_dim=14)
    with pytest.raises(ValueError, match="input_dim"):
        model.forward(torch.randn(2, 5, 12).to(model.device))  # pose only, control missing


def test_rollout_refuses_to_invent_control():
    """With control configured, rollout cannot run without being told where
    the motion should go — in a game that signal comes from the player, and
    the model has no basis to make it up."""
    model = make_model(control_dim=14)
    seed = torch.randn(4, 26).to(model.device)

    with pytest.raises(ValueError, match="control"):
        model.rollout(seed, n_steps=5)


def test_rollout_length_follows_the_control_given_and_returns_poses():
    model = make_model(control_dim=14)
    seed = torch.randn(4, 26).to(model.device)
    control = torch.randn(7, 14).to(model.device)

    out = model.rollout(seed, control=control)

    assert out.shape == (7, 12), "one pose out per control row in, pose-width only"
    assert torch.isfinite(out).all()


def test_control_actually_changes_what_is_generated():
    """The point of the whole exercise: different control must produce
    different motion. If these came out identical, the control input would
    be wired in but ignored, and nothing else would reveal that."""
    torch.manual_seed(0)
    model = make_model(control_dim=14)
    seed = torch.randn(4, 26).to(model.device)
    control = torch.randn(6, 14).to(model.device)

    generated = model.rollout(seed, control=control)
    generated_other = model.rollout(seed, control=-control)

    assert not torch.allclose(generated, generated_other), "control had no effect on the rollout"


def test_no_control_rollout_still_free_runs():
    model = MotionMamba(BASE_CONFIG)
    model.to(model.device)
    out = model.rollout(torch.randn(4, 12).to(model.device), n_steps=5)
    assert out.shape == (5, 12)


# ---------------------------------------------------------------------------
# Against real MotionPrep output, when it has been generated
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CLIPS_DIR.exists() or not list(CLIPS_DIR.glob("*.npz")),
                    reason="no MotionPrep Stage 2 output in processed/with_control")
def test_dataset_windows_match_what_the_model_expects():
    """End-to-end shape contract: what the dataset serves is exactly what
    the model consumes, and the target is the pose the model predicts."""
    from src.gpsm.utils.control_dataset import ControlWindowDataset

    dataset = ControlWindowDataset(str(CLIPS_DIR), k=8)
    x, y = dataset[0]

    assert x.shape == (9, dataset.input_dim)
    assert y.shape == (dataset.pose_dim,)
    assert dataset.input_dim == dataset.pose_dim + dataset.control_dim

    model = MotionMamba({
        **BASE_CONFIG, "state_dim": dataset.pose_dim, "control_dim": dataset.control_dim,
    })
    model.to(model.device)
    prediction = model.predict_next(x.to(model.device))

    assert prediction.shape == y.shape, "the model's output must match the training target"
