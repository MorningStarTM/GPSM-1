"""Training entrypoint for MotionMamba — mirrors main.py (StateMachineGPT's
entrypoint), with a separate checkpoint prefix so the two don't overwrite
each other. See Task 1.6 in doc/Mamba_Implementation_Task_Breakdown.docx:
this is a correctness check ("does the loss curve behave sensibly?"), not
the final, full-budget training run.
"""

import math

from src.gpsm.ssm.motion_mamba_trainer import MotionMambaTrainer
from torch.utils.data import DataLoader
from src.gpsm.utils.dataset import MultiSMPLXNPZNextPoseDataset


config = {
    "n_embd": 256,
    "n_layers": 4,
    "learning_rate": 1e-4,
    "dropout": 0.1,
    "loss_scale": 1.0,
    # Mamba-specific (see motion_mamba.py) — no block_size/max_timestep/
    # n_head here, unlike StateMachineGPT's config: an SSM needs none of
    # those (see MotionMamba's docstring for why).
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    # dt_rank is required explicitly (MotionMamba does not auto-compute
    # it) — ceil(n_embd / 16) is the same heuristic the reference repos
    # default to, used here for a sensible first value, not because it is
    # required to be exactly this.
    "dt_rank": math.ceil(256 / 16),
}

# Same dataset call as main.py, so this trains on the same data
# (strict_dim=False keeps only the files matching the first-loaded file's
# width — in practice the three state_dim=159 files; see
# doc/GPSM_variable_dimension_handling.docx for why that happens).
ds = MultiSMPLXNPZNextPoseDataset(
    "data",
    k=8,
    feature_set="poses+trans",
    include_betas=False,
    include_expression=False,
    include_face=False,
    prefer_joints_if_available=False,
    normalize=True,
    preload=False,
    strict_dim=False,
)

train_loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=True)

config["state_dim"] = ds.D
print(f"Using state_dim={ds.D} (from dataset), {len(ds)} samples across {ds.file_stats()['num_files']} files")

trainer = MotionMambaTrainer(config)
history = trainer.train(
    train_loader,
    val_loader=train_loader,
    epochs=15,
    log_every=1,
    best_path="checkpoints/best_motion_mamba",
)

print("train_loss per epoch:", [f"{v:.6f}" for v in history["train_loss"]])
