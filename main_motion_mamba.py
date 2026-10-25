"""Training entrypoint for MotionMamba — mirrors main.py (StateMachineGPT's
entrypoint), with a separate checkpoint prefix so the two don't overwrite
each other. See Task 1.6 in doc/Mamba_Implementation_Task_Breakdown.docx:
this is a correctness check ("does the loss curve behave sensibly?"), not
the final, full-budget training run.
"""

from src.gpsm.ssm.motion_mamba_trainer import MotionMambaTrainer
from torch.utils.data import DataLoader
from src.gpsm.utils.dataset import MultiSMPLXNPZNextPoseDataset
from src.gpsm.utils.utils import load_yaml_config


# Architecture + core hyperparameters: configs/motion_mamba.yaml, not
# hardcoded here -- single source of truth, shared with
# main_motion_mamba_ddp.py / main_motion_mamba_sft.py /
# main_motion_mamba_sft_ddp.py, so they can't drift apart from each other.
config = load_yaml_config("configs/motion_mamba.yaml")

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
