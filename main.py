from src.gpsm.experiment_model.sm_gpt_trainer import SMTrainer
from torch.utils.data import DataLoader
from src.gpsm.utils.dataset import MultiC3DNextPoseDataset, MultiSMPLXNPZNextPoseDataset
from pathlib import Path
import os


config = {
    'n_embd':256,
    'max_timestep':3000,
    'n_layers':4,
    'learning_rate':1e-4,
    'dropout':0.1,
    'block_size':11,
    "n_head": 4,              # attention heads (head_size = n_embd // n_head, derived automatically)
    'loss_scale': 1.0,        # optional loss scaling factor
}

# DATASET_DIR = Path(r"/kaggle/input/cmu-mocap")
# c3d_files = sorted(DATASET_DIR.rglob("*.c3d"))
#ds = MultiC3DNextPoseDataset(c3d_files, k=10, nan_thresh=0.20, normalize=True, strict_dim=False, preload=False)
ds = MultiSMPLXNPZNextPoseDataset(
        "data",
        k=8,
        feature_set="poses+trans",        # most common
        include_betas=False,              # keep False for motion-only GPT
        include_expression=False,
        include_face=False,
        prefer_joints_if_available=False,
        normalize=True,
        preload=False,
        strict_dim=False,
    )

train_loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=True)

# Derive state_dim from the dataset instead of hardcoding it — files with a
# different feature width than the first-loaded file are silently skipped by
# the dataset (strict_dim=False), so ds.D is the true, consistent width.
config['state_dim'] = ds.D
print(f"Using state_dim={ds.D} (from dataset), {len(ds)} samples across {ds.file_stats()['num_files']} files")

trainer = SMTrainer(config)
history = trainer.train(train_loader, val_loader=train_loader, epochs=2, log_every=1, best_path="checkpoints/best_sm")