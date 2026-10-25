"""SFT phase: fine-tune an already-pretrained MotionMamba checkpoint (see
main_motion_mamba.py / main_motion_mamba_ddp.py for that base training run)
on multi-step autoregressive rollout, instead of single-step teacher
forcing. See MotionMambaSFTTrainer / CachedSFTDataset for why this is a
distinct phase, and src/gpsm/ssm/motion_mamba_sft_trainer.py's module
docstring for the full reasoning.

Reads from the same train.pt / val.pt cache written by
scripts/preprocess_motion_mamba_cache.py and already published as the
"motion-mamba-cache" Kaggle dataset -- no re-preprocessing needed.

Usage (Kaggle notebook cell):
    !python main_motion_mamba_sft.py
"""
import json
import os
from torch.utils.data import DataLoader

from src.gpsm.ssm.motion_mamba_sft_trainer import MotionMambaSFTTrainer
from src.gpsm.utils.cached_dataset import CachedSFTDataset
from src.gpsm.utils.utils import load_yaml_config

# Run settings: configs/motion_mamba_sft.yaml, not hardcoded here -- shared
# with main_motion_mamba_sft_ddp.py, see that file for n_frames=16's
# reasoning (and its fps caveat).
SFT_CONFIG = load_yaml_config("configs/motion_mamba_sft.yaml")

CACHE_DIR = "/kaggle/input/motion-mamba-cache"
BASE_CHECKPOINT = "/kaggle/working/checkpoints/best_motion_mamba"   # the pretrained checkpoint, from main_motion_mamba(_ddp).py
SFT_CHECKPOINT = "/kaggle/working/checkpoints/best_motion_mamba_sft"  # this run resumes from + overwrites this
N_FRAMES = SFT_CONFIG["n_frames"]          # rollout length supervised per sample
CKPT_DIR = "/kaggle/working/checkpoints"

RESUME_FROM = SFT_CHECKPOINT if os.path.exists(f"{SFT_CHECKPOINT}.safetensors") else BASE_CHECKPOINT
print(f"Resuming from: {RESUME_FROM}")

# Reuse the checkpoint's own config so the fresh model MotionMambaSFTTrainer
# builds matches the saved weights exactly -- load_safetensors requires an
# identical config and raises otherwise. No block_size check here, unlike
# sft_main.py's StateMachineGPT version -- an SSM has no such concept (see
# motion_mamba_sft_trainer.py's module docstring).
with open(f"{RESUME_FROM}.json", "r") as f:
    config = json.load(f)["config"]

train_ds = CachedSFTDataset(f"{CACHE_DIR}/train.pt", n_frames=N_FRAMES)
val_ds = CachedSFTDataset(f"{CACHE_DIR}/val.pt", n_frames=N_FRAMES)

if train_ds.state_dim != config["state_dim"]:
    raise ValueError(
        f"Cache state_dim={train_ds.state_dim} does not match pretrained "
        f"state_dim={config['state_dim']} -- check the cache was built from "
        "the same data the base checkpoint was trained on."
    )

train_loader = DataLoader(train_ds, batch_size=SFT_CONFIG["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=SFT_CONFIG["batch_size"], shuffle=False, drop_last=True)

print(f"SFT dataset: state_dim={train_ds.state_dim}, n_frames={N_FRAMES}, "
      f"train={len(train_ds)} samples across {train_ds.file_stats()['num_files']} files, "
      f"val={len(val_ds)} samples across {val_ds.file_stats()['num_files']} files")

trainer = MotionMambaSFTTrainer(config, ckpt_dir=CKPT_DIR, pretrained_checkpoint=RESUME_FROM)
history = trainer.train(
    train_loader,
    val_loader=val_loader,
    epochs=SFT_CONFIG["epochs"],
    log_every=1,
    best_path=SFT_CHECKPOINT,
)

print("train_loss per epoch:", [f"{v:.6f}" for v in history["train_loss"]])
print("val_loss per epoch:", [f"{v:.6f}" for v in history["val_loss"]])
