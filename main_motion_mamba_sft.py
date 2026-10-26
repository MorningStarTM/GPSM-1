"""SFT phase: fine-tune MotionMamba on multi-step autoregressive rollout,
instead of single-step teacher forcing. See MotionMambaSFTTrainer /
CachedSFTDataset for why this is a distinct training objective, and
src/gpsm/ssm/motion_mamba_sft_trainer.py's module docstring for the full
reasoning.

Does NOT require a pretrained checkpoint -- pretraining (main_motion_mamba.py
/ main_motion_mamba_ddp.py) was decided to be skipped in favor of starting
SFT directly from a fresh, randomly-initialized model. If no SFT checkpoint
exists yet either, this starts from scratch using configs/motion_mamba.yaml
for the architecture. If you ever DO want to warm-start from an existing
pretrained checkpoint instead, set BASE_CHECKPOINT below.

Resuming works the same way either way: if best_motion_mamba_sft already
exists (e.g. a later chunk/session), this picks it back up automatically --
that part is unaffected by skipping pretraining.

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
SFT_CHECKPOINT = "/kaggle/working/checkpoints/best_motion_mamba_sft"  # this run resumes from + overwrites this
BASE_CHECKPOINT = None  # optional: a pretrained checkpoint prefix to warm-start from, instead of training from scratch
N_FRAMES = SFT_CONFIG["n_frames"]          # rollout length supervised per sample
CKPT_DIR = "/kaggle/working/checkpoints"

if os.path.exists(f"{SFT_CHECKPOINT}.safetensors"):
    # Resuming an in-progress SFT run -- reuse ITS saved config so the
    # fresh model MotionMambaSFTTrainer builds matches the saved weights
    # exactly (load_safetensors requires an identical config and raises
    # otherwise).
    resume_from = SFT_CHECKPOINT
    with open(f"{resume_from}.json", "r") as f:
        config = json.load(f)["config"]
    print(f"Resuming SFT from: {resume_from}")
elif BASE_CHECKPOINT and os.path.exists(f"{BASE_CHECKPOINT}.safetensors"):
    # Optional warm start from an existing pretrained checkpoint, if you
    # set BASE_CHECKPOINT above.
    resume_from = BASE_CHECKPOINT
    with open(f"{resume_from}.json", "r") as f:
        config = json.load(f)["config"]
    print(f"Starting SFT from pretrained checkpoint: {resume_from}")
else:
    # Fresh start, no pretraining phase: architecture comes from
    # configs/motion_mamba.yaml, state_dim filled in below once the
    # dataset is loaded. No block_size check here, unlike sft_main.py's
    # StateMachineGPT version -- an SSM has no such concept.
    resume_from = None
    config = load_yaml_config("configs/motion_mamba.yaml")
    print("Starting SFT from scratch (no pretrained checkpoint) -- "
          "architecture from configs/motion_mamba.yaml")

train_ds = CachedSFTDataset(f"{CACHE_DIR}/train.pt", n_frames=N_FRAMES)
val_ds = CachedSFTDataset(f"{CACHE_DIR}/val.pt", n_frames=N_FRAMES)

if "state_dim" not in config:
    config["state_dim"] = train_ds.state_dim
elif train_ds.state_dim != config["state_dim"]:
    raise ValueError(
        f"Cache state_dim={train_ds.state_dim} does not match the resumed "
        f"checkpoint's state_dim={config['state_dim']} -- check the cache "
        "was built from the same data that checkpoint was trained on."
    )

train_loader = DataLoader(train_ds, batch_size=SFT_CONFIG["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=SFT_CONFIG["batch_size"], shuffle=False, drop_last=True)

print(f"SFT dataset: state_dim={train_ds.state_dim}, n_frames={N_FRAMES}, "
      f"train={len(train_ds)} samples across {train_ds.file_stats()['num_files']} files, "
      f"val={len(val_ds)} samples across {val_ds.file_stats()['num_files']} files")

trainer = MotionMambaSFTTrainer(config, ckpt_dir=CKPT_DIR, pretrained_checkpoint=resume_from)
history = trainer.train(
    train_loader,
    val_loader=val_loader,
    epochs=SFT_CONFIG["epochs"],
    log_every=1,
    best_path=SFT_CHECKPOINT,
)

print("train_loss per epoch:", [f"{v:.6f}" for v in history["train_loss"]])
print("val_loss per epoch:", [f"{v:.6f}" for v in history["val_loss"]])
