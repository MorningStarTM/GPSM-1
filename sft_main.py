import json
import os
from torch.utils.data import DataLoader
from src.gpsm.experiment_model.sm_gpt_sft_trainer import SFTTrainer
from src.gpsm.utils.dataset import MultiSMPLXNPZSFTDataset


# ---------------------------------------------------------------------------
# SFT phase: fine-tune an already-pretrained StateMachineGPT checkpoint
# (see main.py for that base training run) on multi-step autoregressive
# rollout, instead of single-step teacher forcing. See SFTTrainer /
# MultiSMPLXNPZSFTDataset for why this is a distinct phase.
#
# FILE_START/FILE_END select a chunk of a large file collection (e.g. ~3000
# files won't fit in one Kaggle session) — run this script once per chunk,
# e.g. (0, 500), then (500, 1000), ... Each run automatically resumes from
# the *previous* chunk's SFT checkpoint (falling back to the base pretrained
# checkpoint only the first time, when no SFT checkpoint exists yet), so
# progress carries over between chunks instead of every chunk restarting
# from the base model and overwriting the last chunk's fine-tuning.
# ---------------------------------------------------------------------------

BASE_CHECKPOINT = "checkpoints/best_sm"        # only used for the very first chunk
SFT_CHECKPOINT  = "checkpoints/best_sm_sft"     # every later chunk resumes from + overwrites this
N_FRAMES         = 8                              # rollout length supervised per sample
FILE_START       = 0                              # first file index in this chunk
FILE_END         = 500                            # one-past-last file index in this chunk

RESUME_FROM = SFT_CHECKPOINT if os.path.exists(f"{SFT_CHECKPOINT}.safetensors") else BASE_CHECKPOINT
print(f"Resuming from: {RESUME_FROM}")

# Reuse the checkpoint's own config so the fresh model SFTTrainer builds
# matches the saved weights exactly — load_safetensors requires an
# identical config and raises otherwise.
with open(f"{RESUME_FROM}.json", "r") as f:
    config = json.load(f)["config"]

# block_size must cover the seed frame + the full generated rollout.
assert config["block_size"] >= N_FRAMES + 1, (
    f"block_size={config['block_size']} must be >= N_FRAMES+1={N_FRAMES + 1}"
)

ds = MultiSMPLXNPZSFTDataset(
        "data",
        n_frames=N_FRAMES,
        feature_set="poses+trans",        # must match what the pretrained checkpoint used
        include_betas=False,
        include_expression=False,
        include_face=False,
        prefer_joints_if_available=False,
        normalize=True,
        preload=False,
        strict_dim=False,
        file_range=(FILE_START, FILE_END),
    )

if ds.D != config["state_dim"]:
    raise ValueError(
        f"Dataset feature width D={ds.D} does not match pretrained "
        f"state_dim={config['state_dim']} — check feature_set matches what "
        "the base checkpoint was trained with."
    )

train_loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=True)

print(f"SFT dataset: D={ds.D}, n_frames={N_FRAMES}, files[{FILE_START}:{FILE_END}], "
      f"{len(ds)} samples across {ds.file_stats()['num_files']} files")

trainer = SFTTrainer(config, pretrained_checkpoint=RESUME_FROM)
history = trainer.train(
    train_loader,
    val_loader=train_loader,
    epochs=20,
    log_every=1,
    best_path=SFT_CHECKPOINT,
)
