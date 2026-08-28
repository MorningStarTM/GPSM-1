import json
from torch.utils.data import DataLoader
from src.gpsm.experiment_model.sm_gpt_sft_trainer import SFTTrainer
from src.gpsm.utils.dataset import MultiSMPLXNPZSFTDataset


# ---------------------------------------------------------------------------
# SFT phase: fine-tune an already-pretrained StateMachineGPT checkpoint
# (see main.py for that base training run) on multi-step autoregressive
# rollout, instead of single-step teacher forcing. See SFTTrainer /
# MultiSMPLXNPZSFTDataset for why this is a distinct phase.
# ---------------------------------------------------------------------------

PRETRAINED_CHECKPOINT = "checkpoints/best_sm"   # base model to fine-tune from
N_FRAMES               = 8                        # rollout length supervised per sample

# Reuse the pretrained checkpoint's own config so the fresh model SFTTrainer
# builds matches the saved weights exactly — load_safetensors requires an
# identical config and raises otherwise.
with open(f"{PRETRAINED_CHECKPOINT}.json", "r") as f:
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
    )

if ds.D != config["state_dim"]:
    raise ValueError(
        f"Dataset feature width D={ds.D} does not match pretrained "
        f"state_dim={config['state_dim']} — check feature_set matches what "
        "the base checkpoint was trained with."
    )

train_loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=True)

print(f"SFT dataset: D={ds.D}, n_frames={N_FRAMES}, {len(ds)} samples across "
      f"{ds.file_stats()['num_files']} files")

trainer = SFTTrainer(config, pretrained_checkpoint=PRETRAINED_CHECKPOINT)
history = trainer.train(
    train_loader,
    val_loader=train_loader,
    epochs=20,
    log_every=1,
    best_path="checkpoints/best_sm_sft",
)
