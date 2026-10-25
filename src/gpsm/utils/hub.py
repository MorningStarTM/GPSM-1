"""Push a MotionMamba/StateMachineGPT checkpoint to the Hugging Face Hub.

Separate from the trainer on purpose: the trainer's job is producing a good
checkpoint, this module's job is publishing one, and keeping them apart
means the trainer never needs a Hub token to run. Call this explicitly,
after training, the same way ``download_kit_to_kaggle.py``'s
``publish_to_kaggle`` is a separate step from downloading.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.gpsm.utils.logger import logger


def _hub_token(explicit_token: str = None) -> str:
    """Resolve a Hugging Face token: an explicit argument first, then a
    Kaggle notebook Secret, then the HF_TOKEN env var (e.g. if already
    logged in via `huggingface-cli login`)."""
    if explicit_token:
        return explicit_token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "No Hugging Face token found. Pass token=..., add an HF_TOKEN "
            "Kaggle Secret (Add-ons > Secrets, get the value from "
            "huggingface.co/settings/tokens), or set the HF_TOKEN env var."
        )
    return token


def push_checkpoint_to_hub(
    checkpoint_prefix: str,
    repo_id: str,
    token: str = None,
    private: bool = True,
    include_optimizer: bool = False,
    commit_message: str = "Upload checkpoint",
) -> str:
    """Upload a ``save_safetensors()`` checkpoint to a Hugging Face model repo.

    Args:
        checkpoint_prefix: The same path prefix passed to
            ``model.save_safetensors(...)`` -- e.g. ``"checkpoints/best_motion_mamba"``.
            Uploads ``<prefix>.safetensors`` and ``<prefix>.json``.
        repo_id:           ``"<your-username>/<model-name>"``.
        token:              Hugging Face access token (needs write access).
            If omitted, resolved via Kaggle Secrets / HF_TOKEN env var.
        private:            Create the repo as private if it doesn't exist yet.
        include_optimizer:  Also upload ``<prefix>.optim.safetensors`` (raw
            optimizer momentum state) if present. Off by default -- that
            file is for locally resuming training, not part of a model
            release, and most repos don't expect to find it there.
        commit_message:     Message for this Hub commit.

    Returns:
        The URL of the uploaded repo.
    """
    from huggingface_hub import HfApi

    base = os.path.splitext(checkpoint_prefix)[0]
    weights_path = Path(base + ".safetensors")
    config_path = Path(base + ".json")
    for p in (weights_path, config_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- did training actually save a checkpoint "
                f"at this prefix? (save_safetensors writes "
                f"<prefix>.safetensors + <prefix>.json together)"
            )

    api = HfApi(token=_hub_token(token))
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    uploads = [(weights_path, "model.safetensors"), (config_path, "config.json")]
    if include_optimizer:
        optim_path = Path(base + ".optim.safetensors")
        if optim_path.exists():
            uploads.append((optim_path, "optimizer.safetensors"))

    for local_path, remote_name in uploads:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"Uploaded {local_path} -> {repo_id}/{remote_name}")

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Done: {url}")
    return url
