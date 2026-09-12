"""Trainer for MotionMamba.

This is a minimally-adapted copy of ``sm_gpt_trainer.py``'s ``SMTrainer``
(see Task 1.6 in ``doc/Mamba_Implementation_Task_Breakdown.docx``, which
explicitly allows this rather than requiring ``SMTrainer`` itself to be
generalized). ``SMTrainer`` hardcodes ``StateMachineGPT`` in its
constructor, so it can't be reused unchanged for a different model class —
copying it and swapping the one line was judged lower-risk than editing the
existing, already-working, already-used trainer.

Every other line below (the train loop, loss, checkpointing, early
stopping, DDP handling) is identical to ``SMTrainer`` on purpose. If you
are reading this after also reading ``sm_gpt_trainer.py``, skip ahead to
``__init__`` — that is the only real difference.
"""

import math
import os

import torch
import torch.nn.functional as F

from src.gpsm.experiment_model.distributed import reduce_mean
from src.gpsm.ssm.motion_mamba import MotionMamba
from src.gpsm.utils.logger import logger


class MotionMambaTrainer:
    """Trainer for MotionMamba. See the module docstring — this mirrors
    ``SMTrainer`` with only the model class changed."""

    def __init__(self, config: dict, ckpt_dir: str = "checkpoints"):
        self.config = config
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model = MotionMamba(config)
        self.device = self.model.device
        self.model.to(self.device)
        self.model.print_param_size()

        self.optimizer = self.model.optimizer
        self._eps = 1e-8

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_ddp(self) -> bool:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return isinstance(self.model, DDP)

    def _rank(self) -> int:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0

    def _is_main(self) -> bool:
        return self._rank() == 0

    def _module(self):
        return self.model.module if self._is_ddp() else self.model

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """MSE loss between the last-timestep prediction and the target
        next frame.

        Args:
            logits : ``(B, T, D)`` or ``(T, D)``.
            y      : ``(B, D)`` or ``(D,)``.
        """
        pred = logits[:, -1, :] if logits.dim() == 3 else logits[-1]
        loss = F.mse_loss(pred, y)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {loss.item()}")
        return loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader,
        epochs: int = 10,
        val_loader=None,
        log_every: int = 1,
        best_path: str = None,
        patience: int = 10,
        grad_clip_norm: float = 1.0,
        save_every_epochs: int = None,
    ) -> dict:
        """Train the model. Same behaviour as ``SMTrainer.train()``.

        Args:
            train_loader      : DataLoader yielding ``(x, y)`` batches.
            epochs            : Total training epochs.
            val_loader        : Optional validation DataLoader (same format).
            log_every         : Log interval, in epochs.
            best_path         : Path prefix for the best checkpoint (safetensors).
            patience          : Early-stop patience (epochs without improvement).
            grad_clip_norm    : Gradient clipping max norm (``0`` disables it).
            save_every_epochs : If set, also save a checkpoint every N epochs.

        Returns:
            A history dict with ``"train_loss"`` (and ``"val_loss"`` if
            ``val_loader`` was given).
        """
        history = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        best_metric = math.inf
        best_epoch = 0
        bad_epochs = 0

        if best_path is None:
            best_path = os.path.join(self.ckpt_dir, "best_motion_mamba")

        for ep in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(ep)

            # ---- TRAIN ----
            self.model.train()
            total, n = 0.0, 0

            for batch in train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                    continue  # skip NaN/Inf batches

                logits = self.model(x)
                loss = self._compute_loss(logits, y) * self.config.get("loss_scale", 1.0)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

                total += float(loss.item())
                n += 1

            train_loss = reduce_mean(total / max(n, 1), self.device)
            history["train_loss"].append(train_loss)

            # ---- VALIDATION ----
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                vtotal, vn = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                        logits = self.model(x)
                        vtotal += float(self._compute_loss(logits, y).item())
                        vn += 1
                val_loss = reduce_mean(vtotal / max(vn, 1), self.device)
                history["val_loss"].append(val_loss)

            # ---- CHECKPOINT + EARLY STOPPING ----
            current_metric = val_loss if val_loader is not None else train_loss

            if current_metric < (best_metric - 1e-8):
                best_metric = current_metric
                best_epoch = ep
                bad_epochs = 0
                if self._is_main():
                    self._module().save_safetensors(best_path)
                    logger.info(f"[ckpt] saved best @ epoch {ep}: metric={best_metric:.6f}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if self._is_main():
                        logger.info(
                            f"[early stop] no improvement for {patience} epochs. "
                            f"best_epoch={best_epoch}, best_metric={best_metric:.6f}"
                        )
                    break

            if save_every_epochs and (ep % save_every_epochs) == 0 and self._is_main():
                path = os.path.join(self.ckpt_dir, f"ep_{ep:04d}_motion_mamba")
                self._module().save_safetensors(path)

            # ---- LOG ----
            if (ep % log_every) == 0 and self._is_main():
                if val_loader is None:
                    logger.info(f"Epoch {ep}/{epochs} | train_loss={train_loss:.6f}")
                else:
                    logger.info(
                        f"Epoch {ep}/{epochs} | "
                        f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                    )

        return history

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next pose frame.

        Args:
            x : ``(T, D)`` or ``(B, T, D)``.

        Returns:
            ``(D,)`` or ``(B, D)``.
        """
        self.model.eval()
        x = x.to(self.device)
        logits = self.model(x)
        return logits[-1] if logits.dim() == 2 else logits[:, -1, :]
