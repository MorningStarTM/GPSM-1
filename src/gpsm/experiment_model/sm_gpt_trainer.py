import os
import math
import torch
import torch.nn.functional as F
from src.gpsm.experiment_model.sm_gpt import StateMachineGPT
from src.gpsm.utils.logger import logger


class SMTrainer:
    """
    Trainer for StateMachineGPT.

    FIX 3 — vestigial `pos` tensor:
      The dataset returns (x, pos, y) but StateMachineGPT generates its own
      positional indices internally with torch.arange(T). The `pos` tensor
      from the batch was never passed to the model and never used. We now
      unpack it explicitly as `_pos` (throwaway) so the intent is clear,
      and we document why it is discarded. If you ever need external pos
      (e.g. absolute frame indices for a global sequence), wire it in here.
    """

    def __init__(self, config: dict, ckpt_dir: str = "checkpoints"):
        self.config   = config
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model = StateMachineGPT(config)
        self.device = self.model.device
        self.model.to(self.device)
        self.model.print_param_size()

        self.optimizer = self.model.optimizer
        self._eps      = 1e-8

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
        """
        MSE loss between the last-token prediction and the target next frame.

        logits : (B, T, D) or (T, D)
        y      : (B, D)    or (D,)
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
        """
        Train the model.

        Args:
            train_loader       : DataLoader yielding (x, _pos, y) batches.
                                 _pos is unused — the model generates its own
                                 positional indices internally.
            epochs             : total training epochs
            val_loader         : optional validation DataLoader (same format)
            log_every          : log interval in epochs
            best_path          : path prefix for best checkpoint (safetensors)
            patience           : early-stop patience (epochs without improvement)
            grad_clip_norm     : gradient clipping max norm (0 = disabled)
            save_every_epochs  : if set, save a checkpoint every N epochs

        Returns:
            history dict with 'train_loss' (and 'val_loss' if val_loader given)
        """
        history = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        best_metric = math.inf
        best_epoch  = 0
        bad_epochs  = 0

        if best_path is None:
            best_path = os.path.join(self.ckpt_dir, "best_state_machine_gpt")

        for ep in range(1, epochs + 1):

            # ---- TRAIN ----
            self.model.train()
            total, n = 0.0, 0

            for batch in train_loader:
                # FIX 3: _pos is intentionally discarded — StateMachineGPT
                # builds its own positional indices inside forward().
                x, y = batch

                x = x.to(self.device)
                y = y.to(self.device)

                if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                    continue  # skip NaN/Inf batches

                logits = self.model(x)
                loss   = self._compute_loss(logits, y) * self.config.get("loss_scale", 1.0)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

                total += float(loss.item())
                n     += 1

            train_loss = total / max(n, 1)
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
                        logits   = self.model(x)
                        vtotal  += float(self._compute_loss(logits, y).item())
                        vn      += 1
                val_loss = vtotal / max(vn, 1)
                history["val_loss"].append(val_loss)

            # ---- CHECKPOINT + EARLY STOPPING ----
            current_metric = val_loss if val_loader is not None else train_loss

            if current_metric < (best_metric - 1e-8):
                best_metric = current_metric
                best_epoch  = ep
                bad_epochs  = 0
                self._module().save_safetensors(best_path)
                if self._is_main():
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

            if save_every_epochs and (ep % save_every_epochs) == 0:
                path = os.path.join(self.ckpt_dir, f"ep_{ep:04d}_state_machine_gpt")
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
        """
        Predict the next pose frame.

        Args:
            x : (T, D) or (B, T, D)

        Returns:
            (D,) or (B, D)
        """
        self.model.eval()
        x      = x.to(self.device)
        logits = self.model(x)
        return logits[-1] if logits.dim() == 2 else logits[:, -1, :]