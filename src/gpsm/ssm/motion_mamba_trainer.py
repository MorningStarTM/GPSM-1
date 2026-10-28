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

    def _threshold_accuracy(
        self, logits: torch.Tensor, y: torch.Tensor, threshold: float
    ) -> float:
        """Fraction of frames whose prediction falls within `threshold` of
        the target, measured as per-dimension RMSE (L2 distance / sqrt(D)),
        not raw L2 distance.

        This is NOT classification accuracy -- next-pose prediction is
        regression, which has no natural "correct"/"incorrect" outcome to
        count. This is the PCK-style ("percentage of correct keypoints")
        threshold metric used in pose-estimation papers for exactly this
        reason: pick a tolerance, report what fraction of predictions fall
        within it. Report it in a paper as "frame-level threshold accuracy
        (tau=<threshold>, per-dimension RMSE, normalized units)", not as
        plain "accuracy" -- the value is meaningless without the threshold
        and the fact that poses are per-channel z-score normalized, not in
        physical units.

        Per-dimension RMSE rather than raw L2 distance specifically:
        ``||pred - y||_2`` grows with ``sqrt(D)``, so a fixed threshold
        would silently mean something different for every ``state_dim`` --
        measured directly: with state_dim=159 and threshold=0.5 (raw L2),
        accuracy was 0.0 at every epoch even as loss fell from 0.198 to
        0.032, because a typical error vector's L2 norm at D=159 is far
        above 0.5 even when every individual dimension is well predicted.
        Dividing by sqrt(D) keeps the threshold's meaning ("typical
        per-channel error, in std-deviations") the same across pose widths.
        """
        pred = logits[:, -1, :] if logits.dim() == 3 else logits[-1]
        d = pred.shape[-1]
        per_frame_rmse = torch.linalg.norm(pred - y, dim=-1) / (d ** 0.5)
        return float((per_frame_rmse < threshold).float().mean().item())

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
        acc_threshold: float = 0.5,
        max_steps_per_epoch: int = None,
        on_checkpoint_saved=None,
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
            acc_threshold     : Tolerance (normalized pose-vector L2 distance)
                                for the threshold-accuracy metric -- see
                                ``_threshold_accuracy``. The default is a
                                starting point, not a validated value; pick
                                one deliberately for a paper and say so.
            max_steps_per_epoch: If set, stop each epoch after this many
                                batches. Required under DDP when ranks hold
                                different-sized shards (e.g. different file
                                ranges per GPU) -- every rank MUST call
                                backward() the same number of times per
                                epoch, or the rank(s) that run out of
                                batches first leave the others hanging on
                                the next gradient all-reduce and the whole
                                job deadlocks. Pass the cross-rank MINIMUM
                                batch count here (see
                                main_motion_mamba_ddp.py for how it's
                                computed) -- same mechanism as
                                ``SFTTrainer.train()``'s parameter of the
                                same name.
            on_checkpoint_saved : optional ``callable(best_path)``, called
                                (rank 0 only) right after a new best
                                checkpoint is written to disk -- every
                                epoch that improves, not just once at the
                                end of training. Lets a caller plug in a
                                Hugging Face push without this trainer
                                needing to know anything about HF (see
                                src/gpsm/utils/hub.py's docstring) --
                                pushing only after train() returns would
                                mean a session that dies mid-run never gets
                                anything to HF, even though every earlier
                                epoch's improvement was saved locally the
                                whole time.

        Returns:
            A history dict with ``"train_loss"``/``"train_acc"`` (and the
            ``val_`` equivalents if ``val_loader`` was given). ``*_acc`` is
            threshold accuracy, not classification accuracy -- see
            ``_threshold_accuracy``'s docstring before reporting it.
        """
        history = {"train_loss": [], "train_acc": []}
        if val_loader is not None:
            history["val_loss"] = []
            history["val_acc"] = []

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
            total, acc_total, n = 0.0, 0.0, 0

            for step, batch in enumerate(train_loader):
                # See max_steps_per_epoch's docstring above -- every DDP
                # rank must call backward() the same number of times per
                # epoch, or the all-reduce deadlocks.
                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break

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
                with torch.no_grad():
                    acc_total += self._threshold_accuracy(logits, y, acc_threshold)
                n += 1

            train_loss = reduce_mean(total / max(n, 1), self.device)
            train_acc = reduce_mean(acc_total / max(n, 1), self.device)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # ---- VALIDATION ----
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                vtotal, vacc_total, vn = 0.0, 0.0, 0
                with torch.no_grad():
                    for vstep, batch in enumerate(val_loader):
                        if max_steps_per_epoch is not None and vstep >= max_steps_per_epoch:
                            break
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                        logits = self.model(x)
                        vtotal += float(self._compute_loss(logits, y).item())
                        vacc_total += self._threshold_accuracy(logits, y, acc_threshold)
                        vn += 1
                val_loss = reduce_mean(vtotal / max(vn, 1), self.device)
                val_acc = reduce_mean(vacc_total / max(vn, 1), self.device)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # ---- CHECKPOINT + EARLY STOPPING ----
            current_metric = val_loss if val_loader is not None else train_loss

            if current_metric < (best_metric - 1e-8):
                best_metric = current_metric
                best_epoch = ep
                bad_epochs = 0
                if self._is_main():
                    self._module().save_safetensors(best_path)
                    logger.info(f"[ckpt] saved best @ epoch {ep}: metric={best_metric:.6f}")
                    if on_checkpoint_saved is not None:
                        on_checkpoint_saved(best_path)
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
                    logger.info(
                        f"Epoch {ep}/{epochs} | "
                        f"train_loss={train_loss:.6f} | train_acc@{acc_threshold}={train_acc:.4f}"
                    )
                else:
                    logger.info(
                        f"Epoch {ep}/{epochs} | "
                        f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                        f"train_acc@{acc_threshold}={train_acc:.4f} | val_acc@{acc_threshold}={val_acc:.4f}"
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
