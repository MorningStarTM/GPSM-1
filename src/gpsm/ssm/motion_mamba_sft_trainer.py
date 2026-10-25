"""Supervised fine-tuning (SFT) trainer for MotionMamba.

This is a minimally-adapted copy of ``sm_gpt_sft_trainer.py``'s
``SFTTrainer`` -- same reason ``motion_mamba_trainer.py`` copies
``sm_gpt_trainer.py`` rather than generalizing it: the two model classes
don't share a base class, so copying and swapping the one line was judged
lower-risk than editing the existing, already-working, already-used
trainer. Every line below is identical in spirit to ``SFTTrainer`` except
``_rollout()``, which drops the transformer-specific ``block_size``
context-window truncation -- MotionMamba (an SSM) has no such concept
(see motion_mamba.py's module docstring for why) -- and keeps the window
growing for the whole rollout instead, same as MotionMamba.rollout_batch's
own established pattern.

MotionMambaTrainer (motion_mamba_trainer.py) is the pretraining phase:
teacher-forced, single-step "given k history frames, predict the one next
frame" -- the model never sees its own predictions during training. That
objective can be satisfied almost entirely by learning to copy the last
frame forward, since consecutive mocap frames are already close together
-- measured directly: val_acc reached 0.9987 in epoch 1 of pretraining and
barely moved after. It says nothing about whether the model has learned
real dynamics, because next-frame MSE never sees compounding error -- it
is always handed the real previous frame.

MotionMambaSFTTrainer is the next phase: given a single seed frame, the
model must free-run its own predictions forward for `n_frames` steps (see
`_rollout()` below -- gradient-enabled and DDP-safe, unlike the
inference-only `MotionMamba.rollout()`/`rollout_batch()`), and every
predicted frame in that self-generated sequence is scored against the real
recorded continuation. A model that only learned to copy the last frame
forward diverges sharply from real motion over a multi-step rollout, so
this objective can no longer be satisfied by the trivial solution the way
single-step MSE could.

Expects a torch Dataset like CachedSFTDataset (or MultiSMPLXNPZSFTDataset),
yielding (seed, target) = ((D,), (n_frames, D)) samples.
"""

import os
import math
import torch
import torch.nn.functional as F
from src.gpsm.ssm.motion_mamba import MotionMamba
from src.gpsm.experiment_model.distributed import reduce_mean
from src.gpsm.utils.logger import logger


class MotionMambaSFTTrainer:
    """SFT trainer for MotionMamba. See the module docstring -- this
    mirrors ``SFTTrainer`` with only the model class and ``_rollout()``'s
    window handling changed."""

    def __init__(self, config: dict, ckpt_dir: str = "checkpoints", pretrained_checkpoint: str = None):
        self.config = config
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model = MotionMamba(config)
        self.device = self.model.device
        self.model.to(self.device)

        if pretrained_checkpoint is not None:
            # Fine-tune from an existing pretrained checkpoint: load weights
            # only, not its optimizer state -- SFT is a new training phase
            # with its own optimizer trajectory.
            self.model.load_safetensors(pretrained_checkpoint, device=self.device, load_optimizer=False)
            logger.info(f"[SFT] initialized from pretrained checkpoint: {pretrained_checkpoint}")

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

    def _rollout(self, seed: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Same free-running autoregressive loop as
        MotionMamba.rollout_batch, reimplemented here calling
        `self.model(window)` at every step instead of
        `self.model.rollout_batch(...)`. This matters under DDP: DDP only
        registers its gradient-all-reduce hooks on its own __call__/forward,
        not on custom methods reached through `.module`. Calling through
        `.module` wouldn't even raise an error -- it would silently skip
        gradient synchronization and let every rank's weights drift apart
        independently. Calling `self.model(window)` is correct for both a
        plain MotionMamba and a DDP-wrapped one.

        Unlike SFTTrainer._rollout() (StateMachineGPT), there is no
        block_size to truncate the context to -- an SSM has no fixed
        context window (see motion_mamba.py's module docstring). The
        window grows for the whole rollout, same as
        MotionMamba.rollout_batch's own established pattern.
        """
        window = seed
        preds = []
        for _ in range(n_steps):
            logits = self.model(window)  # self.model(...), NOT self.model.rollout_batch(...)
            next_frame = logits[:, -1:, :]
            preds.append(next_frame)
            window = torch.cat([window, next_frame], dim=1)
        return torch.cat(preds, dim=1)

    def _compute_loss(self, pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        """MSE loss between the full autoregressively-generated sequence
        and the real recorded continuation -- the SFT analogue of an LLM's
        next-token cross-entropy averaged over every generated position,
        not just one.

        pred_seq, target_seq : (B, n_frames, D)
        """
        loss = F.mse_loss(pred_seq, target_seq)
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
        max_steps_per_epoch: int = None,
    ) -> dict:
        """
        Args:
            train_loader       : DataLoader yielding (seed, target) batches,
                                  seed=(B,D), target=(B,n_frames,D) -- e.g.
                                  from CachedSFTDataset.
            epochs              : total training epochs
            val_loader          : optional validation DataLoader (same format)
            log_every           : log interval in epochs
            best_path           : path prefix for best checkpoint (safetensors)
            patience            : early-stop patience (epochs without improvement)
            grad_clip_norm      : gradient clipping max norm (0 = disabled) --
                                   more important here than in
                                   MotionMambaTrainer, since gradients
                                   backprop through the whole unrolled
                                   n_frames-step rollout (similar to BPTT).
            save_every_epochs   : if set, save a checkpoint every N epochs
            max_steps_per_epoch : if set, stop each epoch after this many
                                   batches. Required under DDP when different
                                   ranks hold different-sized data shards --
                                   every rank MUST call backward() the same
                                   number of times per epoch, or the rank(s)
                                   that run out of batches first leave the
                                   others hanging on the next gradient
                                   all-reduce and the whole job deadlocks.
                                   Pass the cross-rank MINIMUM batch count
                                   here.

        Returns:
            history dict with 'train_loss' (and 'val_loss' if val_loader given)
        """
        history = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        best_metric = math.inf
        best_epoch = 0
        bad_epochs = 0

        if best_path is None:
            best_path = os.path.join(self.ckpt_dir, "best_motion_mamba_sft")

        for ep in range(1, epochs + 1):
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(ep)

            # ---- TRAIN ----
            self.model.train()
            total, n = 0.0, 0

            for step, batch in enumerate(train_loader):
                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break

                seed, target = batch  # seed: (B, D), target: (B, n_frames, D)

                seed = seed.to(self.device).unsqueeze(1)  # (B, 1, D) -- _rollout() expects (B, T0, D)
                target = target.to(self.device)

                if not (torch.isfinite(seed).all() and torch.isfinite(target).all()):
                    continue  # skip NaN/Inf batches

                pred_seq = self._rollout(seed, n_steps=target.shape[1])
                loss = self._compute_loss(pred_seq, target) * self.config.get("loss_scale", 1.0)

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
                    for vstep, batch in enumerate(val_loader):
                        if max_steps_per_epoch is not None and vstep >= max_steps_per_epoch:
                            break
                        seed, target = batch
                        seed = seed.to(self.device).unsqueeze(1)
                        target = target.to(self.device)
                        pred_seq = self._rollout(seed, n_steps=target.shape[1])
                        vtotal += float(self._compute_loss(pred_seq, target).item())
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
                path = os.path.join(self.ckpt_dir, f"sft_ep_{ep:04d}_motion_mamba")
                self._module().save_safetensors(path)

            # ---- LOG ----
            if (ep % log_every) == 0 and self._is_main():
                if val_loader is None:
                    logger.info(f"[SFT] Epoch {ep}/{epochs} | train_loss={train_loss:.6f}")
                else:
                    logger.info(
                        f"[SFT] Epoch {ep}/{epochs} | "
                        f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                    )

        return history
