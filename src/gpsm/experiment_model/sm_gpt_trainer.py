import os, math
import torch
import torch.nn.functional as F
from src.gpsm.experiment_model.sm_gpt import StateMachineGPT 
from src.gpsm.utils.logger import logger


# ---------------------------------------------------------
# Trainer for StateMachineGPT 
# ---------------------------------------------------------
class SMTrainer:
    def __init__(self, config, ckpt_dir="checkpoints"):
        """
        config must include:
          - state_dim
          - max_timestep (>= sequence length T you feed)
          - learning_rate
          - (n_embd, n_layers, etc. for model)
        """
        self.config = config
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)


        self.model = StateMachineGPT(config)
        self.device = self.model.device
        self.model.to(self.device)
        self.model.print_param_size()

        # optimizer is already created inside the model
        self.optimizer = self.model.optimizer

        # (optional) small constant for numerical safety checks
        self._eps = 1e-8

    # -------- helpers (same style as your WMTrainer)
    def _is_ddp(self):
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            return isinstance(self.model, DDP)
        except Exception:
            return False

    def _rank(self):
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0

    def _is_main(self):
        return self._rank() == 0

    def _module(self):
        # return underlying module if DDP else model itself
        return self.model.module if self._is_ddp() else self.model

    # -------- core utilities
    def _sanity_batch(self, x, pos, y, ep=None, where="train"):
        """
        x:  (B,T,D) or (T,D)
        pos:(B,T) or (T,) (not used in model, but we validate shape anyway)
        y:  (B,D) or (D,)
        """
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise TypeError(f"[{where}] x and y must be tensors. got x={type(x)} y={type(y)}")

        if x.dim() == 2:
            T, D = x.shape
            if y.dim() != 1 or y.shape[0] != D:
                raise ValueError(f"[{where}] for x(T,D) expected y(D). got x={tuple(x.shape)} y={tuple(y.shape)}")
            if pos is not None:
                if pos.dim() != 1 or pos.shape[0] != T:
                    raise ValueError(f"[{where}] for x(T,D) expected pos(T). got pos={tuple(pos.shape)} T={T}")
        elif x.dim() == 3:
            B, T, D = x.shape
            if y.dim() != 2 or y.shape != (B, D):
                raise ValueError(f"[{where}] for x(B,T,D) expected y(B,D). got x={tuple(x.shape)} y={tuple(y.shape)}")
            if pos is not None:
                if pos.dim() != 2 or pos.shape != (B, T):
                    raise ValueError(f"[{where}] for x(B,T,D) expected pos(B,T). got pos={tuple(pos.shape)}")
        else:
            raise ValueError(f"[{where}] x must be 2D or 3D. got {tuple(x.shape)}")

        # max_timestep guard (positional embedding range)
        max_t = self._module().config["max_timestep"]
        if T > max_t:
            raise ValueError(f"[{where}] T={T} exceeds max_timestep={max_t}. Increase max_timestep in config.")

    def _compute_loss(self, logits, y):
        """
        logits: (B,T,D) or (T,D)
        y:      (B,D) or (D,)
        We train "predict next frame" using last token prediction: logits[:, -1, :]
        """
        if logits.dim() == 2:
            # (T,D)
            pred = logits[-1]           # (D,)
        elif logits.dim() == 3:
            pred = logits[:, -1, :]     # (B,D)
        else:
            raise ValueError(f"logits must be 2D or 3D. got {tuple(logits.shape)}")

        loss = F.mse_loss(pred, y)

        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss: {loss.item()}")
        return loss

    # ---------------------------------------------------------
    # Train (single GPU / non-DDP)
    # ---------------------------------------------------------
    def train(
        self,
        train_loader,
        epochs=10,
        val_loader=None,
        log_every=1,
        best_path=None,
        patience=10,
        grad_clip_norm=1.0,
        save_every_epochs=None,  # optional periodic save
    ):
        history = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []

        best_metric = math.inf
        best_epoch = 0
        bad_epochs = 0

        if best_path is None:
            best_path = os.path.join(self.ckpt_dir, "best_state_machine_gpt")

        for ep in range(1, epochs + 1):
            # ---------- TRAIN ----------
            self.model.train()
            total = 0.0
            n = 0

            for batch in train_loader:
                # dataset returns (x,pos,y)
                x, pos, y = batch

                # move
                x = x.to(self.device)
                y = y.to(self.device)

                # guard: skip bad batches
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    # optional: print once to debug
                    # print("Skipping non-finite batch")
                    continue
                # pos is not used, but move if you want to keep it around for debugging
                # pos = pos.to(self.device)

                # forward
                logits = self.model(x)               # (B,T,D) or (T,D)
                loss = self._compute_loss(logits, y)

                # backward
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()

                total += float(loss.item())
                n += 1

            train_loss = total / max(n, 1)
            history["train_loss"].append(train_loss)

            # ---------- VAL ----------
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                vtotal = 0.0
                vn = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, pos, y = batch
                        self._sanity_batch(x, pos, y, ep=ep, where="val")

                        x = x.to(self.device)
                        y = y.to(self.device)

                        logits = self.model(x)
                        loss = self._compute_loss(logits, y)

                        vtotal += float(loss.item())
                        vn += 1

                val_loss = vtotal / max(vn, 1)
                history["val_loss"].append(val_loss)

            # ---------- CHECKPOINT + EARLY STOP ----------
            current_metric = val_loss if (val_loader is not None) else train_loss

            if current_metric < (best_metric - 1e-8):
                best_metric = current_metric
                best_epoch = ep
                bad_epochs = 0

                # save best (safetensors)
                self._module().save_safetensors(best_path)
                if self._is_main():
                    logger.info(f"[ckpt] saved best @ epoch {ep}: metric={best_metric:.6f} -> {best_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if self._is_main():
                        logger.info(f"[early stop] no improvement for {patience} epochs. best_epoch={best_epoch}, best_metric={best_metric:.6f}")
                    break

            # optional periodic save
            if save_every_epochs is not None and (ep % save_every_epochs) == 0:
                path = os.path.join(self.ckpt_dir, f"ep_{ep:04d}_state_machine_gpt")
                self._module().save_safetensors(path)

            # ---------- LOG ----------
            if (ep % log_every) == 0:
                if val_loader is None:
                    if self._is_main():
                        logger.info(f"Epoch {ep}/{epochs} | train_loss={train_loss:.6f}")
                else:
                    if self._is_main():
                        logger.info(f"Epoch {ep}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        return history

    # ---------------------------------------------------------
    # Simple inference helper: given x -> predict next frame
    # ---------------------------------------------------------
    @torch.no_grad()
    def predict_next(self, x):
        """
        x: (T,D) or (B,T,D)
        Returns:
          next_pred: (D,) or (B,D)
        """
        self.model.eval()
        x = x.to(self.device)
        logits = self.model(x)

        if logits.dim() == 2:
            return logits[-1]          # (D,)
        else:
            return logits[:, -1, :]    # (B,D)
