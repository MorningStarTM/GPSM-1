# gpt architecture for world model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
from src.gpsm.utils.logger import logger
from src.gpsm.utils.utils import _to_jsonable
from safetensors.torch import save_file, load_file


# ---------------------------------------------------------------------------
# Config validation helper
# ---------------------------------------------------------------------------
def _validate_config(config: dict):
    """
    Validate model config and auto-fix the head_size inconsistency.

    FIX 1 — head_size:
      Block used to compute head_size = n_embd // n_head locally, while
      Head and MultiHeadAttention read config['head_size']. If those two
      values differed the projection shape (head_size * n_head -> n_embd)
      would silently mismatch. We now derive head_size from n_embd and
      n_head here, overwriting whatever was in the config, and assert that
      head_size * n_head == n_embd exactly.

    FIX 2 — block_size:
      block_size only needs to be >= the maximum sequence length T you will
      ever feed (i.e. >= k+1 from the dataset). We document this here and
      raise a clear error if it is too small rather than getting a silent
      index-out-of-range inside the causal mask.
    """
    required = ["state_dim", "n_embd", "n_head", "n_layers",
                "block_size", "dropout", "max_timestep", "learning_rate"]
    for key in required:
        if key not in config:
            raise KeyError(f"Config missing required key: '{key}'")

    n_embd = config["n_embd"]
    n_head = config["n_head"]

    if n_embd % n_head != 0:
        raise ValueError(
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head})."
        )

    # FIX 1: derive head_size authoritatively and overwrite config entry
    derived_head_size = n_embd // n_head
    if "head_size" in config and config["head_size"] != derived_head_size:
        logger.warning(
            f"[config] head_size={config['head_size']} overridden to "
            f"n_embd // n_head = {derived_head_size}. "
            "Make sure head_size * n_head == n_embd."
        )
    config["head_size"] = derived_head_size  # single source of truth

    # FIX 2: block_size must cover the full sequence length
    # The causal mask is (block_size, block_size); at runtime we slice
    # tril[:T, :T] so block_size >= T (= k+1 from the dataset).
    # We can't know k here, but we can give a clear lower-bound message.
    if config["block_size"] < 1:
        raise ValueError("block_size must be >= 1.")

    return config


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, config: dict):
        super().__init__()
        # head_size is now guaranteed == n_embd // n_head by _validate_config
        hs = config["head_size"]
        self.key   = nn.Linear(config["n_embd"], hs, bias=False)
        self.query = nn.Linear(config["n_embd"], hs, bias=False)
        self.value = nn.Linear(config["n_embd"], hs, bias=False)
        # block_size sets the max sequence length for the causal mask buffer.
        # At runtime we slice tril[:T, :T] so the mask always matches the
        # actual sequence length T <= block_size.
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config["block_size"], config["block_size"]))
        )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # scaled dot-product attention with causal mask
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)   # (B, T, hs)
        return wei @ v      # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: dict):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config["n_head"])])
        # FIX 1: head_size * n_head == n_embd is now guaranteed, so this
        # projection is always correctly shaped.
        self.proj    = nn.Linear(config["head_size"] * config["n_head"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward block (4x expansion, ReLU)."""

    def __init__(self, config: dict):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["n_embd"], 4 * config["n_embd"]),
            nn.ReLU(),
            nn.Linear(4 * config["n_embd"], config["n_embd"]),
            nn.Dropout(config["dropout"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    Transformer decoder block (pre-norm style).

    FIX 1: head_size is no longer computed locally here. It is set once in
    _validate_config (= n_embd // n_head) and passed through config, so
    Head, MultiHeadAttention, and Block all use the same value.
    """

    def __init__(self, config: dict):
        super().__init__()
        # head_size already in config (set by _validate_config) — no local override
        self.selfAttention = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1  = nn.LayerNorm(config["n_embd"])
        self.ln2  = nn.LayerNorm(config["n_embd"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.selfAttention(x))
        x = self.ln2(x + self.ffwd(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class StateMachineGPT(nn.Module):
    """
    GPT-style causal transformer for next-pose prediction.

    Input  : obs (B, T, state_dim)  — window of T past poses
    Output : logits (B, T, state_dim) — predicted pose at each step
             (use logits[:, -1, :] for the actual next-frame prediction)

    Config keys
    -----------
    state_dim      : dimensionality of one pose frame (e.g. 168 for SMPL-X)
    n_embd         : transformer embedding dimension
    n_head         : number of attention heads  (n_embd must be divisible by n_head)
    n_layers       : number of transformer blocks
    block_size     : maximum sequence length T the causal mask supports.
                     Must be >= (k + 1) where k is your dataset window size.
    dropout        : dropout probability
    max_timestep   : size of the positional embedding table.
                     Must be >= block_size (and >= T at inference).
    learning_rate  : AdamW learning rate
    loss_scale     : (optional) scalar multiplied onto the MSE loss (default 1.0)

    Note — head_size is derived automatically as n_embd // n_head. You do
    not need to set it in the config; if you do, it will be overwritten.
    """

    def __init__(self, config: dict):
        super().__init__()
        # validate + fix config in-place before building any layers
        config = _validate_config(dict(config))  # work on a copy
        self.config = config

        self.prev_pos_embedding    = nn.Linear(config["state_dim"], config["n_embd"])
        self.relative_pos_embedding = nn.Embedding(config["max_timestep"], config["n_embd"])
        self.blocks                = nn.Sequential(*[Block(config) for _ in range(config["n_layers"])])
        self.ln_f                  = nn.LayerNorm(config["n_embd"])
        self.next_pos_head         = nn.Linear(config["n_embd"], config["state_dim"])

        self.optimizer = optim.AdamW(self.parameters(), lr=config["learning_rate"])
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, targets=None) -> torch.Tensor:
        """
        Args:
            obs     : (B, T, D) or (T, D)
            targets : unused — kept for API compatibility

        Returns:
            logits  : same leading dims as obs, last dim = state_dim
        """
        if not torch.is_tensor(obs):
            raise TypeError(f"`obs` must be a torch.Tensor, got {type(obs)}")

        if obs.dim() == 2:
            obs = obs.unsqueeze(0)   # (T, D) -> (1, T, D)
            squeeze_B = True
        elif obs.dim() == 3:
            squeeze_B = False
        else:
            raise ValueError(f"`obs` must be 2D (T,D) or 3D (B,T,D). Got {tuple(obs.shape)}")

        B, T, D = obs.shape

        if T <= 0:
            raise ValueError(f"Sequence length T must be > 0, got T={T}")
        if T > self.config["block_size"]:
            raise ValueError(
                f"T={T} exceeds block_size={self.config['block_size']}. "
                "Increase block_size in config (must be >= your dataset window k+1)."
            )
        if T > self.config["max_timestep"]:
            raise ValueError(
                f"T={T} exceeds max_timestep={self.config['max_timestep']}. "
                "Increase max_timestep in config."
            )

        obs_emb = self.prev_pos_embedding(obs)   # (B, T, n_embd)

        # positional embedding — generated inside the model, not from the dataset
        step    = torch.arange(T, device=obs.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        pos_emb = self.relative_pos_embedding(step)  # (B, T, n_embd)

        x = obs_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.next_pos_head(x)   # (B, T, state_dim)

        if squeeze_B:
            logits = logits.squeeze(0)   # (T, state_dim)
        return logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_next(self, x_one: torch.Tensor, return_last_only: bool = True) -> torch.Tensor:
        """
        Predict the next pose frame for a SINGLE sample.

        Args:
            x_one           : (T, D) or (1, T, D)
            return_last_only: True  -> return (D,) / (1,D) — the predicted next frame
                              False -> return all (T,D) / (1,T,D) predictions

        Returns:
            Tensor of shape (D,) if input was (T,D), or (1,D) if (1,T,D).
        """
        self.eval()

        if not torch.is_tensor(x_one):
            raise TypeError(f"x_one must be a torch.Tensor, got {type(x_one)}")

        if x_one.dim() == 2:
            x_in = x_one.unsqueeze(0)   # (1, T, D)
            squeeze_B = True
        elif x_one.dim() == 3:
            if x_one.shape[0] != 1:
                raise ValueError(f"predict_next expects B=1, got B={x_one.shape[0]}")
            x_in = x_one
            squeeze_B = False
        else:
            raise ValueError(f"x_one must be (T,D) or (1,T,D). Got {tuple(x_one.shape)}")

        x_in   = x_in.to(self.device)
        logits = self.forward(x_in)   # (1, T, state_dim)

        if return_last_only:
            pred = logits[:, -1, :]              # (1, D)
            return pred.squeeze(0) if squeeze_B else pred
        else:
            return logits.squeeze(0) if squeeze_B else logits

    @torch.no_grad()
    def rollout(self, seed_frames: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressive generation: given seed_frames, predict n_steps future poses.

        Args:
            seed_frames : (T, D) — initial context window (T >= 1)
            n_steps     : number of new frames to generate

        Returns:
            generated : (n_steps, D) — the predicted future frames only
        """
        self.eval()
        if seed_frames.dim() != 2:
            raise ValueError(f"seed_frames must be (T, D), got {tuple(seed_frames.shape)}")

        window = seed_frames.to(self.device)   # (T, D)
        k      = self.config["block_size"]     # max context window
        preds  = []

        for _ in range(n_steps):
            ctx        = window[-k:]                   # trim to block_size
            next_frame = self.predict_next(ctx)        # (D,)
            preds.append(next_frame.unsqueeze(0))      # (1, D)
            window = torch.cat([window, next_frame.unsqueeze(0)], dim=0)

        return torch.cat(preds, dim=0)   # (n_steps, D)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_param_size(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters      : {total / 1e6:.3f} M")
        logger.info(f"Trainable parameters  : {trainable / 1e6:.3f} M")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict":     self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config":               self.config,
        }, path)

    def load(self, path: str, device=None):
        if device is None:
            device = next(self.parameters()).device
        ckpt = torch.load(path, map_location=device)
        if "config" in ckpt and ckpt["config"] != self.config:
            raise ValueError("Checkpoint config != current model config.")
        self.load_state_dict(ckpt["model_state_dict"])
        if self.optimizer is not None and ckpt.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        logger.info(f"Loaded model from {path}.")

    def save_safetensors(self, path: str, save_optimizer: bool = True):
        base = os.path.splitext(path)[0]
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

        save_file({k: v.detach().cpu() for k, v in self.state_dict().items()}, base + ".safetensors")

        meta = {
            "format": "safetensors_ckpt_v1",
            "config": _to_jsonable(self.config),
            "has_optimizer": False,
            "optimizer_state_keys": None,
        }

        if save_optimizer and self.optimizer is not None:
            try:
                opt_sd      = self.optimizer.state_dict()
                opt_tensors = {}
                opt_keys    = {}
                opt_nontensor = {}

                for pidx, st in opt_sd.get("state", {}).items():
                    p = str(pidx)
                    opt_keys[p]      = []
                    opt_nontensor[p] = {}
                    for sk, sv in st.items():
                        if torch.is_tensor(sv):
                            opt_tensors[f"state/{p}/{sk}"] = sv.detach().cpu()
                            opt_keys[p].append(sk)
                        else:
                            opt_nontensor[p][sk] = _to_jsonable(sv)

                save_file(opt_tensors, base + ".optim.safetensors")
                meta.update({
                    "has_optimizer":            True,
                    "optimizer_state_keys":     opt_keys,
                    "optimizer_param_groups":   _to_jsonable(opt_sd.get("param_groups", [])),
                    "optimizer_state_nontensor": opt_nontensor,
                })
            except Exception as e:
                meta["optimizer_save_error"] = str(e)

        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_safetensors(self, path: str, device=None, load_optimizer: bool = True, strict: bool = True):
        if device is None:
            device = next(self.parameters()).device
        base = os.path.splitext(path)[0]

        meta = None
        meta_path = base + ".json"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("config") is not None and meta["config"] != _to_jsonable(self.config):
                raise ValueError("Checkpoint config != current model config.")

        model_sd = {k: v.to(device) for k, v in load_file(base + ".safetensors").items()}
        self.load_state_dict(model_sd, strict=strict)

        optim_path = base + ".optim.safetensors"
        if (load_optimizer and self.optimizer is not None
                and meta and meta.get("has_optimizer") and os.path.exists(optim_path)):
            try:
                opt_sd = self.optimizer.state_dict()
                if "optimizer_param_groups" in meta:
                    opt_sd["param_groups"] = meta["optimizer_param_groups"]
                if "optimizer_state_nontensor" in meta:
                    for pidx, fields in meta["optimizer_state_nontensor"].items():
                        pidx_int = int(pidx)
                        opt_sd["state"].setdefault(pidx_int, {}).update(fields)
                for key, tensor in load_file(optim_path).items():
                    _, pidx, sk = key.split("/", 2)
                    opt_sd["state"].setdefault(int(pidx), {})[sk] = tensor.to(device)
                self.optimizer.load_state_dict(opt_sd)
            except Exception as e:
                logger.warning(f"Skipped optimizer restore: {e}")

        logger.info(f"Loaded safetensors checkpoint from {base}.")