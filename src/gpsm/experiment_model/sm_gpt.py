# gpt architecture for world model
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json
import numpy as np
from collections import deque
from src.gpsm.utils.logger import logger
from src.gpsm.utils.utils import _to_jsonable
from safetensors.torch import save_file, load_file



class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.query = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.value = nn.Linear(config['n_embd'], config['head_size'], bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])


    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out




class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config['n_head'])])
        self.proj = nn.Linear(config['head_size'] * config['n_head'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.ReLU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x):
        return self.net(x)
    


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.selfAttention = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])

    def forward(self, x):
        y = self.selfAttention(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
    


class StateMachineGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prev_pos_embedding = nn.Linear(self.config['state_dim'], self.config['n_embd'])

        self.relative_pos_embedding = nn.Embedding(self.config['max_timestep'], self.config['n_embd'])
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config['n_layers'])])

        self.ln_f = nn.LayerNorm(self.config['n_embd'])
        self.next_pos_head = nn.Linear(self.config['n_embd'], config['input_dim'])
        

        self.optimizer = optim.AdamW(self.parameters(), lr=self.config['learning_rate'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, obs, action, step, targets=None):
        #B,T = obs.shape

        obs_emb = self.prev_pos_embedding(obs)

        step = step.to(obs.device).long()                   # (B,T) long on same device
        step = step % self.config['max_timestep']           # safety (even if you already cycle)
        pos = self.relative_pos_embedding(step)
        x = obs_emb + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        next_pos_logits = self.next_pos_head(x)


        return next_pos_logits
    

    def print_param_size(self):
        """
        Print total number of parameters in millions.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Total parameters      : {total_params / 1e6:.3f} M")
        logger.info(f"Trainable parameters  : {trainable_params / 1e6:.3f} M")
    


    def save(self, path: str):
        """
        Save model + (optional) optimizer + config.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        torch.save(ckpt, path)

    def load(self, path: str, device=None):
        if device is None:
            device = next(self.parameters()).device

        ckpt = torch.load(path, map_location=device)

        # Optional safety check: config match
        if "config" in ckpt and ckpt["config"] != self.config:
            raise ValueError("Checkpoint config != current model config (create model with same config before load).")

        self.load_state_dict(ckpt["model_state_dict"])

        if hasattr(self, "optimizer") and self.optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # move optimizer state tensors to device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        logger.info(f"Loaded model from {path}.")

    


    def save_safetensors(self, path: str, save_optimizer: bool = True):
        """
        Save model weights (always) + optimizer state tensors (optional) using safetensors.
        Writes:
          - {path}.safetensors
          - {path}.json
          - {path}.optim.safetensors   (optional)
        """
        base, ext = os.path.splitext(path)
        if ext:  # user passed something like "ckpt.pt" -> use "ckpt"
            base = base
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

        model_path = base + ".safetensors"
        meta_path = base + ".json"
        optim_path = base + ".optim.safetensors"

        # 1) model weights
        model_sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        save_file(model_sd, model_path)

        # 2) metadata (config + optimizer key structure)
        meta = {
            "format": "safetensors_ckpt_v1",
            "config": _to_jsonable(getattr(self, "config", None)),
            "has_optimizer": False,
            "optimizer_state_keys": None,
        }

        # 3) optimizer (optional, best-effort)
        if save_optimizer and hasattr(self, "optimizer") and self.optimizer is not None:
            try:
                opt_sd = self.optimizer.state_dict()

                # Flatten ONLY tensor parts of optimizer state into a safetensors dict
                # Key format: state/{param_idx}/{state_key}
                # (We keep "param_groups" + non-tensor scalars in JSON)
                opt_tensors = {}
                opt_state_keys = {}  # helps debug / validate
                for param_idx, st in opt_sd.get("state", {}).items():
                    param_idx_str = str(param_idx)
                    opt_state_keys[param_idx_str] = []
                    for sk, sv in st.items():
                        if torch.is_tensor(sv):
                            key = f"state/{param_idx_str}/{sk}"
                            opt_tensors[key] = sv.detach().cpu()
                            opt_state_keys[param_idx_str].append(sk)

                # Save optimizer tensor state
                save_file(opt_tensors, optim_path)

                # Save remaining optimizer info in metadata
                meta["has_optimizer"] = True
                meta["optimizer_state_keys"] = opt_state_keys
                meta["optimizer_param_groups"] = _to_jsonable(opt_sd.get("param_groups", []))

                # Also save non-tensor optimizer scalars per param (e.g., step counts if ints)
                # We store them to JSON so load can restore if desired.
                opt_state_nontensor = {}
                for param_idx, st in opt_sd.get("state", {}).items():
                    p = str(param_idx)
                    opt_state_nontensor[p] = {}
                    for sk, sv in st.items():
                        if not torch.is_tensor(sv):
                            opt_state_nontensor[p][sk] = _to_jsonable(sv)
                meta["optimizer_state_nontensor"] = opt_state_nontensor

            except Exception as e:
                # If optimizer save fails, still save model + config
                meta["has_optimizer"] = False
                meta["optimizer_save_error"] = str(e)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_safetensors(self, path: str, device=None, load_optimizer: bool = True, strict: bool = True):
        """
        Load model weights (+ optional optimizer state) from safetensors-based checkpoint.
        Expects:
          - {path}.safetensors
          - {path}.json
          - {path}.optim.safetensors   (optional)
        """
        if device is None:
            device = next(self.parameters()).device

        base, ext = os.path.splitext(path)
        if ext:
            base = base

        model_path = base + ".safetensors"
        meta_path = base + ".json"
        optim_path = base + ".optim.safetensors"

        # metadata (optional but recommended)
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Optional safety check: config match
            if meta.get("config") is not None and hasattr(self, "config"):
                if meta["config"] != _to_jsonable(self.config):
                    raise ValueError("Checkpoint config != current model config (create model with same config before load).")

        # load model weights
        model_sd = load_file(model_path)  # tensors on CPU
        model_sd = {k: v.to(device) for k, v in model_sd.items()}
        self.load_state_dict(model_sd, strict=strict)

        # optimizer (best-effort)
        if (
            load_optimizer
            and hasattr(self, "optimizer")
            and self.optimizer is not None
            and meta is not None
            and meta.get("has_optimizer", False)
            and os.path.exists(optim_path)
        ):
            try:
                opt_sd = self.optimizer.state_dict()

                # Restore param_groups from meta (structure)
                if "optimizer_param_groups" in meta:
                    opt_sd["param_groups"] = meta["optimizer_param_groups"]

                # Restore non-tensor fields (if present)
                if "optimizer_state_nontensor" in meta:
                    for pidx, fields in meta["optimizer_state_nontensor"].items():
                        pidx_int = int(pidx)
                        if pidx_int not in opt_sd["state"]:
                            opt_sd["state"][pidx_int] = {}
                        opt_sd["state"][pidx_int].update(fields)

                # Restore tensor fields from optim safetensors
                opt_tensors = load_file(optim_path)  # CPU tensors
                # Keys are state/{param_idx}/{state_key}
                for key, tensor in opt_tensors.items():
                    _, pidx, sk = key.split("/", 2)
                    pidx_int = int(pidx)
                    if pidx_int not in opt_sd["state"]:
                        opt_sd["state"][pidx_int] = {}
                    opt_sd["state"][pidx_int][sk] = tensor.to(device)

                self.optimizer.load_state_dict(opt_sd)

            except Exception as e:
                # Don’t fail the whole load if optimizer restore breaks
                # (torch version changes often break optimizer states)
                if "logger" in globals():
                    logger.warning(f"Loaded model but skipped optimizer restore due to: {e}")
        if "logger" in globals():
            logger.info(f"Loaded safetensors checkpoint from {base}.")




