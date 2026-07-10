# GPSM-1

**GPSM** (Generative Pre-trained State Machine) is a neural motion model trained on
large-scale motion-capture data to predict and generate human motion, aiming to
replace hand-crafted animation state machines used in traditional motion-matching
systems.

Instead of authoring transition rules and blend trees by hand, GPSM learns the
dynamics of human movement directly from mocap sequences: given a short window of
past poses, it predicts the next pose. Chained autoregressively, this turns into a
generative motion model that can roll a character forward frame by frame.

## Idea

Classic game-animation state machines (walk → run → jump → land, ...) are built by
hand: every transition, blend, and edge case is authored explicitly. This does not
scale well and struggles to generalize beyond the states the designer thought of.

GPSM treats motion generation as a **sequence modeling problem**, the same way GPT
treats text as a sequence of tokens — except here each "token" is a continuous pose
vector (joint rotations / positions for one frame) instead of a discrete word. A
causal transformer is trained to predict pose `t+1` from poses `[t-k, ..., t]`. The
learned model implicitly captures motion transitions, timing, and style from data,
instead of requiring a designer to encode them as explicit states.

## Model architecture

`StateMachineGPT` ([sm_gpt.py](src/gpsm/experiment_model/sm_gpt.py)) is a decoder-only
(GPT-style) causal transformer adapted for continuous-valued pose sequences rather
than a discrete token vocabulary:

```
obs (B, T, state_dim)
   │
   ├─ Linear(state_dim → n_embd)              prev_pos_embedding  (pose  → embedding)
   ├─ Embedding(max_timestep, n_embd)[0..T)    relative_pos_embedding (learned positional embedding)
   │        └─ added together
   ▼
[ Block × n_layers ]
   ├─ LayerNorm → Multi-Head Causal Self-Attention → residual
   ├─ LayerNorm → FeedForward (4x expansion, ReLU) → residual
   ▼
LayerNorm (ln_f)
   ▼
Linear(n_embd → state_dim)                    next_pos_head
   ▼
logits (B, T, state_dim)   — predicted next pose at every timestep
```

Key points:

- **No token embedding table.** Since input frames are continuous vectors (pose
  parameters), `prev_pos_embedding` is a plain `Linear(state_dim, n_embd)` that
  projects a raw pose into embedding space, rather than an `nn.Embedding` lookup.
- **Learned positional embedding**, generated internally from `torch.arange(T)` —
  the dataset does not need to supply position indices.
- **Causal self-attention** (`Head` / `MultiHeadAttention`): standard scaled
  dot-product attention with a lower-triangular mask (`tril`), so frame `t` can only
  attend to frames `≤ t`.
- **Output head** regresses directly to a `state_dim`-sized pose vector (not a
  softmax over a vocabulary) — this is a regression model trained with MSE loss, not
  a classification model trained with cross-entropy.
- `head_size` is always derived as `n_embd // n_head` and validated/overwritten by
  `_validate_config()` so `Head`, `MultiHeadAttention`, and `Block` can never disagree
  on the projection shape.

### Config keys

| Key | Meaning |
|---|---|
| `state_dim` | Width of one pose frame (feature count). Should be derived from the dataset (`ds.D`), not hand-set — different SMPL-X feature sets produce different widths. |
| `n_embd` | Transformer embedding dimension. |
| `n_head` | Number of attention heads (`n_embd` must be divisible by `n_head`). |
| `n_layers` | Number of transformer blocks. |
| `block_size` | Max sequence length the causal mask supports; must be `≥ k + 1` (the dataset window). |
| `max_timestep` | Size of the positional embedding table; must be `≥ block_size`. |
| `dropout` | Dropout probability used in attention and feed-forward layers. |
| `learning_rate` | AdamW learning rate. |
| `loss_scale` | Optional scalar multiplied onto the MSE loss. |

### Inference helpers

- `predict_next(x_one)` — predicts a single next frame from one window.
- `rollout(seed_frames, n_steps)` — autoregressively generates `n_steps` future
  frames by repeatedly predicting the next frame and appending it to the context
  window (trimmed to `block_size`).

## Repository layout

```
main.py                                  example training entrypoint
src/gpsm/
├── experiment_model/
│   ├── sm_gpt.py                        StateMachineGPT model + config validation
│   └── sm_gpt_trainer.py                SMTrainer: training/validation loop
└── utils/
    ├── dataset.py                       mocap → (x, y) window datasets
    ├── logger.py                        loguru-based file + console logger
    └── utils.py                         small config/JSON helpers
```

### `sm_gpt.py` — model

`StateMachineGPT` plus its building blocks (`Head`, `MultiHeadAttention`,
`FeedForward`, `Block`) and `_validate_config`, described above. Also provides
checkpoint I/O: `save`/`load` (plain `torch.save`, includes optimizer state) and
`save_safetensors`/`load_safetensors` (safetensors format, splits model weights and
optimizer state into separate files plus a JSON metadata sidecar).

### `sm_gpt_trainer.py` — training loop

`SMTrainer` wraps a `StateMachineGPT` with:

- an `AdamW` optimizer (owned by the model, config-driven learning rate),
- a training loop with NaN/Inf batch skipping, gradient clipping, and per-epoch
  loss logging,
- optional validation (`val_loader`), tracked as `history["val_loss"]`,
- checkpointing of the best model (by val loss if provided, else train loss) via
  `save_safetensors`,
- early stopping (`patience` epochs without improvement),
- periodic checkpoints via `save_every_epochs`,
- basic DDP-awareness (`_is_ddp` / `_module()` helpers) for multi-GPU setups.

`_compute_loss` uses MSE between the model's last-timestep prediction
(`logits[:, -1, :]`) and the target next frame `y`.

### `dataset.py` — data loading

Two `Dataset` implementations, both producing `(x, y)` samples where `x` is a
`(k+1, D)` history window and `y` is the `(D,)` next-frame target:

- **`MultiC3DNextPoseDataset`** — loads raw marker trajectories from `.c3d` mocap
  files (via `ezc3d`), drops noisy markers (`nan_thresh`), forward/back-fills NaNs,
  and flattens `(T, markers, 3)` into `(T, D)`.
- **`MultiSMPLXNPZNextPoseDataset`** — loads SMPL-X/AMASS-style `.npz` sequences
  (e.g. `poses`, `trans`, `betas`, `expression`, face pose). `build_smplx_features()`
  assembles the requested feature set (`feature_set="poses+trans"` by default) into
  a `(T, D)` matrix. This is the dataset used by `main.py` for the current `data/`
  (SMPL-X `.npz`) training data.

Shared behavior:

- Per-file z-score normalization (`normalize=True`) using each file's own mean/std.
- `strict_dim=False` (default) silently **skips** files whose feature width differs
  from the first successfully loaded file, so `ds.D` stays a single consistent
  `state_dim` across the whole dataset instead of erroring or mixing shapes. Set
  `strict_dim=True` to raise instead.
- `file_range=(start, end)` (SMPL-X dataset only) slices the sorted file list before
  loading, so a large corpus can be trained in chunks (e.g. `(0, 500)`, then
  `(500, 1000)`, ...) instead of building the full index (which can be millions of
  samples across thousands of files) in one go.

### `utils.py` / `logger.py`

- `Config` — recursively turns a nested dict (e.g. loaded from YAML) into an
  attribute-accessible object; `_to_jsonable` converts config values into a
  JSON-safe structure for checkpoint metadata.
- `logger` — a preconfigured `loguru` logger that writes timestamped, rotating log
  files to `logs/` and also mirrors messages to stdout.

## Quickstart

```python
from src.gpsm.experiment_model.sm_gpt_trainer import SMTrainer
from src.gpsm.utils.dataset import MultiSMPLXNPZNextPoseDataset
from torch.utils.data import DataLoader

config = {
    "n_embd": 256,
    "n_layers": 4,
    "n_head": 4,
    "block_size": 11,        # >= k + 1
    "max_timestep": 3000,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "loss_scale": 1.0,
}

ds = MultiSMPLXNPZNextPoseDataset(
    "data",
    k=8,                      # history window size (block_size must be >= k+1)
    feature_set="poses+trans",
    normalize=True,
    strict_dim=False,
)
config["state_dim"] = ds.D    # derive from the data, don't hardcode

train_loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=True)

trainer = SMTrainer(config)
history = trainer.train(
    train_loader,
    epochs=40,
    log_every=1,
    best_path="checkpoints/best_sm",
)
```

See [main.py](main.py) for the full example, including the (currently commented
out) `MultiC3DNextPoseDataset` path for raw `.c3d` marker data.

## Dependencies

- `torch`
- `numpy`
- `loguru`
- `safetensors`
- `ezc3d` (only required for the `.c3d` marker-based dataset path)

No `requirements.txt` is checked in yet — install the above via `pip` as needed for
your environment (e.g. Kaggle notebooks).
