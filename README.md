# GPSM-1

**GPSM** (Generative Pre-trained State Machine) is a neural motion model trained on
large-scale motion-capture data to predict and generate human motion, aiming to
replace hand-crafted animation state machines used in traditional motion-matching
systems.

Instead of authoring transition rules and blend trees by hand, GPSM learns the
dynamics of human movement directly from mocap sequences: given a short window of
past poses, it predicts the next pose. Chained autoregressively, this turns into a
generative motion model that can roll a character forward frame by frame.

## Demo — trained-model inference (next-frame rollout)

The clips below are the **trained checkpoint** (`checkpoints/best_sm`) actually
running inference — not ground-truth mocap. The model is seeded with a single real
pose frame from a held-out `.npz` sequence, then repeatedly predicts its own next
frame and feeds that prediction back in as the next input. This is exactly the
next-token generation loop a language-model GPT runs at inference time — predict
one step ahead, append it to the context, repeat — except each "token" here is a
continuous SMPL-X pose vector instead of a discrete word. Concretely, this is
`StateMachineGPT.rollout(seed_frames, n_steps)`
([sm_gpt.py](src/gpsm/experiment_model/sm_gpt.py)): **1 seed frame in, 8
autoregressively generated frames out**, with no ground truth fed in after the
first frame.

The model itself only outputs a normalized 168-dim vector (SMPL-X `poses` +
`trans`) — not something directly visual. To inspect it, each predicted frame is
denormalized and passed through the real SMPL-X body model (forward kinematics) to
recover 3D joint positions and a posed mesh. Both columns below are the *same*
9-frame predicted sequence (1 seed + 8 predicted), just rendered two ways:


| **Joints (skeleton)** | **Mesh (SMPL-X body)** |
|---------|-------------|
|![App Screenshot](https://github.com/MorningStarTM/GPSM-1/blob/5ffd64cdefd5664d286c25b9b1d1414c2157e4d4/assets/12_L_2_stageii_rollout_joints.gif)|![App Screenshot](https://github.com/MorningStarTM/GPSM-1/blob/5ffd64cdefd5664d286c25b9b1d1414c2157e4d4/assets/12_L_2_stageii_rollout_mesh.gif)

Reproduce it with:

```
python -m src.gpsm.tests.rollout_inference data/0019_lifting_heavy1_poses.npz --checkpoint checkpoints/best_sm --n-steps 128 --model-folder model/SMPLX_FEMALE.npz --mesh --fps 8
```

This checkpoint is only lightly trained — the purpose of this demo is to validate
the **inference pipeline end-to-end** (seed frame → autoregressive rollout →
denormalization → SMPL-X forward kinematics → render), not to showcase polished
motion quality yet. See [`tests/` — inference & visualization
tooling](#tests--inference--visualization-tooling) below for how each piece works.

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
assets/                                  README demo GIFs
model/                                   (local) downloaded SMPL-X model weights — not checked in
src/gpsm/
├── experiment_model/
│   ├── sm_gpt.py                        StateMachineGPT model + config validation
│   └── sm_gpt_trainer.py                SMTrainer: training/validation loop
├── utils/
│   ├── dataset.py                       mocap → (x, y) window datasets
│   ├── logger.py                        loguru-based file + console logger
│   └── utils.py                         small config/JSON helpers
└── tests/
    ├── frame_loader.py                  npz → model-ready feature frame(s)
    ├── visualize_npz.py                 raw npz inspection/plots (no SMPL-X needed)
    ├── simulate_smplx.py                any npz clip -> SMPL-X forward kinematics -> GIF
    ├── rollout_inference.py             trained checkpoint -> autoregressive rollout -> SMPL-X GIF
    └── ground_truth_inference.py        same npz clip, real frames (no model) -> SMPL-X GIF
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

### `tests/` — inference & visualization tooling

Scripts for inspecting data and testing a trained model by actually running it,
rather than just reading loss numbers. All are runnable as modules
(`python -m src.gpsm.tests.<name> ...`) and write their output (GIFs/PNGs) to
`src/gpsm/tests/output/` by default.

- **`frame_loader.py`** — `load_sequence_features(npz_path, ...)` /
  `get_first_frame(npz_path, ...)`: builds the `(T, D)` feature matrix (or just
  frame `0`) for one `.npz` file, using the *same* feature construction and
  per-file z-score normalization as `MultiSMPLXNPZNextPoseDataset`, so a frame
  pulled here is numerically identical to what the model saw during training.
  Also returns the per-file `mean`/`std` needed to denormalize model output back
  to real pose values.
- **`visualize_npz.py`** — quick, dependency-light inspection of a raw `.npz`
  file: prints all keys/shapes, animates `trans` (root motion path) or
  `joints3d` if present, and plots a `poses` heatmap over time. No SMPL-X model
  required.
- **`simulate_smplx.py`** — runs a **ground-truth** mocap sequence through the
  real SMPL-X body model (global_orient/body_pose/hand/jaw/eye pose → forward
  kinematics) and animates the result as joints and/or a full posed mesh.
  Requires the `smplx` pip package and SMPL-X model weights, which are licensed
  and must be downloaded by hand after registering at
  [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) — see the module
  docstring for the exact folder layout expected.
- **`rollout_inference.py`** — the trained-model inference test: loads a
  checkpoint (config + weights from `<checkpoint>.json` / `.safetensors`), seeds
  `StateMachineGPT.rollout()` with the first real frame of a chosen `.npz` file,
  autoregressively predicts `n_steps` future frames, denormalizes them, and
  (optionally, given `--model-folder`) renders the predicted sequence through
  SMPL-X the same way `simulate_smplx.py` does. This is what produced the demo
  GIFs at the top of this README.
- **`ground_truth_inference.py`** — the ground-truth counterpart to
  `rollout_inference.py`: takes the *actual recorded* frames
  `[start : start + n_frames]` straight from an `.npz` file (no model, no
  normalization) and renders them through SMPL-X the same way, producing a
  joints GIF and a mesh GIF. Used to compare "what the model predicted" against
  "what really happened" over the identical frame range — see below.

#### Comparing model prediction vs. ground truth

Run both scripts on the same file with matching frame counts (`--n-steps N` on
one, `--n-frames N+1` on the other — `+1` because the ground-truth clip
includes the shared seed frame that the rollout also starts from) and play the
resulting GIFs side by side:

```
python -m src.gpsm.tests.rollout_inference data/12_L_2_stageii.npz \
    --checkpoint checkpoints/best_sm --n-steps 8 \
    --model-folder model/SMPLX_FEMALE.npz --mesh --fps 4
# -> 12_L_2_stageii_rollout_joints.gif, 12_L_2_stageii_rollout_mesh.gif

python -m src.gpsm.tests.ground_truth_inference data/12_L_2_stageii.npz --n-frames 128 --model-folder model/SMPLX_FEMALE.npz --fps 26
# -> 12_L_2_stageii_groundtruth_joints.gif, 12_L_2_stageii_groundtruth_mesh.gif
```

`--start` (default `0`) picks which real frame the ground-truth clip begins at,
in case you want to compare against a rollout seeded from somewhere other than
frame 0.

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

Core (training):

- `torch`
- `numpy`
- `loguru`
- `safetensors`
- `ezc3d` (only required for the `.c3d` marker-based dataset path)

`tests/` tooling (data inspection, inference demo, SMPL-X visualization):

- `matplotlib` — all GIF/PNG rendering
- `smplx` — only for `simulate_smplx.py` / `rollout_inference.py`'s `--model-folder`
  path (real forward-kinematics visualization). Also requires SMPL-X model
  weights, which are licensed and must be downloaded by hand from
  [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) after registering — see
  `simulate_smplx.py`'s module docstring for the exact steps and folder layout.

No `requirements.txt` is checked in yet — install the above via `pip` as needed for
your environment (e.g. Kaggle notebooks).
