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

python -m src.gpsm.tests.ground_truth_inference data/0019_lifting_heavy1_poses.npz --n-frames 128 --model-folder model/SMPLX_FEMALE.npz --fps 26
# -> 0019_lifting_heavy1_poses_groundtruth_joints.gif, 0019_lifting_heavy1_poses_groundtruth_mesh.gif
```

`--start` (default `0`) picks which real frame the ground-truth clip begins at,
in case you want to compare against a rollout seeded from somewhere other than
frame 0.

## 3D Simulator (SMPL-X live viewer)

`simulator/` is a live 3D viewer for GPSM, built on `pyrender` + `trimesh`. It
sits outside `src/gpsm` on purpose — it's a visualization/testing tool built
on top of a trained checkpoint, not part of the model or training pipeline.
Where the `tests/` scripts above bake a rollout into a pre-rendered GIF, the
simulator opens a real window with a game-engine-editor-style camera (see
`fly_camera.py` below) and plays the posed SMPL-X body back live (or drives
it frame-by-frame from `StateMachineGPT.rollout()` directly).

```
simulator/
├── smplx_scene.py     pyrender scene build/update (ground plane, camera, lighting, body mesh)
├── fly_camera.py         game-engine-style fly camera (WASD + mouse-look) — GameViewer(pyrender.Viewer)
├── model_runner.py     checkpoint/npz -> denormalized rollout -> SMPL-X forward kinematics -> vertices
├── live_viewer.py       interactive GameViewer window (requires a display)
└── offscreen_test.py     headless render-to-GIF/PNG self-test (no display required)
```

Requires `pyrender`, `PyOpenGL`, and `trimesh` in addition to the core/tests
dependencies below (`pip install pyrender PyOpenGL PyOpenGL_accelerate`).

Note (applies to every command below): a checkpoint's `state_dim` must match
the `.npz` file's feature width (`poses` width + `trans`) — see the
`dataset.py` section above (`strict_dim`) and `simulate_smplx.py`'s
`POSES_LAYOUTS`. `checkpoints/best_sm` was trained on `state_dim=159` (the
156-dim hands-only pose layout, e.g. `data/0019_lifting_heavy1_poses.npz`,
`data/0013_knocking1_poses.npz`, `data/0029_jumping2_poses.npz`); the
`12_*_stageii.npz` / `SW_B_*_stageii.npz` files use the 165-dim full
face+hands layout (`D=168`) and will raise a clear "Feature width mismatch"
error against that checkpoint — pass `--ground-truth` (no model involved) to
view those instead.

### `live_viewer.py` — interactive 3D window (main entry point)

Opens a real OpenGL window (via `pyglet`) and plays a motion sequence back
live as a posed SMPL-X body, using a **game-engine-editor-style fly camera**
(`fly_camera.py`, below) — like Unity's Scene view or Unreal's editor
viewport: the camera moves and looks around a fixed world; the ground and
body never rotate. This is what you run to actually *watch* the model. Two
modes, chosen by whether `--ground-truth` is passed:

- **Model rollout (default)** — seeds `StateMachineGPT` with the first frame
  of `npz_path`, autoregressively predicts `--n-steps` future frames via
  `rollout()`, and plays that back.
- **Ground truth** (`--ground-truth`) — plays back the real recorded frames
  from `npz_path` instead, no model/checkpoint involved. Useful as a sanity
  check or to compare against a rollout by eye.

Controls once the window is open: **left-drag** to look around (camera
rotates in place, world stays fixed), **W/A/S/D** to move forward/left/
backward/right, **Space**/**Left-Ctrl** to move up/down, **scroll** to adjust
move speed, **q**/**ESC** to quit. Other pyrender hotkeys still work too
(`f` fullscreen, `l` cycle lighting, `z` reset view, ...). Add `--move-speed`
to change the starting speed (default `3.0` m/s; scroll in-viewer to adjust
live).

```
# model rollout — the main use case
python -m simulator.live_viewer data/0019_lifting_heavy1_poses.npz \
    --checkpoint checkpoints/best_sm --model-folder model/SMPLX_FEMALE.npz --n-steps 90

# ground-truth playback (no --checkpoint needed)
python -m simulator.live_viewer data/12_L_2_stageii.npz \
    --model-folder model/SMPLX_FEMALE.npz --ground-truth --n-steps 150
```

Key flags: `--n-steps` (rollout: frames to predict; ground-truth: frames to
play), `--start` (ground-truth only — starting frame index), `--fps`
(playback speed, default 24), `--device` (`cpu`/`cuda`, default auto),
`--gender`/`--num-betas`/`--use-pca`/`--no-flat-hand-mean` (SMPL-X body model
options), `--keep-root-motion` (show true world-space translation instead of
recentring on the pelvis every frame), `--no-loop` (play once instead of
looping), `--auto-close-seconds N` (close the window automatically after N
seconds — mainly for scripted/automated smoke tests).

### `fly_camera.py` — game-engine-style camera (library, not run directly)

Also not run directly — this is what actually answers "make it move like a
game engine viewport, not orbit the whole scene." `pyrender.Viewer`'s default
camera is a *trackball*: dragging the mouse rotates the entire scene (ground
included) around a pivot point. `fly_camera.py` replaces that with:

- **`FlyCamera`** — a plain yaw/pitch camera (position + heading), Z-up to
  match `smplx_scene.py`'s world convention. `look(dyaw, dpitch)` turns the
  camera in place; `move(forward, right, up)` translates it along its own
  current facing; `to_matrix()` produces the camera-to-world pose pyrender
  needs. None of this ever touches the scene itself.
- **`GameViewer(pyrender.Viewer)`** — a subclass that swaps pyrender's
  trackball mouse/keyboard handling for `FlyCamera`: left-drag looks around,
  WASD moves, Space/Left-Ctrl move up/down, scroll adjusts move speed, ESC
  quits. Every other pyrender hotkey (fullscreen, lighting, wireframe, ...)
  still works unchanged — only the letter keys that collided with WASD were
  overridden. `live_viewer.py` uses `GameViewer` in place of
  `pyrender.Viewer`; nothing else about it changed.

Only worth importing directly if you're building another interactive tool on
top of the same camera; normal usage is through `live_viewer.py` above.

### `offscreen_test.py` — headless render check (no window)

Runs the exact same rendering pipeline as `live_viewer.py` but through
`pyrender.OffscreenRenderer` instead of a live window, so it works with no
display at all, and saves the result as a GIF + one mid-sequence PNG instead
of playing it back. It also actively checks every rendered frame is finite
and non-blank and raises immediately if not. Use this to sanity-check a
checkpoint, a new `.npz` file, or a code change to `smplx_scene.py` /
`model_runner.py`, without needing to sit and watch a window.

```
# model rollout, headless
python -m simulator.offscreen_test data/0019_lifting_heavy1_poses.npz \
    --checkpoint checkpoints/best_sm --model-folder model/SMPLX_FEMALE.npz --n-steps 20

# ground-truth, headless
python -m simulator.offscreen_test data/12_L_2_stageii.npz \
    --model-folder model/SMPLX_FEMALE.npz --ground-truth --n-steps 40
```

Same `--n-steps`/`--start`/`--fps`/`--device`/`--gender`/`--num-betas`/
`--use-pca`/`--no-flat-hand-mean`/`--keep-root-motion`/`--ground-truth` flags
as `live_viewer.py` above, plus `--out-dir` (default `simulator/output/`,
already gitignored) for where the GIF/PNG get written.

### `model_runner.py` — checkpoint/npz → posed vertices (library, not run directly)

Not a script you run from the command line — it's the shared logic both
`live_viewer.py` and `offscreen_test.py` call into. Deliberately reuses the
already-validated rollout/denormalization/SMPL-X logic from
`src/gpsm/tests/rollout_inference.py` and `src/gpsm/tests/simulate_smplx.py`
instead of reimplementing it, and wraps the result in a small `MotionSequence`
container (`.vertices` `(T, V, 3)`, `.joints` `(T, J, 3)`, `.faces` `(F, 3)`,
`.fps`). Exposes two functions:

- `generate_rollout_sequence(npz_path, checkpoint_path, model_folder, n_steps, ...)`
  — seed + autoregressive model rollout → posed vertices.
- `generate_ground_truth_sequence(npz_path, model_folder, n_frames, start, ...)`
  — real recorded frames → posed vertices, no model.

Useful directly if you're scripting something new (e.g. a notebook, a batch
comparison over many checkpoints) rather than using either CLI:

```python
from simulator.model_runner import generate_rollout_sequence

seq = generate_rollout_sequence(
    "data/0019_lifting_heavy1_poses.npz",
    checkpoint_path="checkpoints/best_sm",
    model_folder="model/SMPLX_FEMALE.npz",
    n_steps=90,
)
print(seq.n_frames, seq.vertices.shape, seq.faces.shape)
```

### `smplx_scene.py` — pyrender scene helpers (library, not run directly)

Also not run directly — the low-level pyrender building blocks both the
live and offscreen paths share, kept separate from `model_runner.py` so the
"get posed vertices" logic and the "turn vertices into a pyrender scene"
logic don't get tangled together:

- `build_scene(vertices0, faces, viewport_size=(960, 720))` — builds a fresh
  `pyrender.Scene` for frame 0: a ground plane sized to the body, the posed
  body mesh, a camera framing it, and two directional lights. Returns
  `(scene, body_node, camera_node)`. Built around **world +Z as "up"**,
  verified empirically against this repo's actual SMPL-X output (not
  assumed) — see the module docstring if adapting this for a different SMPL
  variant or dataset convention.
- `update_body_mesh(scene, body_node, vertices, faces)` — swaps the body
  node's mesh for a new frame's pose (pyrender can't mutate a node's mesh
  buffers in place, so this removes and re-adds the node each call). Returns
  the new node — callers must keep using the returned node, not the old one.

Only worth importing directly if you're building a different renderer/viewer
on top of the same scene layout; normal usage is through `live_viewer.py` /
`offscreen_test.py` above.

## Control extraction (prototype)

To steer the model (press left, turn left) it needs a control input, but the
mocap files contain none. `src/gpsm/control/` derives one from the recording
itself: for every frame, where the character's root goes over the next
0.1 / 0.25 / 0.5 s and how much it turns, expressed relative to the
character's own position and heading (so it does not depend on where or which
way the clip was recorded). Works on both `.npz` (AMASS, SMPL-H and SMPL-X)
and raw `.c3d` marker files — for `.c3d` the root is estimated from the four
waist markers (`LFWT`, `RFWT`, `LBWT`, `RBWT`).

```
src/gpsm/control/
├── root_trajectory.py    .npz / .c3d -> RootTrajectory (ground position + heading, metres, Z-up)
├── control_features.py   RootTrajectory -> per-frame control vector (14 numbers) + valid mask
└── inspect_control.py    CLI: summary table, plots, and saved control arrays for a folder
```

```
# summary table + a plot and a control .npz per file, into src/gpsm/control/output/
python -m src.gpsm.control.inspect_control data

# a single file
python -m src.gpsm.control.inspect_control "data/B9 -  Walk turn left 90.c3d"

pytest tests/test_control.py -v
```

Reading `.c3d` needs `pip install c3d` (the existing `dataset.py` loader uses
`ezc3d`, which has no wheel for every platform). On the two labelled clips the
extracted turn while walking is +87 deg for "Walk turn left 90" and +142 deg
for "Walk turn left 135". This is a prototype: it produces the control signal
but does not yet feed it to a model.

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
