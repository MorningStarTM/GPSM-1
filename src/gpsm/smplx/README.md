# src/gpsm/smplx — Mocap Unification

> **Origin.** This package was copied from the Leonidas project
> (`Leonidas/src/smplx`) so GPSM can train on many kinds of mocap. What was
> changed for this repo: it lives at `src/gpsm/smplx/` (internal imports are
> `src.gpsm.smplx...`), and `config.py` now finds the body model in this repo's
> `model/` folder (env var `GPSM_SMPLX_MODEL`; the old `LEONIDAS_SMPLX_MODEL`
> name still works). **Not copied:** the Streamlit viewer `app/` and the two
> design docs (`doc/Mocap_Unification_Design.docx`,
> `doc/SMPLX_Fitting_Methods.docx`) — the sections below that mention `app/`,
> frame budgets, video, or `requirements.txt` describe the original project and
> are kept only as background. Its unit tests are in `tests/mocap_unification/`.
> Needs `numpy`, `scipy`, `torch`, `smplx`, and `c3d` / `bvh` for those formats.


Converts mocap data of any origin into a unified SMPL-X representation
(159-d canonical pose: trans + global_orient + body_pose + both hand
poses). See `doc/Mocap_Unification_Design.docx` and
`doc/SMPLX_Fitting_Methods.docx` for the full design rationale — this file
documents what is actually implemented, tested, and shipped.

## Requirements

- `numpy`, `scipy`, `torch`, `smplx` (pip install them; `c3d` and `bvh` too for those formats)
- An official SMPL-X body model file (`SMPLX_{NEUTRAL,FEMALE,MALE}.npz`),
  registration-gated at https://smpl-x.is.tue.mpg.de and **not** included in
  this repo. Point to it with:

  ```bash
  export GPSM_SMPLX_MODEL=/path/to/SMPLX_FEMALE.npz
  ```

  Without it, Tier 0 (pure parameter conversion) still works fully; any
  fitting (Tier 1+, `fitting/`, `regressor.py`) raises a clear
  `SMPLXModelNotFoundError` rather than failing silently or faking a result.

## What's implemented and tested

| Piece | File(s) | Status |
|---|---|---|
| Canonical schema, part masks, quality scorecard | `schema.py`, `quality.py` | Done, unit-tested |
| 159-d axis-angle <-> 315-d 6D-rotation model features | `features.py` | Done, round-trip tested incl. near-singular rotations |
| fps resampling (SLERP), up-axis, units | `ops.py` | Done, unit-tested |
| Tier 0: SMPL/SMPL-H/SMPL-X params -> canonical | `adapters/params.py` | Done, tested against real AMASS sample data |
| Correspondence layouts (COCO-17, OpenPose-25, MediaPipe-33, H36M-17) | `layouts/` | Done, unit-tested |
| Method 1 — Similarity alignment (Umeyama/Kabsch) | `fitting/align.py` | Done, exact-recovery tested |
| Real SMPL-X body model (forward kinematics, batched, differentiable) | `fitting/body.py` | Done, requires the model file |
| Method 2 — Staged optimization solver | `fitting/solver.py` | Done, end-to-end synthetic-recovery tested against the real body model |
| Method 3 — Learned-regressor interface + 2D reprojection refinement | `regressor.py` | Refinement step is real and tested; the network itself (`NullRegressor`) is a documented stub — see below |

## Known, honest limitations

- **No VPoser.** The pose prior (`fitting/priors.py::GaussianJointPrior`) is
  a per-joint L2 + anatomical-limit substitute, not the VPoser latent-space
  prior the design doc specifies — no VPoser checkpoint was available in
  this environment. It implements the same `PosePrior` interface, so a real
  `VPoserPrior` can replace it without touching `solver.py`.
- **No trained image regressor.** `NullRegressor` raises
  `NotImplementedError` rather than fabricating a pose — running Method 3's
  network step requires downloading a SMPLer-X/NLF/OSX checkpoint, which
  this environment cannot do on your behalf. `refine_from_2d` (the
  reprojection-refinement half of Method 3) is real and tested; it just
  needs a `RegressorEstimate` to start from.
- **Hands are never fit by the Tier-1 skeleton solver.** The built-in
  layouts supply at most a few sparse fingertip points per hand — not
  enough to identify a 45-dof hand pose without overfitting noise. Hand
  pose is left at rest and `part_mask` marks it unobserved, honestly,
  rather than presenting a fabricated hand fit. A layout with real
  per-finger-joint correspondences can add a hand stage.
- **Face is never fit anywhere in this package.** It is out of scope of
  the 159-d canonical layout by design (see the main design doc, section
  2.1).

## Quick usage

```python
from src.gpsm.smplx.adapters.params import ingest_params   # Tier 0
from src.gpsm.smplx.adapters.skeleton3d import ingest_skeleton3d  # Tier 1
from src.gpsm.smplx.fitting.body import SMPLXBody

# Tier 0: already-parametric mocap (AMASS, Motion-X, ...)
motion = ingest_params(raw_npz_dict)  # -> CanonicalMotion

# Tier 1: 3D joints of unknown scale/orientation (BVH, MediaPipe world
# landmarks, a custom skeleton) -> fit to SMPL-X
body = SMPLXBody()  # finds the model file via GPSM_SMPLX_MODEL
motion, quality = ingest_skeleton3d(points, fps=30.0, body=body, layout="mediapipe_33")
print(quality.verdict)  # GOOD | DEGRADED | REJECT
```

## Streamlit viewer app (`app/`)

A two-tab visual UI for exactly this pipeline: upload a `.npz` (SMPL/SMPL-H/
SMPL-X params), `.c3d` (optical mocap markers), or a video, and compare it
raw against a unified SMPL-X fit.

```bash
export GPSM_SMPLX_MODEL=/path/to/SMPLX_FEMALE.npz
streamlit run app/streamlit_app.py
```

- **Tab 1 — Raw Mocap**: the data exactly as the source format measured it,
  always the *complete* clip (see "Frame budgets" below) — `.npz` params
  are shown as a skeleton (direct forward kinematics, no fitting). `.c3d`
  markers are shown as a raw point cloud. `.bvh` files are shown as a
  skeleton using the file's own bone hierarchy. Video is run through
  MediaPipe Pose to extract real 3D world-landmark mocap, shown as a
  skeleton. Unobserved points are drawn in red, never hidden. Playback is
  a native Plotly animation — the browser drives it, not the Python
  backend, so the page stays scrollable and interactive while it plays.
- **Tab 2 — SMPL-X Fit**: the same clip unified onto the real SMPL-X mesh
  and rendered with `pyrender` (a genuine lit, shaded 3D render), plus the
  `FitQuality` verdict (GOOD/DEGRADED/REJECT) and the fitting notes (e.g.
  how many C3D markers or BVH joints matched a known naming convention, or
  that hands are shown at rest). Every frame is pre-rendered once and
  handed to a small client-side HTML/JS player with its own Play/Pause/
  scrub/Fullscreen controls — same reasoning as the raw tab's animation:
  no Python execution drives playback, so scrolling and its own dedicated
  Fullscreen button both work normally while it plays.

### Frame budgets

The Raw Mocap tab always shows the **complete** clip for every format
except video — there is no cost to displaying more raw points, so
truncating it would only hide part of the "as it is" mocap for no reason
(a real bug this project shipped and then fixed: a fixed 200-frame — or,
at one point, a 30-frame default — cap silently truncated real captures
that ran several hundred frames, making the "full simulation" invisible).

Only the **SMPL-X Fit tab** has a frame budget (`max_fit_frames`, a
sidebar slider, default 60): fitting is an iterative optimization and mesh
rendering runs once per frame, so both have a real per-frame cost. A clip
longer than the budget is **evenly subsampled across its entire duration**
(`app/pipeline.py::_fit_frame_selection`), not truncated to its opening
frames — so the fit tab still previews the whole motion, just at a lower
frame density for longer clips.

Video uploads have one additional, unavoidable cap (`max_extract_frames`,
also a sidebar slider): MediaPipe runs a neural network on every extracted
frame, so — unlike the other formats — the raw tab for video can only ever
show however many frames were actually extracted.

`.npz` params need no fitting (Tier 0, pure conversion). `.c3d` markers fit
via `adapters/markers.py`'s `VICON_50` vertex-correspondence layout (built
from real marker->vertex ids extracted from an actual AMASS/SOMA capture
file — see `layouts/builtin.py`), matching whichever of its 50 markers the
upload actually has by label (namespace-prefixed labels like `male2:LFHD`
are recognized too). `.bvh` files are parsed and forward-kinematics'd
ourselves (`adapters/bvh.py`; the `bvh` package only exposes raw channels,
not world positions), auto-detecting both the file's up-axis and its
unit scale — BVH has no standard for either, and the unit heuristic is
careful to use only the per-frame vertical span, not the full bounding-box
diagonal, so a clip where the character travels a long distance
horizontally (a run, a traveling skip) doesn't get mistaken for a giant
body. Joint names are matched against a CMU/Mixamo-style alias table
(`mixamorig:` and similar namespace prefixes are stripped automatically).
Video fits via MediaPipe's real 3D world landmarks through the Tier-1
solver (`mediapipe_33` layout) — this sidesteps needing the (unavailable,
see above) trained regressor checkpoint entirely, since MediaPipe already
provides genuine metric 3D pose.

**MediaPipe's axis convention** (`video_extract.py::_mediapipe_world_to_zup`):
`pose_world_landmarks` use the same Y-down convention as raw image pixels
(rescaled to meters), not this project's Z-up standard — confirmed against
three independent real videos, not assumed. Left uncorrected, this both
makes the raw skeleton view look tipped onto its side and feeds
wrongly-oriented data into the solver. The fix is a 180-degree rotation
about X (not a single-axis sign flip, which would silently mirror left/
right — a determinant -1 reflection, not a rotation), applied once, right
after extraction, before anything else touches the coordinates.

**Fitting hard, wide-motion-range video is a known, disclosed limitation,
not silently hidden.** A single shared body shape (`betas`) is estimated
across every fitted frame; when those frames span a large real range of
motion under real occlusion (e.g. a full barbell squat set, camera behind
a rack, hands gripping a bar behind the neck), the noisier frames can pull
the shared shape estimate off in a way that degrades the fit for every
frame, not just the hard ones — confirmed directly against a real gym
video (a narrow 8-frame standing-only window fit at ~44mm RMSE /
`DEGRADED`; the same clip's full squat range fit at 230mm+ / `REJECT`,
even with more LBFGS iterations, a tighter robust-loss radius, and
temporal-smoothness fully disabled — ruling out those as the cause).
`FitQuality` correctly reports `REJECT` for that case rather than a
falsely reassuring `GOOD`; a proper fix (e.g. not sharing betas so
rigidly, or fitting overlapping windows) is a real, separate piece of
future work, not something papered over here.

A real, generalizable bug *was* found and fixed along the way:
`temporal_smoothness`/`translation_acceleration` (fitting/losses.py) compute
plain frame-to-frame differences with no notion of elapsed real time, so a
heavily-subsampled clip (`app/pipeline.py::subsample_for_fit`, used by
every Tier-1/2 format when a clip exceeds the fit-frame budget) had its
genuine motion penalized as if it were implausible jitter, scaled up by
however many real frames were skipped between samples.
`solver.scale_temporal_weights_for_subsampling` corrects for this (1/step
for the first-difference term, 1/step**2 for the second-difference term),
applied everywhere `subsample_for_fit` is used — a no-op when nothing was
subsampled.

C3D and BVH fits are only as good as how well the uploaded file's marker/
joint placements match the specific protocol our correspondence tables were
derived from — a file using the same marker *names* but a different lab's
placement convention can still land in `DEGRADED`/`REJECT` even though it
loads and fits without error; that's `FitQuality` doing its job, not a bug.

Tested via Streamlit's `AppTest` harness (`tests/test_streamlit_app.py`)
against real sample files, plus `tests/test_app_pipeline.py` for the
upload-processing logic itself, and confirmed to boot as a live server with
no startup errors.

## Running the tests

```bash
python -m pytest tests/mocap_unification -q
```

Run from the repo root (not from inside `src/`) so `from src.gpsm.smplx import
...` resolves correctly and does not collide with the installed `smplx` pip
package. Model-dependent tests skip automatically if no SMPL-X file is
found; the Tier-0 real-data regression tests skip if the GPSM-1 sample data
path isn't present on the machine.
