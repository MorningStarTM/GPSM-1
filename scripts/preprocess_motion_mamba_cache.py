"""One-time CPU-only preprocessing: scan every mocap .npz once, build and
normalize pose features, split by FILE into train/val, and cache the result
as train_1.pt, train_2.pt, ... (val.pt stays one file -- see below).

WHY THIS EXISTS
----------------
MultiSMPLXNPZNextPoseDataset's constructor re-does the expensive part --
loading every .npz, building its feature array, filling gaps, computing
normalization stats -- from scratch every time it's built, with no
progress output and no GPU involvement at all. On a large combined AMASS
corpus that can take many minutes, and it reruns on every notebook restart.
Worse on Kaggle specifically: if that scan runs inside a GPU-enabled
session, it burns your limited weekly GPU-hour quota on work that never
touches the GPU. Run this script once in a CPU-only session, then every
training run (GPU session or not) loads the cached result in roughly the
time it takes to read a file off disk.

WHAT'S CACHED, AND WHY NOT PRE-WINDOWED SAMPLES
--------------------------------------------------
Each file's feature array is cached whole -- (T, D), already built, gap-
filled, and normalized -- not exploded into overlapping (k+1)-length
windows. Windowing was never the slow step; re-parsing .npz files and
rebuilding features was. Pre-exploding windows would bloat the cache by
roughly (k+1)x for a step that costs nothing to do lazily at load time, and
would also bake in a specific `k` -- this cache works for any `k`.

TRAIN/VAL SPLIT IS BY FILE, NOT BY WINDOW
--------------------------------------------
Splitting individual windows would put windows from the same clip on both
sides, so validation would partly measure memorization of clips the model
already trained on. Whole files are randomly assigned to train or val
instead, so every val clip is one the model never saw.

WHY TRAIN IS SPLIT INTO SEVERAL SMALL FILES, AND VAL ISN'T
----------------------------------------------------------------
With millions of frames in the corpus, one giant train.pt is a lot to hold
in memory and a lot to lose if a session times out mid-epoch (see
MotionMambaSFTTrainer's docstring for that exact problem). --frames-per-shard
(default 10,000) controls the target size, and the number of shards this
produces is simply:

    x = total_train_frames / frames_per_shard

Each shard still holds whole files (never splits one file's frames across
two shards -- same reasoning as the train/val split above), so actual shard
sizes vary a little around the target rather than landing on it exactly.
Point a training run's --train-file at ONE shard (e.g. "train_3.pt") to
train on that portion only, then move to the next shard next session --
same chunk-by-chunk idea as sft_main.py's FILE_START/FILE_END, just applied
to the cache instead of raw files.

val.pt stays a single file on purpose: it's read in full every epoch for a
quick quality check, not iterated through piece by piece, so there's no
benefit to sharding it -- shrinking --val-fraction (default lowered to 0.05
here, since training wants as much of the data as it can get) is what keeps
it small instead.

USAGE (Kaggle, CPU-only session -- no GPU needed, see note above)
--------------------------------------------------------------------
Writing to /kaggle/working alone is not enough -- that is this *session's*
output, gone once it ends. --kaggle-dataset-slug pushes everything to an
actual Kaggle dataset in the same run, reusing download_kit_to_kaggle.py's
already-tested publish_to_kaggle (title-collision-safe, catches the Kaggle
CLI's silent-success-on-error bug) rather than a separate hand-run publish
step. Needs the same two Secrets as that script: KAGGLE_USERNAME, KAGGLE_KEY.

    !git clone -q https://github.com/MorningStarTM/GPSM-1.git
    %cd GPSM-1
    # -m, not `python scripts/....py` directly -- the latter does not add
    # the repo root to sys.path, so `from src.gpsm...` fails inside it.
    !python -m scripts.preprocess_motion_mamba_cache /kaggle/input \
        --out /kaggle/working/cache \
        --frames-per-shard 10000 \
        --kaggle-dataset-slug motion-mamba-cache

A later GPU session then attaches the "motion-mamba-cache" dataset and
loads one shard directly -- no scan, no wasted GPU time.

LOADING ONE SHARD FOR TRAINING
----------------------------------
    cache = torch.load("train_3.pt")
    k = cache["meta"]["k"]
    windows = []  # (x, y) pairs, built lazily -- cheap, see module docstring
    for rec in cache["files"]:
        X = rec["X"]                      # (T, D) -- already normalized
        for t in rec["valid_t"]:
            windows.append((X[t - k:t + 1], X[t + 1]))
    # wrap `windows` in any torch.utils.data.Dataset / DataLoader as usual.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List

import torch

from src.gpsm.utils.dataset import MultiSMPLXNPZNextPoseDataset


def _shard_by_frames(records: List[dict], frames_per_shard: int) -> List[List[dict]]:
    """Greedily group whole file-records into shards of roughly
    `frames_per_shard` frames each. Never splits a single file's frames
    across two shards (same "keep a file atomic" reasoning as the
    train/val split), so shard sizes vary a little around the target
    rather than landing on it exactly -- a lone file bigger than
    `frames_per_shard` simply becomes its own oversized shard rather than
    being cut up or causing an infinite loop.
    """
    shards: List[List[dict]] = []
    current: List[dict] = []
    current_frames = 0
    for rec in records:
        n = rec["X"].shape[0]
        if current and current_frames + n > frames_per_shard:
            shards.append(current)
            current, current_frames = [], 0
        current.append(rec)
        current_frames += n
    if current:
        shards.append(current)
    return shards


def build_and_split(
    data_dir: str,
    out_dir: str,
    k: int = 8,
    feature_set: str = "poses+trans",
    val_fraction: float = 0.05,
    frames_per_shard: int = 10000,
    seed: int = 0,
    strict_dim: bool = False,
    kaggle_dataset_slug: str = None,
) -> None:
    """Run the one-time scan, write train_1.pt/train_2.pt/.../val.pt, and
    (if a slug is given) push them to an actual Kaggle dataset in the same
    run -- writing to /kaggle/working alone is this session's output, gone
    once it ends.

    Args:
        data_dir:      Folder of raw mocap .npz, searched recursively (e.g.
                        "/kaggle/input" to cover every attached dataset).
        out_dir:        Where to write the shards + val.pt.
        k:              History window length used only to report sample
                         counts below -- the cache itself does not depend
                         on it (see module docstring).
        feature_set:    Passed straight through to MultiSMPLXNPZNextPoseDataset.
        val_fraction:   Fraction of FILES (not windows) held out for val --
                         kept small by default so most of the corpus goes to
                         training; val.pt is not sharded (see module
                         docstring for why).
        frames_per_shard: Target frame count per train shard -- see module
                         docstring's "x = total_train_frames /
                         frames_per_shard" explanation.
        seed:           For the train/val file split, so it's reproducible.
        strict_dim:     Passed straight through -- False (default) silently
                         drops files whose pose width doesn't match the
                         majority; True raises on the first mismatch instead.
        kaggle_dataset_slug: If given, publish out_dir to
                         <your-username>/<this> as a private Kaggle dataset
                         via download_kit_to_kaggle.py's already-tested
                         publish_to_kaggle (needs KAGGLE_USERNAME/KAGGLE_KEY
                         Secrets). Omit to just save locally.
    """
    print(f"Scanning {data_dir} ...")
    ds = MultiSMPLXNPZNextPoseDataset(
        data_dir,
        k=k,
        feature_set=feature_set,
        include_betas=False,
        include_expression=False,
        include_face=False,
        prefer_joints_if_available=False,
        normalize=True,
        preload=False,
        strict_dim=strict_dim,
    )
    stats = ds.file_stats()
    print(f"Scanned {stats['num_files']} files -> {len(ds)} windowed samples "
          f"(k={k}), state_dim={ds.D}")

    num_files = len(ds._files)
    rng = random.Random(seed)
    file_order = list(range(num_files))
    rng.shuffle(file_order)
    n_val_files = max(1, int(round(num_files * val_fraction))) if num_files > 1 else 0
    val_file_ids = set(file_order[:n_val_files])
    train_file_ids = set(file_order[n_val_files:])

    def pack(file_ids: set) -> list:
        records = []
        for file_i in sorted(file_ids):
            rec = ds._files[file_i]
            records.append({
                "path": str(rec["path"]),
                "X": torch.from_numpy(rec["X_np"]).float(),  # (T, D) -- built, gap-filled, normalized
                "valid_t": rec["valid_t"],
            })
        return records

    train_records = pack(train_file_ids)
    val_records = pack(val_file_ids)

    meta = {
        "state_dim": ds.D,
        "k": k,
        "feature_set": feature_set,
        "seed": seed,
        "val_fraction": val_fraction,
        "frames_per_shard": frames_per_shard,
        "source_dir": str(data_dir),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- train: sharded ---
    total_train_frames = sum(r["X"].shape[0] for r in train_records)
    estimated_shards = max(1, math.ceil(total_train_frames / frames_per_shard)) if train_records else 0
    print(f"Total train frames: {total_train_frames}  |  "
          f"x = total_frames / frames_per_shard = {total_train_frames} / {frames_per_shard} "
          f"~ {estimated_shards} shard(s)")

    train_shards = _shard_by_frames(train_records, frames_per_shard)
    for i, shard in enumerate(train_shards, start=1):
        shard_frames = sum(r["X"].shape[0] for r in shard)
        shard_windows = sum(len(r["valid_t"]) for r in shard)
        torch.save({"files": shard, "meta": meta}, out / f"train_{i}.pt")
        print(f"train_{i}.pt: {len(shard)} files, {shard_frames} frames, "
              f"{shard_windows} windowed samples -> {out / f'train_{i}.pt'}")

    # --- val: one file, not sharded (see module docstring) ---
    torch.save({"files": val_records, "meta": meta}, out / "val.pt")
    val_frames = sum(r["X"].shape[0] for r in val_records)
    val_windows = sum(len(r["valid_t"]) for r in val_records)
    print(f"val.pt:   {len(val_records)} files, {val_frames} frames, "
          f"{val_windows} windowed samples -> {out / 'val.pt'}")

    if kaggle_dataset_slug:
        # Reuse download_kit_to_kaggle.py's credential handling and publish
        # logic rather than duplicating it -- both already handle the
        # title-collision and silent-CLI-success-on-error cases found
        # while building that script.
        from scripts.download_kit_to_kaggle import _secret, publish_to_kaggle

        kaggle_username = _secret("KAGGLE_USERNAME", "KAGGLE_USERNAME")
        kaggle_key = _secret("KAGGLE_KEY", "KAGGLE_KEY")
        publish_to_kaggle(out, kaggle_dataset_slug, kaggle_username, kaggle_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="Folder of raw mocap .npz, searched recursively")
    parser.add_argument("--out", default="processed_cache", help="Where to write the shards + val.pt")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--feature-set", default="poses+trans")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                         help="Fraction of files held out for val -- kept small by default "
                              "so most of the corpus goes to training (default: %(default)s)")
    parser.add_argument("--frames-per-shard", type=int, default=10000,
                         help="Target frame count per train shard (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-dim", action="store_true")
    parser.add_argument("--kaggle-dataset-slug", default=None,
                         help="If given, also push the shards + val.pt to this Kaggle dataset slug")
    args = parser.parse_args()

    build_and_split(
        args.data_dir, args.out, k=args.k, feature_set=args.feature_set,
        val_fraction=args.val_fraction, frames_per_shard=args.frames_per_shard,
        seed=args.seed, strict_dim=args.strict_dim,
        kaggle_dataset_slug=args.kaggle_dataset_slug,
    )


if __name__ == "__main__":
    main()
