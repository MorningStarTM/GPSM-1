"""One-time CPU-only preprocessing: scan every mocap .npz once, build and
normalize pose features, split by FILE into train/val, and cache the result
as train.pt / val.pt.

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
time it takes to read two files off disk.

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

USAGE (Kaggle, CPU-only session -- no GPU needed, see note above)
--------------------------------------------------------------------
Writing to /kaggle/working alone is not enough -- that is this *session's*
output, gone once it ends. --kaggle-dataset-slug pushes train.pt/val.pt to
an actual Kaggle dataset in the same run, reusing download_kit_to_kaggle.py's
already-tested publish_to_kaggle (title-collision-safe, catches the Kaggle
CLI's silent-success-on-error bug) rather than a separate hand-run publish
step. Needs the same two Secrets as that script: KAGGLE_USERNAME, KAGGLE_KEY.

    !git clone -q https://github.com/MorningStarTM/GPSM-1.git
    %cd GPSM-1
    # -m, not `python scripts/....py` directly -- the latter does not add
    # the repo root to sys.path, so `from src.gpsm...` fails inside it.
    !python -m scripts.preprocess_motion_mamba_cache /kaggle/input \
        --out /kaggle/working/cache \
        --kaggle-dataset-slug motion-mamba-cache

A later GPU session then attaches the "motion-mamba-cache" dataset and
loads train.pt/val.pt directly -- no scan, no wasted GPU time.

LOADING train.pt / val.pt FOR TRAINING
------------------------------------------
    cache = torch.load("train.pt")
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
import random
from pathlib import Path

import torch

from src.gpsm.utils.dataset import MultiSMPLXNPZNextPoseDataset


def build_and_split(
    data_dir: str,
    out_dir: str,
    k: int = 8,
    feature_set: str = "poses+trans",
    val_fraction: float = 0.1,
    seed: int = 0,
    strict_dim: bool = False,
    kaggle_dataset_slug: str = None,
) -> None:
    """Run the one-time scan, write train.pt / val.pt, and (if a slug is
    given) push them to an actual Kaggle dataset in the same run -- writing
    to /kaggle/working alone is this session's output, gone once it ends.

    Args:
        data_dir:      Folder of raw mocap .npz, searched recursively (e.g.
                        "/kaggle/input" to cover every attached dataset).
        out_dir:        Where to write train.pt / val.pt.
        k:              History window length used only to report sample
                         counts below -- the cache itself does not depend
                         on it (see module docstring).
        feature_set:    Passed straight through to MultiSMPLXNPZNextPoseDataset.
        val_fraction:   Fraction of FILES (not windows) held out for val.
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
        "source_dir": str(data_dir),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"files": train_records, "meta": meta}, out / "train.pt")
    torch.save({"files": val_records, "meta": meta}, out / "val.pt")

    train_windows = sum(len(r["valid_t"]) for r in train_records)
    val_windows = sum(len(r["valid_t"]) for r in val_records)
    print(f"train.pt: {len(train_records)} files, {train_windows} windowed samples "
          f"-> {out / 'train.pt'}")
    print(f"val.pt:   {len(val_records)} files, {val_windows} windowed samples "
          f"-> {out / 'val.pt'}")

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
    parser.add_argument("--out", default="processed_cache", help="Where to write train.pt / val.pt")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--feature-set", default="poses+trans")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-dim", action="store_true")
    parser.add_argument("--kaggle-dataset-slug", default=None,
                         help="If given, also push train.pt/val.pt to this Kaggle dataset slug")
    args = parser.parse_args()

    build_and_split(
        args.data_dir, args.out, k=args.k, feature_set=args.feature_set,
        val_fraction=args.val_fraction, seed=args.seed, strict_dim=args.strict_dim,
        kaggle_dataset_slug=args.kaggle_dataset_slug,
    )


if __name__ == "__main__":
    main()
