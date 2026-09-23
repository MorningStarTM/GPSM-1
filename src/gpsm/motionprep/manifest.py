"""A running record of what the pipeline did to every file.

Each MotionPrep stage writes one CSV row per input file — including the
files it skipped and why. Two reasons this exists rather than just printing
to the console:

* **Nothing is dropped silently.** A clip rejected for poor fit quality, or
  skipped because its markers were unreadable, leaves a permanent record
  with the reason attached.
* **Big batches survive interruption.** Rows are appended and flushed as
  each file finishes, so killing a run halfway (or a Kaggle session timing
  out) still leaves an accurate record of everything already done — which
  is also what lets a re-run skip that work instead of repeating it.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set


class ManifestWriter:
    """Append-as-you-go CSV writer for one pipeline stage.

    Opens in append mode and writes the header only for a new file, so
    re-running a stage adds to the existing manifest instead of truncating
    the record of earlier work.

    Args:
        path:    Where to write the CSV.
        columns: Column names, in order. Every row must use exactly these.
    """

    def __init__(self, path: Path, columns: List[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.columns)
        if is_new:
            self._writer.writeheader()
            self._handle.flush()

    def append(self, row: Dict[str, object]) -> None:
        """Write one row and flush it to disk immediately.

        Flushing per row (rather than letting the OS buffer) is what makes
        the manifest trustworthy after an interrupted run.

        Raises:
            ValueError: If the row's keys don't match the declared columns —
                catching a typo here beats finding a half-empty column later.
        """
        unexpected = set(row) - set(self.columns)
        missing = set(self.columns) - set(row)
        if unexpected or missing:
            raise ValueError(
                f"Manifest row does not match columns. "
                f"Unexpected: {sorted(unexpected)}; missing: {sorted(missing)}"
            )
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


#: Outcomes treated as final when deciding what a re-run can skip. "failed"
#: is deliberately absent: a failure is usually a crash or a bug (possibly one
#: since fixed), so those files are retried rather than written off forever.
FINAL_STATUSES = ("ok", "skipped")


def already_processed(
    manifest_path: Path,
    column: str = "source_path",
    final_statuses: Set[str] = frozenset(FINAL_STATUSES),
) -> Set[str]:
    """Read back which inputs a previous run finished with for good.

    Used for resumability: a re-run skips these instead of redoing the work.
    A missing manifest simply means "nothing done yet".

    Args:
        manifest_path:  Path to a manifest written by :class:`ManifestWriter`.
        column:         Which column identifies the input.
        final_statuses: Which ``status`` values count as done. Anything else
                        (notably ``"failed"``) is left to be retried.

    Returns:
        The set of input identifiers that need not be processed again.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        return {
            row[column]
            for row in csv.DictReader(handle)
            if row.get(column) and row.get("status") in final_statuses
        }
