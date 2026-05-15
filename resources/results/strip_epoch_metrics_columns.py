#!/usr/bin/env python3
"""
Rewrite epoch_metrics CSV files after dropping selected columns (in-place).

Typical use (Bijie logs): remove train_loss, val_loss, epoch_time before publishing
or before figure bundles that should not treat these as heatmap columns.

  python strip_epoch_metrics_columns.py \\
      --summary-csv SAM/resources/results/bijie_ablation_report/bijie_best_validation_summary.csv \\
      --columns train_loss,val_loss,epoch_time
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Set


def read_summary_sources(summary_csv: Path) -> List[Path]:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: List[Path] = []
    for r in rows:
        p = Path(r.get("source_csv", "")).expanduser()
        if p.is_file():
            out.append(p.resolve())
    return sorted(set(out))


def strip_columns(path: Path, drop: Set[str]) -> tuple[bool, int]:
    """Return (changed, n_cols_removed)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        return False, 0
    lower_drop = {d.strip().lower() for d in drop}
    keep = [c for c in fieldnames if c.strip().lower() not in lower_drop]
    removed = len(fieldnames) - len(keep)
    if removed == 0:
        return False, 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keep})
    return True, removed


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop columns from epoch_metrics CSVs listed in a summary CSV.")
    ap.add_argument("--summary-csv", type=Path, required=True)
    ap.add_argument(
        "--columns",
        type=str,
        default="train_loss,val_loss,epoch_time",
        help="Comma-separated column names to remove (case-insensitive match on header).",
    )
    ap.add_argument(
        "--also-files",
        type=Path,
        nargs="*",
        default=(),
        help="Additional CSV files to strip in-place (e.g. bijie_best_validation_summary.csv).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print paths only; do not write files.")
    args = ap.parse_args()
    summary = args.summary_csv.expanduser().resolve()
    drop = {x.strip() for x in args.columns.split(",") if x.strip()}
    paths = read_summary_sources(summary)
    if not paths:
        raise SystemExit(f"No existing source_csv paths found from {summary}")
    for p in paths:
        if args.dry_run:
            print(f"would process: {p}")
            continue
        changed, n = strip_columns(p, drop)
        if changed:
            print(f"stripped {n} column(s): {p}")
        else:
            print(f"unchanged (columns absent or already removed): {p}")
    for extra in args.also_files:
        ep = extra.expanduser().resolve()
        if not ep.is_file():
            print(f"skip missing: {ep}")
            continue
        if args.dry_run:
            print(f"would process: {ep}")
            continue
        changed, n = strip_columns(ep, drop)
        if changed:
            print(f"stripped {n} column(s): {ep}")
        else:
            print(f"unchanged: {ep}")


if __name__ == "__main__":
    main()
