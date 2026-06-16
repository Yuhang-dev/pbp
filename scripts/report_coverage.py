from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.io import read_jsonl, write_json
from pbp.metrics import histogram_rows, summarize_dense_margins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report coverage and margin histogram.")
    parser.add_argument("--dense-margins", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--histogram-out", required=True)
    parser.add_argument("--histogram-bins", type=int, default=30)
    return parser.parse_args()


def write_histogram_csv(summary: dict, path: str) -> None:
    rows = histogram_rows(summary["histogram"])
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_left", "bin_right", "count"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.dense_margins)
    summary = summarize_dense_margins(records, bins=args.histogram_bins)
    write_json(summary, args.summary_out)
    write_histogram_csv(summary, args.histogram_out)
    print(f"Wrote coverage summary to {args.summary_out}")
    print(f"Wrote margin histogram to {args.histogram_out}")


if __name__ == "__main__":
    main()
