from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.io import read_jsonl, write_json
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.metrics import histogram_rows, summarize_dense_margins
from pbp.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report dense-margin coverage and distribution.")
    parser.add_argument("--dense-margins", required=True)
    parser.add_argument("--out", required=True, help="Coverage summary JSON output path.")
    parser.add_argument("--histogram-out", default=None, help="Histogram CSV output path.")
    parser.add_argument("--histogram-bins", type=int, default=30)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def assert_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def validate_args(args: argparse.Namespace) -> None:
    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")


def default_histogram_path(out_path: Path) -> Path:
    return out_path.with_suffix(".histogram.csv")


def config_for_run(args: argparse.Namespace, out_path: Path, histogram_path: Path) -> dict[str, Any]:
    return {
        "script": "scripts/report_coverage.py",
        "model": None,
        "base_model": None,
        "dataset": None,
        "data_path": args.dense_margins,
        "seed": args.seed,
        "dtype": None,
        "device": "cpu",
        "batch_size": None,
        "max_samples": args.max_samples,
        "output_path": str(out_path),
        "notes": "M4 Coverage@tau reporting",
        "histogram_output_path": str(histogram_path),
        "histogram_bins": args.histogram_bins,
    }


def write_histogram_csv(summary: dict[str, Any], path: Path) -> None:
    rows = histogram_rows(summary["histogram"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_left", "bin_right", "count"])
        writer.writeheader()
        writer.writerows(rows)


def metrics_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "num_examples": int(summary["num_pairs"]),
        "coverage_at_0": float(summary["coverage_at_0"]),
        "coverage_at_q25": float(summary["coverage_at_q25"]),
        "coverage_at_q50": float(summary["coverage_at_q50"]),
        "coverage_at_q75": float(summary["coverage_at_q75"]),
        "preference_accuracy": float(summary["preference_accuracy"]),
        "mean_delta_dense": float(summary["mean_delta_dense"]),
        "median_delta_dense": float(summary["median_delta_dense"]),
        "positive_q25": float(summary["positive_margin_quantiles"]["q25"]),
        "positive_q50": float(summary["positive_margin_quantiles"]["q50"]),
        "positive_q75": float(summary["positive_margin_quantiles"]["q75"]),
    }
    metrics["coverage_metrics_valid"] = all(
        0.0 <= metrics[key] <= 1.0
        for key in ("coverage_at_0", "coverage_at_q25", "coverage_at_q50", "coverage_at_q75")
    )
    metrics["numeric_metrics_finite"] = all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key not in {"coverage_metrics_valid", "numeric_metrics_finite"}
    )
    return metrics


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    out_path = Path(args.out)
    histogram_path = Path(args.histogram_out) if args.histogram_out else default_histogram_path(out_path)

    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args, out_path, histogram_path),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        assert_can_write(out_path, overwrite=args.overwrite)
        assert_can_write(histogram_path, overwrite=args.overwrite)
        records = read_jsonl(args.dense_margins)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        if not records:
            raise ValueError("No dense margin records to report")

        summary = summarize_dense_margins(records, bins=args.histogram_bins)
        write_json(summary, out_path)
        write_histogram_csv(summary, histogram_path)
        metrics = metrics_from_summary(summary)
        finalize_run(run_paths, start_monotonic=start, metrics=metrics)

        message = {
            "run_id": run_paths.run_id,
            "out": str(out_path),
            "histogram_out": str(histogram_path),
            "metrics": metrics,
        }
        print(json.dumps(message, ensure_ascii=False, indent=2))
        logger.stdout(json.dumps(message, ensure_ascii=False))
    except Exception as exc:
        logger.stderr(f"{type(exc).__name__}: {exc}")
        finalize_run(run_paths, start_monotonic=start, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
