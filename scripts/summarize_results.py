from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.io import ensure_parent
from pbp.logging_utils import RunLogger, finalize_run, initialize_run


TABLE_COLUMNS = [
    "model",
    "method",
    "ratio",
    "coverage@0",
    "coverage@q25",
    "bcr@0",
    "bcr@q25",
    "pref_acc",
    "mean_margin_drop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize BCR evaluation JSON files into a pilot CSV table.")
    parser.add_argument("--eval-dir", default="outputs/evals")
    parser.add_argument("--inputs", nargs="*", default=None, help="Explicit BCR JSON files. If omitted, scan --eval-dir.")
    parser.add_argument("--pattern", default="bcr*.json", help="Glob used when --inputs is omitted.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Recorded for protocol consistency.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def config_for_run(args: argparse.Namespace, inputs: list[Path], summary_out: Path) -> dict[str, Any]:
    return {
        "script": "scripts/summarize_results.py",
        "model": None,
        "base_model": None,
        "dataset": None,
        "data_path": str(args.eval_dir),
        "seed": args.seed,
        "dtype": None,
        "device": "cpu",
        "batch_size": None,
        "max_samples": args.max_samples,
        "output_path": args.out,
        "notes": "M9 pilot result table summarization",
        "inputs": [str(path) for path in inputs],
        "summary_out": str(summary_out),
    }


def assert_can_write(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        paths = [Path(path) for path in args.inputs]
    else:
        paths = sorted(Path(args.eval_dir).glob(args.pattern))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input result files: {missing}")
    if not paths:
        raise ValueError("No result files found")
    return paths


def _get(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"Missing required metric; tried keys={keys}")


def _optional_get(payload: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _ratio_from_filename(path: Path) -> float | None:
    text = path.stem.lower()
    match = re.search(r"(?:^|_)(\d+)p(?:_|$)", text)
    if match:
        return float(match.group(1)) / 100.0
    match = re.search(r"(?:^|_)(0?\.\d+)(?:_|$)", text)
    if match:
        return float(match.group(1))
    return None


def _method_from_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ("bcr_qwen2p5_1p5b_", "bcr_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    stem = re.sub(r"_(?:10|20|30)p(?:_|$).*", "", stem)
    stem = re.sub(r"_m9.*$", "", stem)
    return stem or "unknown"


def _stringify_number(value: Any) -> str:
    if value is None:
        return ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value: {value}")
    return f"{number:.12g}"


def row_from_result(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "bcr_at_q25" not in payload and "bcr@q25" not in payload:
        raise ValueError(f"{path} does not look like a BCR result JSON")

    mask_stats = payload.get("mask_stats") or {}
    model = str(payload.get("model_load_id") or payload.get("model") or "unknown")
    method = str(payload.get("mask_method") or payload.get("method") or _method_from_filename(path))
    ratio = _optional_get(mask_stats, "actual_ratio")
    if ratio is None:
        ratio = _optional_get(payload, "mask_actual_ratio", "ratio")
    if ratio is None:
        ratio = _ratio_from_filename(path)

    return {
        "model": model,
        "method": method,
        "ratio": _stringify_number(ratio),
        "coverage@0": _stringify_number(_get(payload, "coverage_at_0", "coverage@0")),
        "coverage@q25": _stringify_number(_get(payload, "coverage_at_q25", "coverage@q25")),
        "bcr@0": _stringify_number(_get(payload, "bcr_at_0", "bcr@0")),
        "bcr@q25": _stringify_number(_get(payload, "bcr_at_q25", "bcr@q25")),
        "pref_acc": _stringify_number(_get(payload, "preference_accuracy_pruned", "preference_accuracy")),
        "mean_margin_drop": _stringify_number(_get(payload, "mean_margin_drop")),
    }


def sort_key(row: dict[str, str]) -> tuple[str, float, str]:
    try:
        ratio = float(row["ratio"])
    except ValueError:
        ratio = math.inf
    return row["method"], ratio, row["model"]


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    inputs = discover_inputs(args)
    out_path = Path(args.out)
    summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".json")
    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args, inputs, summary_out),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        assert_can_write([out_path, summary_out], overwrite=args.overwrite)
        rows = sorted((row_from_result(path) for path in inputs), key=sort_key)
        write_csv(rows, out_path)
        summary = {
            "num_input_files": len(inputs),
            "num_rows": len(rows),
            "columns": TABLE_COLUMNS,
            "inputs": [str(path) for path in inputs],
            "rows": rows,
        }
        ensure_parent(summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "num_input_files": len(inputs),
            "num_rows": len(rows),
            "out": str(out_path),
            "summary_out": str(summary_out),
        }
        finalize_run(run_paths, start_monotonic=start, metrics=metrics)
        message = {"run_id": run_paths.run_id, "out": str(out_path), "metrics": metrics}
        print(json.dumps(message, ensure_ascii=False, indent=2))
        logger.stdout(json.dumps(message, ensure_ascii=False))
    except Exception as exc:
        logger.stderr(f"{type(exc).__name__}: {exc}")
        finalize_run(run_paths, start_monotonic=start, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
