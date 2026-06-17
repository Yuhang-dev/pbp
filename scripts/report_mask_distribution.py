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


TABLE_COLUMNS = ["method", "ratio", "layer", "total_units", "pruned_units", "pruned_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report layer-wise pruning distribution for masked M9 models.")
    parser.add_argument("--mask-dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def assert_can_write(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def _fmt_ratio(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite ratio: {value}")
    return f"{number:.12g}"


def _layer_from_module_name(module_name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|\.)h\.(\d+)(?:\.|$)", module_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not infer layer index from module name: {module_name}")


def _module_to_layer(config: dict[str, Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for group in config.get("groups", []):
        module_name = str(group.get("module_name", ""))
        if not module_name:
            continue
        if "layer" in group:
            mapping[module_name] = int(group["layer"])
        else:
            mapping[module_name] = _layer_from_module_name(module_name)
    return mapping


def rows_for_mask_dir(mask_dir: str | Path) -> list[dict[str, str]]:
    path = Path(mask_dir)
    config_path = path / "mask_config.json"
    masks_path = path / "masks.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing mask_config.json in {path}")
    if not masks_path.is_file():
        raise FileNotFoundError(f"Missing masks.json in {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    masks = json.loads(masks_path.read_text(encoding="utf-8"))
    method = str(config.get("method", "unknown"))
    ratio = float(config.get("actual_ratio", config.get("ratio", 0.0)))
    module_to_layer = _module_to_layer(config)

    rows: list[dict[str, str]] = []
    for module_name, values in masks.items():
        layer = module_to_layer.get(module_name)
        if layer is None:
            layer = _layer_from_module_name(module_name)
        total_units = len(values)
        pruned_units = sum(1 for value in values if int(value) == 0)
        pruned_ratio = pruned_units / total_units if total_units else 0.0
        rows.append(
            {
                "method": method,
                "ratio": _fmt_ratio(ratio),
                "layer": str(layer),
                "total_units": str(total_units),
                "pruned_units": str(pruned_units),
                "pruned_ratio": _fmt_ratio(pruned_ratio),
            }
        )
    return sorted(rows, key=lambda row: (row["method"], float(row["ratio"]), int(row["layer"])))


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def config_for_run(args: argparse.Namespace, summary_out: Path) -> dict[str, Any]:
    return {
        "script": "scripts/report_mask_distribution.py",
        "model": None,
        "base_model": None,
        "dataset": None,
        "data_path": None,
        "seed": args.seed,
        "dtype": None,
        "device": "cpu",
        "batch_size": None,
        "max_samples": None,
        "output_path": args.out,
        "notes": "M10B layer-wise masked pruning distribution",
        "mask_dirs": args.mask_dirs,
        "summary_out": str(summary_out),
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    summary_out = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".json")
    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args, summary_out),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        assert_can_write([out_path, summary_out], overwrite=args.overwrite)
        rows: list[dict[str, str]] = []
        for mask_dir in args.mask_dirs:
            rows.extend(rows_for_mask_dir(mask_dir))
        rows = sorted(rows, key=lambda row: (row["method"], float(row["ratio"]), int(row["layer"])))
        write_csv(rows, out_path)
        summary = {
            "num_mask_dirs": len(args.mask_dirs),
            "num_rows": len(rows),
            "columns": TABLE_COLUMNS,
            "mask_dirs": args.mask_dirs,
            "out": str(out_path),
        }
        ensure_parent(summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "num_mask_dirs": len(args.mask_dirs),
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
