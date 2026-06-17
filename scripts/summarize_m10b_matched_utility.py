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

from pbp.io import ensure_parent
from pbp.logging_utils import RunLogger, finalize_run, initialize_run


TABLE_COLUMNS = [
    "model",
    "method",
    "ratio",
    "ppl",
    "arc_c",
    "hellaswag",
    "bcr@q25",
    "bcr@0",
    "pref_acc",
    "mean_margin_drop",
    "ppl_relative_delta",
    "arc_c_drop",
    "hellaswag_drop",
    "matched_utility_flag",
]

METHOD_ORDER = {
    "dense": 0,
    "random": 1,
    "magnitude": 2,
    "activation": 3,
    "boundary_taylor_weighted": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the M10B matched-utility table and summary.")
    parser.add_argument("--general-inputs", nargs="+", required=True)
    parser.add_argument("--bcr-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ppl-relative-delta", type=float, default=0.10)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def assert_can_write(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def _finite_float(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {field}: {value}")
    return number


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value: {value}")
    return f"{number:.12g}"


def _ratio_key(value: Any) -> str:
    return f"{float(value):.6f}"


def _is_ratio(value: Any, target: float) -> bool:
    return math.isclose(float(value), target, rel_tol=0.0, abs_tol=1e-6)


def read_bcr_rows(path: Path) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"BCR table is empty: {path}")
    by_key = {
        (row["method"], _ratio_key(row["ratio"])): row
        for row in rows
    }
    return by_key, rows[0]


def read_general(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in ("model", "method", "ratio", "ppl", "arc_c", "hellaswag"):
        if field not in payload:
            raise KeyError(f"{path} missing required field {field!r}")
    return payload


def utility_components(
    *,
    ppl: float,
    arc_c: float,
    hellaswag: float,
    dense_ppl: float,
    dense_arc_c: float,
    dense_hellaswag: float,
) -> dict[str, float]:
    return {
        "ppl_relative_delta": (ppl - dense_ppl) / dense_ppl if dense_ppl else 0.0,
        "arc_c_drop": dense_arc_c - arc_c,
        "hellaswag_drop": dense_hellaswag - hellaswag,
    }


def matched_flag(components: dict[str, float], *, max_ppl_relative_delta: float, max_accuracy_drop: float) -> bool:
    return (
        components["ppl_relative_delta"] <= max_ppl_relative_delta
        and components["arc_c_drop"] <= max_accuracy_drop
        and components["hellaswag_drop"] <= max_accuracy_drop
    )


def row_sort_key(row: dict[str, str]) -> tuple[int, float, str]:
    return METHOD_ORDER.get(row["method"], 99), float(row["ratio"]), row["method"]


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    bcr_by_key, first_bcr_row = read_bcr_rows(Path(args.bcr_table))
    general_payloads = [read_general(Path(path)) for path in args.general_inputs]
    dense_payloads = [payload for payload in general_payloads if payload["method"] == "dense"]
    if len(dense_payloads) != 1:
        raise ValueError("Exactly one dense general-utility payload is required")

    dense = dense_payloads[0]
    dense_ppl = _finite_float(dense["ppl"], field="dense.ppl")
    dense_arc_c = _finite_float(dense["arc_c"], field="dense.arc_c")
    dense_hellaswag = _finite_float(dense["hellaswag"], field="dense.hellaswag")
    dense_pref_acc = first_bcr_row.get("coverage@0")

    rows: list[dict[str, str]] = []
    for payload in general_payloads:
        method = str(payload["method"])
        ratio = _finite_float(payload["ratio"], field=f"{method}.ratio")
        ppl = _finite_float(payload["ppl"], field=f"{method}.ppl")
        arc_c = _finite_float(payload["arc_c"], field=f"{method}.arc_c")
        hellaswag = _finite_float(payload["hellaswag"], field=f"{method}.hellaswag")
        components = utility_components(
            ppl=ppl,
            arc_c=arc_c,
            hellaswag=hellaswag,
            dense_ppl=dense_ppl,
            dense_arc_c=dense_arc_c,
            dense_hellaswag=dense_hellaswag,
        )
        is_matched = method == "dense" or matched_flag(
            components,
            max_ppl_relative_delta=args.max_ppl_relative_delta,
            max_accuracy_drop=args.max_accuracy_drop,
        )

        if method == "dense":
            bcr_q25 = 0.0
            bcr_0 = 0.0
            pref_acc = dense_pref_acc
            mean_margin_drop = 0.0
        else:
            bcr_row = bcr_by_key.get((method, _ratio_key(ratio)))
            if bcr_row is None:
                raise KeyError(f"No BCR row found for method={method}, ratio={ratio}")
            bcr_q25 = bcr_row["bcr@q25"]
            bcr_0 = bcr_row["bcr@0"]
            pref_acc = bcr_row["pref_acc"]
            mean_margin_drop = bcr_row["mean_margin_drop"]

        rows.append(
            {
                "model": str(payload["model"]),
                "method": method,
                "ratio": _fmt(ratio),
                "ppl": _fmt(ppl),
                "arc_c": _fmt(arc_c),
                "hellaswag": _fmt(hellaswag),
                "bcr@q25": _fmt(bcr_q25),
                "bcr@0": _fmt(bcr_0),
                "pref_acc": _fmt(pref_acc),
                "mean_margin_drop": _fmt(mean_margin_drop),
                "ppl_relative_delta": _fmt(components["ppl_relative_delta"]),
                "arc_c_drop": _fmt(components["arc_c_drop"]),
                "hellaswag_drop": _fmt(components["hellaswag_drop"]),
                "matched_utility_flag": _fmt(is_matched),
            }
        )

    rows = sorted(rows, key=row_sort_key)
    summary = build_summary(
        rows,
        dense_reference={
            "ppl": dense_ppl,
            "arc_c": dense_arc_c,
            "hellaswag": dense_hellaswag,
        },
        thresholds={
            "max_ppl_relative_delta": args.max_ppl_relative_delta,
            "max_accuracy_drop": args.max_accuracy_drop,
        },
    )
    return rows, summary


def _row_identity(row: dict[str, str]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "ratio": float(row["ratio"]),
        "bcr@q25": float(row["bcr@q25"]),
        "ppl": float(row["ppl"]),
        "arc_c": float(row["arc_c"]),
        "hellaswag": float(row["hellaswag"]),
    }


def build_summary(
    rows: list[dict[str, str]],
    *,
    dense_reference: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    pruned_rows = [row for row in rows if row["method"] != "dense"]
    matched_rows = [row for row in pruned_rows if row["matched_utility_flag"] == "true"]
    matched_10p = [row for row in matched_rows if _is_ratio(row["ratio"], 0.10)]
    matched_20p = [row for row in matched_rows if _is_ratio(row["ratio"], 0.20)]
    best_matched = min(matched_rows, key=lambda row: float(row["bcr@q25"])) if matched_rows else None

    no_20p_message = "20% is not a mild regime under current masking."
    yes_20p_message = "At least one 20% pruned model is matched utility under current thresholds."
    return {
        "num_rows": len(rows),
        "num_pruned_rows": len(pruned_rows),
        "num_matched_pruned_rows": len(matched_rows),
        "columns": TABLE_COLUMNS,
        "dense_reference": dense_reference,
        "thresholds": thresholds,
        "answers": {
            "is_any_10p_pruned_model_matched_utility": bool(matched_10p),
            "matched_10p_models": [_row_identity(row) for row in matched_10p],
            "is_any_20p_pruned_model_matched_utility": bool(matched_20p),
            "matched_20p_models": [_row_identity(row) for row in matched_20p],
            "lowest_bcr_q25_among_matched_utility_models": _row_identity(best_matched) if best_matched else None,
            "twenty_percent_mild_regime_assessment": yes_20p_message if matched_20p else no_20p_message,
        },
    }


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def config_for_run(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/summarize_m10b_matched_utility.py",
        "model": None,
        "base_model": None,
        "dataset": None,
        "data_path": args.bcr_table,
        "seed": args.seed,
        "dtype": None,
        "device": "cpu",
        "batch_size": None,
        "max_samples": None,
        "output_path": args.out,
        "notes": "M10B matched utility table and summary",
        "general_inputs": args.general_inputs,
        "summary_out": args.summary_out,
        "max_ppl_relative_delta": args.max_ppl_relative_delta,
        "max_accuracy_drop": args.max_accuracy_drop,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    summary_out = Path(args.summary_out)
    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        assert_can_write([out_path, summary_out], overwrite=args.overwrite)
        rows, summary = build_rows(args)
        write_csv(rows, out_path)
        ensure_parent(summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "num_rows": summary["num_rows"],
            "num_pruned_rows": summary["num_pruned_rows"],
            "num_matched_pruned_rows": summary["num_matched_pruned_rows"],
            "is_any_10p_pruned_model_matched_utility": summary["answers"][
                "is_any_10p_pruned_model_matched_utility"
            ],
            "is_any_20p_pruned_model_matched_utility": summary["answers"][
                "is_any_20p_pruned_model_matched_utility"
            ],
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
