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
    "utility_delta_vs_dense",
    "matched_utility_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join M10A general utility JSON files with the M9 BCR table.")
    parser.add_argument("--general-inputs", nargs="+", required=True)
    parser.add_argument("--bcr-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
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


def read_bcr_rows(path: Path) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, str] | None]:
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


def utility_delta(
    *,
    ppl: float,
    arc_c: float,
    hellaswag: float,
    dense_ppl: float,
    dense_arc_c: float,
    dense_hellaswag: float,
) -> tuple[float, dict[str, float]]:
    ppl_rel_delta = (ppl - dense_ppl) / dense_ppl if dense_ppl else 0.0
    arc_drop = dense_arc_c - arc_c
    hellaswag_drop = dense_hellaswag - hellaswag
    aggregate = (ppl_rel_delta + arc_drop + hellaswag_drop) / 3.0
    return aggregate, {
        "ppl_relative_delta": ppl_rel_delta,
        "arc_c_drop": arc_drop,
        "hellaswag_drop": hellaswag_drop,
    }


def matched_flag(components: dict[str, float], *, max_ppl_relative_delta: float, max_accuracy_drop: float) -> bool:
    return (
        components["ppl_relative_delta"] <= max_ppl_relative_delta
        and components["arc_c_drop"] <= max_accuracy_drop
        and components["hellaswag_drop"] <= max_accuracy_drop
    )


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
    dense_pref_acc = first_bcr_row.get("coverage@0") if first_bcr_row else None

    rows: list[dict[str, str]] = []
    components_by_method: dict[str, dict[str, float]] = {}
    for payload in sorted(general_payloads, key=lambda item: (float(item["ratio"]), str(item["method"]))):
        method = str(payload["method"])
        ratio = _finite_float(payload["ratio"], field=f"{method}.ratio")
        ppl = _finite_float(payload["ppl"], field=f"{method}.ppl")
        arc_c = _finite_float(payload["arc_c"], field=f"{method}.arc_c")
        hellaswag = _finite_float(payload["hellaswag"], field=f"{method}.hellaswag")
        delta, components = utility_delta(
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
        components_by_method[method] = components

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
                "utility_delta_vs_dense": _fmt(delta),
                "matched_utility_flag": _fmt(is_matched),
            }
        )

    by_method = {row["method"]: row for row in rows}
    boundary = by_method.get("boundary_taylor_weighted")
    activation = by_method.get("activation")
    comparison: dict[str, Any] = {}
    if boundary and activation:
        comparison = {
            "boundary_bcr_q25": float(boundary["bcr@q25"]),
            "activation_bcr_q25": float(activation["bcr@q25"]),
            "boundary_lower_bcr_q25_than_activation": float(boundary["bcr@q25"]) < float(activation["bcr@q25"]),
            "boundary_matched_utility": boundary["matched_utility_flag"] == "true",
            "activation_matched_utility": activation["matched_utility_flag"] == "true",
        }
    summary = {
        "num_general_inputs": len(general_payloads),
        "num_rows": len(rows),
        "columns": TABLE_COLUMNS,
        "dense_reference": {
            "ppl": dense_ppl,
            "arc_c": dense_arc_c,
            "hellaswag": dense_hellaswag,
        },
        "thresholds": {
            "max_ppl_relative_delta": args.max_ppl_relative_delta,
            "max_accuracy_drop": args.max_accuracy_drop,
        },
        "utility_components_vs_dense": components_by_method,
        "boundary_vs_activation": comparison,
    }
    return rows, summary


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def config_for_run(args: argparse.Namespace, summary_out: Path) -> dict[str, Any]:
    return {
        "script": "scripts/summarize_m10a_matched_utility.py",
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
        "notes": "M10A matched utility table",
        "general_inputs": args.general_inputs,
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
        rows, summary = build_rows(args)
        write_csv(rows, out_path)
        ensure_parent(summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "num_general_inputs": summary["num_general_inputs"],
            "num_rows": summary["num_rows"],
            "out": str(out_path),
            "summary_out": str(summary_out),
            **summary.get("boundary_vs_activation", {}),
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
