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
    "utility_method",
    "boundary_method",
    "ratio",
    "alpha",
    "ppl",
    "arc_c",
    "hellaswag",
    "matched",
    "bcr@q25",
    "bcr@0",
    "bcr_eval_samples",
    "pref_acc",
    "mean_margin_drop",
    "ppl_relative_delta",
    "arc_c_drop",
    "hellaswag_drop",
    "selection_scope",
    "protection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize M12 hybrid utility-boundary pruning results.")
    parser.add_argument("--general-inputs", nargs="+", required=True)
    parser.add_argument("--bcr-inputs", nargs="+", required=True)
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
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value: {value}")
    return f"{number:.12g}"


def _ratio_key(value: Any) -> str:
    return f"{float(value):.6f}"


def _alpha_key(value: Any) -> str:
    if value is None or value == "":
        return "none"
    return f"{float(value):.6f}"


def _identity(*, method: str, selection_scope: str, protection: str, ratio: Any, alpha: Any) -> tuple[str, str, str, str, str]:
    return method, selection_scope, protection, _ratio_key(ratio), _alpha_key(alpha)


def read_general(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in ("model", "method", "ppl", "arc_c", "hellaswag"):
        if field not in payload:
            raise KeyError(f"{path} missing required field {field!r}")
    payload.setdefault("selection_scope", "dense" if payload["method"] == "dense" else "global")
    payload.setdefault("protection", "none")
    payload.setdefault("requested_ratio", payload.get("ratio", 0.0))
    payload.setdefault("actual_global_ratio", payload.get("ratio", 0.0))
    payload.setdefault("actual_unprotected_ratio", payload.get("ratio", 0.0))
    payload.setdefault("alpha", None)
    payload.setdefault("utility_method", None)
    payload.setdefault("boundary_method", None)
    return payload


def read_bcr(path: Path) -> tuple[tuple[str, str, str, str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = str(payload.get("mask_method") or payload.get("method"))
    selection_scope = str(payload.get("selection_scope") or payload.get("mask_stats", {}).get("selection_scope", "global"))
    protection = str(payload.get("protection") or payload.get("mask_stats", {}).get("protection", "none"))
    ratio = payload.get("requested_ratio")
    if ratio is None:
        ratio = payload.get("mask_stats", {}).get("requested_ratio", payload.get("mask_stats", {}).get("actual_ratio"))
    if ratio is None:
        raise KeyError(f"{path} missing requested ratio")
    alpha = payload.get("alpha")
    return _identity(method=method, selection_scope=selection_scope, protection=protection, ratio=ratio, alpha=alpha), payload


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


def choose_largest_bcr_inputs(paths: list[str]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for path in paths:
        key, payload = read_bcr(Path(path))
        current = out.get(key)
        current_n = int(current.get("num_examples", 0)) if current else -1
        candidate_n = int(payload.get("num_examples", 0))
        if current is None or candidate_n >= current_n:
            out[key] = payload
    return out


def row_identity(row: dict[str, str]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "utility_method": row["utility_method"] or None,
        "boundary_method": row["boundary_method"] or None,
        "ratio": float(row["ratio"]),
        "alpha": float(row["alpha"]) if row["alpha"] else None,
        "ppl": float(row["ppl"]),
        "arc_c": float(row["arc_c"]),
        "hellaswag": float(row["hellaswag"]),
        "bcr@q25": float(row["bcr@q25"]),
        "bcr@0": float(row["bcr@0"]),
        "bcr_eval_samples": int(row["bcr_eval_samples"]) if row["bcr_eval_samples"] else None,
    }


def compare_hybrids(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {
        (row["method"], _ratio_key(row["ratio"]), _alpha_key(row["alpha"])): row
        for row in rows
        if row["method"] != "dense"
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] not in {"activation_boundary", "general_taylor_boundary"}:
            continue
        if row["matched"] != "true":
            continue
        baseline_method = row["utility_method"]
        if not baseline_method:
            continue
        baseline = by_key.get((baseline_method, _ratio_key(row["ratio"]), "none"))
        if baseline is None:
            continue
        same_samples = row["bcr_eval_samples"] == baseline["bcr_eval_samples"]
        hybrid_bcr = float(row["bcr@q25"])
        baseline_bcr = float(baseline["bcr@q25"])
        comparisons.append(
            {
                "hybrid_method": row["method"],
                "baseline_method": baseline_method,
                "ratio": float(row["ratio"]),
                "alpha": float(row["alpha"]) if row["alpha"] else None,
                "hybrid_bcr_q25": hybrid_bcr,
                "baseline_bcr_q25": baseline_bcr,
                "hybrid_bcr_lower": hybrid_bcr < baseline_bcr,
                "relative_bcr_reduction": (baseline_bcr - hybrid_bcr) / baseline_bcr if baseline_bcr else None,
                "same_bcr_eval_samples": same_samples,
                "bcr_eval_samples": int(row["bcr_eval_samples"]) if row["bcr_eval_samples"] else None,
            }
        )
    return comparisons


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    general_payloads = [read_general(Path(path)) for path in args.general_inputs]
    dense_payloads = [payload for payload in general_payloads if payload["method"] == "dense"]
    if len(dense_payloads) != 1:
        raise ValueError("Exactly one dense general-utility payload is required")
    dense = dense_payloads[0]
    dense_ppl = _finite_float(dense["ppl"], field="dense.ppl")
    dense_arc_c = _finite_float(dense["arc_c"], field="dense.arc_c")
    dense_hellaswag = _finite_float(dense["hellaswag"], field="dense.hellaswag")
    bcr_by_key = choose_largest_bcr_inputs(args.bcr_inputs)

    rows: list[dict[str, str]] = []
    for payload in general_payloads:
        method = str(payload["method"])
        selection_scope = str(payload.get("selection_scope", "dense" if method == "dense" else "global"))
        protection = str(payload.get("protection", "none"))
        ratio = _finite_float(payload.get("requested_ratio", payload.get("ratio", 0.0)), field=f"{method}.ratio")
        alpha = payload.get("alpha")
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
            pref_acc = None
            mean_margin_drop = 0.0
            bcr_eval_samples = None
        else:
            key = _identity(method=method, selection_scope=selection_scope, protection=protection, ratio=ratio, alpha=alpha)
            bcr = bcr_by_key.get(key)
            if bcr is None:
                raise KeyError(f"Missing BCR result for {key}")
            bcr_q25 = bcr.get("bcr@q25", bcr.get("bcr_at_q25"))
            bcr_0 = bcr.get("bcr@0", bcr.get("bcr_at_0"))
            pref_acc = bcr.get("preference_accuracy_pruned", bcr.get("preference_accuracy"))
            mean_margin_drop = bcr["mean_margin_drop"]
            bcr_eval_samples = bcr.get("num_examples")

        rows.append(
            {
                "model": str(payload["model"]),
                "method": method,
                "utility_method": str(payload.get("utility_method") or ""),
                "boundary_method": str(payload.get("boundary_method") or ""),
                "ratio": _fmt(ratio),
                "alpha": _fmt(alpha),
                "ppl": _fmt(ppl),
                "arc_c": _fmt(arc_c),
                "hellaswag": _fmt(hellaswag),
                "matched": _fmt(is_matched),
                "bcr@q25": _fmt(bcr_q25),
                "bcr@0": _fmt(bcr_0),
                "bcr_eval_samples": _fmt(bcr_eval_samples),
                "pref_acc": _fmt(pref_acc),
                "mean_margin_drop": _fmt(mean_margin_drop),
                "ppl_relative_delta": _fmt(components["ppl_relative_delta"]),
                "arc_c_drop": _fmt(components["arc_c_drop"]),
                "hellaswag_drop": _fmt(components["hellaswag_drop"]),
                "selection_scope": selection_scope,
                "protection": protection,
            }
        )

    rows = sorted(
        rows,
        key=lambda row: (
            row["method"],
            float(row["ratio"]),
            float(row["alpha"]) if row["alpha"] else -1.0,
        ),
    )
    matched_rows = [row for row in rows if row["method"] != "dense" and row["matched"] == "true"]
    hybrid_rows = [row for row in rows if row["method"] in {"activation_boundary", "general_taylor_boundary"}]
    matched_hybrid_rows = [row for row in hybrid_rows if row["matched"] == "true"]
    comparisons = compare_hybrids(rows)
    fair_lower = [item for item in comparisons if item["same_bcr_eval_samples"] and item["hybrid_bcr_lower"]]
    summary = {
        "num_rows": len(rows),
        "num_pruned_rows": len([row for row in rows if row["method"] != "dense"]),
        "num_matched_pruned_rows": len(matched_rows),
        "num_hybrid_rows": len(hybrid_rows),
        "num_matched_hybrid_rows": len(matched_hybrid_rows),
        "columns": TABLE_COLUMNS,
        "thresholds": {
            "max_ppl_relative_delta": args.max_ppl_relative_delta,
            "max_accuracy_drop": args.max_accuracy_drop,
        },
        "dense_reference": {
            "ppl": dense_ppl,
            "arc_c": dense_arc_c,
            "hellaswag": dense_hellaswag,
        },
        "answers": {
            "matched_utility_settings": [row_identity(row) for row in matched_rows],
            "matched_hybrid_settings": [row_identity(row) for row in matched_hybrid_rows],
            "hybrid_lower_bcr_q25_than_corresponding_utility_baseline": bool(fair_lower),
            "hybrid_lower_bcr_q25_comparisons": comparisons,
            "lowest_bcr_q25_among_matched_settings": (
                row_identity(min(matched_rows, key=lambda row: float(row["bcr@q25"]))) if matched_rows else None
            ),
            "lowest_bcr_q25_among_matched_hybrids": (
                row_identity(min(matched_hybrid_rows, key=lambda row: float(row["bcr@q25"])))
                if matched_hybrid_rows
                else None
            ),
        },
    }
    return rows, summary


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def config_for_run(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/summarize_m12_hybrid.py",
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
        "notes": "M12 hybrid utility-boundary alpha sweep summary",
        "general_inputs": args.general_inputs,
        "bcr_inputs": args.bcr_inputs,
        "summary_out": args.summary_out,
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
            "num_hybrid_rows": summary["num_hybrid_rows"],
            "num_matched_hybrid_rows": summary["num_matched_hybrid_rows"],
            "hybrid_lower_bcr_q25_than_corresponding_utility_baseline": summary["answers"][
                "hybrid_lower_bcr_q25_than_corresponding_utility_baseline"
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
