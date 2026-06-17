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
    "selection_scope",
    "protection",
    "ratio",
    "requested_ratio",
    "actual_global_ratio",
    "actual_unprotected_ratio",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize M11A layerwise utility/BCR results.")
    parser.add_argument("--general-inputs", nargs="+", required=True)
    parser.add_argument("--bcr-inputs", nargs="*", default=[])
    parser.add_argument("--mask-distribution", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ppl-relative-delta", type=float, default=0.10)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument("--layerwise-ratio-tolerance", type=float, default=0.005)
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


def _identity(*, method: str, selection_scope: str, protection: str, ratio: Any) -> tuple[str, str, str, str]:
    return method, selection_scope, protection, _ratio_key(ratio)


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
    return payload


def read_bcr(path: Path) -> tuple[tuple[str, str, str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    method = str(payload.get("mask_method") or payload.get("method"))
    selection_scope = str(payload.get("selection_scope") or payload.get("mask_stats", {}).get("selection_scope", "global"))
    protection = str(payload.get("protection") or payload.get("mask_stats", {}).get("protection", "none"))
    ratio = payload.get("requested_ratio")
    if ratio is None:
        ratio = payload.get("mask_stats", {}).get("requested_ratio", payload.get("mask_stats", {}).get("actual_ratio"))
    if ratio is None:
        raise KeyError(f"{path} missing requested ratio")
    return _identity(method=method, selection_scope=selection_scope, protection=protection, ratio=ratio), payload


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


def row_identity(row: dict[str, str]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "selection_scope": row["selection_scope"],
        "protection": row["protection"],
        "ratio": float(row["ratio"]),
        "bcr@q25": float(row["bcr@q25"]),
        "ppl": float(row["ppl"]),
        "arc_c": float(row["arc_c"]),
        "hellaswag": float(row["hellaswag"]),
    }


def read_mask_distribution(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def analyze_mask_distribution(rows: list[dict[str, str]], *, tolerance: float) -> dict[str, Any]:
    if not rows:
        return {
            "available": False,
            "layerwise_avoids_early_layer_pruning_collapse": False,
            "protected_layers_have_zero_pruned_units": False,
        }
    max_deviation = 0.0
    protected_ok = True
    early_collapse_ok = True
    setting_summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        selection_scope = row.get("selection_scope", "global")
        protection = row.get("protection", "none")
        requested = float(row.get("requested_ratio") or row.get("ratio") or 0.0)
        pruned_ratio = float(row["pruned_ratio"])
        layer = int(row["layer"])
        key = "|".join([row["method"], selection_scope, protection, _ratio_key(row["ratio"])])
        entry = setting_summaries.setdefault(
            key,
            {
                "method": row["method"],
                "selection_scope": selection_scope,
                "protection": protection,
                "ratio": float(row["ratio"]),
                "max_pruned_ratio": 0.0,
                "max_abs_deviation_from_requested": 0.0,
            },
        )
        entry["max_pruned_ratio"] = max(entry["max_pruned_ratio"], pruned_ratio)
        if selection_scope == "layerwise" and pruned_ratio > 0:
            deviation = abs(pruned_ratio - requested)
            max_deviation = max(max_deviation, deviation)
            entry["max_abs_deviation_from_requested"] = max(entry["max_abs_deviation_from_requested"], deviation)
            if layer <= 3 and pruned_ratio > requested + tolerance:
                early_collapse_ok = False
        if protection != "none" and pruned_ratio == 0.0:
            continue
        if protection.startswith("protect_first") and layer <= 3 and pruned_ratio != 0.0:
            protected_ok = False
    return {
        "available": True,
        "layerwise_avoids_early_layer_pruning_collapse": early_collapse_ok and max_deviation <= tolerance,
        "protected_layers_have_zero_pruned_units": protected_ok,
        "max_layerwise_abs_deviation_from_requested": max_deviation,
        "settings": list(setting_summaries.values()),
    }


def utility_damage(row: dict[str, str]) -> float:
    return float(row["ppl_relative_delta"]) + float(row["arc_c_drop"]) + float(row["hellaswag_drop"])


def analyze_protection(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_key = {
        (row["method"], row["selection_scope"], row["protection"], _ratio_key(row["ratio"])): row
        for row in rows
        if row["method"] != "dense"
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] == "dense" or row["protection"] == "none":
            continue
        baseline = by_key.get((row["method"], row["selection_scope"], "none", _ratio_key(row["ratio"])))
        if baseline is None:
            continue
        comparisons.append(
            {
                "method": row["method"],
                "ratio": float(row["ratio"]),
                "protection": row["protection"],
                "protected_damage": utility_damage(row),
                "unprotected_damage": utility_damage(baseline),
                "improved": utility_damage(row) < utility_damage(baseline),
            }
        )
    return {
        "has_protection_comparisons": bool(comparisons),
        "protected_layerwise_improves_utility_retention": any(item["improved"] for item in comparisons),
        "comparisons": comparisons,
    }


def recommend_ratio(matched_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not matched_rows:
        return {
            "recommended_mild_ratio": None,
            "reason": "No matched-utility setting found; recommend next ratio sweep at 0.005, 0.01, 0.015, 0.02.",
            "next_ratio_sweep": [0.005, 0.01, 0.015, 0.02],
        }
    best = min(matched_rows, key=lambda row: float(row["bcr@q25"]))
    return {
        "recommended_mild_ratio": float(best["ratio"]),
        "reason": "Selected from matched-utility settings by lowest BCR@q25.",
        "setting": row_identity(best),
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, Any]]:
    general_payloads = [read_general(Path(path)) for path in args.general_inputs]
    dense_payloads = [payload for payload in general_payloads if payload["method"] == "dense"]
    if len(dense_payloads) != 1:
        raise ValueError("Exactly one dense general-utility payload is required")
    dense = dense_payloads[0]
    dense_ppl = _finite_float(dense["ppl"], field="dense.ppl")
    dense_arc_c = _finite_float(dense["arc_c"], field="dense.arc_c")
    dense_hellaswag = _finite_float(dense["hellaswag"], field="dense.hellaswag")
    bcr_by_key = dict(read_bcr(Path(path)) for path in args.bcr_inputs)

    rows: list[dict[str, str]] = []
    for payload in general_payloads:
        method = str(payload["method"])
        selection_scope = str(payload.get("selection_scope", "dense" if method == "dense" else "global"))
        protection = str(payload.get("protection", "none"))
        ratio = _finite_float(payload.get("requested_ratio", payload.get("ratio", 0.0)), field=f"{method}.ratio")
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
        else:
            key = _identity(method=method, selection_scope=selection_scope, protection=protection, ratio=ratio)
            bcr = bcr_by_key.get(key)
            if bcr is None:
                raise KeyError(f"Missing BCR result for {key}")
            bcr_q25 = bcr.get("bcr@q25", bcr.get("bcr_at_q25"))
            bcr_0 = bcr.get("bcr@0", bcr.get("bcr_at_0"))
            pref_acc = bcr.get("preference_accuracy_pruned", bcr.get("preference_accuracy"))
            mean_margin_drop = bcr["mean_margin_drop"]

        rows.append(
            {
                "model": str(payload["model"]),
                "method": method,
                "selection_scope": selection_scope,
                "protection": protection,
                "ratio": _fmt(ratio),
                "requested_ratio": _fmt(payload.get("requested_ratio", ratio)),
                "actual_global_ratio": _fmt(payload.get("actual_global_ratio", ratio)),
                "actual_unprotected_ratio": _fmt(payload.get("actual_unprotected_ratio", ratio)),
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

    rows = sorted(rows, key=lambda row: (row["method"], row["selection_scope"], row["protection"], float(row["ratio"])))
    matched_rows = [row for row in rows if row["method"] != "dense" and row["matched_utility_flag"] == "true"]
    best_matched = min(matched_rows, key=lambda row: float(row["bcr@q25"])) if matched_rows else None
    mask_rows = read_mask_distribution(Path(args.mask_distribution))
    mask_analysis = analyze_mask_distribution(mask_rows, tolerance=args.layerwise_ratio_tolerance)
    protection_analysis = analyze_protection(rows)
    summary = {
        "num_rows": len(rows),
        "num_pruned_rows": len([row for row in rows if row["method"] != "dense"]),
        "num_matched_pruned_rows": len(matched_rows),
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
            "lowest_bcr_q25_among_matched_utility_settings": row_identity(best_matched) if best_matched else None,
            "layerwise_selection_avoids_early_layer_pruning_collapse": mask_analysis[
                "layerwise_avoids_early_layer_pruning_collapse"
            ],
            "protected_layerwise_selection_improves_utility_retention": protection_analysis[
                "protected_layerwise_improves_utility_retention"
            ],
            "mild_pruning_ratio_recommendation": recommend_ratio(matched_rows),
            "no_matched_utility_setting_message": (
                "No M11A setting satisfies matched utility; recommend next ratio sweep."
                if not matched_rows
                else None
            ),
        },
        "mask_distribution_analysis": mask_analysis,
        "protection_analysis": protection_analysis,
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
        "script": "scripts/summarize_m11a_layerwise.py",
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
        "notes": "M11A layerwise utility/BCR summary",
        "general_inputs": args.general_inputs,
        "bcr_inputs": args.bcr_inputs,
        "mask_distribution": args.mask_distribution,
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
            "layerwise_selection_avoids_early_layer_pruning_collapse": summary["answers"][
                "layerwise_selection_avoids_early_layer_pruning_collapse"
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
