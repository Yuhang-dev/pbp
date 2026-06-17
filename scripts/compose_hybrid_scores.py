from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.ffn_units import CoupledFFNUnitGroup
from pbp.io import ensure_parent
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.pruning import mask_plan_stats
from pbp.scoring import (
    flatten_scores,
    hybrid_scores,
    nonzero_score_stats,
    score_stats,
    scores_by_module,
    select_lowest_score_mask_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose utility-boundary hybrid pruning scores.")
    parser.add_argument("--utility-scores", required=True, help="Score artifact for activation or general_taylor.")
    parser.add_argument("--boundary-scores", required=True, help="Score artifact for boundary_taylor_weighted.")
    parser.add_argument("--method", default=None, help="Hybrid method label. Defaults to '<utility>_boundary'.")
    parser.add_argument("--alpha", type=float, required=True, help="Boundary score weight in utility + alpha * boundary.")
    parser.add_argument("--ratio", type=float, required=True, help="Ratio used for the emitted preview mask.")
    parser.add_argument("--selection-scope", choices=["global", "layerwise"], default="layerwise")
    parser.add_argument("--normalization-scope", choices=["global", "layerwise"], default="layerwise")
    parser.add_argument("--protect-first-n-layers", type=int, default=0)
    parser.add_argument("--protect-last-n-layers", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def group_from_dict(record: dict[str, Any]) -> CoupledFFNUnitGroup:
    return CoupledFFNUnitGroup(
        layer=int(record["layer"]),
        module_name=str(record["module_name"]),
        intermediate_size=int(record["intermediate_size"]),
        gate_shape=tuple(int(value) for value in record["gate_shape"]),
        up_shape=tuple(int(value) for value in record["up_shape"]),
        down_shape=tuple(int(value) for value in record["down_shape"]),
    )


def load_score_artifact(path: str | Path) -> tuple[dict[str, Any], list[CoupledFFNUnitGroup]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "pruning_importance_scores":
        raise ValueError(f"{path} is not a pruning importance score artifact")
    if "groups" not in payload or "scores_by_module" not in payload:
        raise ValueError(f"{path} is missing groups or scores_by_module")
    groups = [group_from_dict(record) for record in payload["groups"]]
    return payload, groups


def assert_compatible_artifacts(
    utility_payload: dict[str, Any],
    boundary_payload: dict[str, Any],
    utility_groups: list[CoupledFFNUnitGroup],
    boundary_groups: list[CoupledFFNUnitGroup],
) -> None:
    if utility_payload["model"] != boundary_payload["model"]:
        raise ValueError("Utility and boundary score artifacts must use the same model")
    if [group.to_dict() for group in utility_groups] != [group.to_dict() for group in boundary_groups]:
        raise ValueError("Utility and boundary score artifacts have incompatible FFN groups")


def write_score_artifact(
    *,
    out_path: Path,
    model_id: str,
    method: str,
    args: argparse.Namespace,
    groups: list[CoupledFFNUnitGroup],
    scores: list[Any],
    mask_plan: dict[str, Any],
    stats: dict[str, Any],
    method_info: dict[str, Any],
) -> None:
    payload = {
        "artifact_type": "pruning_importance_scores",
        "model": model_id,
        "method": method,
        "ratio": args.ratio,
        "alpha": args.alpha,
        "utility_method": method_info["utility_method"],
        "boundary_method": method_info["boundary_method"],
        "hybrid_normalization_scope": args.normalization_scope,
        "utility_scores": args.utility_scores,
        "boundary_scores": args.boundary_scores,
        "selection_scope": mask_plan.get("selection_scope", args.selection_scope),
        "protect_first_n_layers": mask_plan.get("protect_first_n_layers", args.protect_first_n_layers),
        "protect_last_n_layers": mask_plan.get("protect_last_n_layers", args.protect_last_n_layers),
        "protected_layers": mask_plan.get("protected_layers", []),
        "protection": mask_plan.get("protection"),
        "seed": args.seed,
        "score_semantics": "larger means more important; emitted mask prunes lowest scores",
        "hybrid_formula": "rank_norm(utility) + alpha * rank_norm(boundary)",
        "groups": [group.to_dict() for group in groups],
        "scores_by_module": scores_by_module(scores),
        "mask_format": "1=keep, 0=prune",
        "masks_by_module": mask_plan["masks_by_module"],
        "mask_stats": mask_plan_stats(mask_plan),
        "score_stats": stats,
        "method_info": method_info,
    }
    out_path = ensure_parent(out_path)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output file: {out_path}. Use --overwrite.")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def config_for_run(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/compose_hybrid_scores.py",
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
        "notes": "M12 hybrid utility-boundary score composition",
        "utility_scores": args.utility_scores,
        "boundary_scores": args.boundary_scores,
        "alpha": args.alpha,
        "ratio": args.ratio,
        "selection_scope": args.selection_scope,
        "normalization_scope": args.normalization_scope,
    }


def main() -> None:
    args = parse_args()
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative")
    if args.protect_first_n_layers < 0 or args.protect_last_n_layers < 0:
        raise ValueError("Protected layer counts must be non-negative")
    out_path = Path(args.out)
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
        utility_payload, utility_groups = load_score_artifact(args.utility_scores)
        boundary_payload, boundary_groups = load_score_artifact(args.boundary_scores)
        assert_compatible_artifacts(utility_payload, boundary_payload, utility_groups, boundary_groups)
        utility_method = str(utility_payload["method"])
        boundary_method = str(boundary_payload["method"])
        method = args.method or f"{utility_method}_boundary"

        utility = flatten_scores(utility_payload["scores_by_module"], utility_groups)
        boundary = flatten_scores(boundary_payload["scores_by_module"], utility_groups)
        scores = hybrid_scores(
            utility_groups,
            utility,
            boundary,
            alpha=args.alpha,
            normalization_scope=args.normalization_scope,
        )
        stats = score_stats(scores)
        stats.update(nonzero_score_stats(scores))
        mask_plan = select_lowest_score_mask_plan(
            utility_groups,
            scores,
            ratio=args.ratio,
            method=method,
            seed=args.seed,
            selection_scope=args.selection_scope,
            protect_first_n_layers=args.protect_first_n_layers,
            protect_last_n_layers=args.protect_last_n_layers,
        )
        mask_plan.update(
            {
                "alpha": args.alpha,
                "utility_method": utility_method,
                "boundary_method": boundary_method,
                "hybrid_normalization_scope": args.normalization_scope,
                "utility_scores": args.utility_scores,
                "boundary_scores": args.boundary_scores,
            }
        )
        method_info = {
            "utility_method": utility_method,
            "boundary_method": boundary_method,
            "alpha": args.alpha,
            "hybrid_normalization_scope": args.normalization_scope,
            "hybrid_formula": "rank_norm(utility) + alpha * rank_norm(boundary)",
            "utility_scores": args.utility_scores,
            "boundary_scores": args.boundary_scores,
            "utility_score_stats": utility_payload.get("score_stats", {}),
            "boundary_score_stats": boundary_payload.get("score_stats", {}),
        }
        write_score_artifact(
            out_path=out_path,
            model_id=str(utility_payload["model"]),
            method=method,
            args=args,
            groups=utility_groups,
            scores=scores,
            mask_plan=mask_plan,
            stats=stats,
            method_info=method_info,
        )
        mask_stats = mask_plan_stats(mask_plan)
        metrics = {
            "method": method,
            "utility_method": utility_method,
            "boundary_method": boundary_method,
            "alpha": args.alpha,
            "normalization_scope": args.normalization_scope,
            "selection_scope": mask_stats["selection_scope"],
            "requested_ratio": args.ratio,
            "actual_global_ratio": float(mask_stats["actual_global_ratio"]),
            "actual_unprotected_ratio": float(mask_stats["actual_unprotected_ratio"]),
            "num_pruned_units": int(mask_stats["num_pruned_units"]),
            "scores_finite": bool(stats["scores_finite"]),
            "out": str(out_path),
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
