from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.ffn_units import discover_coupled_ffn_unit_groups
from pbp.io import ensure_parent, read_jsonl
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.pruning import mask_plan_stats
from pbp.scoring import (
    activation_scores,
    magnitude_scores,
    random_scores,
    score_stats,
    scores_by_module,
    select_lowest_score_mask_plan,
)
from pbp.utils import model_id_to_slug, set_seed, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score coupled FFN intermediate-neuron pruning importance.")
    parser.add_argument("--model", default=None, help="Model ID or local path for random/magnitude/activation scoring.")
    parser.add_argument("--instruct-model", default=None, help="Alias for --model, kept for later milestone commands.")
    parser.add_argument("--data", default=None, help="Calibration JSONL preference pairs. Required for activation.")
    parser.add_argument("--method", choices=["random", "magnitude", "activation"], required=True)
    parser.add_argument("--ratio", type=float, default=0.10, help="Fraction selected for pruning in the emitted mask.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--text-mode", choices=["prompt", "chosen", "rejected", "chosen_rejected"], default="chosen_rejected")
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def resolve_model(args: argparse.Namespace) -> str:
    model = args.model or args.instruct_model
    if not model:
        raise ValueError("Provide --model or --instruct-model")
    return model


def resolve_output_path(args: argparse.Namespace, model_id: str) -> Path:
    if args.out:
        return Path(args.out)
    if args.out_dir:
        return Path(args.out_dir) / f"{model_id_to_slug(model_id)}_{args.method}_scores.json"
    raise ValueError("Provide --out or --out-dir")


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.ratio < 1.0):
        raise ValueError("--ratio must be in (0, 1)")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.method == "activation" and not args.data:
        raise ValueError("--data is required for activation scoring")


def config_for_run(args: argparse.Namespace, model_id: str, out_path: Path) -> dict[str, Any]:
    return {
        "script": "scripts/score_pruning_importance.py",
        "model": model_id,
        "base_model": None,
        "dataset": None,
        "data_path": args.data,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "auto" if args.device_map else "manual",
        "batch_size": args.batch_size if args.method == "activation" else None,
        "max_samples": args.max_samples,
        "output_path": str(out_path),
        "notes": f"M6 {args.method} coupled FFN pruning importance scoring",
        "method": args.method,
        "ratio": args.ratio,
        "device_map": args.device_map,
        "local_files_only": args.local_files_only,
        "text_mode": args.text_mode if args.method == "activation" else None,
        "max_length": args.max_length if args.method == "activation" else None,
    }


def load_model(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install transformers to score pruning importance") from exc

    import torch

    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not args.device_map:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def load_tokenizer(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to score activation importance") from exc

    return AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.data:
        return []
    records = read_jsonl(args.data)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if args.method == "activation" and not records:
        raise ValueError("No calibration records available for activation scoring")
    return records


def write_score_artifact(
    *,
    out_path: Path,
    model_id: str,
    args: argparse.Namespace,
    groups: list[Any],
    scores: list[Any],
    mask_plan: dict[str, Any],
    stats: dict[str, Any],
    method_info: dict[str, Any],
) -> None:
    payload = {
        "artifact_type": "pruning_importance_scores",
        "model": model_id,
        "method": args.method,
        "ratio": args.ratio,
        "seed": args.seed,
        "score_semantics": "larger means more important; emitted mask prunes lowest scores",
        "groups": [group.to_dict() for group in groups],
        "scores_by_module": scores_by_module(scores),
        "mask_format": "1=keep, 0=prune",
        "masks_by_module": mask_plan["masks_by_module"],
        "mask_stats": mask_plan_stats(mask_plan),
        "score_stats": stats,
        "method_info": method_info,
    }
    out_path = ensure_parent(out_path)
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {out_path}")
    out_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_id = resolve_model(args)
    out_path = resolve_output_path(args, model_id)
    set_seed(args.seed)

    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args, model_id, out_path),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        if out_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output file: {out_path}")

        model = load_model(model_id, args)
        groups = discover_coupled_ffn_unit_groups(model)
        method_info: dict[str, Any] = {}
        if args.method == "random":
            scores = random_scores(groups, seed=args.seed)
        elif args.method == "magnitude":
            scores = magnitude_scores(model, groups)
        elif args.method == "activation":
            tokenizer = load_tokenizer(model_id, args)
            records = load_records(args)
            scores, method_info = activation_scores(
                model,
                tokenizer,
                records,
                groups,
                batch_size=args.batch_size,
                max_length=args.max_length,
                text_mode=args.text_mode,
                use_chat_template=not args.no_chat_template,
            )
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        stats = score_stats(scores)
        mask_plan = select_lowest_score_mask_plan(
            groups,
            scores,
            ratio=args.ratio,
            method=args.method,
            seed=args.seed,
        )
        write_score_artifact(
            out_path=out_path,
            model_id=model_id,
            args=args,
            groups=groups,
            scores=scores,
            mask_plan=mask_plan,
            stats=stats,
            method_info=method_info,
        )

        mask_stats = mask_plan_stats(mask_plan)
        metrics = {
            "method": args.method,
            "num_groups": len(groups),
            "total_units": int(mask_stats["total_units"]),
            "num_scores": int(stats["num_scores"]),
            "scores_finite": bool(stats["scores_finite"]),
            "min_score": float(stats["min_score"]),
            "max_score": float(stats["max_score"]),
            "mean_score": float(stats["mean_score"]),
            "std_score": float(stats["std_score"]),
            "requested_ratio": args.ratio,
            "actual_ratio": float(mask_stats["actual_ratio"]),
            "num_pruned_units": int(mask_stats["num_pruned_units"]),
            "max_samples": args.max_samples,
            **method_info,
        }
        finalize_run(run_paths, start_monotonic=start, metrics=metrics)
        message = {
            "run_id": run_paths.run_id,
            "out": str(out_path),
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
