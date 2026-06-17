from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_omp_threads = os.environ.get("OMP_NUM_THREADS", "1").strip()
if not _omp_threads.isdigit() or int(_omp_threads) <= 0:
    _omp_threads = "1"
os.environ["OMP_NUM_THREADS"] = _omp_threads

from pbp.eval_general import (
    compute_lm_perplexity,
    finite_general_metrics,
    load_arc_challenge_items,
    load_hellaswag_items,
    load_wikitext_texts,
    score_multiple_choice_accuracy,
)
from pbp.io import ensure_parent
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.pruning import apply_mask_plan_to_model, mask_plan_stats
from pbp.utils import set_seed, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight general-utility evaluation for dense or masked models.")
    parser.add_argument("--model", required=True, help="Dense model ID or masked pruning artifact directory.")
    parser.add_argument("--mask-config", default=None, help="Optional explicit mask_config.json path.")
    parser.add_argument("--method", default=None, help="Recorded method label. Inferred from mask_config when omitted.")
    parser.add_argument("--ratio", type=float, default=None, help="Recorded pruning ratio. Inferred from mask_config when omitted.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Use local model files only.")
    parser.add_argument("--datasets-local-files-only", action="store_true", help="Use local dataset cache only.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ppl-dataset", default="Salesforce/wikitext")
    parser.add_argument("--ppl-config", default="wikitext-2-raw-v1")
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--ppl-samples", type=int, default=64)
    parser.add_argument("--arc-dataset", default="allenai/ai2_arc")
    parser.add_argument("--arc-config", default="ARC-Challenge")
    parser.add_argument("--arc-split", default="validation")
    parser.add_argument("--arc-samples", type=int, default=100)
    parser.add_argument("--hellaswag-dataset", default="Rowan/hellaswag")
    parser.add_argument("--hellaswag-split", default="validation")
    parser.add_argument("--hellaswag-samples", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length < 16:
        raise ValueError("--max-length must be at least 16")
    for name in ("ppl_samples", "arc_samples", "hellaswag_samples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def assert_can_write(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def resolve_mask_config(model_arg: str, explicit_mask_config: str | None) -> Path | None:
    if explicit_mask_config:
        return Path(explicit_mask_config)
    model_path = Path(model_arg)
    candidate = model_path / "mask_config.json"
    if model_path.is_dir() and candidate.is_file():
        return candidate
    if model_arg.startswith(("outputs/", "outputs\\")) or model_path.is_absolute():
        raise FileNotFoundError(
            f"Expected masked model artifact directory with mask_config.json, but it was not found: {model_arg}"
        )
    return None


def load_mask_plan(mask_config_path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if mask_config_path is None:
        return None, {}
    config = json.loads(mask_config_path.read_text(encoding="utf-8"))
    masks_path = mask_config_path.parent / "masks.json"
    if not masks_path.is_file():
        raise FileNotFoundError(f"Missing masks.json next to {mask_config_path}")
    masks = json.loads(masks_path.read_text(encoding="utf-8"))
    return {
        "method": config.get("method", "unknown"),
        "ratio": config.get("ratio"),
        "seed": config.get("seed"),
        "total_units": config.get("total_units"),
        "num_pruned_units": config.get("num_pruned_units"),
        "actual_ratio": config.get("actual_ratio"),
        "masks_by_module": masks,
    }, config


def resolve_model_load_id(model_arg: str, mask_config: dict[str, Any]) -> str:
    if mask_config:
        return str(mask_config["model"])
    return model_arg


def load_tokenizer(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run general utility evaluation") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install transformers to run general utility evaluation") from exc

    import torch

    model_kwargs = {
        "dtype": torch_dtype_from_name(args.dtype),
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


def cleanup_model(model: Any) -> None:
    try:
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()


def config_for_run(args: argparse.Namespace, model_load_id: str | None = None) -> dict[str, Any]:
    return {
        "script": "scripts/evaluate_general.py",
        "model": args.model,
        "model_load_id": model_load_id,
        "base_model": None,
        "dataset": "Salesforce/wikitext, ARC-Challenge, and HellaSwag subsets",
        "data_path": None,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "auto" if args.device_map else "manual",
        "batch_size": args.batch_size,
        "max_samples": {
            "ppl": args.ppl_samples,
            "arc_challenge": args.arc_samples,
            "hellaswag": args.hellaswag_samples,
        },
        "max_length": args.max_length,
        "output_path": args.out,
        "notes": "M10A lightweight matched-utility evaluation",
        "local_files_only": args.local_files_only,
        "datasets_local_files_only": args.datasets_local_files_only,
    }


def evaluate(args: argparse.Namespace, logger: RunLogger) -> dict[str, Any]:
    mask_config_path = resolve_mask_config(args.model, args.mask_config)
    mask_plan, mask_config = load_mask_plan(mask_config_path)
    model_load_id = resolve_model_load_id(args.model, mask_config)
    method = args.method or mask_config.get("method") or "dense"
    ratio = args.ratio
    if ratio is None:
        ratio = mask_config.get("actual_ratio", mask_config.get("ratio", 0.0)) if mask_config else 0.0

    tokenizer = load_tokenizer(model_load_id, args)
    model = load_model(model_load_id, args)
    applied_mask_stats: dict[str, Any] | None = None
    if mask_plan is not None:
        applied_mask_stats = apply_mask_plan_to_model(model, mask_plan)
        logger.stdout(json.dumps({"applied_mask_stats": applied_mask_stats}, ensure_ascii=False))

    ppl_texts = load_wikitext_texts(
        dataset_name=args.ppl_dataset,
        dataset_config=args.ppl_config,
        split=args.ppl_split,
        max_samples=args.ppl_samples,
        cache_dir=args.dataset_cache_dir,
        local_files_only=args.datasets_local_files_only,
    )
    ppl_metrics = compute_lm_perplexity(
        model,
        tokenizer,
        ppl_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    arc_items = load_arc_challenge_items(
        dataset_name=args.arc_dataset,
        dataset_config=args.arc_config,
        split=args.arc_split,
        max_samples=args.arc_samples,
        cache_dir=args.dataset_cache_dir,
        local_files_only=args.datasets_local_files_only,
    )
    arc_metrics = score_multiple_choice_accuracy(model, tokenizer, arc_items, batch_size=args.batch_size)

    hellaswag_items = load_hellaswag_items(
        dataset_name=args.hellaswag_dataset,
        split=args.hellaswag_split,
        max_samples=args.hellaswag_samples,
        cache_dir=args.dataset_cache_dir,
        local_files_only=args.datasets_local_files_only,
    )
    hellaswag_metrics = score_multiple_choice_accuracy(model, tokenizer, hellaswag_items, batch_size=args.batch_size)

    summary = {
        "artifact_type": "general_utility_eval",
        "backend": "lightweight",
        "model": model_load_id,
        "model_arg": args.model,
        "method": method,
        "ratio": float(ratio),
        "loaded_successfully": True,
        "mask_config": str(mask_config_path) if mask_config_path is not None else None,
        "mask_stats": mask_plan_stats(mask_plan) if mask_plan is not None else None,
        "applied_mask_stats": applied_mask_stats,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "ppl": float(ppl_metrics["ppl"]),
        "arc_c": float(arc_metrics["accuracy"]),
        "hellaswag": float(hellaswag_metrics["accuracy"]),
        "task_metrics": {
            "ppl": ppl_metrics,
            "arc_challenge": {
                key: value for key, value in arc_metrics.items() if key != "records"
            },
            "hellaswag": {
                key: value for key, value in hellaswag_metrics.items() if key != "records"
            },
        },
        "datasets": {
            "ppl": {
                "dataset": args.ppl_dataset,
                "config": args.ppl_config,
                "split": args.ppl_split,
                "max_samples": args.ppl_samples,
            },
            "arc_challenge": {
                "dataset": args.arc_dataset,
                "config": args.arc_config,
                "split": args.arc_split,
                "max_samples": args.arc_samples,
            },
            "hellaswag": {
                "dataset": args.hellaswag_dataset,
                "split": args.hellaswag_split,
                "max_samples": args.hellaswag_samples,
            },
        },
    }
    summary["general_utility_finite"] = finite_general_metrics(summary)
    cleanup_model(model)
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

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
        assert_can_write([out_path], overwrite=args.overwrite)
        summary = evaluate(args, logger)
        ensure_parent(out_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = {
            "model": summary["model"],
            "method": summary["method"],
            "ratio": summary["ratio"],
            "loaded_successfully": bool(summary["loaded_successfully"]),
            "ppl": float(summary["ppl"]),
            "arc_c": float(summary["arc_c"]),
            "hellaswag": float(summary["hellaswag"]),
            "general_utility_finite": bool(summary["general_utility_finite"]),
            "metrics_finite": all(
                math.isfinite(float(summary[key])) for key in ("ppl", "arc_c", "hellaswag")
            ),
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
