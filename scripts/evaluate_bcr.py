from __future__ import annotations

import argparse
import gc
import json
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.chat_format import format_prompt
from pbp.io import ensure_parent, read_jsonl, read_jsonl_map, write_jsonl
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.logprobs import compute_response_logprobs_batch
from pbp.margins import compute_preference_margin, length_normalized_from_logprob
from pbp.metrics import summarize_bcr
from pbp.pruning import apply_mask_plan_to_model, mask_plan_stats
from pbp.utils import batched, infer_model_device, set_seed, sha256_text, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BCR for a pruned or masked model.")
    parser.add_argument("--model", required=True, help="Pruned model path, mask artifact directory, or dense model ID.")
    parser.add_argument("--mask-config", default=None, help="Optional explicit mask_config.json path.")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--base-logprobs", default=None)
    parser.add_argument("--dense-margins", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--records-out", default=None)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if not args.base_model and not args.base_logprobs:
        raise ValueError("Provide --base-model or --base-logprobs")


def config_for_run(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/evaluate_bcr.py",
        "model": args.model,
        "base_model": args.base_model,
        "dataset": None,
        "data_path": args.data,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "auto" if args.device_map else "manual",
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "output_path": args.out,
        "notes": "M7 BCR evaluation for pruned model",
        "dense_margins": args.dense_margins,
        "base_logprobs": args.base_logprobs,
        "mask_config": args.mask_config,
        "local_files_only": args.local_files_only,
    }


def assert_can_write(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def load_tokenizer(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run BCR evaluation") from exc

    return AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )


def load_model(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install transformers to run BCR evaluation") from exc

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


def resolve_mask_config(model_arg: str, explicit_mask_config: str | None) -> Path | None:
    if explicit_mask_config:
        return Path(explicit_mask_config)
    model_path = Path(model_arg)
    candidate = model_path / "mask_config.json"
    if model_path.is_dir() and candidate.is_file():
        return candidate
    if model_arg.startswith(("outputs/", "outputs\\")) or model_path.is_absolute():
        raise FileNotFoundError(
            f"Expected masked model artifact directory with mask_config.json, but it was not found: {model_arg}. "
            "Run scripts/apply_mask_pruning.py for this method/ratio before evaluate_bcr.py."
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


def score_model(
    *,
    model: Any,
    tokenizer: Any,
    formatted_prompts: list[str],
    examples: list[dict[str, Any]],
    batch_size: int,
    desc: str,
) -> dict[str, dict[str, Any]]:
    device = infer_model_device(model)
    expanded: list[tuple[str, str, str, str]] = []
    outputs_by_id: dict[str, dict[str, Any]] = {}
    for example, formatted_prompt in zip(examples, formatted_prompts, strict=True):
        outputs_by_id[example["id"]] = {"prompt_sha256": sha256_text(formatted_prompt)}
        expanded.append((example["id"], "chosen", formatted_prompt, example["chosen"]))
        expanded.append((example["id"], "rejected", formatted_prompt, example["rejected"]))

    for chunk in tqdm(batched(expanded, batch_size), desc=desc):
        logprobs = compute_response_logprobs_batch(
            model,
            tokenizer,
            [(item[2], item[3]) for item in chunk],
            device=device,
        )
        for item, logprob in zip(chunk, logprobs, strict=True):
            example_id, response_key, _, _ = item
            outputs_by_id[example_id][response_key] = logprob.to_dict()
    return outputs_by_id


def load_base_logprobs(path: str) -> dict[str, dict[str, Any]]:
    return read_jsonl_map(path, key="id")


def pruned_margin_record(
    *,
    example_id: str,
    dense_record: dict[str, Any],
    pruned_record: dict[str, Any],
    base_record: dict[str, Any],
) -> dict[str, Any]:
    ell_pruned_chosen = length_normalized_from_logprob(pruned_record["chosen"])
    ell_pruned_rejected = length_normalized_from_logprob(pruned_record["rejected"])
    ell_base_chosen = length_normalized_from_logprob(base_record["chosen"])
    ell_base_rejected = length_normalized_from_logprob(base_record["rejected"])
    delta_pruned = compute_preference_margin(
        ell_pruned_chosen,
        ell_pruned_rejected,
        ell_base_chosen,
        ell_base_rejected,
    )
    delta_dense = float(dense_record["delta_dense"])
    return {
        "id": example_id,
        "delta_dense": delta_dense,
        "delta_pruned": delta_pruned,
        "margin_drop": delta_dense - delta_pruned,
        "boundary_crossed": delta_dense > 0.0 and delta_pruned <= 0.0,
        "ell_pruned_chosen": ell_pruned_chosen,
        "ell_pruned_rejected": ell_pruned_rejected,
        "ell_base_chosen": ell_base_chosen,
        "ell_base_rejected": ell_base_rejected,
        "prompt_sha256": pruned_record.get("prompt_sha256"),
    }


def compute_records(args: argparse.Namespace, logger: RunLogger) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = read_jsonl(args.data)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("No examples to evaluate")

    dense_by_id = read_jsonl_map(args.dense_margins, key="id")
    missing_dense = [example["id"] for example in examples if example["id"] not in dense_by_id]
    if missing_dense:
        raise KeyError(f"Dense margins missing {len(missing_dense)} requested examples")

    mask_config_path = resolve_mask_config(args.model, args.mask_config)
    mask_plan, mask_config = load_mask_plan(mask_config_path)
    model_load_id = resolve_model_load_id(args.model, mask_config)

    pruned_tokenizer = load_tokenizer(model_load_id, args)
    formatted_prompts = [format_prompt(example["prompt"], pruned_tokenizer) for example in examples]

    pruned_model = load_model(model_load_id, args)
    applied_mask_stats: dict[str, Any] | None = None
    if mask_plan is not None:
        applied_mask_stats = apply_mask_plan_to_model(pruned_model, mask_plan)
        logger.stdout(json.dumps({"applied_mask_stats": applied_mask_stats}, ensure_ascii=False))
    pruned_by_id = score_model(
        model=pruned_model,
        tokenizer=pruned_tokenizer,
        formatted_prompts=formatted_prompts,
        examples=examples,
        batch_size=args.batch_size,
        desc="pruned logprobs",
    )
    cleanup_model(pruned_model)

    if args.base_logprobs:
        base_by_id = load_base_logprobs(args.base_logprobs)
    else:
        base_tokenizer = load_tokenizer(args.base_model, args)
        base_model = load_model(args.base_model, args)
        base_by_id = score_model(
            model=base_model,
            tokenizer=base_tokenizer,
            formatted_prompts=formatted_prompts,
            examples=examples,
            batch_size=args.batch_size,
            desc="base logprobs",
        )
        cleanup_model(base_model)

    records: list[dict[str, Any]] = []
    for example in examples:
        example_id = example["id"]
        if example_id not in base_by_id:
            raise KeyError(f"Base logprobs missing id={example_id}")
        records.append(
            pruned_margin_record(
                example_id=example_id,
                dense_record=dense_by_id[example_id],
                pruned_record=pruned_by_id[example_id],
                base_record=base_by_id[example_id],
            )
        )

    stats = mask_plan_stats(mask_plan) if mask_plan is not None else None
    method_info = {
        "model_load_id": model_load_id,
        "mask_config": str(mask_config_path) if mask_config_path is not None else None,
        "mask_method": mask_config.get("method") if mask_config else None,
        "selection_scope": mask_config.get("selection_scope") if mask_config else None,
        "protection": mask_config.get("protection") if mask_config else None,
        "alpha": mask_config.get("alpha") if mask_config else None,
        "utility_method": mask_config.get("utility_method") if mask_config else None,
        "boundary_method": mask_config.get("boundary_method") if mask_config else None,
        "hybrid_normalization_scope": mask_config.get("hybrid_normalization_scope") if mask_config else None,
        "requested_ratio": mask_config.get("requested_ratio", mask_config.get("ratio")) if mask_config else None,
        "actual_global_ratio": stats.get("actual_global_ratio") if stats else None,
        "actual_unprotected_ratio": stats.get("actual_unprotected_ratio") if stats else None,
        "num_protected_layers": stats.get("num_protected_layers") if stats else None,
        "mask_stats": stats,
        "used_base_logprobs": bool(args.base_logprobs),
        "num_examples": len(records),
    }
    return records, method_info


def finite_metrics(summary: dict[str, Any]) -> bool:
    def walk(value: Any) -> list[float]:
        if isinstance(value, dict):
            out: list[float] = []
            for nested in value.values():
                out.extend(walk(nested))
            return out
        if isinstance(value, (int, float)):
            return [float(value)]
        return []

    return all(math.isfinite(value) for value in walk(summary))


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    out_path = Path(args.out)
    records_path = Path(args.records_out) if args.records_out else None
    write_paths = [out_path] + ([records_path] if records_path is not None else [])
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
        assert_can_write(write_paths, overwrite=args.overwrite)
        records, method_info = compute_records(args, logger)
        summary = summarize_bcr(records)
        summary.update(method_info)
        if records_path is not None:
            write_jsonl(records, records_path)
            summary["records_out"] = str(records_path)
        ensure_parent(out_path).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metrics = {
            "num_examples": int(summary["num_pairs"]),
            "coverage_at_0": float(summary["coverage_at_0"]),
            "coverage_at_q25": float(summary["coverage_at_q25"]),
            "bcr_at_0": float(summary["bcr_at_0"]),
            "bcr_at_q25": float(summary["bcr_at_q25"]),
            "bcr_at_q50": float(summary["bcr_at_q50"]),
            "bcr_at_q75": float(summary["bcr_at_q75"]),
            "preference_accuracy_dense": float(summary["preference_accuracy_dense"]),
            "preference_accuracy_pruned": float(summary["preference_accuracy_pruned"]),
            "mean_margin_drop": float(summary["mean_margin_drop"]),
            "alpha": method_info.get("alpha"),
            "metrics_finite": finite_metrics(summary),
        }
        if method_info["mask_stats"] is not None:
            metrics.update(
                {
                    "mask_actual_ratio": float(method_info["mask_stats"]["actual_ratio"]),
                    "mask_num_pruned_units": int(method_info["mask_stats"]["num_pruned_units"]),
                    "mask_total_units": int(method_info["mask_stats"]["total_units"]),
                }
            )
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
