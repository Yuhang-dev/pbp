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
from pbp.io import read_jsonl, read_jsonl_map, write_jsonl
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.logprobs import (
    ResponseLogProb,
    build_response_token_mask,
    compute_response_logprobs_batch,
    response_logprob_token_count,
)
from pbp.margins import dense_margin_record
from pbp.utils import batched, infer_model_device, model_id_to_slug, set_seed, sha256_text, torch_dtype_from_name


class DryRunCharTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        if add_special_tokens:
            raise ValueError("DryRunCharTokenizer expects add_special_tokens=False")
        return {"input_ids": [ord(ch) for ch in text]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute base-reference-normalized dense margins.")
    parser.add_argument("--instruct-model", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--base-logprobs", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default="outputs/margins")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not load models; validate margin schema only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def output_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    dense_slug = model_id_to_slug(args.instruct_model)
    base_slug = model_id_to_slug(args.base_model) if args.base_model else "cached_base"
    return Path(args.out_dir) / f"dense_margins_{dense_slug}_vs_{base_slug}.jsonl"


def assert_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if not args.base_model and not args.base_logprobs:
        raise ValueError("Provide --base-model or --base-logprobs")


def config_for_run(args: argparse.Namespace, out_path: Path) -> dict[str, Any]:
    return {
        "script": "scripts/compute_dense_margins.py",
        "model": args.instruct_model,
        "base_model": args.base_model,
        "dataset": None,
        "data_path": args.data,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "auto" if args.device_map else "manual",
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "output_path": str(out_path),
        "notes": "M3 dense/base margin computation" + (" dry-run" if args.dry_run else ""),
        "device_map": args.device_map,
        "dry_run": args.dry_run,
        "local_files_only": args.local_files_only,
        "base_logprobs": args.base_logprobs,
    }


def load_tokenizer(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run real margin computation") from exc

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
        raise SystemExit("Install transformers to run real margin computation") from exc

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


def cleanup_model(model: Any) -> None:
    try:
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()


def dry_run_logprob(
    tokenizer: Any,
    formatted_prompt: str,
    response: str,
    *,
    offset: float,
) -> ResponseLogProb:
    _, response_mask = build_response_token_mask(tokenizer, formatted_prompt, response)
    count = response_logprob_token_count(response_mask)
    if count <= 0:
        raise ValueError("No response tokens available after causal shift")
    length_normalized = -1.0 + offset
    return ResponseLogProb(
        sum_logprob=length_normalized * count,
        num_response_tokens=count,
        length_normalized_logprob=length_normalized,
    )


def score_real_model(
    *,
    model_id: str,
    tokenizer: Any,
    formatted_prompts: list[str],
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    logger: RunLogger,
) -> dict[str, dict[str, Any]]:
    model = load_model(model_id, args)
    device = infer_model_device(model)

    expanded: list[tuple[str, str, str, str]] = []
    outputs_by_id: dict[str, dict[str, Any]] = {}
    for example, formatted_prompt in zip(examples, formatted_prompts, strict=True):
        outputs_by_id[example["id"]] = {"prompt_sha256": sha256_text(formatted_prompt)}
        expanded.append((example["id"], "chosen", formatted_prompt, example["chosen"]))
        expanded.append((example["id"], "rejected", formatted_prompt, example["rejected"]))

    for chunk in tqdm(batched(expanded, args.batch_size), desc=f"{model_id} logprobs"):
        pairs = [(item[2], item[3]) for item in chunk]
        logprobs = compute_response_logprobs_batch(model, tokenizer, pairs, device=device)
        for item, logprob in zip(chunk, logprobs, strict=True):
            example_id, response_key, _, _ = item
            outputs_by_id[example_id][response_key] = logprob.to_dict()

    logger.stdout(f"Computed response logprobs for {len(examples)} examples with {model_id}")
    cleanup_model(model)
    return outputs_by_id


def score_dry_run(
    *,
    tokenizer: Any,
    formatted_prompts: list[str],
    examples: list[dict[str, Any]],
    dense: bool,
) -> dict[str, dict[str, Any]]:
    outputs_by_id: dict[str, dict[str, Any]] = {}
    for index, (example, formatted_prompt) in enumerate(zip(examples, formatted_prompts, strict=True)):
        # Deterministic offsets make delta_dense finite and non-zero in fixture tests.
        dense_bonus = 0.10 + 0.01 * index if dense else 0.0
        rejected_bonus = 0.03 if dense else 0.0
        outputs_by_id[example["id"]] = {
            "prompt_sha256": sha256_text(formatted_prompt),
            "chosen": dry_run_logprob(
                tokenizer,
                formatted_prompt,
                example["chosen"],
                offset=dense_bonus,
            ).to_dict(),
            "rejected": dry_run_logprob(
                tokenizer,
                formatted_prompt,
                example["rejected"],
                offset=rejected_bonus,
            ).to_dict(),
        }
    return outputs_by_id


def compute_margin_records(args: argparse.Namespace, logger: RunLogger) -> list[dict[str, Any]]:
    examples = read_jsonl(args.data)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("No examples to score")

    if args.dry_run:
        instruct_tokenizer = DryRunCharTokenizer()
        base_tokenizer = instruct_tokenizer
    else:
        instruct_tokenizer = load_tokenizer(args.instruct_model, args)
        base_tokenizer = load_tokenizer(args.base_model, args) if args.base_model and not args.base_logprobs else None

    formatted_prompts = [format_prompt(example["prompt"], instruct_tokenizer) for example in examples]

    if args.dry_run:
        dense_by_id = score_dry_run(
            tokenizer=instruct_tokenizer,
            formatted_prompts=formatted_prompts,
            examples=examples,
            dense=True,
        )
        base_by_id = score_dry_run(
            tokenizer=base_tokenizer,
            formatted_prompts=formatted_prompts,
            examples=examples,
            dense=False,
        )
    else:
        dense_by_id = score_real_model(
            model_id=args.instruct_model,
            tokenizer=instruct_tokenizer,
            formatted_prompts=formatted_prompts,
            examples=examples,
            args=args,
            logger=logger,
        )
        if args.base_logprobs:
            base_by_id = read_jsonl_map(args.base_logprobs, key="id")
        else:
            if args.base_model is None or base_tokenizer is None:
                raise ValueError("--base-model is required when --base-logprobs is not provided")
            base_by_id = score_real_model(
                model_id=args.base_model,
                tokenizer=base_tokenizer,
                formatted_prompts=formatted_prompts,
                examples=examples,
                args=args,
                logger=logger,
            )

    margin_records: list[dict[str, Any]] = []
    for example in examples:
        example_id = example["id"]
        dense_record = dense_by_id[example_id]
        base_record = base_by_id[example_id]
        if dense_record["prompt_sha256"] != base_record["prompt_sha256"]:
            raise ValueError(f"Prompt hash mismatch for id={example_id}")
        record = dense_margin_record(
            example_id,
            dense_record["chosen"],
            dense_record["rejected"],
            base_record,
            prompt_sha256=dense_record["prompt_sha256"],
        )
        record["instruct_model"] = args.instruct_model
        record["base_model"] = args.base_model
        record["dry_run"] = args.dry_run
        margin_records.append(record)
    return margin_records


def metrics_from_records(records: list[dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    deltas = [float(record["delta_dense"]) for record in records]
    finite = all(math.isfinite(delta) for delta in deltas)
    return {
        "num_examples": len(records),
        "delta_dense_finite": finite,
        "mean_delta_dense": sum(deltas) / len(deltas),
        "min_delta_dense": min(deltas),
        "max_delta_dense": max(deltas),
        "dry_run": dry_run,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    out_path = output_path(args)
    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args, out_path),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        assert_can_write(out_path, overwrite=args.overwrite)
        records = compute_margin_records(args, logger)
        write_jsonl(records, out_path)
        metrics = metrics_from_records(records, dry_run=args.dry_run)
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
