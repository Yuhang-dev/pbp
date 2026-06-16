from __future__ import annotations

import argparse
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
from pbp.io import read_jsonl, write_jsonl
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.logprobs import (
    ResponseLogProb,
    build_response_token_mask,
    compute_response_logprobs_batch,
    response_logprob_token_count,
)
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
    parser = argparse.ArgumentParser(description="Compute response-only length-normalized log-probs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--chat-template-model", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default="outputs/logprobs")
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
    parser.add_argument("--dry-run", action="store_true", help="Do not load a model; validate masking/schema only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def output_path(args: argparse.Namespace) -> Path:
    if args.out:
        return Path(args.out)
    return Path(args.out_dir) / f"{model_id_to_slug(args.model)}_logprobs.jsonl"


def assert_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def config_for_run(args: argparse.Namespace, out_path: Path) -> dict[str, Any]:
    return {
        "script": "scripts/compute_logprobs.py",
        "model": args.model,
        "base_model": None,
        "dataset": None,
        "data_path": args.data,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "auto" if args.device_map else "manual",
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "output_path": str(out_path),
        "notes": "M2 response-only logprob computation" + (" dry-run" if args.dry_run else ""),
        "chat_template_model": args.chat_template_model,
        "device_map": args.device_map,
        "dry_run": args.dry_run,
        "local_files_only": args.local_files_only,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")


def load_model_and_tokenizers(args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run real logprob computation") from exc

    import torch

    tokenizer_kwargs = {
        "use_fast": True,
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    template_model = args.chat_template_model or args.model
    if template_model == args.model:
        template_tokenizer = tokenizer
    else:
        template_tokenizer = AutoTokenizer.from_pretrained(template_model, **tokenizer_kwargs)

    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.device_map:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, tokenizer, template_tokenizer


def dry_run_logprob(tokenizer: Any, formatted_prompt: str, response: str) -> ResponseLogProb:
    _, response_mask = build_response_token_mask(tokenizer, formatted_prompt, response)
    count = response_logprob_token_count(response_mask)
    if count <= 0:
        raise ValueError("No response tokens available after causal shift")
    sum_logprob = -1.0 * count
    return ResponseLogProb(
        sum_logprob=sum_logprob,
        num_response_tokens=count,
        length_normalized_logprob=sum_logprob / count,
    )


def make_record(
    example: dict[str, Any],
    formatted_prompt: str,
    chosen_logprob: ResponseLogProb,
    rejected_logprob: ResponseLogProb,
    *,
    model: str,
    chat_template_model: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "id": example["id"],
        "source": example.get("source", ""),
        "model": model,
        "chat_template_model": chat_template_model,
        "prompt_sha256": sha256_text(formatted_prompt),
        "chosen": chosen_logprob.to_dict(),
        "rejected": rejected_logprob.to_dict(),
        "dry_run": dry_run,
    }


def compute_records(args: argparse.Namespace, logger: RunLogger) -> list[dict[str, Any]]:
    examples = read_jsonl(args.data)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("No examples to score")

    if args.dry_run:
        tokenizer = DryRunCharTokenizer()
        template_tokenizer = tokenizer
        model = None
        device = None
    else:
        model, tokenizer, template_tokenizer = load_model_and_tokenizers(args)
        device = infer_model_device(model)

    template_model = args.chat_template_model or args.model
    formatted_prompts = [format_prompt(example["prompt"], template_tokenizer) for example in examples]

    if args.dry_run:
        records = []
        for example, formatted_prompt in zip(examples, formatted_prompts, strict=True):
            records.append(
                make_record(
                    example,
                    formatted_prompt,
                    dry_run_logprob(tokenizer, formatted_prompt, example["chosen"]),
                    dry_run_logprob(tokenizer, formatted_prompt, example["rejected"]),
                    model=args.model,
                    chat_template_model=template_model,
                    dry_run=True,
                )
            )
        return records

    outputs_by_id: dict[str, dict[str, Any]] = {}
    expanded: list[tuple[str, str, str, str]] = []
    for example, formatted_prompt in zip(examples, formatted_prompts, strict=True):
        outputs_by_id[example["id"]] = {
            "example": example,
            "formatted_prompt": formatted_prompt,
        }
        expanded.append((example["id"], "chosen", formatted_prompt, example["chosen"]))
        expanded.append((example["id"], "rejected", formatted_prompt, example["rejected"]))

    for chunk in tqdm(batched(expanded, args.batch_size), desc="response logprobs"):
        pairs = [(item[2], item[3]) for item in chunk]
        logprobs = compute_response_logprobs_batch(model, tokenizer, pairs, device=device)
        for item, logprob in zip(chunk, logprobs, strict=True):
            example_id, response_key, _, _ = item
            outputs_by_id[example_id][response_key] = logprob

    records = []
    for example in examples:
        row = outputs_by_id[example["id"]]
        records.append(
            make_record(
                example,
                row["formatted_prompt"],
                row["chosen"],
                row["rejected"],
                model=args.model,
                chat_template_model=template_model,
                dry_run=False,
            )
        )
    logger.stdout(f"Computed logprobs for {len(records)} examples")
    return records


def metrics_from_records(records: list[dict[str, Any]], *, dry_run: bool) -> dict[str, Any]:
    chosen_counts = [record["chosen"]["num_response_tokens"] for record in records]
    rejected_counts = [record["rejected"]["num_response_tokens"] for record in records]
    all_ell = [
        record[key]["length_normalized_logprob"]
        for record in records
        for key in ("chosen", "rejected")
    ]
    finite = all(math.isfinite(float(value)) for value in all_ell)
    return {
        "num_examples": len(records),
        "num_scored_responses": len(records) * 2,
        "min_chosen_response_tokens": min(chosen_counts),
        "min_rejected_response_tokens": min(rejected_counts),
        "mean_chosen_response_tokens": sum(chosen_counts) / len(chosen_counts),
        "mean_rejected_response_tokens": sum(rejected_counts) / len(rejected_counts),
        "length_normalized_logprobs_finite": finite,
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
        records = compute_records(args, logger)
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
