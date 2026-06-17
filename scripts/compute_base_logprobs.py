from __future__ import annotations

import argparse
import json
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
from pbp.logprobs import compute_response_logprobs_batch
from pbp.utils import batched, infer_model_device, set_seed, sha256_text, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cached base-model response log-probs.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--chat-template-model", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", default="compute_base_logprobs")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def resolve_max_samples(args: argparse.Namespace) -> int | None:
    if args.max_examples is not None and args.max_samples is not None and args.max_examples != args.max_samples:
        raise ValueError("--max-examples and --max-samples disagree")
    return args.max_samples if args.max_samples is not None else args.max_examples


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    max_samples = resolve_max_samples(args)
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")


def config_for_run(args: argparse.Namespace, max_samples: int | None) -> dict[str, Any]:
    return {
        "script": "scripts/compute_base_logprobs.py",
        "model": args.base_model,
        "base_model": args.base_model,
        "dataset": None,
        "data_path": args.data,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "auto" if args.device_map else "manual",
        "batch_size": args.batch_size,
        "max_samples": max_samples,
        "output_path": args.out,
        "notes": "Cached base-model response logprobs with instruct chat-template context",
        "chat_template_model": args.chat_template_model,
        "device_map": args.device_map,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }


def assert_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --overwrite.")


def load_tokenizer(model_id: str, args: argparse.Namespace):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run this script") from exc

    return AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )


def load_model(args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit("Install transformers to run this script") from exc

    import torch

    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if not args.device_map:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def compute_base_logprob_records(args: argparse.Namespace, logger: RunLogger) -> list[dict[str, Any]]:
    tokenizer = load_tokenizer(args.base_model, args)
    template_model = args.chat_template_model or args.base_model
    template_tokenizer = load_tokenizer(template_model, args)
    model = load_model(args)
    device = infer_model_device(model)

    max_samples = resolve_max_samples(args)
    examples = read_jsonl(args.data)
    if max_samples is not None:
        examples = examples[:max_samples]
    if not examples:
        raise ValueError("No examples to score")

    outputs_by_id: dict[str, dict] = {}
    expanded: list[tuple[str, str, str, str]] = []
    for example in examples:
        formatted_prompt = format_prompt(example["prompt"], template_tokenizer)
        prompt_hash = sha256_text(formatted_prompt)
        outputs_by_id[example["id"]] = {
            "id": example["id"],
            "source": example.get("source", ""),
            "prompt_sha256": prompt_hash,
            "chat_template_model": template_model,
        }
        expanded.append((example["id"], "chosen", formatted_prompt, example["chosen"]))
        expanded.append((example["id"], "rejected", formatted_prompt, example["rejected"]))

    for chunk in tqdm(batched(expanded, args.batch_size), desc="base logprobs"):
        pairs = [(item[2], item[3]) for item in chunk]
        logprobs = compute_response_logprobs_batch(model, tokenizer, pairs, device=device)
        for item, logprob in zip(chunk, logprobs, strict=True):
            example_id, response_key, _, _ = item
            outputs_by_id[example_id][response_key] = logprob.to_dict()

    logger.stdout(f"Computed base response logprobs for {len(examples)} examples")
    return [outputs_by_id[example["id"]] for example in examples]


def finite_logprobs(records: list[dict[str, Any]]) -> bool:
    import math

    values: list[float] = []
    for record in records:
        values.append(float(record["chosen"]["length_normalized_logprob"]))
        values.append(float(record["rejected"]["length_normalized_logprob"]))
    return all(math.isfinite(value) for value in values)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    out_path = Path(args.out)
    max_samples = resolve_max_samples(args)
    start = time.monotonic()
    run_paths = initialize_run(
        args.run_name,
        config=config_for_run(args, max_samples),
        command=command_string(),
        out_root=args.runs_dir,
        cwd=Path.cwd(),
    )
    logger = RunLogger(run_paths)

    try:
        assert_can_write(out_path, overwrite=args.overwrite)
        records = compute_base_logprob_records(args, logger)
        write_jsonl(records, out_path)
        metrics = {
            "num_examples": len(records),
            "num_scored_responses": len(records) * 2,
            "length_normalized_logprobs_finite": finite_logprobs(records),
            "chat_template_model": args.chat_template_model or args.base_model,
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
