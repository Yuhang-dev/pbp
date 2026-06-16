from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.chat_format import format_prompt
from pbp.io import read_jsonl, read_jsonl_map, write_json, write_jsonl
from pbp.logprobs import compute_response_logprobs_batch
from pbp.margins import dense_margin_record
from pbp.metrics import histogram_rows, summarize_dense_margins
from pbp.utils import batched, infer_model_device, sha256_text, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute dense/base-normalized preference margins.")
    parser.add_argument("--instruct-model", required=True)
    parser.add_argument("--base-logprobs", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--histogram-out", default=None)
    parser.add_argument("--histogram-bins", type=int, default=30)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def write_histogram_csv(summary: dict, path: str) -> None:
    rows = histogram_rows(summary["histogram"])
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_left", "bin_right", "count"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run this script") from exc

    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        args.instruct_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.instruct_model, **model_kwargs)
    if not args.device_map:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    device = infer_model_device(model)

    examples = read_jsonl(args.data)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    base_by_id = read_jsonl_map(args.base_logprobs)

    outputs_by_id: dict[str, dict] = {}
    expanded = []
    for example in examples:
        example_id = example["id"]
        if example_id not in base_by_id:
            raise KeyError(f"Missing base logprobs for id={example_id}")
        formatted_prompt = format_prompt(example["prompt"], tokenizer)
        prompt_hash = sha256_text(formatted_prompt)
        base_hash = base_by_id[example_id].get("prompt_sha256")
        if base_hash is not None and base_hash != prompt_hash:
            raise ValueError(
                f"Formatted prompt hash mismatch for id={example_id}. "
                "Recompute base logprobs with the instruct chat template."
            )
        outputs_by_id[example_id] = {"prompt_sha256": prompt_hash}
        expanded.append((example_id, "chosen", formatted_prompt, example["chosen"]))
        expanded.append((example_id, "rejected", formatted_prompt, example["rejected"]))

    for chunk in tqdm(list(batched(expanded, args.batch_size)), desc="dense logprobs"):
        pairs = [(item[2], item[3]) for item in chunk]
        logprobs = compute_response_logprobs_batch(model, tokenizer, pairs, device=device)
        for item, logprob in zip(chunk, logprobs, strict=True):
            example_id, response_key, _, _ = item
            outputs_by_id[example_id][response_key] = logprob.to_dict()

    margin_records = []
    for example in examples:
        example_id = example["id"]
        dense_record = outputs_by_id[example_id]
        margin_records.append(
            dense_margin_record(
                example_id,
                dense_record["chosen"],
                dense_record["rejected"],
                base_by_id[example_id],
                prompt_sha256=dense_record["prompt_sha256"],
            )
        )

    write_jsonl(margin_records, args.out)
    print(f"Wrote {len(margin_records)} dense margin records to {args.out}")

    if args.summary_out or args.histogram_out:
        summary = summarize_dense_margins(margin_records, bins=args.histogram_bins)
        if args.summary_out:
            write_json(summary, args.summary_out)
            print(f"Wrote coverage summary to {args.summary_out}")
        if args.histogram_out:
            write_histogram_csv(summary, args.histogram_out)
            print(f"Wrote margin histogram to {args.histogram_out}")


if __name__ == "__main__":
    main()
