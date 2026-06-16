from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.chat_format import format_prompt
from pbp.io import read_jsonl, write_jsonl
from pbp.logprobs import compute_response_logprobs_batch
from pbp.utils import batched, infer_model_device, sha256_text, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute cached base-model response log-probs.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--chat-template-model", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


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
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    template_model = args.chat_template_model or args.base_model
    template_tokenizer = AutoTokenizer.from_pretrained(
        template_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if not args.device_map:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    device = infer_model_device(model)

    examples = read_jsonl(args.data)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    outputs_by_id: dict[str, dict] = {}
    expanded = []
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

    for chunk in tqdm(list(batched(expanded, args.batch_size)), desc="base logprobs"):
        pairs = [(item[2], item[3]) for item in chunk]
        logprobs = compute_response_logprobs_batch(model, tokenizer, pairs, device=device)
        for item, logprob in zip(chunk, logprobs, strict=True):
            example_id, response_key, _, _ = item
            outputs_by_id[example_id][response_key] = logprob.to_dict()

    write_jsonl((outputs_by_id[example["id"]] for example in examples), args.out)
    print(f"Wrote {len(examples)} base logprob records to {args.out}")


if __name__ == "__main__":
    main()
