from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.ffn_units import CoupledFFNUnitGroup, discover_coupled_ffn_unit_groups
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.pruning import apply_mask_plan_to_model, create_global_random_mask_plan, save_mask_artifacts
from pbp.utils import infer_model_device, set_seed, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply mask-based coupled FFN intermediate-neuron pruning.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["random"], default="random")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Reserved for protocol consistency.")
    parser.add_argument("--smoke-generate", action="store_true")
    parser.add_argument("--smoke-prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Do not load a model; write toy mask artifacts only.")
    parser.add_argument("--dry-run-layers", type=int, default=2)
    parser.add_argument("--dry-run-intermediate-size", type=int, default=8)
    return parser.parse_args()


def command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.ratio < 1.0):
        raise ValueError("--ratio must be in (0, 1)")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.dry_run_layers <= 0 or args.dry_run_intermediate_size <= 1:
        raise ValueError("--dry-run-layers must be > 0 and --dry-run-intermediate-size must be > 1")


def config_for_run(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script": "scripts/apply_mask_pruning.py",
        "model": args.model,
        "base_model": None,
        "dataset": None,
        "data_path": None,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": "auto" if args.device_map else "manual",
        "batch_size": None,
        "max_samples": args.max_samples,
        "output_path": args.out,
        "notes": "M5 mask-based coupled FFN pruning" + (" dry-run" if args.dry_run else ""),
        "method": args.method,
        "ratio": args.ratio,
        "device_map": args.device_map,
        "dry_run": args.dry_run,
        "local_files_only": args.local_files_only,
        "smoke_generate": args.smoke_generate,
    }


def toy_groups(num_layers: int, intermediate_size: int) -> list[CoupledFFNUnitGroup]:
    return [
        CoupledFFNUnitGroup(
            layer=layer,
            module_name=f"model.layers.{layer}.mlp",
            intermediate_size=intermediate_size,
            gate_shape=(intermediate_size, 4),
            up_shape=(intermediate_size, 4),
            down_shape=(4, intermediate_size),
        )
        for layer in range(num_layers)
    ]


def load_model_and_tokenizer(args: argparse.Namespace):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run real mask pruning") from exc

    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
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
    return model, tokenizer


def run_generation_smoke(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    import torch

    device = infer_model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    new_tokens = int(generated.shape[-1] - encoded["input_ids"].shape[-1])
    return {
        "generation_success": True,
        "generated_new_tokens": new_tokens,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

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
        out_dir = Path(args.out)
        if out_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output directory: {out_dir}")

        if args.dry_run:
            groups = toy_groups(args.dry_run_layers, args.dry_run_intermediate_size)
            mask_plan = create_global_random_mask_plan(groups, ratio=args.ratio, seed=args.seed)
            artifact_info = save_mask_artifacts(
                out_dir=out_dir,
                model_id=args.model,
                method=args.method,
                ratio=args.ratio,
                seed=args.seed,
                groups=groups,
                mask_plan=mask_plan,
                dry_run=True,
            )
            generation_info = {"generation_success": None, "generated_new_tokens": None}
        else:
            model, tokenizer = load_model_and_tokenizer(args)
            groups = discover_coupled_ffn_unit_groups(model)
            mask_plan = create_global_random_mask_plan(groups, ratio=args.ratio, seed=args.seed)
            applied_stats = apply_mask_plan_to_model(model, mask_plan)
            logger.stdout(json.dumps({"applied_mask_stats": applied_stats}, ensure_ascii=False))
            artifact_info = save_mask_artifacts(
                out_dir=out_dir,
                model_id=args.model,
                method=args.method,
                ratio=args.ratio,
                seed=args.seed,
                groups=groups,
                mask_plan=mask_plan,
                dry_run=False,
            )
            tokenizer.save_pretrained(out_dir)
            try:
                model.config.save_pretrained(out_dir)
            except Exception:
                pass
            if args.smoke_generate:
                generation_info = run_generation_smoke(
                    model,
                    tokenizer,
                    args.smoke_prompt,
                    args.max_new_tokens,
                )
            else:
                generation_info = {"generation_success": None, "generated_new_tokens": None}

        metrics = {
            "method": args.method,
            "requested_ratio": args.ratio,
            "total_units": int(artifact_info["total_units"]),
            "num_pruned_units": int(artifact_info["num_pruned_units"]),
            "num_kept_units": int(artifact_info["num_kept_units"]),
            "actual_ratio": float(artifact_info["actual_ratio"]),
            "num_masked_modules": int(artifact_info["num_masked_modules"]),
            "dry_run": args.dry_run,
            **generation_info,
        }
        finalize_run(run_paths, start_monotonic=start, metrics=metrics)
        message = {
            "run_id": run_paths.run_id,
            "out": args.out,
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
