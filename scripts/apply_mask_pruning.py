from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pbp.ffn_units import CoupledFFNUnitGroup, discover_coupled_ffn_unit_groups
from pbp.logging_utils import RunLogger, finalize_run, initialize_run
from pbp.pruning import apply_mask_plan_to_model, create_global_random_mask_plan, save_mask_artifacts
from pbp.scoring import flatten_scores, select_lowest_score_mask_plan
from pbp.utils import infer_model_device, set_seed, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply mask-based coupled FFN intermediate-neuron pruning.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scores", default=None, help="Pruning score artifact produced by score_pruning_importance.py.")
    parser.add_argument("--method", choices=["random"], default="random")
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--selection-scope", choices=["global", "layerwise"], default=None)
    parser.add_argument("--protect-first-n-layers", type=int, default=None)
    parser.add_argument("--protect-last-n-layers", type=int, default=None)
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
    if not args.model and not args.scores:
        raise ValueError("Provide --model for random masks or --scores for score-derived masks")
    if args.scores is None and args.ratio is None:
        raise ValueError("--ratio is required for random mask pruning")
    if args.ratio is not None and not (0.0 < args.ratio < 1.0):
        raise ValueError("--ratio must be in (0, 1)")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.protect_first_n_layers is not None and args.protect_first_n_layers < 0:
        raise ValueError("--protect-first-n-layers must be non-negative")
    if args.protect_last_n_layers is not None and args.protect_last_n_layers < 0:
        raise ValueError("--protect-last-n-layers must be non-negative")
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
        "scores": args.scores,
        "selection_scope": args.selection_scope,
        "protect_first_n_layers": args.protect_first_n_layers,
        "protect_last_n_layers": args.protect_last_n_layers,
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


def group_from_dict(record: dict[str, Any]) -> CoupledFFNUnitGroup:
    return CoupledFFNUnitGroup(
        layer=int(record["layer"]),
        module_name=str(record["module_name"]),
        intermediate_size=int(record["intermediate_size"]),
        gate_shape=tuple(int(value) for value in record["gate_shape"]),
        up_shape=tuple(int(value) for value in record["up_shape"]),
        down_shape=tuple(int(value) for value in record["down_shape"]),
    )


def load_score_artifact_mask_plan(
    score_path: str | Path,
    *,
    ratio: float | None,
    seed: int,
    selection_scope: str | None,
    protect_first_n_layers: int | None,
    protect_last_n_layers: int | None,
) -> tuple[str, str, float, list[CoupledFFNUnitGroup], dict[str, Any]]:
    payload = json.loads(Path(score_path).read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "pruning_importance_scores":
        raise ValueError(f"{score_path} is not a pruning importance score artifact")
    if "groups" not in payload or "scores_by_module" not in payload:
        raise ValueError(f"{score_path} is missing groups or scores_by_module")

    model_id = str(payload["model"])
    method = str(payload.get("method", "from_scores"))
    artifact_seed = int(payload.get("seed", seed))
    groups = [group_from_dict(record) for record in payload["groups"]]
    selected_ratio = float(ratio if ratio is not None else payload.get("ratio"))
    selected_scope = str(selection_scope or payload.get("selection_scope") or payload.get("mask_stats", {}).get("selection_scope") or "global")
    selected_protect_first = int(
        protect_first_n_layers
        if protect_first_n_layers is not None
        else payload.get("protect_first_n_layers", payload.get("mask_stats", {}).get("protect_first_n_layers", 0))
    )
    selected_protect_last = int(
        protect_last_n_layers
        if protect_last_n_layers is not None
        else payload.get("protect_last_n_layers", payload.get("mask_stats", {}).get("protect_last_n_layers", 0))
    )

    scores = flatten_scores(payload["scores_by_module"], groups)
    mask_plan = select_lowest_score_mask_plan(
        groups,
        scores,
        ratio=selected_ratio,
        method=method,
        seed=artifact_seed,
        selection_scope=selected_scope,
        protect_first_n_layers=selected_protect_first,
        protect_last_n_layers=selected_protect_last,
    )
    return model_id, method, selected_ratio, groups, mask_plan


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
        "dtype": torch_dtype_from_name(args.dtype),
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

        if args.scores:
            if args.dry_run:
                raise ValueError("--dry-run cannot be combined with --scores")
            model_id, method, ratio, groups, mask_plan = load_score_artifact_mask_plan(
                args.scores,
                ratio=args.ratio,
                seed=args.seed,
                selection_scope=args.selection_scope,
                protect_first_n_layers=args.protect_first_n_layers,
                protect_last_n_layers=args.protect_last_n_layers,
            )
            if args.model and args.model != model_id:
                raise ValueError(f"--model={args.model!r} does not match score artifact model={model_id!r}")
            artifact_info = save_mask_artifacts(
                out_dir=out_dir,
                model_id=model_id,
                method=method,
                ratio=ratio,
                seed=int(mask_plan.get("seed", args.seed)),
                groups=groups,
                mask_plan=mask_plan,
                dry_run=False,
            )
            generation_info = {"generation_success": None, "generated_new_tokens": None}
            args_method = method
            requested_ratio = ratio
        elif args.dry_run:
            groups = toy_groups(args.dry_run_layers, args.dry_run_intermediate_size)
            if args.ratio is None:
                raise ValueError("--ratio is required for dry-run mask pruning")
            mask_plan = create_global_random_mask_plan(
                groups,
                ratio=args.ratio,
                seed=args.seed,
                selection_scope=args.selection_scope or "global",
                protect_first_n_layers=args.protect_first_n_layers or 0,
                protect_last_n_layers=args.protect_last_n_layers or 0,
            )
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
            args_method = args.method
            requested_ratio = args.ratio
        else:
            if args.ratio is None:
                raise ValueError("--ratio is required for random mask pruning")
            model, tokenizer = load_model_and_tokenizer(args)
            groups = discover_coupled_ffn_unit_groups(model)
            mask_plan = create_global_random_mask_plan(
                groups,
                ratio=args.ratio,
                seed=args.seed,
                selection_scope=args.selection_scope or "global",
                protect_first_n_layers=args.protect_first_n_layers or 0,
                protect_last_n_layers=args.protect_last_n_layers or 0,
            )
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
            args_method = args.method
            requested_ratio = args.ratio

        metrics = {
            "method": args_method,
            "requested_ratio": requested_ratio,
            "total_units": int(artifact_info["total_units"]),
            "num_pruned_units": int(artifact_info["num_pruned_units"]),
            "num_kept_units": int(artifact_info["num_kept_units"]),
            "actual_ratio": float(artifact_info["actual_ratio"]),
            "actual_global_ratio": float(artifact_info["actual_global_ratio"]),
            "actual_unprotected_ratio": float(artifact_info["actual_unprotected_ratio"]),
            "selection_scope": artifact_info["selection_scope"],
            "protection": artifact_info.get("protection", "none"),
            "protect_first_n_layers": int(artifact_info["protect_first_n_layers"]),
            "protect_last_n_layers": int(artifact_info["protect_last_n_layers"]),
            "num_protected_layers": int(artifact_info["num_protected_layers"]),
            "num_masked_modules": int(artifact_info["num_masked_modules"]),
            "dry_run": args.dry_run,
            "source_scores": args.scores,
            "global_ratio_matches_request": math.isclose(
                float(artifact_info["actual_global_ratio"]), float(requested_ratio), rel_tol=0.0, abs_tol=1e-12
            ),
            "unprotected_ratio_matches_request": math.isclose(
                float(artifact_info["actual_unprotected_ratio"]), float(requested_ratio), rel_tol=0.0, abs_tol=1e-12
            ),
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
