from __future__ import annotations

import json
import random
from pathlib import Path
from types import MethodType
from typing import Any

from pbp.ffn_units import CoupledFFNUnitGroup
from pbp.io import ensure_parent


def validate_pruning_ratio(ratio: float) -> None:
    if not (0.0 < ratio < 1.0):
        raise ValueError("ratio must be in (0, 1)")


def create_global_random_mask_plan(
    groups: list[CoupledFFNUnitGroup],
    *,
    ratio: float,
    seed: int,
) -> dict[str, Any]:
    validate_pruning_ratio(ratio)
    total_units = sum(group.intermediate_size for group in groups)
    if total_units <= 0:
        raise ValueError("No FFN units available for pruning")

    num_pruned = int(round(total_units * ratio))
    num_pruned = max(1, min(num_pruned, total_units - 1))

    rng = random.Random(seed)
    pruned_global = set(rng.sample(range(total_units), num_pruned))

    masks_by_module: dict[str, list[int]] = {}
    cursor = 0
    for group in groups:
        mask: list[int] = []
        for local_index in range(group.intermediate_size):
            global_index = cursor + local_index
            mask.append(0 if global_index in pruned_global else 1)
        masks_by_module[group.module_name] = mask
        cursor += group.intermediate_size

    return {
        "method": "random",
        "ratio": ratio,
        "seed": seed,
        "total_units": total_units,
        "num_pruned_units": num_pruned,
        "actual_ratio": num_pruned / total_units,
        "masks_by_module": masks_by_module,
    }


def mask_plan_stats(mask_plan: dict[str, Any]) -> dict[str, Any]:
    masks_by_module = mask_plan["masks_by_module"]
    total_units = sum(len(mask) for mask in masks_by_module.values())
    num_pruned = sum(1 for mask in masks_by_module.values() for value in mask if int(value) == 0)
    return {
        "total_units": total_units,
        "num_pruned_units": num_pruned,
        "num_kept_units": total_units - num_pruned,
        "actual_ratio": num_pruned / total_units if total_units else 0.0,
        "num_masked_modules": len(masks_by_module),
    }


def get_module_by_name(model: Any, module_name: str) -> Any:
    current = model
    if not module_name:
        return current
    for part in module_name.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _masked_swiglu_forward(self, x):
    hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    mask = self._pbp_intermediate_mask.to(device=hidden.device, dtype=hidden.dtype)
    view_shape = [1] * (hidden.ndim - 1) + [mask.numel()]
    hidden = hidden * mask.view(*view_shape)
    return self.down_proj(hidden)


def apply_mask_plan_to_model(model: Any, mask_plan: dict[str, Any]) -> dict[str, Any]:
    import torch

    for module_name, mask_values in mask_plan["masks_by_module"].items():
        module = get_module_by_name(model, module_name)
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError(f"Module {module_name} is not a supported SwiGLU MLP")
        mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
        if hasattr(module, "_pbp_intermediate_mask"):
            module._pbp_intermediate_mask = mask_tensor
        else:
            module.register_buffer("_pbp_intermediate_mask", mask_tensor, persistent=True)
        if not hasattr(module, "_pbp_original_forward"):
            module._pbp_original_forward = module.forward
        module.forward = MethodType(_masked_swiglu_forward, module)
    return mask_plan_stats(mask_plan)


def save_mask_artifacts(
    *,
    out_dir: str | Path,
    model_id: str,
    method: str,
    ratio: float,
    seed: int,
    groups: list[CoupledFFNUnitGroup],
    mask_plan: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=False)
    stats = mask_plan_stats(mask_plan)
    config = {
        "artifact_type": "masked_structured_pruning",
        "model": model_id,
        "method": method,
        "ratio": ratio,
        "seed": seed,
        "dry_run": dry_run,
        **stats,
        "groups": [group.to_dict() for group in groups],
        "mask_format": "1=keep, 0=prune",
    }

    try:
        import torch

        torch_masks = {
            module_name: torch.tensor(mask_values, dtype=torch.float32)
            for module_name, mask_values in mask_plan["masks_by_module"].items()
        }
        torch.save(torch_masks, ensure_parent(out_path / "masks.pt"))
        config["masks_pt"] = "masks.pt"
    except Exception:
        pass

    config_path = out_path / "mask_config.json"
    masks_path = out_path / "masks.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    masks_path.write_text(
        json.dumps(mask_plan["masks_by_module"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_path),
        "mask_config": str(config_path),
        "masks_json": str(masks_path),
        "masks_pt": str(out_path / "masks.pt") if "masks_pt" in config else None,
        **stats,
    }
