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
    selection_scope: str = "global",
    protect_first_n_layers: int = 0,
    protect_last_n_layers: int = 0,
) -> dict[str, Any]:
    validate_pruning_ratio(ratio)
    if selection_scope not in {"global", "layerwise"}:
        raise ValueError("selection_scope must be 'global' or 'layerwise'")
    if protect_first_n_layers < 0 or protect_last_n_layers < 0:
        raise ValueError("Protected layer counts must be non-negative")
    total_units = sum(group.intermediate_size for group in groups)
    if total_units <= 0:
        raise ValueError("No FFN units available for pruning")

    rng = random.Random(seed)
    layers = sorted({group.layer for group in groups})
    protected_layers = set(layers[:protect_first_n_layers])
    if protect_last_n_layers:
        protected_layers.update(layers[-protect_last_n_layers:])
    unprotected_layers = [layer for layer in layers if layer not in protected_layers]
    if not unprotected_layers:
        raise ValueError("All layers are protected; no units are available for pruning")

    units_by_layer: dict[int, list[tuple[str, int]]] = {}
    all_units: list[tuple[str, int, int]] = []
    for group in groups:
        layer_units = units_by_layer.setdefault(group.layer, [])
        for local_index in range(group.intermediate_size):
            layer_units.append((group.module_name, local_index))
            all_units.append((group.module_name, local_index, group.layer))

    pruned: set[tuple[str, int]] = set()
    if selection_scope == "global":
        eligible = [(module_name, unit_index) for module_name, unit_index, layer in all_units if layer not in protected_layers]
        num_pruned = int(round(len(eligible) * ratio))
        num_pruned = max(1, min(num_pruned, len(eligible) - 1))
        pruned.update(rng.sample(eligible, num_pruned))
    else:
        for layer in unprotected_layers:
            eligible = units_by_layer[layer]
            num_layer_pruned = int(round(len(eligible) * ratio))
            num_layer_pruned = max(1, min(num_layer_pruned, len(eligible) - 1))
            pruned.update(rng.sample(eligible, num_layer_pruned))

    masks_by_module: dict[str, list[int]] = {}
    for group in groups:
        mask: list[int] = []
        for local_index in range(group.intermediate_size):
            mask.append(0 if (group.module_name, local_index) in pruned else 1)
        masks_by_module[group.module_name] = mask

    num_pruned = len(pruned)
    unprotected_units = sum(len(units_by_layer[layer]) for layer in unprotected_layers)
    if protect_first_n_layers and protect_last_n_layers:
        protection = f"protect_first{protect_first_n_layers}_last{protect_last_n_layers}"
    elif protect_first_n_layers:
        protection = f"protect_first{protect_first_n_layers}"
    elif protect_last_n_layers:
        protection = f"protect_last{protect_last_n_layers}"
    else:
        protection = "none"

    return {
        "method": "random",
        "ratio": ratio,
        "requested_ratio": ratio,
        "seed": seed,
        "total_units": total_units,
        "unprotected_units": unprotected_units,
        "num_pruned_units": num_pruned,
        "num_pruned_unprotected_units": num_pruned,
        "actual_ratio": num_pruned / total_units,
        "actual_global_ratio": num_pruned / total_units,
        "actual_unprotected_ratio": num_pruned / unprotected_units,
        "selection_scope": selection_scope,
        "selection_rule": "random",
        "protect_first_n_layers": protect_first_n_layers,
        "protect_last_n_layers": protect_last_n_layers,
        "protected_layers": sorted(protected_layers),
        "num_protected_layers": len(protected_layers),
        "protection": protection,
        "masks_by_module": masks_by_module,
    }


def mask_plan_stats(mask_plan: dict[str, Any]) -> dict[str, Any]:
    masks_by_module = mask_plan["masks_by_module"]
    total_units = sum(len(mask) for mask in masks_by_module.values())
    num_pruned = sum(1 for mask in masks_by_module.values() for value in mask if int(value) == 0)
    stats = {
        "total_units": total_units,
        "num_pruned_units": num_pruned,
        "num_kept_units": total_units - num_pruned,
        "actual_ratio": num_pruned / total_units if total_units else 0.0,
        "num_masked_modules": len(masks_by_module),
    }
    unprotected_units = int(mask_plan.get("unprotected_units", total_units))
    stats.update(
        {
            "requested_ratio": mask_plan.get("requested_ratio", mask_plan.get("ratio")),
            "actual_global_ratio": num_pruned / total_units if total_units else 0.0,
            "actual_unprotected_ratio": num_pruned / unprotected_units if unprotected_units else 0.0,
            "unprotected_units": unprotected_units,
            "num_pruned_unprotected_units": num_pruned,
            "selection_scope": mask_plan.get("selection_scope", "global"),
            "selection_rule": mask_plan.get("selection_rule"),
            "protect_first_n_layers": int(mask_plan.get("protect_first_n_layers", 0)),
            "protect_last_n_layers": int(mask_plan.get("protect_last_n_layers", 0)),
            "protected_layers": list(mask_plan.get("protected_layers", [])),
            "num_protected_layers": int(mask_plan.get("num_protected_layers", 0)),
            "protection": mask_plan.get("protection", "none"),
        }
    )
    for key in (
        "alpha",
        "utility_method",
        "boundary_method",
        "hybrid_normalization_scope",
        "utility_scores",
        "boundary_scores",
    ):
        if key in mask_plan:
            stats[key] = mask_plan[key]
    return stats


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
    protect_first = int(stats.get("protect_first_n_layers", 0))
    protect_last = int(stats.get("protect_last_n_layers", 0))
    if protect_first and protect_last:
        protection = f"protect_first{protect_first}_last{protect_last}"
    elif protect_first:
        protection = f"protect_first{protect_first}"
    elif protect_last:
        protection = f"protect_last{protect_last}"
    else:
        protection = "none"
    config = {
        "artifact_type": "masked_structured_pruning",
        "model": model_id,
        "method": method,
        "ratio": ratio,
        "requested_ratio": stats.get("requested_ratio", ratio),
        "seed": seed,
        "dry_run": dry_run,
        **stats,
        "protection": protection,
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
