from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any

from tqdm import tqdm

from pbp.chat_format import format_prompt
from pbp.ffn_units import CoupledFFNUnitGroup
from pbp.pruning import get_module_by_name, validate_pruning_ratio
from pbp.utils import batched, infer_model_device


@dataclass(frozen=True)
class UnitScore:
    layer: int
    module_name: str
    unit_index: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num_pruned(total_units: int, ratio: float) -> int:
    validate_pruning_ratio(ratio)
    if total_units <= 0:
        raise ValueError("No units available for selection")
    count = int(round(total_units * ratio))
    return max(1, min(count, total_units - 1))


def scores_by_module(scores: list[UnitScore]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for item in scores:
        module_scores = out.setdefault(item.module_name, [])
        if len(module_scores) != item.unit_index:
            raise ValueError(f"Non-contiguous scores for {item.module_name}")
        module_scores.append(float(item.score))
    return out


def flatten_scores(group_scores: dict[str, list[float]], groups: list[CoupledFFNUnitGroup]) -> list[UnitScore]:
    records: list[UnitScore] = []
    for group in groups:
        values = group_scores[group.module_name]
        if len(values) != group.intermediate_size:
            raise ValueError(
                f"Score length mismatch for {group.module_name}: "
                f"expected {group.intermediate_size}, got {len(values)}"
            )
        for unit_index, score in enumerate(values):
            records.append(
                UnitScore(
                    layer=group.layer,
                    module_name=group.module_name,
                    unit_index=unit_index,
                    score=float(score),
                )
            )
    return records


def score_stats(scores: list[UnitScore]) -> dict[str, Any]:
    values = [float(item.score) for item in scores]
    finite = all(math.isfinite(value) for value in values)
    if not values:
        raise ValueError("No scores to summarize")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "num_scores": len(values),
        "scores_finite": finite,
        "min_score": min(values),
        "max_score": max(values),
        "mean_score": mean,
        "std_score": math.sqrt(variance),
    }


def random_scores(groups: list[CoupledFFNUnitGroup], *, seed: int) -> list[UnitScore]:
    rng = random.Random(seed)
    records: list[UnitScore] = []
    for group in groups:
        for unit_index in range(group.intermediate_size):
            records.append(
                UnitScore(
                    layer=group.layer,
                    module_name=group.module_name,
                    unit_index=unit_index,
                    score=rng.random(),
                )
            )
    return records


def magnitude_scores(model: Any, groups: list[CoupledFFNUnitGroup]) -> list[UnitScore]:
    import torch

    records: list[UnitScore] = []
    with torch.no_grad():
        for group in groups:
            module = get_module_by_name(model, group.module_name)
            gate = module.gate_proj.weight.detach().float()
            up = module.up_proj.weight.detach().float()
            down = module.down_proj.weight.detach().float()
            gate_score = gate.pow(2).mean(dim=1).sqrt()
            up_score = up.pow(2).mean(dim=1).sqrt()
            down_score = down.pow(2).mean(dim=0).sqrt()
            combined = gate_score + up_score + down_score
            if int(combined.numel()) != group.intermediate_size:
                raise ValueError(f"Magnitude score shape mismatch at {group.module_name}")
            for unit_index, score in enumerate(combined.cpu().tolist()):
                records.append(
                    UnitScore(
                        layer=group.layer,
                        module_name=group.module_name,
                        unit_index=unit_index,
                        score=float(score),
                    )
                )
    return records


def calibration_texts_from_pairs(
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    text_mode: str,
    use_chat_template: bool = True,
) -> list[str]:
    texts: list[str] = []
    for record in records:
        prompt = str(record["prompt"])
        formatted_prompt = format_prompt(
            prompt,
            tokenizer,
            use_chat_template=use_chat_template,
            add_generation_prompt=True,
        )
        if text_mode == "prompt":
            texts.append(formatted_prompt)
        elif text_mode == "chosen":
            texts.append(formatted_prompt + str(record["chosen"]))
        elif text_mode == "rejected":
            texts.append(formatted_prompt + str(record["rejected"]))
        elif text_mode == "chosen_rejected":
            texts.append(formatted_prompt + str(record["chosen"]))
            texts.append(formatted_prompt + str(record["rejected"]))
        else:
            raise ValueError(f"Unsupported text_mode: {text_mode}")
    return texts


def _make_activation_forward(
    module_name: str,
    sums: dict[str, Any],
    counts: dict[str, int],
    active_mask: dict[str, Any],
):
    def forward(self, x):
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        values = hidden.detach().abs().float()
        mask = active_mask.get("attention_mask")
        if mask is not None and values.ndim == 3:
            weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
            sums[module_name] += (values * weights).sum(dim=(0, 1)).cpu()
            counts[module_name] += int(mask.sum().item())
        else:
            flat = values.reshape(-1, values.shape[-1])
            sums[module_name] += flat.sum(dim=0).cpu()
            counts[module_name] += int(flat.shape[0])
        return self.down_proj(hidden)

    return forward


def activation_scores(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    groups: list[CoupledFFNUnitGroup],
    *,
    batch_size: int,
    max_length: int,
    text_mode: str = "chosen_rejected",
    use_chat_template: bool = True,
) -> tuple[list[UnitScore], dict[str, Any]]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    texts = calibration_texts_from_pairs(records, tokenizer, text_mode=text_mode, use_chat_template=use_chat_template)
    if not texts:
        raise ValueError("No calibration texts available for activation scoring")
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    sums = {group.module_name: torch.zeros(group.intermediate_size, dtype=torch.float32) for group in groups}
    counts = {group.module_name: 0 for group in groups}
    active_mask: dict[str, Any] = {}
    original_forwards: dict[str, Any] = {}
    for group in groups:
        module = get_module_by_name(model, group.module_name)
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError(f"Module {group.module_name} is not a supported SwiGLU MLP")
        original_forwards[group.module_name] = module.forward
        module.forward = MethodType(_make_activation_forward(group.module_name, sums, counts, active_mask), module)

    device = infer_model_device(model)
    try:
        with torch.no_grad():
            for batch in tqdm(batched(texts, batch_size), desc="activation scoring"):
                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                active_mask["attention_mask"] = encoded.get("attention_mask")
                model(**encoded, use_cache=False)
    finally:
        for group in groups:
            module = get_module_by_name(model, group.module_name)
            module.forward = original_forwards[group.module_name]

    group_scores: dict[str, list[float]] = {}
    for group in groups:
        count = counts[group.module_name]
        if count <= 0:
            raise ValueError(f"No activation tokens collected for {group.module_name}")
        group_scores[group.module_name] = (sums[group.module_name] / count).tolist()

    return flatten_scores(group_scores, groups), {
        "num_calibration_pairs": len(records),
        "num_calibration_texts": len(texts),
        "text_mode": text_mode,
        "max_length": max_length,
        "batch_size": batch_size,
    }


def select_lowest_score_mask_plan(
    groups: list[CoupledFFNUnitGroup],
    scores: list[UnitScore],
    *,
    ratio: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    total_units = sum(group.intermediate_size for group in groups)
    num_pruned = _num_pruned(total_units, ratio)
    expected = {(group.module_name, unit_index) for group in groups for unit_index in range(group.intermediate_size)}
    observed = {(item.module_name, item.unit_index) for item in scores}
    if observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise ValueError(f"Score coverage mismatch: missing={missing}, extra={extra}")

    ranked = sorted(scores, key=lambda item: (item.score, item.layer, item.module_name, item.unit_index))
    pruned = {(item.module_name, item.unit_index) for item in ranked[:num_pruned]}
    masks_by_module: dict[str, list[int]] = {}
    for group in groups:
        masks_by_module[group.module_name] = [
            0 if (group.module_name, unit_index) in pruned else 1 for unit_index in range(group.intermediate_size)
        ]

    return {
        "method": method,
        "ratio": ratio,
        "seed": seed,
        "total_units": total_units,
        "num_pruned_units": num_pruned,
        "actual_ratio": num_pruned / total_units,
        "masks_by_module": masks_by_module,
        "selection_rule": "prune_lowest_score",
    }
