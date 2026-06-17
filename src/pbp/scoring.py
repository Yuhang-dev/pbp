from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any

import numpy as np
from tqdm import tqdm

from pbp.chat_format import format_prompt
from pbp.ffn_units import CoupledFFNUnitGroup
from pbp.logprobs import build_response_token_mask
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


def nonzero_score_stats(scores: list[UnitScore]) -> dict[str, Any]:
    values = [abs(float(item.score)) for item in scores]
    if not values:
        raise ValueError("No scores to summarize")
    return {
        "num_nonzero_scores": int(sum(value > 0.0 for value in values)),
        "all_scores_zero": not any(value > 0.0 for value in values),
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


def threshold_from_dense_margins(
    dense_margins: list[float],
    *,
    tau_mode: str,
    tau_value: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    if tau_mode == "all":
        return None, {"tau_mode": tau_mode, "tau_calib": None}
    if tau_mode == "value":
        if tau_value is None:
            raise ValueError("--tau-value is required when tau_mode='value'")
        return float(tau_value), {"tau_mode": tau_mode, "tau_calib": float(tau_value)}

    positive = np.asarray([value for value in dense_margins if value > 0.0], dtype=float)
    if positive.size == 0:
        raise ValueError("Cannot compute boundary threshold: no positive dense margins")
    if tau_mode == "0":
        tau = 0.0
    elif tau_mode == "q25":
        tau = float(np.quantile(positive, 0.25))
    elif tau_mode == "q50":
        tau = float(np.quantile(positive, 0.50))
    elif tau_mode == "q75":
        tau = float(np.quantile(positive, 0.75))
    else:
        raise ValueError(f"Unsupported tau_mode: {tau_mode}")
    return tau, {"tau_mode": tau_mode, "tau_calib": tau}


def select_boundary_records(
    records: list[dict[str, Any]],
    dense_margin_by_id: dict[str, float],
    *,
    tau_mode: str,
    tau_value: float | None = None,
    margin_eps: float = 1e-6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dense_values = [float(dense_margin_by_id[str(record["id"])]) for record in records]
    tau, threshold_info = threshold_from_dense_margins(
        dense_values,
        tau_mode=tau_mode,
        tau_value=tau_value,
    )
    if tau is None:
        selected = list(records)
    else:
        selected = [
            record
            for record in records
            if float(dense_margin_by_id[str(record["id"])]) > tau
        ]
    if not selected:
        raise ValueError("No calibration pairs selected for Taylor scoring")

    raw_weights: list[float] = []
    for record in selected:
        delta = float(dense_margin_by_id[str(record["id"])])
        if tau is None:
            raw_weights.append(1.0)
        else:
            raw_weights.append(1.0 / max(delta - tau, margin_eps))
    mean_weight = sum(raw_weights) / len(raw_weights)
    normalized_weights = [weight / mean_weight for weight in raw_weights]
    selected_ids = [str(record["id"]) for record in selected]
    method_info = {
        **threshold_info,
        "num_calibration_pairs": len(records),
        "num_selected_pairs": len(selected),
        "selected_fraction": len(selected) / len(records),
        "selected_ids": selected_ids,
        "weight_min": min(normalized_weights),
        "weight_max": max(normalized_weights),
        "weight_mean": sum(normalized_weights) / len(normalized_weights),
    }
    weight_by_id = dict(zip(selected_ids, normalized_weights, strict=True))
    return selected, {**method_info, "weight_by_id": weight_by_id}


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def differentiable_response_logprobs_batch(
    model: Any,
    tokenizer: Any,
    prompt_response_pairs: list[tuple[str, str]],
    *,
    device: Any,
    max_length: int | None = None,
):
    import torch

    if max_length is not None and max_length <= 1:
        raise ValueError("max_length must be greater than 1 when provided")

    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for formatted_prompt, response in prompt_response_pairs:
        input_ids, response_mask = build_response_token_mask(tokenizer, formatted_prompt, response)
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            response_mask = response_mask[-max_length:]
        sequences.append(input_ids)
        masks.append(response_mask)

    max_len = max(len(sequence) for sequence in sequences)
    pad_id = _pad_token_id(tokenizer)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for row, (sequence, mask) in enumerate(zip(sequences, masks, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        response_mask[row, :length] = torch.tensor(mask, dtype=torch.bool, device=device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_response_mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
    token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)
    counts = shift_response_mask.sum(dim=1).clamp_min(1)
    length_normalized = (token_logprobs * shift_response_mask.float()).sum(dim=1) / counts
    return length_normalized, attention_mask


def _make_taylor_forward(
    module_name: str,
    sums: dict[str, Any],
    active_mask: dict[str, Any],
    *,
    method: str,
):
    import torch

    def forward(self, x):
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        def grad_hook(grad):
            values = hidden.detach() * grad.detach()
            if method in {"boundary_taylor_drop", "boundary_taylor_weighted"}:
                values = values.clamp_min(0.0)
            elif method in {"boundary_taylor_abs", "general_taylor"}:
                values = values.abs()
            else:
                raise ValueError(f"Unsupported Taylor method: {method}")

            mask = active_mask.get("attention_mask")
            if mask is not None and values.ndim == 3:
                weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
                reduced = (values * weights).sum(dim=(0, 1), dtype=torch.float32)
            else:
                reduced = values.reshape(-1, values.shape[-1]).sum(dim=0, dtype=torch.float32)
            sums[module_name] += reduced.cpu()
            return grad

        if hidden.requires_grad:
            hidden.register_hook(grad_hook)
        return self.down_proj(hidden)

    return forward


def taylor_scores(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    groups: list[CoupledFFNUnitGroup],
    *,
    method: str,
    dense_margin_by_id: dict[str, float] | None,
    batch_size: int,
    max_length: int | None = None,
    tau_mode: str = "q25",
    tau_value: float | None = None,
    margin_eps: float = 1e-6,
    use_chat_template: bool = True,
) -> tuple[list[UnitScore], dict[str, Any]]:
    import torch

    if method not in {"boundary_taylor_drop", "boundary_taylor_weighted", "boundary_taylor_abs", "general_taylor"}:
        raise ValueError(f"Unsupported Taylor method: {method}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not records:
        raise ValueError("No calibration records available for Taylor scoring")

    if method == "general_taylor":
        selected = list(records)
        method_info: dict[str, Any] = {
            "tau_mode": "all",
            "tau_calib": None,
            "num_calibration_pairs": len(records),
            "num_selected_pairs": len(records),
            "selected_fraction": 1.0,
            "selected_ids": [str(record["id"]) for record in records],
            "weight_min": 1.0,
            "weight_max": 1.0,
            "weight_mean": 1.0,
            "weight_by_id": {str(record["id"]): 1.0 for record in records},
        }
    else:
        if dense_margin_by_id is None:
            raise ValueError("dense_margin_by_id is required for boundary Taylor scoring")
        selected, method_info = select_boundary_records(
            records,
            dense_margin_by_id,
            tau_mode=tau_mode,
            tau_value=tau_value,
            margin_eps=margin_eps,
        )

    sums = {group.module_name: torch.zeros(group.intermediate_size, dtype=torch.float32) for group in groups}
    active_mask: dict[str, Any] = {}
    original_forwards: dict[str, Any] = {}
    for group in groups:
        module = get_module_by_name(model, group.module_name)
        if not all(hasattr(module, attr) for attr in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            raise ValueError(f"Module {group.module_name} is not a supported SwiGLU MLP")
        original_forwards[group.module_name] = module.forward
        module.forward = MethodType(
            _make_taylor_forward(group.module_name, sums, active_mask, method=method),
            module,
        )

    device = infer_model_device(model)
    try:
        model.eval()
        for batch in tqdm(batched(selected, batch_size), desc=f"{method} scoring"):
            for record in batch:
                formatted_prompt = format_prompt(
                    str(record["prompt"]),
                    tokenizer,
                    use_chat_template=use_chat_template,
                    add_generation_prompt=True,
                )
                weight = float(method_info["weight_by_id"][str(record["id"])])
                if method == "boundary_taylor_weighted":
                    chosen_sign = weight
                    rejected_sign = -weight
                elif method in {"boundary_taylor_drop", "boundary_taylor_abs"}:
                    chosen_sign = 1.0
                    rejected_sign = -1.0
                else:
                    chosen_sign = 1.0
                    rejected_sign = 1.0

                # Keep Taylor response micro-batch at 1. Chosen and rejected share a
                # calibration pair, but evaluating them together doubles the training
                # graph peak on long HH-RLHF examples.
                for response, sign in (
                    (str(record["chosen"]), chosen_sign),
                    (str(record["rejected"]), rejected_sign),
                ):
                    model.zero_grad(set_to_none=True)
                    length_normalized, attention_mask = differentiable_response_logprobs_batch(
                        model,
                        tokenizer,
                        [(formatted_prompt, response)],
                        device=device,
                        max_length=max_length,
                    )
                    active_mask["attention_mask"] = attention_mask
                    objective = length_normalized[0] * float(sign)
                    objective.backward()
                    active_mask.clear()
    finally:
        for group in groups:
            module = get_module_by_name(model, group.module_name)
            module.forward = original_forwards[group.module_name]

    group_scores = {
        group.module_name: (sums[group.module_name] / max(1, len(selected))).tolist()
        for group in groups
    }
    clean_info = {key: value for key, value in method_info.items() if key != "weight_by_id"}
    clean_info.update(
        {
            "taylor_objective": "delta_margin" if method != "general_taylor" else "logprob_magnitude",
            "score_transform": method,
            "batch_size": batch_size,
            "response_micro_batch_size": 1,
            "max_length": max_length,
        }
    )
    return flatten_scores(group_scores, groups), clean_info


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
