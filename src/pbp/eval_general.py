from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pbp.logprobs import compute_response_logprobs_batch
from pbp.utils import batched, infer_model_device


@dataclass(frozen=True)
class MultipleChoiceItem:
    id: str
    prompt: str
    choices: list[str]
    answer_index: int


def _dataset_kwargs(cache_dir: str | None, local_files_only: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if local_files_only:
        kwargs["download_mode"] = "reuse_dataset_if_exists"
    return kwargs


def _clean_text(text: Any) -> str:
    return " ".join(str(text).strip().split())


def load_wikitext_texts(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    max_samples: int,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    min_words: int = 16,
) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        **_dataset_kwargs(cache_dir, local_files_only),
    )
    texts: list[str] = []
    for record in dataset:
        text = _clean_text(record.get("text", ""))
        if len(text.split()) >= min_words:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError(f"No usable perplexity texts found in {dataset_name}/{dataset_config}:{split}")
    return texts


def compute_lm_perplexity(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    import torch

    if not texts:
        raise ValueError("texts must be non-empty")
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = infer_model_device(model)
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for chunk in batched(texts, batch_size):
            encoded = tokenizer(
                chunk,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].bool()
            token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)
            total_nll += float((-token_logprobs[shift_mask]).sum().item())
            total_tokens += int(shift_mask.sum().item())

    if total_tokens <= 0:
        raise ValueError("No valid tokens for perplexity")
    mean_nll = total_nll / total_tokens
    return {
        "ppl": float(math.exp(mean_nll)),
        "mean_nll": float(mean_nll),
        "num_texts": len(texts),
        "num_tokens": total_tokens,
    }


def load_arc_challenge_items(
    *,
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_samples: int,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> list[MultipleChoiceItem]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        **_dataset_kwargs(cache_dir, local_files_only),
    )
    items: list[MultipleChoiceItem] = []
    for idx, record in enumerate(dataset):
        choices_payload = record.get("choices", {})
        labels = [str(label) for label in choices_payload.get("label", [])]
        texts = [_clean_text(text) for text in choices_payload.get("text", [])]
        answer_key = str(record.get("answerKey", "")).strip()
        if not labels or len(labels) != len(texts) or answer_key not in labels:
            continue
        question = _clean_text(record.get("question", ""))
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts, strict=True))
        prompt = f"Question: {question}\nChoices:\n{options}\nAnswer:"
        responses = [f" {label}" for label in labels]
        items.append(
            MultipleChoiceItem(
                id=str(record.get("id", f"arc_challenge_{idx}")),
                prompt=prompt,
                choices=responses,
                answer_index=labels.index(answer_key),
            )
        )
        if len(items) >= max_samples:
            break
    if not items:
        raise ValueError(f"No usable ARC-Challenge examples found in {dataset_name}/{dataset_config}:{split}")
    return items


def load_hellaswag_items(
    *,
    dataset_name: str,
    split: str,
    max_samples: int,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> list[MultipleChoiceItem]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        split=split,
        **_dataset_kwargs(cache_dir, local_files_only),
    )
    items: list[MultipleChoiceItem] = []
    for idx, record in enumerate(dataset):
        endings = [_clean_text(text) for text in record.get("endings", [])]
        try:
            answer_index = int(record.get("label"))
        except (TypeError, ValueError):
            continue
        if len(endings) != 4 or not (0 <= answer_index < len(endings)):
            continue
        prompt = _clean_text(record.get("ctx", ""))
        if not prompt:
            continue
        responses = [f" {ending}" if not ending.startswith(" ") else ending for ending in endings]
        items.append(
            MultipleChoiceItem(
                id=str(record.get("ind", f"hellaswag_{idx}")),
                prompt=prompt,
                choices=responses,
                answer_index=answer_index,
            )
        )
        if len(items) >= max_samples:
            break
    if not items:
        raise ValueError(f"No usable HellaSwag examples found in {dataset_name}:{split}")
    return items


def score_multiple_choice_accuracy(
    model: Any,
    tokenizer: Any,
    items: list[MultipleChoiceItem],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if not items:
        raise ValueError("items must be non-empty")
    device = infer_model_device(model)
    correct = 0
    records: list[dict[str, Any]] = []
    for item in items:
        pairs = [(item.prompt, choice) for choice in item.choices]
        scores: list[float] = []
        for chunk in batched(pairs, batch_size):
            scores.extend(
                result.length_normalized_logprob
                for result in compute_response_logprobs_batch(model, tokenizer, chunk, device=device)
            )
        prediction = max(range(len(scores)), key=scores.__getitem__)
        is_correct = prediction == item.answer_index
        correct += int(is_correct)
        records.append(
            {
                "id": item.id,
                "prediction": prediction,
                "answer_index": item.answer_index,
                "correct": is_correct,
                "scores": scores,
            }
        )
    return {
        "accuracy": correct / len(items),
        "num_examples": len(items),
        "num_correct": correct,
        "records": records,
    }


def finite_general_metrics(payload: dict[str, Any]) -> bool:
    keys = ("ppl", "arc_c", "hellaswag")
    return all(isinstance(payload.get(key), (int, float)) and math.isfinite(float(payload[key])) for key in keys)
