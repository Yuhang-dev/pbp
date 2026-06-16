from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from pbp.utils import infer_model_device


@dataclass(frozen=True)
class ResponseLogProb:
    sum_logprob: float
    num_response_tokens: int
    length_normalized_logprob: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _extract_input_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    return list(encoded.input_ids)


def encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = _extract_input_ids(encoded)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _try_encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except Exception:
        return None

    input_ids = _extract_input_ids(encoded)
    offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    if offsets is None or len(offsets) != len(input_ids):
        return None
    return [int(x) for x in input_ids], [(int(start), int(end)) for start, end in offsets]


def build_response_token_mask(
    tokenizer: Any,
    formatted_prompt: str,
    response: str,
) -> tuple[list[int], list[int]]:
    """Return full input ids and a token-position mask where only response tokens are 1."""
    if not response:
        raise ValueError("response must be non-empty")
    full_text = formatted_prompt + response

    encoded_with_offsets = _try_encode_with_offsets(tokenizer, full_text)
    if encoded_with_offsets is not None:
        full_ids, offsets = encoded_with_offsets
        prompt_chars = len(formatted_prompt)
        mask = [1 if end > prompt_chars else 0 for start, end in offsets]
        if any(mask):
            return full_ids, mask

    prompt_ids = encode_without_special_tokens(tokenizer, formatted_prompt)
    full_ids = encode_without_special_tokens(tokenizer, full_text)
    prompt_len = len(prompt_ids)
    if prompt_len >= len(full_ids):
        raise ValueError(
            "response produced no additional tokens; check prompt/response formatting"
        )
    mask = [0] * len(full_ids)
    for idx in range(prompt_len, len(full_ids)):
        mask[idx] = 1
    return full_ids, mask


def response_logprob_token_count(response_mask: list[int]) -> int:
    """Count response labels after the causal next-token shift."""
    return int(sum(response_mask[1:]))


def _pad_batch(
    sequences: list[list[int]],
    masks: list[list[int]],
    pad_token_id: int,
    device,
):
    import torch

    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    response_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, (seq, mask) in enumerate(zip(sequences, masks, strict=True)):
        length = len(seq)
        input_ids[row, :length] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, :length] = 1
        response_mask[row, :length] = torch.tensor(mask, dtype=torch.bool)
    return input_ids.to(device), attention_mask.to(device), response_mask.to(device)


def _pad_token_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is not None:
        return int(pad_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return int(eos_id)
    return 0


def compute_response_logprobs_batch(
    model: Any,
    tokenizer: Any,
    prompt_response_pairs: Iterable[tuple[str, str]],
    *,
    device=None,
) -> list[ResponseLogProb]:
    import torch

    pairs = list(prompt_response_pairs)
    if not pairs:
        return []

    sequences: list[list[int]] = []
    masks: list[list[int]] = []
    for formatted_prompt, response in pairs:
        input_ids, response_mask = build_response_token_mask(tokenizer, formatted_prompt, response)
        sequences.append(input_ids)
        masks.append(response_mask)

    if device is None:
        device = infer_model_device(model)

    input_ids, attention_mask, response_mask = _pad_batch(
        sequences,
        masks,
        _pad_token_id(tokenizer),
        device,
    )

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if logits.shape[:2] != input_ids.shape:
            raise ValueError("model logits must have shape [batch, sequence, vocab]")

        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_response_mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
        token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

    results: list[ResponseLogProb] = []
    for row in range(input_ids.shape[0]):
        row_mask = shift_response_mask[row]
        count = int(row_mask.sum().item())
        if count <= 0:
            raise ValueError("No response tokens available after causal shift")
        sum_logprob = float(token_logprobs[row][row_mask].sum().item())
        results.append(
            ResponseLogProb(
                sum_logprob=sum_logprob,
                num_response_tokens=count,
                length_normalized_logprob=sum_logprob / count,
            )
        )
    return results


def compute_response_logprob(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    response: str,
    *,
    device=None,
) -> ResponseLogProb:
    return compute_response_logprobs_batch(
        model,
        tokenizer,
        [(formatted_prompt, response)],
        device=device,
    )[0]
