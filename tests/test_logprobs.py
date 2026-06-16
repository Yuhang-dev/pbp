from __future__ import annotations

from types import SimpleNamespace

import pytest

from pbp.logprobs import build_response_token_mask, compute_response_logprob


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return {"input_ids": [ord(ch) for ch in text]}


class PositionAwareToyModel:
    def __init__(self, high_ids: set[int], vocab_size: int = 256):
        self.high_ids = high_ids
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def parameters(self):
        import torch

        yield torch.empty((), dtype=torch.float32)

    def __call__(self, input_ids, attention_mask=None):
        import torch

        batch, seq_len = input_ids.shape
        logits = torch.zeros((batch, seq_len, self.vocab_size), device=input_ids.device)
        for row in range(batch):
            for pos in range(seq_len - 1):
                target = int(input_ids[row, pos + 1].item())
                if target in self.high_ids:
                    logits[row, pos, :] = -100.0
                    logits[row, pos, target] = 100.0
                else:
                    logits[row, pos, target] = -100.0
        return SimpleNamespace(logits=logits)


def test_build_response_token_mask_excludes_prompt_tokens():
    tokenizer = CharTokenizer()

    input_ids, mask = build_response_token_mask(tokenizer, "ab", "XY")

    assert input_ids == [ord("a"), ord("b"), ord("X"), ord("Y")]
    assert mask == [0, 0, 1, 1]


def test_response_logprob_ignores_prompt_targets_after_causal_shift():
    pytest.importorskip("torch")
    tokenizer = CharTokenizer()
    model = PositionAwareToyModel(high_ids={ord("X"), ord("Y")})

    result = compute_response_logprob(model, tokenizer, "ab", "XY")

    assert result.num_response_tokens == 2
    assert result.sum_logprob > -1e-4
    assert result.length_normalized_logprob > -1e-4


def test_response_mask_rejects_empty_response():
    tokenizer = CharTokenizer()

    with pytest.raises(ValueError, match="response must be non-empty"):
        build_response_token_mask(tokenizer, "prompt", "")
