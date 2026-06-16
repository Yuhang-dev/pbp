from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from pbp.logprobs import build_response_token_mask, compute_response_logprob


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return {"input_ids": [ord(ch) for ch in text]}


class OffsetTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        assert add_special_tokens is False
        if return_offsets_mapping:
            return {
                "input_ids": [10, 20],
                "offset_mapping": [(0, 2), (2, len(text))],
            }
        return {"input_ids": [10, 20]}


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


def test_build_response_token_mask_prefers_offsets_for_boundary_alignment():
    tokenizer = OffsetTokenizer()

    input_ids, mask = build_response_token_mask(tokenizer, "ab", "cd")

    assert input_ids == [10, 20]
    assert mask == [0, 1]


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


def test_compute_logprobs_cli_dry_run_fixture(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    data = repo_root / "tests" / "fixtures" / "hh_rlhf_processed_fixture.jsonl"
    out = tmp_path / "logprobs.jsonl"
    runs_dir = tmp_path / "runs"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "compute_logprobs.py"),
            "--model",
            "dry-run-model",
            "--data",
            str(data),
            "--out",
            str(out),
            "--runs-dir",
            str(runs_dir),
            "--run-name",
            "m2_logprob_dry_run_test",
            "--max-samples",
            "1",
            "--seed",
            "42",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["metrics"]["num_examples"] == 1
    assert payload["metrics"]["num_scored_responses"] == 2
    assert payload["metrics"]["length_normalized_logprobs_finite"] is True
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["dry_run"] is True
    assert records[0]["chosen"]["num_response_tokens"] > 0
    assert records[0]["rejected"]["num_response_tokens"] > 0
    run_dirs = list(runs_dir.glob("*_m2_logprob_dry_run_test"))
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "metrics.json").is_file()
    assert (run_dirs[0] / "status.json").is_file()
