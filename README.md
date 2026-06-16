# Preference Boundary Pruning

Milestone 1 implementation for preference-boundary diagnostics before any pruning work.

This repository currently supports:

- HH-RLHF preprocessing into `prompt`, `chosen`, `rejected` JSONL records.
- Response-token-only, length-normalized log-probability computation.
- Base-reference-normalized dense margin computation.
- Coverage and preference-accuracy reporting.
- Margin histogram export.
- Unit tests for response masking and margin/coverage computation.

Pruning is intentionally not implemented yet. It starts in Milestone 2.

## Setup

```bash
cd preference-boundary-pruning
pip install -e ".[test]"
```

For model execution, install the CUDA-compatible `torch` build appropriate for the remote machine.

## Milestone 1 Commands

Prepare 1k HH-RLHF evaluation pairs:

```bash
python scripts/prepare_hh_rlhf.py \
  --dataset Anthropic/hh-rlhf \
  --eval-size 1000 \
  --seed 42 \
  --out-dir data/processed
```

Compute base-model response log-probs. Pass the instruct tokenizer as the chat-template source so the base and instruct models see the same formatted prompt string.

```bash
python scripts/compute_base_logprobs.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --chat-template-model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/hh_rlhf_eval.jsonl \
  --out outputs/logprobs/base_qwen2p5_1p5b_eval.jsonl \
  --dtype bfloat16 \
  --batch-size 4
```

Compute dense instruct-model margins:

```bash
python scripts/compute_dense_margins.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-logprobs outputs/logprobs/base_qwen2p5_1p5b_eval.jsonl \
  --data data/processed/hh_rlhf_eval.jsonl \
  --out outputs/margins/dense_qwen2p5_1p5b_eval.jsonl \
  --summary-out outputs/evals/coverage_dense_qwen2p5_1p5b_eval.json \
  --histogram-out outputs/evals/dense_margin_histogram_qwen2p5_1p5b_eval.csv \
  --dtype bfloat16 \
  --batch-size 4
```

Recompute only the coverage report from saved margins:

```bash
python scripts/report_coverage.py \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_eval.jsonl \
  --summary-out outputs/evals/coverage_dense_qwen2p5_1p5b_eval.json \
  --histogram-out outputs/evals/dense_margin_histogram_qwen2p5_1p5b_eval.csv
```

Run local tests:

```bash
python -m pytest
```

See `EXPERIMENTS.md` for remote execution notes and storage layout.
