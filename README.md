# Preference Boundary Pruning

Milestone 1 implementation for preference-boundary diagnostics before any pruning work.

This repository currently supports:

- HH-RLHF preprocessing into `prompt`, `chosen`, `rejected` JSONL records.
- Response-token-only, length-normalized log-probability computation.
- Base-reference-normalized dense margin computation.
- Coverage and preference-accuracy reporting.
- Margin histogram export.
- Mask-based coupled SwiGLU FFN pruning artifacts.
- Random, magnitude, activation, and boundary-aware Taylor scoring.
- BCR evaluation for dense or masked pruned models.
- Pilot result-table summarization.
- Lightweight general-utility evaluation for M10A matched-utility checks.
- M10A matched-utility table summarization.
- M10B stale run-status cleanup, layer-wise mask distribution reporting, and all-ratio matched-utility summarization.
- M11A layer-wise/protected layer-wise pruning selection and utility/BCR summarization.
- M12 hybrid utility-boundary score composition and alpha-sweep summarization.
- Unit tests for response masking and margin/coverage computation.

Physical dimension-changing pruning is not implemented yet. The current MVP uses masked structured pruning.

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

Run local syntax/static checks only:

```bash
python -m compileall src scripts tests
```

Run functional tests, smoke runs, model loading, pruning, and evaluation on the remote machine. See `EXPERIMENTS.md` for remote execution notes and storage layout.

## Milestone 9 Pilot Table

M9 is remote-only. The command block is in `EXPERIMENTS.md` under `M9 Remote Pilot Table`, with matrix:

```text
methods = random, magnitude, activation, boundary_taylor_weighted
ratios = 0.10, 0.20
eval_samples = 1000 HH-RLHF pairs
```

The expected table output is:

```text
outputs/tables/m9_qwen2p5_1p5b_pilot_1k.csv
```

## M10A Matched Utility

M10A is remote-only and limited to dense Qwen2.5-1.5B-Instruct plus the 20% M9 masked pruned models. The command block is in `EXPERIMENTS.md` under `M10A Matched Utility 20% Check`.

Expected outputs:

```text
outputs/evals/general_m10a_*.json
outputs/tables/m10a_matched_utility_20p.csv
```

M10A completed on the remote `1 x NVIDIA RTX PRO 6000 96GB` setup. At 20% pruning, `boundary_taylor_weighted` reduced `BCR@q25` versus activation, but all 20% pruned models failed the configured matched-utility thresholds.

## M10B Larger Matched Utility

M10B is remote-only and limited to dense Qwen2.5-1.5B-Instruct plus all M9 10% and 20% masked pruned models. The command block is in `EXPERIMENTS.md` under `M10B Larger Matched Utility and Mask Distribution`.

## M12 Hybrid Utility-Boundary Pruning

M12 is remote-only and keeps Qwen2.5-1.5B-Instruct as the only model. Hybrid score artifacts are composed from existing utility and boundary score artifacts:

```text
I_hybrid(g) = rank_norm(I_utility(g)) + alpha * rank_norm(I_boundary(g))
```

Expected summary outputs:

```text
outputs/tables/m12_alpha_sweep.csv
outputs/tables/m12_hybrid_summary.json
```

Expected outputs:

```text
outputs/tables/m10b_stale_status_cleanup.json
outputs/tables/m10b_mask_distribution.csv
outputs/tables/m10b_matched_utility_all.csv
outputs/tables/m10b_matched_utility_summary.json
```

M10B completed remotely as a larger smoke/checkpoint run. No 10% or 20% M9 pruned model met the configured matched-utility thresholds, so matched utility is not established under current masking.

## M11A Layer-Wise Pruning

M11A is remote-only and limited to Qwen2.5-1.5B-Instruct. It introduces `--selection-scope layerwise` plus optional protected layers to find a utility-preserving pruning regime before comparing BCR.

Expected outputs:

```text
outputs/tables/m11a_layerwise_utility_bcr.csv
outputs/tables/m11a_mask_distribution.csv
outputs/tables/m11a_summary.json
```
