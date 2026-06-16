# Experiment Notes

These notes are separate from the implementation docs so remote experiment setup can evolve without changing code.

## Local Machine

The local machine is used only for Python/conda testing:

```bash
cd preference-boundary-pruning
pip install -e .
python -m pytest
```

Do not run Qwen model inference locally unless the environment has a suitable GPU and enough memory.

## Remote Machine

Expected remote hardware:

- GPU: 2 x RTX 4090
- RAM: 200 GB
- System disk: 30 GB
- Data disk: 50 GB

Because the system disk is small, put caches, processed data, outputs, and downloaded models on the data disk.

Example Linux environment variables:

```bash
export PBP_ROOT=/data/preference-boundary-pruning
export HF_HOME=/data/hf_cache
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/transformers
export TORCH_HOME=/data/torch_cache
export TOKENIZERS_PARALLELISM=false
```

Recommended first remote smoke test:

```bash
python scripts/prepare_hh_rlhf.py \
  --dataset Anthropic/hh-rlhf \
  --eval-size 100 \
  --seed 42 \
  --out-dir data/processed

python scripts/compute_base_logprobs.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --chat-template-model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/hh_rlhf_eval.jsonl \
  --out outputs/logprobs/base_qwen2p5_1p5b_eval_smoke.jsonl \
  --dtype bfloat16 \
  --batch-size 2

python scripts/compute_dense_margins.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-logprobs outputs/logprobs/base_qwen2p5_1p5b_eval_smoke.jsonl \
  --data data/processed/hh_rlhf_eval.jsonl \
  --out outputs/margins/dense_qwen2p5_1p5b_eval_smoke.jsonl \
  --summary-out outputs/evals/coverage_dense_qwen2p5_1p5b_eval_smoke.json \
  --histogram-out outputs/evals/dense_margin_histogram_qwen2p5_1p5b_eval_smoke.csv \
  --dtype bfloat16 \
  --batch-size 2
```

After the smoke test, run the 1k Milestone 1 evaluation from `README.md`.

## Milestone Boundary

Milestone 1 stops after dense margins and coverage reporting. Do not run pruning, Taylor scoring, post-pruning recovery, DPO, or LoRA in this milestone.
