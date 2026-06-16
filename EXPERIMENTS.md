# Experiment Notes

These notes are separate from the implementation docs so remote experiment setup can evolve without changing code.

## Local Machine

The local machine is used only for syntax/static sanity checks and command editing. Do not use local runs as milestone validation.

```bash
cd preference-boundary-pruning
python -m compileall src scripts tests
```

Do not run `pytest`, fixture smoke tests, dry-run validation scripts, Qwen model inference, pruning, or evaluation locally. Functional validation belongs on the remote machine.

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

## M5 Remote Smoke

Run this on the remote machine only, after pulling the latest commit:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
export OMP_NUM_THREADS=1

python scripts/apply_mask_pruning.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method random \
  --ratio 0.10 \
  --out outputs/pruned_models/qwen2p5_1p5b_random_mask_10p \
  --dtype bfloat16 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m5_random_mask_10p_smoke \
  --smoke-generate \
  --max-new-tokens 16
```

Check the generated artifacts:

```bash
cat outputs/pruned_models/qwen2p5_1p5b_random_mask_10p/mask_config.json
cat outputs/runs/*_m5_random_mask_10p_smoke/metrics.json
cat outputs/runs/*_m5_random_mask_10p_smoke/status.json
```

Expected smoke criteria:

- `status.json` has `"status": "success"`.
- `metrics.json` has `"requested_ratio": 0.1`, `"actual_ratio": 0.1`, and `"generation_success": true`.
- `mask_config.json` reports the Qwen MLP groups and no shape errors occurred.

## Milestone Boundary

Current M5 work stops after mask-based random pruning support and remote smoke verification. Do not run M6 scoring baselines, Taylor scoring, post-pruning recovery, DPO, or LoRA until explicitly approved.
