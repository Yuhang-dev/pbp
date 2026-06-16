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

## M6 Remote Smoke

Run these on the remote machine only, after pulling the latest commit:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
export OMP_NUM_THREADS=1

python scripts/score_pruning_importance.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method random \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_random_smoke.json \
  --dtype bfloat16 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m6_random_score_smoke

python scripts/score_pruning_importance.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method magnitude \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_magnitude_smoke.json \
  --dtype bfloat16 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m6_magnitude_score_smoke

python scripts/score_pruning_importance.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/hh_rlhf_calib.jsonl \
  --method activation \
  --max-samples 50 \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_activation_smoke.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 1024 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m6_activation_score_smoke
```

Check metrics and compare selected masks:

```bash
cat outputs/runs/*_m6_random_score_smoke/metrics.json
cat outputs/runs/*_m6_magnitude_score_smoke/metrics.json
cat outputs/runs/*_m6_activation_score_smoke/metrics.json
cat outputs/runs/*_m6_random_score_smoke/status.json
cat outputs/runs/*_m6_magnitude_score_smoke/status.json
cat outputs/runs/*_m6_activation_score_smoke/status.json

python - <<'PY'
import json

paths = {
    "random": "outputs/scores/qwen2p5_1p5b_random_smoke.json",
    "magnitude": "outputs/scores/qwen2p5_1p5b_magnitude_smoke.json",
    "activation": "outputs/scores/qwen2p5_1p5b_activation_smoke.json",
}

def pruned_units(path):
    payload = json.load(open(path, encoding="utf-8"))
    out = set()
    for module_name, mask in payload["masks_by_module"].items():
        out.update((module_name, idx) for idx, value in enumerate(mask) if value == 0)
    return out, payload["score_stats"]

sets = {}
for name, path in paths.items():
    units, stats = pruned_units(path)
    print(name, len(units), stats["num_scores"], stats["scores_finite"])
    sets[name] = units

assert all(pruned for pruned in sets.values())
assert all(json.load(open(path, encoding="utf-8"))["score_stats"]["scores_finite"] for path in paths.values())
assert sets["random"] != sets["magnitude"]
assert sets["random"] != sets["activation"]
assert sets["magnitude"] != sets["activation"]
PY
```

Expected smoke criteria:

- All three `status.json` files have `"status": "success"`.
- All three `metrics.json` files report `scores_finite=true`, `total_units=250880`, and `actual_ratio=0.1`.
- The mask comparison script succeeds, confirming selected pruning masks differ across methods.

## M7 Remote Smoke

Run these on the remote machine only, after pulling the latest commit:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
export OMP_NUM_THREADS=1

python scripts/evaluate_bcr.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --base-model Qwen/Qwen2.5-1.5B \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl \
  --data data/processed/hh_rlhf_eval.jsonl \
  --max-samples 20 \
  --out outputs/evals/bcr_dense_self_smoke.json \
  --records-out outputs/evals/bcr_dense_self_smoke_records.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m7_bcr_dense_self_smoke

python scripts/evaluate_bcr.py \
  --model outputs/pruned_models/qwen2p5_1p5b_random_mask_10p \
  --base-model Qwen/Qwen2.5-1.5B \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl \
  --data data/processed/hh_rlhf_eval.jsonl \
  --max-samples 20 \
  --out outputs/evals/bcr_random_10p_smoke.json \
  --records-out outputs/evals/bcr_random_10p_smoke_records.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m7_bcr_random_10p_smoke
```

Check metrics:

```bash
cat outputs/runs/*_m7_bcr_dense_self_smoke/metrics.json
cat outputs/runs/*_m7_bcr_dense_self_smoke/status.json
cat outputs/runs/*_m7_bcr_random_10p_smoke/metrics.json
cat outputs/runs/*_m7_bcr_random_10p_smoke/status.json
cat outputs/evals/bcr_random_10p_smoke.json
```

Expected smoke criteria:

- Both `status.json` files have `"status": "success"`.
- Dense self run has `bcr_at_0=0`, `bcr_at_q25=0`, `bcr_at_q50=0`, and `bcr_at_q75=0`.
- Random 10% run reports finite numeric metrics, `mask_actual_ratio=0.1`, and non-null `mean_margin_drop`.
- Coverage values are based on dense margins and therefore match the M4 20-example smoke coverage.

## Milestone Boundary

Current M7 work stops after BCR evaluation support plus remote smoke verification. Do not run M8 Taylor scoring, post-pruning recovery, DPO, or LoRA until explicitly approved.
