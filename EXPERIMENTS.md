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

- GPU: 1 x RTX PRO 6000 96GB
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

Recommended dependency upgrade for RTX PRO 6000 96GB:

```bash
conda activate pbp
python -m pip install -U pip setuptools wheel
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
python -m pip install -U "transformers>=5.0" "accelerate>=1.14" "datasets>=5.0" "huggingface_hub[cli]>=1.0" numpy pyyaml tqdm matplotlib

python - <<'PY'
import torch, transformers, accelerate, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("capability", f"{props.major}.{props.minor}", "memory_gb", round(props.total_memory / 1024**3, 1))
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
PY
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
set -euo pipefail
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

## M8 Remote Smoke

Run this on the remote machine only, after pulling the latest commit:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
export OMP_NUM_THREADS=1

python scripts/score_pruning_importance.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-model Qwen/Qwen2.5-1.5B \
  --data data/processed/hh_rlhf_calib.jsonl \
  --method boundary_taylor_weighted \
  --max-samples 20 \
  --tau-mode q25 \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_smoke.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m8_boundary_taylor_smoke
```

Compare the selected pruning mask against M6 activation pruning:

```bash
cat outputs/runs/*_m8_boundary_taylor_smoke/metrics.json
cat outputs/runs/*_m8_boundary_taylor_smoke/status.json

python - <<'PY'
import json

boundary_path = "outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_smoke.json"
activation_path = "outputs/scores/qwen2p5_1p5b_activation_smoke.json"

def load(path):
    payload = json.load(open(path, encoding="utf-8"))
    pruned = set()
    for module_name, mask in payload["masks_by_module"].items():
        pruned.update((module_name, idx) for idx, value in enumerate(mask) if value == 0)
    return payload, pruned

boundary, boundary_pruned = load(boundary_path)
activation, activation_pruned = load(activation_path)
stats = boundary["score_stats"]
print("boundary", len(boundary_pruned), stats["num_scores"], stats["scores_finite"], stats["all_scores_zero"])
print("activation", len(activation_pruned), activation["score_stats"]["num_scores"], activation["score_stats"]["scores_finite"])
assert stats["scores_finite"]
assert not stats["all_scores_zero"]
assert boundary_pruned != activation_pruned
PY
```

Expected smoke criteria:

- `status.json` has `"status": "success"`.
- `metrics.json` reports `scores_finite=true`, `all_scores_zero=false`, `num_scores=250880`, and `actual_ratio=0.1`.
- `method_info` reports `tau_mode=q25` and `num_selected_pairs > 0`.
- The mask comparison script succeeds, confirming boundary Taylor selects a different pruning mask than activation pruning.

## M9 Remote Pilot Table

Run this on the remote machine only, after pulling the latest commit. This produces the first 1k-pair pilot table for Qwen2.5-1.5B with methods `random`, `magnitude`, `activation`, and `boundary_taylor_weighted` at ratios `0.10` and `0.20`.

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
BASE_MODEL="Qwen/Qwen2.5-1.5B"
DATA_DIR="data/processed/m9_pilot"

python scripts/prepare_hh_rlhf.py \
  --dataset Anthropic/hh-rlhf \
  --calib-size 1000 \
  --eval-size 1000 \
  --seed 42 \
  --out-dir "$DATA_DIR" \
  --run-name m9_prepare_hh_rlhf_1k

python scripts/compute_base_logprobs.py \
  --base-model "$BASE_MODEL" \
  --chat-template-model "$INSTRUCT_MODEL" \
  --data "$DATA_DIR/hh_rlhf_eval.jsonl" \
  --max-samples 1000 \
  --out outputs/logprobs/base_qwen2p5_1p5b_m9_eval_1k.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m9_base_logprobs_eval_1k

python scripts/compute_dense_margins.py \
  --instruct-model "$INSTRUCT_MODEL" \
  --base-logprobs outputs/logprobs/base_qwen2p5_1p5b_m9_eval_1k.jsonl \
  --data "$DATA_DIR/hh_rlhf_eval.jsonl" \
  --max-samples 1000 \
  --out outputs/margins/dense_qwen2p5_1p5b_m9_eval_1k.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m9_dense_margins_eval_1k

python scripts/compute_dense_margins.py \
  --instruct-model "$INSTRUCT_MODEL" \
  --base-model "$BASE_MODEL" \
  --data "$DATA_DIR/hh_rlhf_calib.jsonl" \
  --max-samples 1000 \
  --out outputs/margins/dense_qwen2p5_1p5b_m9_calib_1k.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m9_dense_margins_calib_1k

python scripts/score_pruning_importance.py \
  --model "$INSTRUCT_MODEL" \
  --method random \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_random_m9_calib_1k.json \
  --dtype bfloat16 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m9_score_random_1k

python scripts/score_pruning_importance.py \
  --model "$INSTRUCT_MODEL" \
  --method magnitude \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_magnitude_m9_calib_1k.json \
  --dtype bfloat16 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m9_score_magnitude_1k

python scripts/score_pruning_importance.py \
  --model "$INSTRUCT_MODEL" \
  --data "$DATA_DIR/hh_rlhf_calib.jsonl" \
  --method activation \
  --max-samples 1000 \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_activation_m9_calib_1k.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 1024 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m9_score_activation_1k

python scripts/score_pruning_importance.py \
  --instruct-model "$INSTRUCT_MODEL" \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_m9_calib_1k.jsonl \
  --data "$DATA_DIR/hh_rlhf_calib.jsonl" \
  --method boundary_taylor_weighted \
  --max-samples 1000 \
  --tau-mode q25 \
  --ratio 0.10 \
  --out outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_m9_calib_1k.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 2048 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m9_score_boundary_taylor_weighted_1k

METHODS=(random magnitude activation boundary_taylor_weighted)
RATIOS=("0.10:10p" "0.20:20p")

for pair in "${RATIOS[@]}"; do
  ratio="${pair%%:*}"
  label="${pair##*:}"
  for method in "${METHODS[@]}"; do
    python scripts/apply_mask_pruning.py \
      --scores "outputs/scores/qwen2p5_1p5b_${method}_m9_calib_1k.json" \
      --ratio "$ratio" \
      --out "outputs/pruned_models/qwen2p5_1p5b_${method}_mask_${label}_m9" \
      --run-name "m9_apply_${method}_${label}"
  done
done

for pair in "${RATIOS[@]}"; do
  label="${pair##*:}"
  for method in "${METHODS[@]}"; do
    python scripts/evaluate_bcr.py \
      --model "outputs/pruned_models/qwen2p5_1p5b_${method}_mask_${label}_m9" \
      --base-logprobs outputs/logprobs/base_qwen2p5_1p5b_m9_eval_1k.jsonl \
      --dense-margins outputs/margins/dense_qwen2p5_1p5b_m9_eval_1k.jsonl \
      --data "$DATA_DIR/hh_rlhf_eval.jsonl" \
      --max-samples 1000 \
      --out "outputs/evals/bcr_qwen2p5_1p5b_${method}_${label}_m9_1k.json" \
      --records-out "outputs/evals/bcr_qwen2p5_1p5b_${method}_${label}_m9_1k_records.jsonl" \
      --dtype bfloat16 \
      --batch-size 1 \
      --cache-dir "$HF_HUB_CACHE" \
      --local-files-only \
      --run-name "m9_bcr_${method}_${label}_1k"
  done
done

python scripts/summarize_results.py \
  --inputs \
    outputs/evals/bcr_qwen2p5_1p5b_random_10p_m9_1k.json \
    outputs/evals/bcr_qwen2p5_1p5b_magnitude_10p_m9_1k.json \
    outputs/evals/bcr_qwen2p5_1p5b_activation_10p_m9_1k.json \
    outputs/evals/bcr_qwen2p5_1p5b_boundary_taylor_weighted_10p_m9_1k.json \
    outputs/evals/bcr_qwen2p5_1p5b_random_20p_m9_1k.json \
    outputs/evals/bcr_qwen2p5_1p5b_magnitude_20p_m9_1k.json \
    outputs/evals/bcr_qwen2p5_1p5b_activation_20p_m9_1k.json \
    outputs/evals/bcr_qwen2p5_1p5b_boundary_taylor_weighted_20p_m9_1k.json \
  --out outputs/tables/m9_qwen2p5_1p5b_pilot_1k.csv \
  --summary-out outputs/tables/m9_qwen2p5_1p5b_pilot_1k.json \
  --run-name m9_summarize_pilot_1k
```

Check the table and run statuses:

```bash
cat outputs/tables/m9_qwen2p5_1p5b_pilot_1k.csv
cat outputs/runs/*_m9_summarize_pilot_1k/status.json
cat outputs/runs/*_m9_summarize_pilot_1k/metrics.json
```

Expected pilot criteria:

- The CSV has 8 rows: 4 methods x 2 ratios.
- Every BCR evaluation run has `"status": "success"`.
- The table contains the required columns: `model`, `method`, `ratio`, `coverage@0`, `coverage@q25`, `bcr@0`, `bcr@q25`, `pref_acc`, and `mean_margin_drop`.
- Any failed remote run must be recorded in `KNOWN_ISSUES.md` before marking M9 passed.

If rerunning after a partial M9 run, do not paste the whole block unless you first remove or rename the existing M9 outputs. The scripts intentionally refuse to overwrite existing output files and directories.

## Milestone Boundary

M9 has passed after the remote Qwen2.5-1.5B 1k pilot table completed with 8 rows. Do not run post-pruning recovery, DPO, LoRA, 3B/7B scaling, general utility evaluation, or M10 work until explicitly approved.
