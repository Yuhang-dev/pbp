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
unset OMP_NUM_THREADS
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

## M10A Matched Utility 20% Check

M10A is remote-only. Run only dense Qwen2.5-1.5B-Instruct and the M9 20% masked pruned models. Do not run 10%, 3B/7B, DPO, LoRA, or post-pruning recovery.

This uses the repository lightweight evaluator rather than assuming `lm-evaluation-harness`, because the M9 pruned artifacts are masked models that need in-process mask injection. The metrics are subset checks for matched-utility diagnosis, not full benchmark claims.

Download/cache the M10A model and datasets first. Run this block separately before evaluation:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

hf download "$INSTRUCT_MODEL" \
  --cache-dir "$HF_HUB_CACHE"

python - <<'PY'
import os
from datasets import load_dataset

cache_dir = os.environ.get("HF_DATASETS_CACHE")
kwargs = {"cache_dir": cache_dir} if cache_dir else {}

jobs = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
    ("allenai/ai2_arc", "ARC-Challenge", "validation"),
    ("Rowan/hellaswag", None, "validation"),
]

for name, config, split in jobs:
    if config is None:
        dataset = load_dataset(name, split=split, **kwargs)
    else:
        dataset = load_dataset(name, config, split=split, **kwargs)
    print(name, config, split, len(dataset))
PY
```

After the cache step succeeds, run M10A evaluation offline from the cache:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1

INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

python scripts/evaluate_general.py \
  --model "$INSTRUCT_MODEL" \
  --method dense \
  --ratio 0.0 \
  --out outputs/evals/general_m10a_dense.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 2048 \
  --ppl-samples 64 \
  --arc-samples 100 \
  --hellaswag-samples 100 \
  --cache-dir "$HF_HUB_CACHE" \
  --dataset-cache-dir "$HF_DATASETS_CACHE" \
  --local-files-only \
  --datasets-local-files-only \
  --run-name m10a_general_dense

for method in activation boundary_taylor_weighted random magnitude; do
  python scripts/evaluate_general.py \
    --model "outputs/pruned_models/qwen2p5_1p5b_${method}_mask_20p_m9" \
    --out "outputs/evals/general_m10a_${method}_20p.json" \
    --dtype bfloat16 \
    --batch-size 1 \
    --max-length 2048 \
    --ppl-samples 64 \
    --arc-samples 100 \
    --hellaswag-samples 100 \
    --cache-dir "$HF_HUB_CACHE" \
    --dataset-cache-dir "$HF_DATASETS_CACHE" \
    --local-files-only \
    --datasets-local-files-only \
    --run-name "m10a_general_${method}_20p"
done

python scripts/summarize_m10a_matched_utility.py \
  --general-inputs \
    outputs/evals/general_m10a_dense.json \
    outputs/evals/general_m10a_activation_20p.json \
    outputs/evals/general_m10a_boundary_taylor_weighted_20p.json \
    outputs/evals/general_m10a_random_20p.json \
    outputs/evals/general_m10a_magnitude_20p.json \
  --bcr-table outputs/tables/m9_qwen2p5_1p5b_pilot_1k.csv \
  --out outputs/tables/m10a_matched_utility_20p.csv \
  --summary-out outputs/tables/m10a_matched_utility_20p.json \
  --run-name m10a_summarize_matched_utility_20p
```

Check the M10A outputs:

```bash
cat outputs/tables/m10a_matched_utility_20p.csv
cat outputs/tables/m10a_matched_utility_20p.json
cat outputs/runs/*_m10a_general_*/*status.json
cat outputs/runs/*_m10a_summarize_matched_utility_20p/status.json
cat outputs/runs/*_m10a_summarize_matched_utility_20p/metrics.json
```

Expected M10A criteria:

- Five general-utility JSON files exist under `outputs/evals/general_m10a_*.json`: dense plus activation, boundary_taylor_weighted, random, and magnitude at 20%.
- `outputs/tables/m10a_matched_utility_20p.csv` has the required columns: `model`, `method`, `ratio`, `ppl`, `arc_c`, `hellaswag`, `bcr@q25`, `bcr@0`, `pref_acc`, `mean_margin_drop`, `utility_delta_vs_dense`, and `matched_utility_flag`.
- Every model reports `loaded_successfully=true` and `general_utility_finite=true`.
- Stop after this table and inspect whether `boundary_taylor_weighted` has lower `bcr@q25` than activation under a similar utility flag.

## M10B Larger Matched Utility and Mask Distribution

M10B is remote-only. It expands the general-utility check to dense plus all eight M9 pruned models at 10% and 20%, and adds layer-wise mask distribution analysis. Do not run 3B/7B, DPO, LoRA, post-pruning recovery, or change the research question.

First cache the datasets needed for larger evaluation:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
unset HF_DATASETS_OFFLINE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import os
from datasets import load_dataset

cache_dir = os.environ.get("HF_DATASETS_CACHE")
kwargs = {"cache_dir": cache_dir} if cache_dir else {}

jobs = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
    ("allenai/ai2_arc", "ARC-Challenge", "validation"),
    ("Rowan/hellaswag", None, "validation"),
]

for name, config, split in jobs:
    if config is None:
        dataset = load_dataset(name, split=split, **kwargs)
    else:
        dataset = load_dataset(name, config, split=split, **kwargs)
    print(name, config, split, len(dataset))
PY
```

Clean stale M10A `status.json` files that were left as `running` by failed or aborted attempts. This does not delete any run directories:

```bash
export HF_DATASETS_OFFLINE=1

python scripts/clean_run_status.py \
  --runs-dir outputs/runs \
  --runs-dir-name-contains m10a \
  --older-than-minutes 10 \
  --to-status interrupted \
  --note "Stale M10A attempt superseded by final successful M10A/M10B workflow; marked interrupted, not deleted." \
  --out outputs/tables/m10b_stale_status_cleanup.json \
  --run-name m10b_clean_stale_statuses
```

Run larger general-utility evaluation. PPL uses 500 WikiText examples, ARC-Challenge requests 500 examples and therefore uses full validation if fewer are available, and HellaSwag uses 1000 validation examples:

```bash
INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

COMMON_GENERAL_ARGS=(
  --dtype bfloat16
  --batch-size 2
  --max-length 2048
  --ppl-samples 500
  --arc-samples 500
  --hellaswag-samples 1000
  --cache-dir "$HF_HUB_CACHE"
  --dataset-cache-dir "$HF_DATASETS_CACHE"
  --local-files-only
  --datasets-local-files-only
)

python scripts/evaluate_general.py \
  --model "$INSTRUCT_MODEL" \
  --method dense \
  --ratio 0.0 \
  --out outputs/evals/general_m10b_dense.json \
  "${COMMON_GENERAL_ARGS[@]}" \
  --run-name m10b_general_dense

METHODS=(random magnitude activation boundary_taylor_weighted)
RATIOS=("0.10:10p" "0.20:20p")

for pair in "${RATIOS[@]}"; do
  label="${pair##*:}"
  for method in "${METHODS[@]}"; do
    python scripts/evaluate_general.py \
      --model "outputs/pruned_models/qwen2p5_1p5b_${method}_mask_${label}_m9" \
      --out "outputs/evals/general_m10b_${method}_${label}.json" \
      "${COMMON_GENERAL_ARGS[@]}" \
      --run-name "m10b_general_${method}_${label}"
  done
done
```

Produce layer-wise mask distribution:

```bash
python scripts/report_mask_distribution.py \
  --mask-dirs \
    outputs/pruned_models/qwen2p5_1p5b_random_mask_10p_m9 \
    outputs/pruned_models/qwen2p5_1p5b_random_mask_20p_m9 \
    outputs/pruned_models/qwen2p5_1p5b_magnitude_mask_10p_m9 \
    outputs/pruned_models/qwen2p5_1p5b_magnitude_mask_20p_m9 \
    outputs/pruned_models/qwen2p5_1p5b_activation_mask_10p_m9 \
    outputs/pruned_models/qwen2p5_1p5b_activation_mask_20p_m9 \
    outputs/pruned_models/qwen2p5_1p5b_boundary_taylor_weighted_mask_10p_m9 \
    outputs/pruned_models/qwen2p5_1p5b_boundary_taylor_weighted_mask_20p_m9 \
  --out outputs/tables/m10b_mask_distribution.csv \
  --summary-out outputs/tables/m10b_mask_distribution.json \
  --run-name m10b_mask_distribution
```

Summarize M10B matched utility:

```bash
python scripts/summarize_m10b_matched_utility.py \
  --general-inputs \
    outputs/evals/general_m10b_dense.json \
    outputs/evals/general_m10b_random_10p.json \
    outputs/evals/general_m10b_random_20p.json \
    outputs/evals/general_m10b_magnitude_10p.json \
    outputs/evals/general_m10b_magnitude_20p.json \
    outputs/evals/general_m10b_activation_10p.json \
    outputs/evals/general_m10b_activation_20p.json \
    outputs/evals/general_m10b_boundary_taylor_weighted_10p.json \
    outputs/evals/general_m10b_boundary_taylor_weighted_20p.json \
  --bcr-table outputs/tables/m9_qwen2p5_1p5b_pilot_1k.csv \
  --out outputs/tables/m10b_matched_utility_all.csv \
  --summary-out outputs/tables/m10b_matched_utility_summary.json \
  --run-name m10b_summarize_matched_utility_all
```

Check M10B outputs:

```bash
cat outputs/tables/m10b_mask_distribution.csv
cat outputs/tables/m10b_matched_utility_all.csv
cat outputs/tables/m10b_matched_utility_summary.json
cat outputs/runs/*_m10b_summarize_matched_utility_all/status.json
cat outputs/runs/*_m10b_summarize_matched_utility_all/metrics.json
```

Expected M10B criteria:

- `outputs/tables/m10b_mask_distribution.csv` has columns `method`, `ratio`, `layer`, `total_units`, `pruned_units`, and `pruned_ratio`.
- `outputs/tables/m10b_matched_utility_all.csv` has dense plus 8 pruned rows and the required matched-utility columns.
- `outputs/tables/m10b_matched_utility_summary.json` explicitly answers whether any 10% or 20% pruned model is matched utility, identifies the lowest `BCR@q25` among matched-utility models when one exists, and says `20% is not a mild regime under current masking.` if no 20% model is matched.
- Stop after M10B.

## M11A Utility-Preserving Layer-wise Pruning Regime

M11A is remote-only and uses Qwen2.5-1.5B-Instruct only. Do not run 3B/7B, DPO/LoRA recovery, PAT, Wanda, safety datasets, UltraFeedback, or M11B.

M11A changes the pruning selection regime, not the research question. It adds:

- `--selection-scope layerwise`
- `--protect-first-n-layers`
- `--protect-last-n-layers`

The default remains global unless `--selection-scope layerwise` is explicitly set. M11A must use layerwise selection.

Prepare the remote shell and dataset cache:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
unset HF_DATASETS_OFFLINE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY'
import os
from datasets import load_dataset

cache_dir = os.environ.get("HF_DATASETS_CACHE")
kwargs = {"cache_dir": cache_dir} if cache_dir else {}
jobs = [
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
    ("allenai/ai2_arc", "ARC-Challenge", "validation"),
    ("Rowan/hellaswag", None, "validation"),
]
for name, config, split in jobs:
    if config is None:
        dataset = load_dataset(name, split=split, **kwargs)
    else:
        dataset = load_dataset(name, config, split=split, **kwargs)
    print(name, config, split, len(dataset))
PY
export HF_DATASETS_OFFLINE=1
```

### M11A Smoke

Run this smoke before any full grid. It checks random layerwise 2%, protected-layer mask logic, utility evaluation, BCR evaluation, and finite outputs.

```bash
INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
BASE_LOGPROBS="outputs/logprobs/base_qwen2p5_1p5b_m9_eval_1k.jsonl"
DENSE_MARGINS_EVAL="outputs/margins/dense_qwen2p5_1p5b_m9_eval_1k.jsonl"
DATA_DIR="data/processed/m9_pilot"

python scripts/score_pruning_importance.py \
  --model "$INSTRUCT_MODEL" \
  --method random \
  --ratio 0.02 \
  --selection-scope layerwise \
  --out outputs/scores/qwen2p5_1p5b_random_m11a_smoke.json \
  --dtype bfloat16 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m11a_score_random_layerwise_2p_smoke

python scripts/apply_mask_pruning.py \
  --scores outputs/scores/qwen2p5_1p5b_random_m11a_smoke.json \
  --ratio 0.02 \
  --selection-scope layerwise \
  --out outputs/pruned_models/qwen2p5_1p5b_random_layerwise_2p_m11a_smoke \
  --run-name m11a_apply_random_layerwise_2p_smoke

python scripts/apply_mask_pruning.py \
  --scores outputs/scores/qwen2p5_1p5b_random_m11a_smoke.json \
  --ratio 0.02 \
  --selection-scope layerwise \
  --protect-first-n-layers 4 \
  --protect-last-n-layers 2 \
  --out outputs/pruned_models/qwen2p5_1p5b_random_layerwise_protect_first4_last2_2p_m11a_smoke \
  --run-name m11a_apply_random_layerwise_protect_first4_last2_2p_smoke

python scripts/report_mask_distribution.py \
  --mask-dirs \
    outputs/pruned_models/qwen2p5_1p5b_random_layerwise_2p_m11a_smoke \
    outputs/pruned_models/qwen2p5_1p5b_random_layerwise_protect_first4_last2_2p_m11a_smoke \
  --out outputs/tables/m11a_smoke_mask_distribution.csv \
  --summary-out outputs/tables/m11a_smoke_mask_distribution.json \
  --run-name m11a_smoke_mask_distribution

python scripts/evaluate_general.py \
  --model outputs/pruned_models/qwen2p5_1p5b_random_layerwise_2p_m11a_smoke \
  --out outputs/evals/general_m11a_smoke_random_layerwise_2p.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 2048 \
  --ppl-samples 64 \
  --arc-samples 100 \
  --hellaswag-samples 100 \
  --cache-dir "$HF_HUB_CACHE" \
  --dataset-cache-dir "$HF_DATASETS_CACHE" \
  --local-files-only \
  --datasets-local-files-only \
  --run-name m11a_general_smoke_random_layerwise_2p

python scripts/evaluate_bcr.py \
  --model outputs/pruned_models/qwen2p5_1p5b_random_layerwise_2p_m11a_smoke \
  --base-logprobs "$BASE_LOGPROBS" \
  --dense-margins "$DENSE_MARGINS_EVAL" \
  --data "$DATA_DIR/hh_rlhf_eval.jsonl" \
  --max-samples 100 \
  --out outputs/evals/bcr_m11a_smoke_random_layerwise_2p.json \
  --records-out outputs/evals/bcr_m11a_smoke_random_layerwise_2p_records.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m11a_bcr_smoke_random_layerwise_2p
```

Smoke checks:

```bash
cat outputs/tables/m11a_smoke_mask_distribution.csv
cat outputs/evals/general_m11a_smoke_random_layerwise_2p.json
cat outputs/evals/bcr_m11a_smoke_random_layerwise_2p.json
cat outputs/runs/*_m11a_bcr_smoke_random_layerwise_2p/status.json
```

Expected smoke criteria:

- Layerwise smoke has approximately 2% pruned units in every unprotected layer.
- Protected smoke has zero pruned units in first 4 and last 2 layers.
- General utility and BCR outputs are finite.

### M11A Priority 1 Grid

Run this only after smoke passes. This evaluates `selection_scope=layerwise`, methods `random`, `activation`, `general_taylor`, and `boundary_taylor_weighted`, ratios `0.02`, `0.05`, `0.075`, and `0.10`.

```bash
INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
BASE_LOGPROBS="outputs/logprobs/base_qwen2p5_1p5b_m9_eval_1k.jsonl"
DENSE_MARGINS_EVAL="outputs/margins/dense_qwen2p5_1p5b_m9_eval_1k.jsonl"
DENSE_MARGINS_CALIB="outputs/margins/dense_qwen2p5_1p5b_m9_calib_1k.jsonl"
DATA_DIR="data/processed/m9_pilot"

python scripts/evaluate_general.py \
  --model "$INSTRUCT_MODEL" \
  --method dense \
  --ratio 0.0 \
  --out outputs/evals/general_m11a_dense.json \
  --dtype bfloat16 \
  --batch-size 2 \
  --max-length 2048 \
  --ppl-samples 500 \
  --arc-samples 500 \
  --hellaswag-samples 1000 \
  --cache-dir "$HF_HUB_CACHE" \
  --dataset-cache-dir "$HF_DATASETS_CACHE" \
  --local-files-only \
  --datasets-local-files-only \
  --run-name m11a_general_dense

python scripts/score_pruning_importance.py \
  --model "$INSTRUCT_MODEL" \
  --method random \
  --ratio 0.02 \
  --selection-scope layerwise \
  --out outputs/scores/qwen2p5_1p5b_random_m11a_layerwise.json \
  --dtype bfloat16 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m11a_score_random_layerwise

python scripts/score_pruning_importance.py \
  --model "$INSTRUCT_MODEL" \
  --data "$DATA_DIR/hh_rlhf_calib.jsonl" \
  --method activation \
  --max-samples 1000 \
  --ratio 0.02 \
  --selection-scope layerwise \
  --out outputs/scores/qwen2p5_1p5b_activation_m11a_layerwise.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 2048 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m11a_score_activation_layerwise

python scripts/score_pruning_importance.py \
  --model "$INSTRUCT_MODEL" \
  --data "$DATA_DIR/hh_rlhf_calib.jsonl" \
  --method general_taylor \
  --max-samples 1000 \
  --ratio 0.02 \
  --selection-scope layerwise \
  --out outputs/scores/qwen2p5_1p5b_general_taylor_m11a_layerwise.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 2048 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m11a_score_general_taylor_layerwise

python scripts/score_pruning_importance.py \
  --instruct-model "$INSTRUCT_MODEL" \
  --dense-margins "$DENSE_MARGINS_CALIB" \
  --data "$DATA_DIR/hh_rlhf_calib.jsonl" \
  --method boundary_taylor_weighted \
  --max-samples 1000 \
  --tau-mode q25 \
  --ratio 0.02 \
  --selection-scope layerwise \
  --out outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_m11a_layerwise.json \
  --dtype bfloat16 \
  --batch-size 1 \
  --max-length 2048 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m11a_score_boundary_taylor_weighted_layerwise

declare -A SCORE_BY_METHOD
SCORE_BY_METHOD[random]="outputs/scores/qwen2p5_1p5b_random_m11a_layerwise.json"
SCORE_BY_METHOD[activation]="outputs/scores/qwen2p5_1p5b_activation_m11a_layerwise.json"
SCORE_BY_METHOD[general_taylor]="outputs/scores/qwen2p5_1p5b_general_taylor_m11a_layerwise.json"
SCORE_BY_METHOD[boundary_taylor_weighted]="outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_m11a_layerwise.json"

METHODS=(random activation general_taylor boundary_taylor_weighted)
RATIOS=("0.02:2p" "0.05:5p" "0.075:7p5" "0.10:10p")
GENERAL_INPUTS=(outputs/evals/general_m11a_dense.json)
BCR_INPUTS=()
MASK_DIRS=()

for method in "${METHODS[@]}"; do
  for pair in "${RATIOS[@]}"; do
    ratio="${pair%%:*}"
    label="${pair##*:}"
    out_dir="outputs/pruned_models/qwen2p5_1p5b_${method}_layerwise_${label}_m11a"
    general_out="outputs/evals/general_m11a_${method}_layerwise_${label}.json"
    bcr_out="outputs/evals/bcr_m11a_${method}_layerwise_${label}.json"

    python scripts/apply_mask_pruning.py \
      --scores "${SCORE_BY_METHOD[$method]}" \
      --ratio "$ratio" \
      --selection-scope layerwise \
      --out "$out_dir" \
      --run-name "m11a_apply_${method}_layerwise_${label}"

    python scripts/evaluate_general.py \
      --model "$out_dir" \
      --out "$general_out" \
      --dtype bfloat16 \
      --batch-size 2 \
      --max-length 2048 \
      --ppl-samples 500 \
      --arc-samples 500 \
      --hellaswag-samples 1000 \
      --cache-dir "$HF_HUB_CACHE" \
      --dataset-cache-dir "$HF_DATASETS_CACHE" \
      --local-files-only \
      --datasets-local-files-only \
      --run-name "m11a_general_${method}_layerwise_${label}"

    python scripts/evaluate_bcr.py \
      --model "$out_dir" \
      --base-logprobs "$BASE_LOGPROBS" \
      --dense-margins "$DENSE_MARGINS_EVAL" \
      --data "$DATA_DIR/hh_rlhf_eval.jsonl" \
      --max-samples 1000 \
      --out "$bcr_out" \
      --records-out "outputs/evals/bcr_m11a_${method}_layerwise_${label}_records.jsonl" \
      --dtype bfloat16 \
      --batch-size 1 \
      --cache-dir "$HF_HUB_CACHE" \
      --local-files-only \
      --run-name "m11a_bcr_${method}_layerwise_${label}"

    GENERAL_INPUTS+=("$general_out")
    BCR_INPUTS+=("$bcr_out")
    MASK_DIRS+=("$out_dir")
  done
done

python scripts/report_mask_distribution.py \
  --mask-dirs "${MASK_DIRS[@]}" \
  --out outputs/tables/m11a_mask_distribution.csv \
  --summary-out outputs/tables/m11a_mask_distribution.json \
  --run-name m11a_mask_distribution_priority1

python scripts/summarize_m11a_layerwise.py \
  --general-inputs "${GENERAL_INPUTS[@]}" \
  --bcr-inputs "${BCR_INPUTS[@]}" \
  --mask-distribution outputs/tables/m11a_mask_distribution.csv \
  --out outputs/tables/m11a_layerwise_utility_bcr.csv \
  --summary-out outputs/tables/m11a_summary.json \
  --run-name m11a_summarize_priority1
```

Check M11A Priority 1 outputs:

```bash
cat outputs/tables/m11a_layerwise_utility_bcr.csv
cat outputs/tables/m11a_summary.json
cat outputs/tables/m11a_mask_distribution.csv | head -80
cat outputs/runs/*_m11a_summarize_priority1/status.json
cat outputs/runs/*_m11a_summarize_priority1/metrics.json
```

Stop after Priority 1 if it finds a clear matched-utility regime. Only run protected-layer Priority 2/3 after inspecting `outputs/tables/m11a_summary.json` and deciding that layerwise without protection is insufficient.

## M12 Hybrid Boundary-Utility Pruning

M12 is remote-only and uses Qwen2.5-1.5B-Instruct only. Do not run 3B/7B, DPO/LoRA recovery, PAT, Wanda, safety datasets, UltraFeedback, or M13.

M12 tests hybrid pruning scores:

```text
I_hybrid(g) = rank_norm(I_utility(g)) + alpha * rank_norm(I_boundary(g))
```

where `I_utility` is either `activation` or `general_taylor`, `I_boundary` is `boundary_taylor_weighted`, and normalization is layerwise rank percentile normalization. Pruning still removes the lowest hybrid-score units.

Run the 1k alpha sweep first:

```bash
source /etc/network_turbo || true
conda activate pbp
cd ~/autodl-tmp/preference-boundary-pruning
git pull origin main

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
BASE_LOGPROBS="outputs/logprobs/base_qwen2p5_1p5b_m9_eval_1k.jsonl"
DENSE_MARGINS_EVAL="outputs/margins/dense_qwen2p5_1p5b_m9_eval_1k.jsonl"
DATA_DIR="data/processed/m9_pilot"

mkdir -p outputs/evals outputs/scores outputs/pruned_models outputs/tables outputs/logs

if [ ! -f outputs/evals/general_m12_dense.json ]; then
  python scripts/evaluate_general.py \
    --model "$INSTRUCT_MODEL" \
    --method dense \
    --ratio 0.0 \
    --out outputs/evals/general_m12_dense.json \
    --dtype bfloat16 \
    --batch-size 2 \
    --max-length 2048 \
    --ppl-samples 500 \
    --arc-samples 500 \
    --hellaswag-samples 1000 \
    --cache-dir "$HF_HUB_CACHE" \
    --dataset-cache-dir "$HF_DATASETS_CACHE" \
    --local-files-only \
    --datasets-local-files-only \
    --run-name m12_general_dense
fi

declare -A BASE_SCORE_BY_METHOD
BASE_SCORE_BY_METHOD[activation]="outputs/scores/qwen2p5_1p5b_activation_m11a_layerwise.json"
BASE_SCORE_BY_METHOD[general_taylor]="outputs/scores/qwen2p5_1p5b_general_taylor_m11a_layerwise.json"
BASE_SCORE_BY_METHOD[boundary_taylor_weighted]="outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_m11a_layerwise.json"

BASE_METHODS=(activation general_taylor boundary_taylor_weighted)
HYBRID_UTILITIES=(activation general_taylor)
ALPHAS=("0.25:0p25" "0.5:0p5" "1.0:1p0" "2.0:2p0")
RATIOS=("0.02:2p" "0.03:3p" "0.05:5p")

declare -A SCORE_BY_METHOD_ALPHA

for utility in "${HYBRID_UTILITIES[@]}"; do
  for alpha_pair in "${ALPHAS[@]}"; do
    alpha="${alpha_pair%%:*}"
    alpha_label="${alpha_pair##*:}"
    method="${utility}_boundary"
    score_out="outputs/scores/qwen2p5_1p5b_${method}_alpha${alpha_label}_m12_layerwise.json"
    if [ ! -f "$score_out" ]; then
      python scripts/compose_hybrid_scores.py \
        --utility-scores "${BASE_SCORE_BY_METHOD[$utility]}" \
        --boundary-scores "${BASE_SCORE_BY_METHOD[boundary_taylor_weighted]}" \
        --method "$method" \
        --alpha "$alpha" \
        --ratio 0.02 \
        --selection-scope layerwise \
        --normalization-scope layerwise \
        --out "$score_out" \
        --run-name "m12_compose_${method}_alpha${alpha_label}"
    fi
    SCORE_BY_METHOD_ALPHA["${method}:${alpha_label}"]="$score_out"
  done
done

GENERAL_INPUTS=(outputs/evals/general_m12_dense.json)
BCR_INPUTS=()

run_model_eval_1k() {
  method="$1"
  ratio="$2"
  label="$3"
  score_path="$4"
  alpha_label="${5:-}"

  if [ -n "$alpha_label" ]; then
    suffix="${method}_alpha${alpha_label}_layerwise_${label}"
  else
    suffix="${method}_layerwise_${label}"
  fi
  out_dir="outputs/pruned_models/qwen2p5_1p5b_${suffix}_m12"
  general_out="outputs/evals/general_m12_${suffix}.json"
  bcr_out="outputs/evals/bcr_m12_${suffix}.json"

  if [ ! -d "$out_dir" ]; then
    python scripts/apply_mask_pruning.py \
      --scores "$score_path" \
      --ratio "$ratio" \
      --selection-scope layerwise \
      --out "$out_dir" \
      --run-name "m12_apply_${suffix}"
  fi

  if [ ! -f "$general_out" ]; then
    python scripts/evaluate_general.py \
      --model "$out_dir" \
      --out "$general_out" \
      --dtype bfloat16 \
      --batch-size 2 \
      --max-length 2048 \
      --ppl-samples 500 \
      --arc-samples 500 \
      --hellaswag-samples 1000 \
      --cache-dir "$HF_HUB_CACHE" \
      --dataset-cache-dir "$HF_DATASETS_CACHE" \
      --local-files-only \
      --datasets-local-files-only \
      --run-name "m12_general_${suffix}"
  fi

  if [ ! -f "$bcr_out" ]; then
    python scripts/evaluate_bcr.py \
      --model "$out_dir" \
      --base-logprobs "$BASE_LOGPROBS" \
      --dense-margins "$DENSE_MARGINS_EVAL" \
      --data "$DATA_DIR/hh_rlhf_eval.jsonl" \
      --max-samples 1000 \
      --out "$bcr_out" \
      --records-out "outputs/evals/bcr_m12_${suffix}_records.jsonl" \
      --dtype bfloat16 \
      --batch-size 1 \
      --cache-dir "$HF_HUB_CACHE" \
      --local-files-only \
      --run-name "m12_bcr_${suffix}_1k"
  fi

  GENERAL_INPUTS+=("$general_out")
  BCR_INPUTS+=("$bcr_out")
}

for method in "${BASE_METHODS[@]}"; do
  for pair in "${RATIOS[@]}"; do
    ratio="${pair%%:*}"
    label="${pair##*:}"
    run_model_eval_1k "$method" "$ratio" "$label" "${BASE_SCORE_BY_METHOD[$method]}"
  done
done

for utility in "${HYBRID_UTILITIES[@]}"; do
  method="${utility}_boundary"
  for alpha_pair in "${ALPHAS[@]}"; do
    alpha_label="${alpha_pair##*:}"
    score_path="${SCORE_BY_METHOD_ALPHA[${method}:${alpha_label}]}"
    for pair in "${RATIOS[@]}"; do
      ratio="${pair%%:*}"
      label="${pair##*:}"
      run_model_eval_1k "$method" "$ratio" "$label" "$score_path" "$alpha_label"
    done
  done
done

python scripts/summarize_m12_hybrid.py \
  --general-inputs "${GENERAL_INPUTS[@]}" \
  --bcr-inputs "${BCR_INPUTS[@]}" \
  --out outputs/tables/m12_alpha_sweep.csv \
  --summary-out outputs/tables/m12_hybrid_summary.json \
  --run-name m12_summarize_alpha_sweep_1k \
  --overwrite

cat outputs/tables/m12_alpha_sweep.csv
cat outputs/tables/m12_hybrid_summary.json
```

Then run 5k BCR only for matched candidates and their corresponding utility baselines:

```bash
python - <<'PY'
import csv
from pathlib import Path

def ratio_label(value):
    return f"{int(round(float(value) * 100))}p"

def alpha_label(value):
    if not value:
        return ""
    mapping = {0.25: "0p25", 0.5: "0p5", 1.0: "1p0", 2.0: "2p0"}
    return mapping.get(round(float(value), 2), str(value).replace(".", "p"))

rows = list(csv.DictReader(open("outputs/tables/m12_alpha_sweep.csv", newline="")))
targets = set()
for row in rows:
    if row["matched"] != "true" or row["method"] == "dense":
        continue
    targets.add((row["method"], row["ratio"], row["alpha"]))
    if row["method"] in {"activation_boundary", "general_taylor_boundary"}:
        targets.add((row["utility_method"], row["ratio"], ""))

out = Path("outputs/tables/m12_5k_bcr_targets.tsv")
with out.open("w", encoding="utf-8") as f:
    f.write("method\tratio\talpha\tmodel_dir\tbcr_out\trecords_out\n")
    for method, ratio, alpha in sorted(targets):
        rlabel = ratio_label(ratio)
        alabel = alpha_label(alpha)
        if alpha:
            suffix = f"{method}_alpha{alabel}_layerwise_{rlabel}"
        else:
            suffix = f"{method}_layerwise_{rlabel}"
        f.write(
            f"{method}\t{ratio}\t{alpha}\t"
            f"outputs/pruned_models/qwen2p5_1p5b_{suffix}_m12\t"
            f"outputs/evals/bcr_m12_5k_{suffix}.json\t"
            f"outputs/evals/bcr_m12_5k_{suffix}_records.jsonl\n"
        )
print(out)
PY

tail -n +2 outputs/tables/m12_5k_bcr_targets.tsv | while IFS=$'\t' read -r method ratio alpha model_dir bcr_out records_out; do
  if [ ! -f "$bcr_out" ]; then
    python scripts/evaluate_bcr.py \
      --model "$model_dir" \
      --base-logprobs "$BASE_LOGPROBS" \
      --dense-margins "$DENSE_MARGINS_EVAL" \
      --data "$DATA_DIR/hh_rlhf_eval.jsonl" \
      --max-samples 5000 \
      --out "$bcr_out" \
      --records-out "$records_out" \
      --dtype bfloat16 \
      --batch-size 1 \
      --cache-dir "$HF_HUB_CACHE" \
      --local-files-only \
      --run-name "m12_bcr_5k_${method}_${ratio}_${alpha:-baseline}"
  fi
done

GENERAL_INPUTS=(outputs/evals/general_m12_dense.json outputs/evals/general_m12_*_layerwise_*.json)
BCR_INPUTS=(outputs/evals/bcr_m12_*_layerwise_*.json outputs/evals/bcr_m12_5k_*.json)

python scripts/summarize_m12_hybrid.py \
  --general-inputs "${GENERAL_INPUTS[@]}" \
  --bcr-inputs "${BCR_INPUTS[@]}" \
  --out outputs/tables/m12_alpha_sweep.csv \
  --summary-out outputs/tables/m12_hybrid_summary.json \
  --run-name m12_summarize_alpha_sweep_with_5k \
  --overwrite

cat outputs/tables/m12_alpha_sweep.csv
cat outputs/tables/m12_hybrid_summary.json
```

Expected M12 outputs:

```text
outputs/tables/m12_alpha_sweep.csv
outputs/tables/m12_hybrid_summary.json
outputs/tables/m12_5k_bcr_targets.tsv
```

The M12 decision rule is: a hybrid setting must have `matched=true` and lower `BCR@q25` than its corresponding utility-only baseline at the same ratio and BCR sample size.

## Milestone Boundary

M9 has passed after the remote Qwen2.5-1.5B 1k pilot table completed with 8 rows and all 8 BCR inputs summarized successfully on `1 x NVIDIA RTX PRO 6000 96GB`.

M10A has passed after the remote 20% matched-utility table completed with 5 rows. `boundary_taylor_weighted` had lower `BCR@q25` than activation at 20%, but all 20% pruned models had `matched_utility_flag=false` under the configured thresholds.

M10B has passed as a larger remote smoke/checkpoint run. It produced the all-ratio matched-utility table and mask-distribution table, but no 10% or 20% pruned model satisfied the configured matched-utility thresholds. Under current global masking, 20% is not a mild regime.

M11A passed after the layerwise Priority 1 grid found matched utility at 2% and avoided early-layer collapse. Activation 2% had the lowest matched `BCR@q25`, so M12 is approved to test hybrid utility-boundary scores before any scaling or recovery methods.

M12 is limited to the hybrid alpha sweep above. Do not run post-pruning recovery, DPO, LoRA, 3B/7B scaling, PAT, Wanda, M13, safety datasets, UltraFeedback, or work beyond M12 until explicitly approved.
