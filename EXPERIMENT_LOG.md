# EXPERIMENT LOG

## Run: 20260616_223733_m0_run_dir_helper_smoke

Date: 2026-06-16 22:37  
Milestone: M0  
Purpose: Verify that the run-directory helper creates the protocol-required layout and metadata files.  
Command: `D:\anaconda3\python.exe -c <pbp.logging_utils smoke>`  
Config file: `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/config.yaml`  
Git commit: null, workspace is not a git repository  
Model: none  
Dataset: none  
Seed: 42  
GPU: not used  
Runtime: 0.680926 seconds  
Status: success

Inputs:
- `src/pbp/logging_utils.py`

Outputs:
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/config.yaml`
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/command.sh`
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/stdout.log`
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/stderr.log`
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/metrics.json`
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/status.json`
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/environment.json`
- `outputs/runs/20260616_223733_m0_run_dir_helper_smoke/artifacts/`

Metrics:

```json
{
  "required_files_created": 7,
  "run_directory_helper_created": true
}
```

Notes:
- This is a local lightweight smoke test and does not load models or download datasets.
- Remote experiment smoke tests are not executed locally per the updated protocol.

## Run: M0_LOCAL_CHECKS_20260616_2237

Date: 2026-06-16 22:37  
Milestone: M0  
Purpose: Run local unit tests and bytecode compile checks.  
Command: `D:\anaconda3\python.exe -m pytest tests/` and `D:\anaconda3\python.exe -m compileall src scripts`  
Config file: none  
Git commit: null, workspace is not a git repository  
Model: none  
Dataset: none  
Seed: not applicable  
GPU: not used  
Runtime: pytest 1.42 seconds; compileall approximately 2 seconds  
Status: success

Inputs:
- `src/pbp/`
- `scripts/`
- `tests/`

Outputs:
- pytest result: `10 passed, 1 skipped`
- compileall result: success

Metrics:

```json
{
  "tests_passed": 10,
  "tests_skipped": 1,
  "compileall_success": true
}
```

Notes:
- The skipped test requires `torch`, which is not installed in the local conda environment. This is acceptable for local lightweight validation.

## Run: 20260617_004902_m1_hh_rlhf_fixture_smoke

Date: 2026-06-17 00:47  
Milestone: M1  
Purpose: Local lightweight fixture smoke test for HH-RLHF preprocessing without downloading datasets.  
Command: `D:\anaconda3\python.exe scripts\prepare_hh_rlhf.py --input-jsonl tests\fixtures\hh_rlhf_raw_fixture.jsonl --calib-size 2 --eval-size 2 --seed 42 --out-dir outputs\m1_fixture_smoke_20260617_0049\processed --runs-dir outputs\runs --run-name m1_hh_rlhf_fixture_smoke`  
Config file: `outputs/runs/20260617_004902_m1_hh_rlhf_fixture_smoke/config.yaml`  
Git commit: `fd1ee6f` plus uncommitted M1 working-tree changes  
Model: none  
Dataset: local fixture `tests/fixtures/hh_rlhf_raw_fixture.jsonl`  
Seed: 42  
GPU: not used  
Runtime: 0.280856 seconds  
Status: success

Inputs:
- `tests/fixtures/hh_rlhf_raw_fixture.jsonl`
- `scripts/prepare_hh_rlhf.py`
- `src/pbp/data.py`

Outputs:
- `outputs/m1_fixture_smoke_20260617_0049/processed/hh_rlhf_calib.jsonl`
- `outputs/m1_fixture_smoke_20260617_0049/processed/hh_rlhf_eval.jsonl`
- `outputs/runs/20260617_004902_m1_hh_rlhf_fixture_smoke/`

Metrics:

```json
{
  "num_raw_records": 5,
  "num_calib_records": 2,
  "num_eval_records": 2,
  "num_total_records": 4,
  "num_skipped_records": 0,
  "calib_eval_disjoint": true,
  "empty_chosen_or_rejected": 0
}
```

Notes:
- Local-only fixture smoke per remote execution policy.
- Real HH-RLHF preprocessing remains remote pending.

## Run: 20260617_012943_m1_hh_rlhf_smoke

Date: 2026-06-17 01:29  
Milestone: M1  
Purpose: Remote real HH-RLHF preprocessing smoke test.  
Command:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning

python scripts/prepare_hh_rlhf.py \
  --dataset Anthropic/hh-rlhf \
  --calib-size 100 \
  --eval-size 200 \
  --seed 42 \
  --out-dir data/processed \
  --run-name m1_hh_rlhf_smoke
```

Config file: `outputs/runs/20260617_012943_m1_hh_rlhf_smoke/config.yaml`  
Git commit: `2411147`  
Model: none  
Dataset: `Anthropic/hh-rlhf`  
Seed: 42  
GPU: not used  
Runtime: 46.790961 seconds  
Status: success

Inputs:
- `Anthropic/hh-rlhf`

Outputs:
- `data/processed/hh_rlhf_calib.jsonl`
- `data/processed/hh_rlhf_eval.jsonl`
- `outputs/runs/20260617_012943_m1_hh_rlhf_smoke/`

Metrics:

```json
{
  "num_raw_records": 160800,
  "num_calib_records": 100,
  "num_eval_records": 200,
  "num_total_records": 300,
  "num_skipped_records": 0,
  "calib_eval_disjoint": true,
  "empty_chosen_or_rejected": 0
}
```

Notes:
- Executed remotely because the protocol forbids local dataset downloads.
- Checked line counts: `100 data/processed/hh_rlhf_calib.jsonl`, `200 data/processed/hh_rlhf_eval.jsonl`.

## Run: 20260617_013822_m2_logprob_dry_run

Date: 2026-06-17 01:38  
Milestone: M2  
Purpose: Local lightweight dry-run smoke for response-only logprob schema and masking without loading a model.  
Command: `D:\anaconda3\python.exe scripts\compute_logprobs.py --model dry-run-model --data tests\fixtures\hh_rlhf_processed_fixture.jsonl --out outputs\m2_logprob_dry_run_20260617_0140\logprobs.jsonl --runs-dir outputs\runs --run-name m2_logprob_dry_run --max-samples 2 --seed 42 --dry-run`  
Config file: `outputs/runs/20260617_013822_m2_logprob_dry_run/config.yaml`  
Git commit: `12086fa` plus uncommitted M2 working-tree changes  
Model: dry-run-model  
Dataset: local fixture `tests/fixtures/hh_rlhf_processed_fixture.jsonl`  
Seed: 42  
GPU: not used  
Runtime: 0.269153 seconds  
Status: success

Inputs:
- `tests/fixtures/hh_rlhf_processed_fixture.jsonl`
- `scripts/compute_logprobs.py`
- `src/pbp/logprobs.py`

Outputs:
- `outputs/m2_logprob_dry_run_20260617_0140/logprobs.jsonl`
- `outputs/runs/20260617_013822_m2_logprob_dry_run/`

Metrics:

```json
{
  "num_examples": 2,
  "num_scored_responses": 4,
  "min_chosen_response_tokens": 15,
  "min_rejected_response_tokens": 11,
  "mean_chosen_response_tokens": 15.5,
  "mean_rejected_response_tokens": 14.5,
  "length_normalized_logprobs_finite": true,
  "dry_run": true
}
```

Notes:
- Local-only dry run per remote execution policy.
- Real Qwen logprob smoke remains remote pending.

## Run: 20260617_014506_m2_logprob_smoke

Date: 2026-06-17 01:45  
Milestone: M2  
Purpose: Remote real Qwen response-only logprob smoke test on 5 HH-RLHF examples.  
Command:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull

python scripts/compute_logprobs.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/hh_rlhf_eval.jsonl \
  --max-samples 5 \
  --out outputs/logprobs/smoke_instruct_5.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m2_logprob_smoke
```

Config file: `outputs/runs/20260617_014506_m2_logprob_smoke/config.yaml`  
Git commit: `6e0f1c6`  
Model: `Qwen/Qwen2.5-1.5B-Instruct`  
Dataset: `data/processed/hh_rlhf_eval.jsonl`  
Seed: 42  
GPU: remote  
Runtime: 7.739926 seconds  
Status: success

Inputs:
- `data/processed/hh_rlhf_eval.jsonl`
- cached `Qwen/Qwen2.5-1.5B-Instruct`

Outputs:
- `outputs/logprobs/smoke_instruct_5.jsonl`
- `outputs/runs/20260617_014506_m2_logprob_smoke/`

Metrics:

```json
{
  "num_examples": 5,
  "num_scored_responses": 10,
  "min_chosen_response_tokens": 5,
  "min_rejected_response_tokens": 4,
  "mean_chosen_response_tokens": 37.2,
  "mean_rejected_response_tokens": 87.2,
  "length_normalized_logprobs_finite": true,
  "dry_run": false
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading and GPU inference.
- Output line count checked: `5 outputs/logprobs/smoke_instruct_5.jsonl`.
- Remote stderr included non-fatal warnings: invalid `OMP_NUM_THREADS` value and Transformers `torch_dtype` deprecation.

## Run: 20260617_014918_m3_dense_margin_dry_run

Date: 2026-06-17 01:49  
Milestone: M3  
Purpose: Local lightweight dry-run smoke for dense/base margin schema without loading models.  
Command: `D:\anaconda3\python.exe scripts\compute_dense_margins.py --instruct-model dense-dry-run --base-model base-dry-run --data tests\fixtures\hh_rlhf_processed_fixture.jsonl --out outputs\m3_dense_margin_dry_run_20260617_0150\dense_margins.jsonl --runs-dir outputs\runs --run-name m3_dense_margin_dry_run --max-samples 2 --seed 42 --dry-run`  
Config file: `outputs/runs/20260617_014918_m3_dense_margin_dry_run/config.yaml`  
Git commit: `9a8ae5f` plus uncommitted M3 working-tree changes  
Model: `dense-dry-run`  
Dataset: local fixture `tests/fixtures/hh_rlhf_processed_fixture.jsonl`  
Seed: 42  
GPU: not used  
Runtime: 0.279244 seconds  
Status: success

Inputs:
- `tests/fixtures/hh_rlhf_processed_fixture.jsonl`
- `scripts/compute_dense_margins.py`
- `src/pbp/margins.py`

Outputs:
- `outputs/m3_dense_margin_dry_run_20260617_0150/dense_margins.jsonl`
- `outputs/runs/20260617_014918_m3_dense_margin_dry_run/`

Metrics:

```json
{
  "num_examples": 2,
  "delta_dense_finite": true,
  "mean_delta_dense": 0.07499999999999996,
  "min_delta_dense": 0.06999999999999995,
  "max_delta_dense": 0.07999999999999996,
  "dry_run": true
}
```

Notes:
- Local-only dry run per remote execution policy.
- Real dense/base margin smoke remains remote pending.

## Run: 20260617_015131_m3_dense_margin_smoke

Date: 2026-06-17 01:51  
Milestone: M3  
Purpose: Remote real Qwen dense/base margin smoke test on 20 HH-RLHF examples.  
Command:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull
export OMP_NUM_THREADS=1

python scripts/compute_dense_margins.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-model Qwen/Qwen2.5-1.5B \
  --data data/processed/hh_rlhf_eval.jsonl \
  --max-samples 20 \
  --out outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl \
  --dtype bfloat16 \
  --batch-size 1 \
  --cache-dir "$HF_HUB_CACHE" \
  --local-files-only \
  --run-name m3_dense_margin_smoke
```

Config file: `outputs/runs/20260617_015131_m3_dense_margin_smoke/config.yaml`  
Git commit: `63894cc`  
Model: `Qwen/Qwen2.5-1.5B-Instruct`  
Reference: `Qwen/Qwen2.5-1.5B`  
Dataset: `data/processed/hh_rlhf_eval.jsonl`  
Seed: 42  
GPU: remote  
Runtime: 11.792054 seconds  
Status: success

Inputs:
- `data/processed/hh_rlhf_eval.jsonl`
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- cached `Qwen/Qwen2.5-1.5B`

Outputs:
- `outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl`
- `outputs/runs/20260617_015131_m3_dense_margin_smoke/`

Metrics:

```json
{
  "num_examples": 20,
  "delta_dense_finite": true,
  "mean_delta_dense": 0.044380287931350404,
  "min_delta_dense": -0.4868914150299237,
  "max_delta_dense": 0.8073058234320745,
  "dry_run": false
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading and GPU inference.
- Output line count checked: `20 outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl`.

## Run: 20260617_015536_m4_coverage_fixture

Date: 2026-06-17 01:55  
Milestone: M4  
Purpose: Local lightweight fixture smoke for Coverage@tau reporting and histogram CSV generation.  
Command: `D:\anaconda3\python.exe scripts\report_coverage.py --dense-margins tests\fixtures\dense_margins_fixture.jsonl --out outputs\m4_coverage_fixture_20260617_0158\coverage.json --histogram-out outputs\m4_coverage_fixture_20260617_0158\histogram.csv --histogram-bins 2 --runs-dir outputs\runs --run-name m4_coverage_fixture --seed 42`  
Config file: `outputs/runs/20260617_015536_m4_coverage_fixture/config.yaml`  
Git commit: `993050f` plus uncommitted M4 working-tree changes  
Model: none  
Dataset: local fixture `tests/fixtures/dense_margins_fixture.jsonl`  
Seed: 42  
GPU: not used  
Runtime: 0.277853 seconds  
Status: success

Inputs:
- `tests/fixtures/dense_margins_fixture.jsonl`
- `scripts/report_coverage.py`
- `src/pbp/metrics.py`

Outputs:
- `outputs/m4_coverage_fixture_20260617_0158/coverage.json`
- `outputs/m4_coverage_fixture_20260617_0158/histogram.csv`
- `outputs/runs/20260617_015536_m4_coverage_fixture/`

Metrics:

```json
{
  "num_examples": 4,
  "coverage_at_0": 0.75,
  "coverage_at_q25": 0.5,
  "coverage_at_q50": 0.25,
  "coverage_at_q75": 0.25,
  "preference_accuracy": 0.75,
  "mean_delta_dense": 0.10000000000000002,
  "median_delta_dense": 0.15000000000000002,
  "positive_q25": 0.15000000000000002,
  "positive_q50": 0.2,
  "positive_q75": 0.25,
  "coverage_metrics_valid": true,
  "numeric_metrics_finite": true
}
```

Notes:
- Local-only fixture smoke per remote execution policy.
- Real dense-margin coverage smoke remains remote pending.

## Run: 20260617_020235_m4_coverage_smoke

Date: 2026-06-17 02:02  
Milestone: M4  
Purpose: Remote Coverage@tau report on real dense-margin smoke output.  
Command:

```bash
source /root/.pbp_env
cd /root/autodl-tmp/preference-boundary-pruning
git pull

python scripts/report_coverage.py \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl \
  --out outputs/evals/coverage_qwen2p5_1p5b_smoke.json \
  --histogram-out outputs/evals/coverage_qwen2p5_1p5b_smoke_histogram.csv \
  --run-name m4_coverage_smoke
```

Config file: `outputs/runs/20260617_020235_m4_coverage_smoke/config.yaml`  
Git commit: `1a50c9d`  
Model: none  
Dataset: `outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl`  
Seed: 42  
GPU: not used  
Runtime: 0.047963 seconds  
Status: success

Inputs:
- `outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl`

Outputs:
- `outputs/evals/coverage_qwen2p5_1p5b_smoke.json`
- `outputs/evals/coverage_qwen2p5_1p5b_smoke_histogram.csv`
- `outputs/runs/20260617_020235_m4_coverage_smoke/`

Metrics:

```json
{
  "num_examples": 20,
  "coverage_at_0": 0.6,
  "coverage_at_q25": 0.45,
  "coverage_at_q50": 0.3,
  "coverage_at_q75": 0.15,
  "preference_accuracy": 0.6,
  "mean_delta_dense": 0.044380287931350404,
  "median_delta_dense": 0.023675782385889743,
  "positive_q25": 0.03139724181309256,
  "positive_q50": 0.16378641597378119,
  "positive_q75": 0.26488588120991663,
  "coverage_metrics_valid": true,
  "numeric_metrics_finite": true
}
```

Notes:
- Executed remotely because the protocol treats real experiment outputs as remote-only.
- Output files checked: `outputs/evals/coverage_qwen2p5_1p5b_smoke.json` and `outputs/evals/coverage_qwen2p5_1p5b_smoke_histogram.csv`.

## Run: 20260617_020948_m5_mask_pruning_dry_run

Date: 2026-06-17 02:09
Milestone: M5
Purpose: Local lightweight dry-run for mask artifact generation and global pruning ratio accounting without loading a model.
Command: `D:\anaconda3\python.exe scripts\apply_mask_pruning.py --model dry-run-model --method random --ratio 0.25 --out outputs\m5_mask_pruning_dry_run_20260617_1420 --runs-dir outputs\runs --run-name m5_mask_pruning_dry_run --seed 42 --dry-run --dry-run-layers 2 --dry-run-intermediate-size 8`
Config file: `outputs/runs/20260617_020948_m5_mask_pruning_dry_run/config.yaml`
Git commit: `9a4f211` plus uncommitted M5 working-tree changes
Model: dry-run-model
Dataset: none
Seed: 42
GPU: not used
Runtime: 0.307234 seconds
Status: success

Inputs:
- `scripts/apply_mask_pruning.py`
- `src/pbp/ffn_units.py`
- `src/pbp/pruning.py`

Outputs:
- `outputs/m5_mask_pruning_dry_run_20260617_1420/mask_config.json`
- `outputs/m5_mask_pruning_dry_run_20260617_1420/masks.json`
- `outputs/runs/20260617_020948_m5_mask_pruning_dry_run/`

Metrics:

```json
{
  "method": "random",
  "requested_ratio": 0.25,
  "total_units": 16,
  "num_pruned_units": 4,
  "num_kept_units": 12,
  "actual_ratio": 0.25,
  "num_masked_modules": 2,
  "dry_run": true,
  "generation_success": null,
  "generated_new_tokens": null
}
```

Notes:
- Local-only dry run per remote execution policy.
- The torch-dependent forward masking unit test is skipped locally because torch is not installed in the local conda environment.
- Real Qwen mask-pruning smoke completed remotely in `outputs/runs/20260617_021524_m5_random_mask_10p_smoke`.

## Run: 20260617_021524_m5_random_mask_10p_smoke

Date: 2026-06-17 02:15
Milestone: M5
Purpose: Remote real Qwen mask-based coupled FFN pruning smoke test.
Command:

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

Config file: `outputs/runs/20260617_021524_m5_random_mask_10p_smoke/config.yaml`
Git commit: `8e707a8` inferred from run time before the later protocol-only commit
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Dataset: none
Seed: 42
GPU: remote
Runtime: 6.523119 seconds
Status: success

Inputs:
- cached `Qwen/Qwen2.5-1.5B-Instruct`

Outputs:
- `outputs/pruned_models/qwen2p5_1p5b_random_mask_10p/mask_config.json`
- `outputs/pruned_models/qwen2p5_1p5b_random_mask_10p/masks.json`
- `outputs/pruned_models/qwen2p5_1p5b_random_mask_10p/masks.pt`
- `outputs/runs/20260617_021524_m5_random_mask_10p_smoke/`

Metrics:

```json
{
  "actual_ratio": 0.1,
  "dry_run": false,
  "generated_new_tokens": 16,
  "generation_success": true,
  "method": "random",
  "num_kept_units": 225792,
  "num_masked_modules": 28,
  "num_pruned_units": 25088,
  "requested_ratio": 0.1,
  "total_units": 250880
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading, pruning, and generation.
- The mask artifact reports 28 Qwen MLP groups, each with intermediate size 8960, for 250880 total coupled FFN units.
- M5 pass criteria were met: model loaded, generation produced 16 new tokens, exact 10% global mask ratio was applied, and no shape errors were reported.

## Run: M6_REMOTE_SMOKE_PENDING_20260617

Date: 2026-06-17
Milestone: M6
Purpose: Remote validation for random, magnitude, and activation coupled FFN importance scoring.
Command:

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

Config file: generated at `outputs/runs/*_m6_*_score_smoke/config.yaml` when run remotely
Git commit: pending
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Dataset: `data/processed/hh_rlhf_calib.jsonl` for activation, none for random/magnitude
Seed: 42
GPU: remote
Runtime: pending
Status: remote_pending

Inputs:
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- `data/processed/hh_rlhf_calib.jsonl` for activation

Outputs:
- `outputs/scores/qwen2p5_1p5b_random_smoke.json`
- `outputs/scores/qwen2p5_1p5b_magnitude_smoke.json`
- `outputs/scores/qwen2p5_1p5b_activation_smoke.json`
- `outputs/runs/*_m6_random_score_smoke/`
- `outputs/runs/*_m6_magnitude_score_smoke/`
- `outputs/runs/*_m6_activation_score_smoke/`

Metrics:

```json
{}
```

Notes:
- Execute remotely only. These commands load Qwen and activation scoring runs model forward passes.
- M6 remains blocked until remote runs report finite scores for all 250880 coupled FFN units, exact 10% selected masks, and different selected masks across random/magnitude/activation.
