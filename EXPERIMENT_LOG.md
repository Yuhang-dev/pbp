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

## Run: 20260617_023019_m6_random_score_smoke

Date: 2026-06-17 02:30
Milestone: M6
Purpose: Remote random coupled FFN importance scoring smoke test.
Command: `python scripts/score_pruning_importance.py --model Qwen/Qwen2.5-1.5B-Instruct --method random --ratio 0.10 --out outputs/scores/qwen2p5_1p5b_random_smoke.json --dtype bfloat16 --cache-dir "$HF_HUB_CACHE" --local-files-only --run-name m6_random_score_smoke`
Config file: `outputs/runs/20260617_023019_m6_random_score_smoke/config.yaml`
Git commit: `7cbeb5c`
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Dataset: none
Seed: 42
GPU: remote
Runtime: 5.791209 seconds
Status: success

Inputs:
- cached `Qwen/Qwen2.5-1.5B-Instruct`

Outputs:
- `outputs/scores/qwen2p5_1p5b_random_smoke.json`
- `outputs/runs/20260617_023019_m6_random_score_smoke/`

Metrics:

```json
{
  "actual_ratio": 0.1,
  "max_samples": null,
  "max_score": 0.9999978438570691,
  "mean_score": 0.5004347969936106,
  "method": "random",
  "min_score": 6.402053533083318e-06,
  "num_groups": 28,
  "num_pruned_units": 25088,
  "num_scores": 250880,
  "requested_ratio": 0.1,
  "scores_finite": true,
  "std_score": 0.2889341039910362,
  "total_units": 250880
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading.
- Produced finite scores for all 250880 coupled FFN units and selected 25088 units for a 10% pruning mask.

## Run: 20260617_023027_m6_magnitude_score_smoke

Date: 2026-06-17 02:30
Milestone: M6
Purpose: Remote magnitude coupled FFN importance scoring smoke test.
Command: `python scripts/score_pruning_importance.py --model Qwen/Qwen2.5-1.5B-Instruct --method magnitude --ratio 0.10 --out outputs/scores/qwen2p5_1p5b_magnitude_smoke.json --dtype bfloat16 --cache-dir "$HF_HUB_CACHE" --local-files-only --run-name m6_magnitude_score_smoke`
Config file: `outputs/runs/20260617_023027_m6_magnitude_score_smoke/config.yaml`
Git commit: `7cbeb5c`
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Dataset: none
Seed: 42
GPU: remote
Runtime: 6.163203 seconds
Status: success

Inputs:
- cached `Qwen/Qwen2.5-1.5B-Instruct`

Outputs:
- `outputs/scores/qwen2p5_1p5b_magnitude_smoke.json`
- `outputs/runs/20260617_023027_m6_magnitude_score_smoke/`

Metrics:

```json
{
  "actual_ratio": 0.1,
  "max_samples": null,
  "max_score": 0.5416341423988342,
  "mean_score": 0.08382441661196192,
  "method": "magnitude",
  "min_score": 0.03869392350316048,
  "num_groups": 28,
  "num_pruned_units": 25088,
  "num_scores": 250880,
  "requested_ratio": 0.1,
  "scores_finite": true,
  "std_score": 0.0083053994296318,
  "total_units": 250880
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading.
- Produced finite scores for all 250880 coupled FFN units and selected 25088 units for a 10% pruning mask.

## Run: 20260617_023036_m6_activation_score_smoke

Date: 2026-06-17 02:30
Milestone: M6
Purpose: Remote activation coupled FFN importance scoring smoke test.
Command: `python scripts/score_pruning_importance.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/processed/hh_rlhf_calib.jsonl --method activation --max-samples 50 --ratio 0.10 --out outputs/scores/qwen2p5_1p5b_activation_smoke.json --dtype bfloat16 --batch-size 1 --max-length 1024 --cache-dir "$HF_HUB_CACHE" --local-files-only --run-name m6_activation_score_smoke`
Config file: `outputs/runs/20260617_023036_m6_activation_score_smoke/config.yaml`
Git commit: `7cbeb5c`
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Dataset: `data/processed/hh_rlhf_calib.jsonl`
Seed: 42
GPU: remote
Runtime: 10.205065 seconds
Status: success

Inputs:
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- `data/processed/hh_rlhf_calib.jsonl`

Outputs:
- `outputs/scores/qwen2p5_1p5b_activation_smoke.json`
- `outputs/runs/20260617_023036_m6_activation_score_smoke/`

Metrics:

```json
{
  "actual_ratio": 0.1,
  "batch_size": 1,
  "max_length": 1024,
  "max_samples": 50,
  "max_score": 38.02402114868164,
  "mean_score": 0.12280862410650548,
  "method": "activation",
  "min_score": 0.0003806173917837441,
  "num_calibration_pairs": 50,
  "num_calibration_texts": 100,
  "num_groups": 28,
  "num_pruned_units": 25088,
  "num_scores": 250880,
  "requested_ratio": 0.1,
  "scores_finite": true,
  "std_score": 0.23592877354164266,
  "text_mode": "chosen_rejected",
  "total_units": 250880
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading and GPU inference.
- Produced finite scores for all 250880 coupled FFN units and selected 25088 units for a 10% pruning mask.
- The remote mask comparison printed `random 25088 250880 True`, `magnitude 25088 250880 True`, and `activation 25088 250880 True`; the comparison assertions did not fail, so selected masks differ across methods.

## Run: 20260617_024441_m7_bcr_dense_self_smoke

Date: 2026-06-17 02:44
Milestone: M7
Purpose: Remote dense self BCR sanity smoke test.
Command: `python scripts/evaluate_bcr.py --model Qwen/Qwen2.5-1.5B-Instruct --base-model Qwen/Qwen2.5-1.5B --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl --data data/processed/hh_rlhf_eval.jsonl --max-samples 20 --out outputs/evals/bcr_dense_self_smoke.json --records-out outputs/evals/bcr_dense_self_smoke_records.jsonl --dtype bfloat16 --batch-size 1 --cache-dir "$HF_HUB_CACHE" --local-files-only --run-name m7_bcr_dense_self_smoke`
Config file: `outputs/runs/20260617_024441_m7_bcr_dense_self_smoke/config.yaml`
Git commit: `c50f845`
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Reference: `Qwen/Qwen2.5-1.5B`
Dataset: `data/processed/hh_rlhf_eval.jsonl`
Seed: 42
GPU: remote
Runtime: 11.083842 seconds
Status: success

Inputs:
- `outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl`
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- cached `Qwen/Qwen2.5-1.5B`

Outputs:
- `outputs/evals/bcr_dense_self_smoke.json`
- `outputs/evals/bcr_dense_self_smoke_records.jsonl`
- `outputs/runs/20260617_024441_m7_bcr_dense_self_smoke/`

Metrics:

```json
{
  "bcr_at_0": 0.0,
  "bcr_at_q25": 0.0,
  "bcr_at_q50": 0.0,
  "bcr_at_q75": 0.0,
  "coverage_at_0": 0.6,
  "coverage_at_q25": 0.45,
  "mean_margin_drop": 0.0,
  "metrics_finite": true,
  "num_examples": 20,
  "preference_accuracy_dense": 0.6,
  "preference_accuracy_pruned": 0.6
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading.
- Dense self sanity passed: all BCR thresholds are zero and mean margin drop is zero.

## Run: 20260617_024502_m7_bcr_random_10p_smoke

Date: 2026-06-17 02:45
Milestone: M7
Purpose: Remote BCR smoke test for M5 random 10% masked pruning.
Command: `python scripts/evaluate_bcr.py --model outputs/pruned_models/qwen2p5_1p5b_random_mask_10p --base-model Qwen/Qwen2.5-1.5B --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl --data data/processed/hh_rlhf_eval.jsonl --max-samples 20 --out outputs/evals/bcr_random_10p_smoke.json --records-out outputs/evals/bcr_random_10p_smoke_records.jsonl --dtype bfloat16 --batch-size 1 --cache-dir "$HF_HUB_CACHE" --local-files-only --run-name m7_bcr_random_10p_smoke`
Config file: `outputs/runs/20260617_024502_m7_bcr_random_10p_smoke/config.yaml`
Git commit: `c50f845`
Model: masked `Qwen/Qwen2.5-1.5B-Instruct`
Reference: `Qwen/Qwen2.5-1.5B`
Dataset: `data/processed/hh_rlhf_eval.jsonl`
Seed: 42
GPU: remote
Runtime: 10.672482 seconds
Status: success

Inputs:
- `outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl`
- `outputs/pruned_models/qwen2p5_1p5b_random_mask_10p/mask_config.json`
- `outputs/pruned_models/qwen2p5_1p5b_random_mask_10p/masks.json`
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- cached `Qwen/Qwen2.5-1.5B`

Outputs:
- `outputs/evals/bcr_random_10p_smoke.json`
- `outputs/evals/bcr_random_10p_smoke_records.jsonl`
- `outputs/runs/20260617_024502_m7_bcr_random_10p_smoke/`

Metrics:

```json
{
  "bcr_at_0": 0.08333333333333333,
  "bcr_at_q25": 0.0,
  "bcr_at_q50": 0.0,
  "bcr_at_q75": 0.0,
  "coverage_at_0": 0.6,
  "coverage_at_q25": 0.45,
  "mask_actual_ratio": 0.1,
  "mask_num_pruned_units": 25088,
  "mask_total_units": 250880,
  "mean_margin_drop": -0.08233094341192532,
  "metrics_finite": true,
  "num_examples": 20,
  "preference_accuracy_dense": 0.6,
  "preference_accuracy_pruned": 0.7
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading, pruning, and GPU inference.
- Coverage matches the M4 20-example smoke output: `coverage_at_0=0.6`, `coverage_at_q25=0.45`, `coverage_at_q50=0.3`, `coverage_at_q75=0.15`.
- Random 10% masked pruning has finite BCR metrics and exact mask ratio.

## Run: 20260617_145653_m8_boundary_taylor_smoke

Date: 2026-06-17 14:56
Milestone: M8
Purpose: Remote boundary-aware Taylor scoring smoke validation.
Command: `python scripts/score_pruning_importance.py --instruct-model Qwen/Qwen2.5-1.5B-Instruct --base-model Qwen/Qwen2.5-1.5B --data data/processed/hh_rlhf_calib.jsonl --method boundary_taylor_weighted --max-samples 20 --tau-mode q25 --ratio 0.10 --out outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_smoke.json --dtype bfloat16 --batch-size 1 --cache-dir "$HF_HUB_CACHE" --local-files-only --run-name m8_boundary_taylor_smoke`
Config file: `outputs/runs/20260617_145653_m8_boundary_taylor_smoke/config.yaml`
Git commit: `e9e4918`
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Reference: `Qwen/Qwen2.5-1.5B`
Dataset: `data/processed/hh_rlhf_calib.jsonl`
Seed: 42
GPU: remote
Runtime: 13.717101 seconds
Status: success

Inputs:
- `data/processed/hh_rlhf_calib.jsonl`
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- cached `Qwen/Qwen2.5-1.5B`
- `outputs/scores/qwen2p5_1p5b_activation_smoke.json` for mask comparison

Outputs:
- `outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_smoke.json`
- `outputs/runs/20260617_145653_m8_boundary_taylor_smoke/`

Metrics:

```json
{
  "actual_ratio": 0.1,
  "all_scores_zero": false,
  "batch_size": 1,
  "max_samples": 20,
  "max_score": 0.1306699961423874,
  "mean_score": 0.001690736932569507,
  "method": "boundary_taylor_weighted",
  "min_score": 1.4469124835159164e-05,
  "num_calibration_pairs": 20,
  "num_groups": 28,
  "num_nonzero_scores": 250880,
  "num_pruned_units": 25088,
  "num_scores": 250880,
  "num_selected_pairs": 9,
  "requested_ratio": 0.1,
  "score_transform": "boundary_taylor_weighted",
  "scores_finite": true,
  "selected_fraction": 0.45,
  "std_score": 0.0011907935751317327,
  "tau_calib": 0.12528753158359246,
  "tau_mode": "q25",
  "taylor_objective": "delta_margin",
  "total_units": 250880,
  "weight_max": 3.347670805869662,
  "weight_mean": 0.9999999999999998,
  "weight_min": 0.0658627126318743
}
```

Notes:
- Executed remotely because the protocol forbids local Qwen model loading and gradient-based scoring.
- Selected 9 of 20 calibration pairs above `tau_calib=0.12528753158359246` for boundary Taylor scoring.
- The remote mask comparison printed `boundary 25088 250880 True False` and `activation 25088 250880 True`; the comparison assertions did not fail, so selected masks differ from activation pruning.

## Run: M9_REMOTE_PLAN_20260617_1535

Date: 2026-06-17 15:35
Milestone: M9
Purpose: Prepared the remote-only Qwen2.5-1.5B 1k pilot table command plan and implementation support.
Command: See `EXPERIMENTS.md` section `M9 Remote Pilot Table`.
Config file: `configs/m9_pilot_qwen2p5_1p5b.yaml`
Git commit: pending
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Reference: `Qwen/Qwen2.5-1.5B`
Dataset: `Anthropic/hh-rlhf`
Seed: 42
GPU: remote `2 x RTX 4090`
Runtime: not run yet
Status: partial

Inputs:
- `data/processed/m9_pilot/hh_rlhf_calib.jsonl`
- `data/processed/m9_pilot/hh_rlhf_eval.jsonl`
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- cached `Qwen/Qwen2.5-1.5B`

Planned outputs:
- `outputs/logprobs/base_qwen2p5_1p5b_m9_eval_1k.jsonl`
- `outputs/margins/dense_qwen2p5_1p5b_m9_eval_1k.jsonl`
- `outputs/scores/qwen2p5_1p5b_{random,magnitude,activation,boundary_taylor_weighted}_m9_calib_1k.json`
- `outputs/pruned_models/qwen2p5_1p5b_{method}_mask_{10p,20p}_m9/`
- `outputs/evals/bcr_qwen2p5_1p5b_{method}_{10p,20p}_m9_1k.json`
- `outputs/tables/m9_qwen2p5_1p5b_pilot_1k.csv`
- `outputs/tables/m9_qwen2p5_1p5b_pilot_1k.json`

Metrics:

```json
{
  "remote_validation": "pending",
  "expected_rows": 8,
  "methods": ["random", "magnitude", "activation", "boundary_taylor_weighted"],
  "ratios": [0.1, 0.2],
  "eval_samples": 1000
}
```

Notes:
- This is a command/config tracking entry only. The local machine was not used for functional validation.
- M9 should not be marked passed until the remote table exists and all required run directories report success.

## Run: 20260617_153128_m9_score_boundary_taylor_weighted_1k

Date: 2026-06-17 15:31
Milestone: M9
Purpose: Remote boundary Taylor weighted scoring for the 1k pilot table.
Command: `python scripts/score_pruning_importance.py --instruct-model Qwen/Qwen2.5-1.5B-Instruct --base-model Qwen/Qwen2.5-1.5B --data data/processed/m9_pilot/hh_rlhf_calib.jsonl --method boundary_taylor_weighted --max-samples 1000 --tau-mode q25 --ratio 0.10 --out outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_m9_calib_1k.json ...`
Config file: `outputs/runs/20260617_153128_m9_score_boundary_taylor_weighted_1k/config.yaml`
Git commit: `cf1b2c3`
Model: `Qwen/Qwen2.5-1.5B-Instruct`
Reference: `Qwen/Qwen2.5-1.5B`
Dataset: `data/processed/m9_pilot/hh_rlhf_calib.jsonl`
Seed: 42
GPU: remote
Runtime: 142.586693 seconds
Status: failed

Inputs:
- `data/processed/m9_pilot/hh_rlhf_calib.jsonl`
- cached `Qwen/Qwen2.5-1.5B-Instruct`
- cached `Qwen/Qwen2.5-1.5B`

Outputs:
- No score artifact was produced.

Metrics:

```json
{}
```

Notes:
- Failure reason: CUDA OOM while boundary scoring computed calibration dense margins inline with `--base-model`.
- Follow-up fix: precompute calibration dense margins with `scripts/compute_dense_margins.py`, then rerun boundary scoring with `--dense-margins outputs/margins/dense_qwen2p5_1p5b_m9_calib_1k.jsonl`.
- The failed boundary apply/evaluate/summarize runs after this are downstream failures from the missing boundary score artifact.
