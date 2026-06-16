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
