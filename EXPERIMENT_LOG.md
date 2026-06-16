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
