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
