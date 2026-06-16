# RUNS

| run_id | date | milestone | command | status | output_dir | key_metric |
|---|---|---|---|---|---|---|
| 20260616_223733_m0_run_dir_helper_smoke | 2026-06-16 22:37 | M0 | `D:\anaconda3\python.exe -c <pbp.logging_utils smoke>` | success | `outputs/runs/20260616_223733_m0_run_dir_helper_smoke` | `run_directory_helper_created=true` |
| M0_LOCAL_CHECKS_20260616_2237 | 2026-06-16 22:37 | M0 | `D:\anaconda3\python.exe -m pytest tests/`; `D:\anaconda3\python.exe -m compileall src scripts` | success | local terminal only | `10 passed, 1 skipped; compileall_success=true` |
