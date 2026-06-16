# RUNS

| run_id | date | milestone | command | status | output_dir | key_metric |
|---|---|---|---|---|---|---|
| 20260616_223733_m0_run_dir_helper_smoke | 2026-06-16 22:37 | M0 | `D:\anaconda3\python.exe -c <pbp.logging_utils smoke>` | success | `outputs/runs/20260616_223733_m0_run_dir_helper_smoke` | `run_directory_helper_created=true` |
| M0_LOCAL_CHECKS_20260616_2237 | 2026-06-16 22:37 | M0 | `D:\anaconda3\python.exe -m pytest tests/`; `D:\anaconda3\python.exe -m compileall src scripts` | success | local terminal only | `10 passed, 1 skipped; compileall_success=true` |
| 20260617_004902_m1_hh_rlhf_fixture_smoke | 2026-06-17 00:49 | M1 | `D:\anaconda3\python.exe scripts\prepare_hh_rlhf.py --input-jsonl tests\fixtures\hh_rlhf_raw_fixture.jsonl ...` | success | `outputs/runs/20260617_004902_m1_hh_rlhf_fixture_smoke` | `calib=2 eval=2 disjoint=true` |
| 20260617_012943_m1_hh_rlhf_smoke | 2026-06-17 01:29 | M1 | `python scripts/prepare_hh_rlhf.py --dataset Anthropic/hh-rlhf --calib-size 100 --eval-size 200 ...` | success | `outputs/runs/20260617_012943_m1_hh_rlhf_smoke` | `calib=100 eval=200 disjoint=true` |
| 20260617_013822_m2_logprob_dry_run | 2026-06-17 01:38 | M2 | `D:\anaconda3\python.exe scripts\compute_logprobs.py --model dry-run-model ... --dry-run` | success | `outputs/runs/20260617_013822_m2_logprob_dry_run` | `examples=2 finite=true` |
| 20260617_014506_m2_logprob_smoke | 2026-06-17 01:45 | M2 | `python scripts/compute_logprobs.py --model Qwen/Qwen2.5-1.5B-Instruct --max-samples 5 ...` | success | `outputs/runs/20260617_014506_m2_logprob_smoke` | `examples=5 finite=true` |
| 20260617_014918_m3_dense_margin_dry_run | 2026-06-17 01:49 | M3 | `D:\anaconda3\python.exe scripts\compute_dense_margins.py --instruct-model dense-dry-run ... --dry-run` | success | `outputs/runs/20260617_014918_m3_dense_margin_dry_run` | `examples=2 delta_finite=true` |
| 20260617_015131_m3_dense_margin_smoke | 2026-06-17 01:51 | M3 | `python scripts/compute_dense_margins.py --instruct-model Qwen/Qwen2.5-1.5B-Instruct --base-model Qwen/Qwen2.5-1.5B --max-samples 20 ...` | success | `outputs/runs/20260617_015131_m3_dense_margin_smoke` | `examples=20 delta_finite=true` |
| 20260617_015536_m4_coverage_fixture | 2026-06-17 01:55 | M4 | `D:\anaconda3\python.exe scripts\report_coverage.py --dense-margins tests\fixtures\dense_margins_fixture.jsonl ...` | success | `outputs/runs/20260617_015536_m4_coverage_fixture` | `coverage_at_0=0.75 coverage_at_q25=0.5` |
| M4_REMOTE_COVERAGE_SMOKE_PENDING | 2026-06-17 01:55 | M4 | `python scripts/report_coverage.py --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl ...` | remote_pending | `outputs/runs/*_m4_coverage_smoke` | pending |
