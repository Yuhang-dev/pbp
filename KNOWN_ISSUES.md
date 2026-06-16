# KNOWN ISSUES

## Active

- The repository contains code for later milestone surfaces (`prepare_hh_rlhf.py`, log-prob scripts, margin scripts, metrics, and placeholders) that was created before `AGENT_IMPLEMENTATION_PROTOCOL.md` was applied. These are not marked as protocol-passed and must be revalidated milestone by milestone.
- Local conda Python does not have `torch`; torch-dependent unit tests are skipped locally. Real model and forward-mask tests must run remotely or in a torch-enabled environment.
- The local environment has no bare `python` command on PATH. Local checks use `D:\anaconda3\python.exe`.
- Remote environment emitted `libgomp: Invalid value for environment variable OMP_NUM_THREADS`; M2 completed successfully, but the env var should be fixed before larger runs.
- Transformers emitted a non-fatal `torch_dtype` deprecation warning; switch to `dtype` in a later cleanup.
- M5 real Qwen mask-pruning smoke is remote pending. M5 cannot be marked passed until the remote run confirms model loading, generation success, exact mask ratio, and no shape errors.

## Resolved

- The protocol now explicitly states that local execution is limited to lightweight tests and that all HH-RLHF downloads, Qwen loading, GPU inference, pruning, and evaluations are remote-only.
- M1 real HH-RLHF smoke completed remotely in `outputs/runs/20260617_012943_m1_hh_rlhf_smoke` with 100 calibration and 200 evaluation records.
- The workspace is now a git repository with `main` tracking `origin/main`.
- M2 real Qwen logprob smoke completed remotely in `outputs/runs/20260617_014506_m2_logprob_smoke` with 5 examples and finite response-only logprobs.
- M3 real dense/base margin smoke completed remotely in `outputs/runs/20260617_015131_m3_dense_margin_smoke` with 20 examples and finite dense margins.
- M4 real Coverage@tau report completed remotely in `outputs/runs/20260617_020235_m4_coverage_smoke` with valid coverage metrics and histogram output.
