# KNOWN ISSUES

## Active

- The repository contains code for later milestone surfaces (`prepare_hh_rlhf.py`, log-prob scripts, margin scripts, metrics, and placeholders) that was created before `AGENT_IMPLEMENTATION_PROTOCOL.md` was applied. These are not marked as protocol-passed and must be revalidated milestone by milestone.
- Local conda Python does not have `torch`; one toy inference unit test is skipped locally. Real model tests must run remotely.
- The local environment has no bare `python` command on PATH. Local checks use `D:\anaconda3\python.exe`.
- Remote environment emitted `libgomp: Invalid value for environment variable OMP_NUM_THREADS`; M2 completed successfully, but the env var should be fixed before larger runs.
- Transformers emitted a non-fatal `torch_dtype` deprecation warning; switch to `dtype` in a later cleanup.
- M4 real Coverage@tau report is pending remote execution. Local fixture reporting passed, but `outputs/evals/coverage_qwen2p5_1p5b_smoke.json` has not been produced remotely yet.

## Resolved

- The protocol now explicitly states that local execution is limited to lightweight tests and that all HH-RLHF downloads, Qwen loading, GPU inference, pruning, and evaluations are remote-only.
- M1 real HH-RLHF smoke completed remotely in `outputs/runs/20260617_012943_m1_hh_rlhf_smoke` with 100 calibration and 200 evaluation records.
- The workspace is now a git repository with `main` tracking `origin/main`.
- M2 real Qwen logprob smoke completed remotely in `outputs/runs/20260617_014506_m2_logprob_smoke` with 5 examples and finite response-only logprobs.
- M3 real dense/base margin smoke completed remotely in `outputs/runs/20260617_015131_m3_dense_margin_smoke` with 20 examples and finite dense margins.
