# KNOWN ISSUES

## Active

- The repository contains code for later milestone surfaces (`prepare_hh_rlhf.py`, log-prob scripts, margin scripts, metrics, and placeholders) that was created before `AGENT_IMPLEMENTATION_PROTOCOL.md` was applied. These are not marked as protocol-passed and must be revalidated milestone by milestone.
- Local conda Python does not have `torch`, and local execution is no longer used for functional validation. Real tests, model execution, and forward-mask checks must run remotely.
- The local environment has no bare `python` command on PATH. Syntax/static checks use `D:\anaconda3\python.exe`.
- Remote environment emitted `libgomp: Invalid value for environment variable OMP_NUM_THREADS`; M2 completed successfully, but the env var should be fixed before larger runs.
- M10B showed no matched-utility setting under global coupled-FFN pruning at 10% or 20%. Global selection can over-prune early layers, so M11A introduces layer-wise and protected layer-wise selection to find a utility-preserving pruning regime.

## Resolved

- The protocol now explicitly states that local execution is limited to syntax/static sanity checks and that all functional validation, HH-RLHF downloads, Qwen loading, GPU inference, pruning, and evaluations are remote-only.
- M1 real HH-RLHF smoke completed remotely in `outputs/runs/20260617_012943_m1_hh_rlhf_smoke` with 100 calibration and 200 evaluation records.
- The workspace is now a git repository with `main` tracking `origin/main`.
- M2 real Qwen logprob smoke completed remotely in `outputs/runs/20260617_014506_m2_logprob_smoke` with 5 examples and finite response-only logprobs.
- M3 real dense/base margin smoke completed remotely in `outputs/runs/20260617_015131_m3_dense_margin_smoke` with 20 examples and finite dense margins.
- M4 real Coverage@tau report completed remotely in `outputs/runs/20260617_020235_m4_coverage_smoke` with valid coverage metrics and histogram output.
- M5 real Qwen random mask-pruning smoke completed remotely in `outputs/runs/20260617_021524_m5_random_mask_10p_smoke` with exact 10% global mask ratio and successful generation.
- M6 random/magnitude/activation scoring smoke completed remotely in `outputs/runs/20260617_023019_m6_random_score_smoke`, `outputs/runs/20260617_023027_m6_magnitude_score_smoke`, and `outputs/runs/20260617_023036_m6_activation_score_smoke` with finite scores for all 250880 coupled FFN units and different selected masks across methods.
- M7 BCR smoke completed remotely in `outputs/runs/20260617_024441_m7_bcr_dense_self_smoke` and `outputs/runs/20260617_024502_m7_bcr_random_10p_smoke`; dense-self BCR is zero and random 10% masked pruning reports finite BCR metrics.
- M8 boundary-aware Taylor smoke completed remotely in `outputs/runs/20260617_145653_m8_boundary_taylor_smoke` with finite non-zero scores for all 250880 coupled FFN units and a selected mask different from activation pruning.
- Model loading now passes `dtype=` to Transformers instead of deprecated `torch_dtype=`.
- The RTX PRO 6000 Blackwell remote was upgraded from an incompatible `torch 2.3.0+cu121` environment to a CUDA 13-compatible PyTorch stack before completing M9.
- M9 boundary Taylor OOMs were resolved by precomputing calibration dense margins, capping successful Taylor scoring at `max_length=2048`, and splitting chosen/rejected Taylor backward passes with `response_micro_batch_size=1`. Runs with `max_length > 2048` OOMed on the `1 x NVIDIA RTX PRO 6000 96GB` remote.
- M9 remote 1k pilot table completed successfully with 8 rows in `outputs/tables/m9_qwen2p5_1p5b_pilot_1k.csv`.
- M10A first dense general-utility attempt failed because current `datasets`/`huggingface_hub` rejects the bare `wikitext` dataset ID. The fix switched to `Salesforce/wikitext`, after which M10A completed successfully.
- M10A remote 20% matched-utility table completed successfully with 5 rows in `outputs/tables/m10a_matched_utility_20p.csv`. All final general-utility metrics were finite, but all 20% pruned models had `matched_utility_flag=false` under the configured thresholds.
