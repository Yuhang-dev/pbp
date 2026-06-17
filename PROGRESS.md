# PROGRESS

## Current Milestone
Milestone: M12
Status: completed; 5k BCR validation analyzed
Last updated: 2026-06-18 01:20

## Completed Milestones
- [x] M0: Repository skeleton
- [x] M1: HH-RLHF preprocessing
- [x] M2: response-only logprob computation
- [x] M3: dense/base margin computation
- [x] M4: Coverage@τ reporting
- [x] M5: mask-based coupled FFN pruning
- [x] M6: basic pruning baselines
- [x] M7: BCR evaluation for pruned model
- [x] M8: boundary-aware Taylor scoring
- [x] M9: pilot experiment table
- [x] M10A: lightweight matched-utility check for 20% M9 pruned models
- [x] M10B: larger matched-utility check and mask distribution for all M9 pruned models
- [x] M11A: layerwise mild pruning regime and utility/BCR sweep
- [x] M12: hybrid utility-boundary pruning alpha sweep with 5k HH-RLHF BCR validation

## Current Blockers
- M12B 5k validation is mixed. `general_taylor_boundary` consistently lowers `BCR@q25` versus `general_taylor` at 2% and 3% under matched utility. `activation_boundary` only slightly improves activation at 3% (`0.01860` vs `0.01911`) and does not beat activation at 2%. The strongest absolute matched 5k `BCR@q25` remains activation 2% (`0.00981`), while the best hybrid is `general_taylor_boundary` 2% alpha 0.25 (`0.01033`).

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Stop after M12B and decide the next experiment. Do not run M13, 3B/7B, PAT, Wanda, DPO/LoRA recovery, safety datasets, UltraFeedback, or work beyond M12 without explicit approval.
