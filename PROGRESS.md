# PROGRESS

## Current Milestone
Milestone: M12
Status: implementation ready; remote hybrid alpha sweep pending
Last updated: 2026-06-17 23:20

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

## Current Blockers
- M11A found matched utility only at 2% layerwise pruning. Activation 2% had the lowest `BCR@q25` among matched settings, so M12 must test whether hybrid utility-boundary scores can reduce BCR below the corresponding utility-only baseline under matched utility.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run M12 only on the remote GPU: compose hybrid scores from existing activation/general_taylor and boundary_taylor_weighted score artifacts, sweep alpha at 2%, 3%, and 5%, then run 5k HH-RLHF BCR only for matched candidates and their corresponding utility baselines. Do not run M13, 3B/7B, PAT, Wanda, DPO/LoRA recovery, safety datasets, UltraFeedback, or work beyond M12.
