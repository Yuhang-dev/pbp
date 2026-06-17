# PROGRESS

## Current Milestone
Milestone: M10B
Status: implementation ready; remote validation pending
Last updated: 2026-06-17 20:55

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

## Current Blockers
- M10B has not yet been run on the remote GPU. Local validation remains limited to syntax/static checks.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run M10B only on the remote GPU: clean stale M10A run statuses, evaluate larger general utility for dense plus all M9 10%/20% pruned models, report layer-wise mask distribution, and summarize matched utility. Do not run 3B/7B, DPO, LoRA, post-pruning recovery, M11, or any work beyond M10B.
