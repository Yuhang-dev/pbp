# PROGRESS

## Current Milestone
Milestone: M10B
Status: passed
Last updated: 2026-06-17 21:15

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

## Current Blockers
- None for M10B. Current empirical blocker for the research claim: no 10% or 20% M9 pruned model satisfies the configured matched-utility thresholds.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Stop and wait for explicit approval before running M11, 3B/7B scaling, DPO, LoRA, post-pruning recovery, new pruning criteria, or additional ablations.
