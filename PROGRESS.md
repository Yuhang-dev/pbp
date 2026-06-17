# PROGRESS

## Current Milestone
Milestone: M11A
Status: implementation ready; remote smoke pending
Last updated: 2026-06-17 21:45

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
- M11A has not yet been run on the remote GPU. Current empirical blocker for the research claim: global pruning did not find matched utility at 10% or 20%.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run M11A only on the remote GPU: first the random layerwise 2% smoke, then Priority 1 layerwise grid if smoke passes. Do not run M11B, 3B/7B, PAT, Wanda, DPO/LoRA recovery, safety datasets, UltraFeedback, or work beyond M11A.
