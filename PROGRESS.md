# PROGRESS

## Current Milestone
Milestone: M4
Status: passed
Last updated: 2026-06-17 02:03

## Completed Milestones
- [x] M0: Repository skeleton
- [x] M1: HH-RLHF preprocessing
- [x] M2: response-only logprob computation
- [x] M3: dense/base margin computation
- [x] M4: Coverage@τ reporting
- [ ] M5: mask-based coupled FFN pruning
- [ ] M6: basic pruning baselines
- [ ] M7: BCR evaluation for pruned model
- [ ] M8: boundary-aware Taylor scoring
- [ ] M9: pilot experiment table

## Current Blockers
- None for M4.

## Execution Boundary
- Local machine: lightweight tests, compile checks, fixture tests, import checks, run-directory validation only.
- Remote machine: HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Stop and wait for explicit approval to begin M5: mask-based coupled FFN pruning.
