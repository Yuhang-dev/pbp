# PROGRESS

## Current Milestone
Milestone: M8
Status: blocked
Last updated: 2026-06-17 14:36

## Completed Milestones
- [x] M0: Repository skeleton
- [x] M1: HH-RLHF preprocessing
- [x] M2: response-only logprob computation
- [x] M3: dense/base margin computation
- [x] M4: Coverage@τ reporting
- [x] M5: mask-based coupled FFN pruning
- [x] M6: basic pruning baselines
- [x] M7: BCR evaluation for pruned model
- [ ] M8: boundary-aware Taylor scoring
- [ ] M9: pilot experiment table

## Current Blockers
- M8 implementation is present, but remote boundary-aware Taylor smoke validation is pending. M8 cannot be marked passed until the remote run confirms finite non-zero scores and selected pruning mask differs from activation pruning.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run the M8 remote smoke command from `EXPERIMENTS.md`, then update M8 to passed if remote metrics and mask comparison succeed.
