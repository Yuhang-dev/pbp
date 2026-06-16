# PROGRESS

## Current Milestone
Milestone: M6
Status: blocked
Last updated: 2026-06-17 02:22

## Completed Milestones
- [x] M0: Repository skeleton
- [x] M1: HH-RLHF preprocessing
- [x] M2: response-only logprob computation
- [x] M3: dense/base margin computation
- [x] M4: Coverage@τ reporting
- [x] M5: mask-based coupled FFN pruning
- [ ] M6: basic pruning baselines
- [ ] M7: BCR evaluation for pruned model
- [ ] M8: boundary-aware Taylor scoring
- [ ] M9: pilot experiment table

## Current Blockers
- M6 implementation is present, but remote random/magnitude/activation scoring smoke validation is pending. M6 cannot be marked passed until the remote runs confirm finite scores for all coupled FFN units and different selected masks across methods.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run the M6 remote smoke commands from `EXPERIMENTS.md`, then update M6 to passed if remote metrics and mask comparison succeed.
