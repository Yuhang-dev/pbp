# PROGRESS

## Current Milestone
Milestone: M9
Status: in_progress
Last updated: 2026-06-17 15:35

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
- [ ] M9: pilot experiment table

## Current Blockers
- M9 remote 1k pilot table has not been run yet.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run the M9 remote pilot commands in `EXPERIMENTS.md`, then paste back the table and run metrics so M9 can be marked passed or failed.
