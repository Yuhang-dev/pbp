# PROGRESS

## Current Milestone
Milestone: M5
Status: blocked
Last updated: 2026-06-17 02:16

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
- M5 implementation is present, but real Qwen mask-pruning smoke is remote pending. M5 cannot be marked passed until the remote run confirms model loading, generation, exact mask counts, and no shape errors.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run the M5 remote smoke command from `EXPERIMENTS.md`, then update M5 to passed if the remote metrics and status files are successful.
