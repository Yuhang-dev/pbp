# PROGRESS

## Current Milestone
Milestone: M7
Status: blocked
Last updated: 2026-06-17 02:37

## Completed Milestones
- [x] M0: Repository skeleton
- [x] M1: HH-RLHF preprocessing
- [x] M2: response-only logprob computation
- [x] M3: dense/base margin computation
- [x] M4: Coverage@τ reporting
- [x] M5: mask-based coupled FFN pruning
- [x] M6: basic pruning baselines
- [ ] M7: BCR evaluation for pruned model
- [ ] M8: boundary-aware Taylor scoring
- [ ] M9: pilot experiment table

## Current Blockers
- M7 implementation is present, but remote BCR smoke validation is pending. M7 cannot be marked passed until the remote run reports finite BCR metrics and a dense-vs-dense sanity check gives zero BCR.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run the M7 remote smoke commands from `EXPERIMENTS.md`, then update M7 to passed if remote metrics succeed.
