# PROGRESS

## Current Milestone
Milestone: M0
Status: passed
Last updated: 2026-06-16 22:37

## Completed Milestones
- [x] M0: Repository skeleton
- [ ] M1: HH-RLHF preprocessing
- [ ] M2: response-only logprob computation
- [ ] M3: dense/base margin computation
- [ ] M4: Coverage@τ reporting
- [ ] M5: mask-based coupled FFN pruning
- [ ] M6: basic pruning baselines
- [ ] M7: BCR evaluation for pruned model
- [ ] M8: boundary-aware Taylor scoring
- [ ] M9: pilot experiment table

## Current Blockers
- None for M0.

## Execution Boundary
- Local machine: lightweight tests, compile checks, fixture tests, import checks, run-directory validation only.
- Remote machine: HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Wait for explicit approval to begin M1 under the protocol. Existing M1-M4-style code must be revalidated milestone by milestone before being marked passed.
