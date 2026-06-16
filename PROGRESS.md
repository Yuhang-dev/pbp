# PROGRESS

## Current Milestone
Milestone: M1
Status: in_progress
Last updated: 2026-06-17 00:47

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
- Remote HH-RLHF smoke run is pending. Local fixture preprocessing passed, but real dataset download/preprocessing must run on the remote machine.

## Execution Boundary
- Local machine: lightweight tests, compile checks, fixture tests, import checks, run-directory validation only.
- Remote machine: HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run the M1 remote smoke command on the remote machine and provide the run output/logs. After remote success, mark M1 passed.
