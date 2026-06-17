# PROGRESS

## Current Milestone
Milestone: M10A
Status: remote commands ready; remote validation pending
Last updated: 2026-06-17 19:45

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

## Current Blockers
- M10A first dense run failed on the remote because bare dataset ID `wikitext` is rejected by the current `datasets`/`huggingface_hub` stack. Fix prepared: use `Salesforce/wikitext`, re-cache datasets, then rerun M10A.

## Execution Boundary
- Local machine: syntax/static checks only, such as `compileall`, `py_compile`, file-existence checks, and command/config drafting.
- Remote machine: all functional validation, tests, smoke runs, HH-RLHF downloads, Qwen model loading, GPU inference, pruning, and evaluation experiments.

## Next Action
Run M10A only on the remote GPU: lightweight general-utility evaluation for dense Qwen2.5-1.5B-Instruct and the 20% M9 masked pruned models. Do not run 10%, 3B/7B, DPO, LoRA, or post-pruning recovery.
