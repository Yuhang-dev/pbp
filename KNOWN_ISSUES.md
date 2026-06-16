# KNOWN ISSUES

## Active

- The repository contains code for later milestone surfaces (`prepare_hh_rlhf.py`, log-prob scripts, margin scripts, metrics, and placeholders) that was created before `AGENT_IMPLEMENTATION_PROTOCOL.md` was applied. These are not marked as protocol-passed and must be revalidated milestone by milestone.
- Local conda Python does not have `torch`; one toy inference unit test is skipped locally. Real model tests must run remotely.
- The workspace is not currently a git repository, so run metadata records `git_commit: null`.
- The local environment has no bare `python` command on PATH. Local checks use `D:\anaconda3\python.exe`.

## Resolved

- The protocol now explicitly states that local execution is limited to lightweight tests and that all HH-RLHF downloads, Qwen loading, GPU inference, pruning, and evaluations are remote-only.
