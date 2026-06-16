# AGENT_IMPLEMENTATION_PROTOCOL.md

You are the implementation agent for the project described in `PROJECT_SPEC.md`.

Your job is to implement the project step by step, with strict milestone gates, reproducible experiment tracking, and explicit checks.

Do not change the research question.

Do not redesign the project.

Do not implement everything at once.

Do not run large experiments before the current milestone passes its checks.

Do not run dataset downloads, Qwen model loading, GPU inference, pruning, or evaluation experiments on the local machine. Local execution is restricted to lightweight Python tests, static checks, fixture-based smoke tests, and run-directory/logging validation. All real experiments are run remotely by the user.

---

# 0. Project Scope Reminder

The project studies:

> Preference Boundary Crossing under mild coupled SwiGLU FFN intermediate-neuron structured pruning of Qwen2.5-Instruct models.

Main experiment:

* Model: Qwen2.5-Instruct dense/pruned
* Reference: Qwen2.5-Base same size
* Dataset: HH-RLHF
* Metric: base-reference-normalized preference margin and BCR@τ
* Main pruning: coupled SwiGLU FFN intermediate-neuron pruning
* Main method: margin-drop-aware Taylor pruning
* Main setting: no post-pruning recovery

Do not introduce DPO/LoRA recovery into the main pipeline.

---

# 1. Working Rules

You must work in milestones.

For each milestone:

1. State the exact objective.
2. List files you will create or modify.
3. Implement only the current milestone.
4. Add minimal tests.
5. Run the tests.
6. Run a small smoke test. If the smoke test requires downloading HH-RLHF, loading Qwen models, GPU inference, pruning, or benchmark evaluation, prepare the remote command/config/run-log entry but do not execute it locally.
7. Save all commands, configs, logs, and outputs.
8. Update progress documents.
9. Stop and report status.

Do not continue to the next milestone until I explicitly approve.

---

# 1.1 Local and Remote Execution Policy

The local machine is only for lightweight implementation validation.

Allowed locally:

* unit tests using synthetic fixtures or tiny fake models;
* `python -m compileall`;
* import checks;
* JSONL/YAML parsing tests on hand-written fixture files;
* run-directory helper checks;
* command/config generation for remote runs.

Forbidden locally:

* downloading HH-RLHF or other large datasets;
* loading Qwen or other large Hugging Face models;
* GPU inference;
* log-probability computation on real models;
* pruning application on real models;
* benchmark evaluation;
* any run expected to consume substantial disk, memory, network, or GPU resources.

Remote machine assumptions:

```text
GPU: 2 x RTX 4090
RAM: 200 GB
System disk: 30 GB
Data disk: 50 GB
```

Remote experiments should place Hugging Face caches, datasets, model weights, and outputs on the data disk. The agent should produce reproducible remote commands and configs, but the user runs them remotely unless explicitly stated otherwise.

When a milestone definition says "Smoke command" and that command requires real data/model execution, treat it as a **remote smoke command**. For local completion of that milestone, run only the corresponding lightweight tests and record the real smoke command as `remote_pending` until the user provides remote logs or results.

---

# 2. Required Repository Tracking Files

Create and maintain these files:

```text
PROJECT_SPEC.md
AGENT_IMPLEMENTATION_PROTOCOL.md
PROGRESS.md
EXPERIMENT_LOG.md
RUNS.md
CONFIG_REGISTRY.md
KNOWN_ISSUES.md
```

## 2.1 PROGRESS.md

This file tracks implementation progress.

Format:

```markdown
# PROGRESS

## Current Milestone
Milestone: M1
Status: in_progress / passed / failed / blocked
Last updated: YYYY-MM-DD HH:MM

## Completed Milestones
- [ ] M0: Repository skeleton
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
- None

## Next Action
...
```

## 2.2 EXPERIMENT_LOG.md

This file records all experiment runs.

Each run must include:

````markdown
## Run: RUN_ID

Date:
Milestone:
Purpose:
Command:
Config file:
Git commit:
Model:
Dataset:
Seed:
GPU:
Runtime:
Status: success / failed / partial

Inputs:
- ...

Outputs:
- ...

Metrics:
```json
{}
````

Notes:

* ...

````

## 2.3 RUNS.md

This file is a compact table of runs.

```markdown
| run_id | date | milestone | command | status | output_dir | key_metric |
|---|---|---|---|---|---|
````

## 2.4 CONFIG_REGISTRY.md

Record every config file and what it controls.

```markdown
| config | purpose | model | data | seed | notes |
|---|---|---|---|---|---|
```

## 2.5 KNOWN_ISSUES.md

Record bugs, caveats, failed runs, and unresolved implementation risks.

---

# 3. Required Run Directory Format

Every script that performs a real computation must write to a unique run directory:

```text
outputs/runs/{YYYYMMDD_HHMMSS}_{run_name}/
```

Each run directory must contain:

```text
config.yaml
command.sh
stdout.log
stderr.log
metrics.json
status.json
environment.json
artifacts/
```

## 3.1 config.yaml

Must contain:

```yaml
run_id:
timestamp:
script:
model:
base_model:
dataset:
data_path:
seed:
dtype:
device:
batch_size:
max_samples:
output_path:
notes:
```

## 3.2 command.sh

The exact command used.

## 3.3 metrics.json

All numeric outputs.

Examples:

```json
{
  "num_examples": 1000,
  "coverage_at_0": 0.61,
  "coverage_at_q25": 0.45,
  "preference_accuracy": 0.61,
  "mean_delta_dense": 0.087
}
```

## 3.4 status.json

Must contain:

```json
{
  "status": "success",
  "start_time": "...",
  "end_time": "...",
  "runtime_seconds": 123.4,
  "error": null
}
```

If failed:

```json
{
  "status": "failed",
  "error": "..."
}
```

## 3.5 environment.json

Must contain:

```json
{
  "python_version": "...",
  "torch_version": "...",
  "transformers_version": "...",
  "datasets_version": "...",
  "cuda_available": true,
  "gpu_name": "...",
  "git_commit": "..."
}
```

---

# 4. Global Coding Requirements

Use Python.

Use Hugging Face `transformers` and `datasets`.

Use PyTorch.

Use YAML configs.

Use deterministic seeds where possible.

Use tqdm for progress bars.

Use structured JSONL for intermediate outputs.

Use small sample sizes for smoke tests.

Do not silently overwrite outputs.

Do not hide exceptions.

All scripts must have:

```bash
--config path/to/config.yaml
```

or explicit CLI arguments.

Every script must support:

```bash
--max-samples
--seed
--out-dir
```

where applicable.

---

# 5. Milestone Gates

## M0: Repository Skeleton

Goal:

Create repository structure and basic utility modules.

Files:

```text
README.md
requirements.txt
pyproject.toml
configs/
scripts/
src/pbp/
tests/
outputs/
```

Must implement:

```text
src/pbp/io.py
src/pbp/utils.py
src/pbp/logging_utils.py
```

Checks:

```bash
python -m pytest tests/
python -m compileall src scripts
```

Definition of Done:

* repo imports successfully;
* all required tracking files exist;
* output run directory helper works;
* no model loading yet.

Stop after M0.

---

## M1: HH-RLHF Preprocessing

Goal:

Prepare HH-RLHF into JSONL preference pairs.

Implement:

```text
scripts/prepare_hh_rlhf.py
src/pbp/data.py
```

Output:

```text
data/processed/hh_rlhf_calib.jsonl
data/processed/hh_rlhf_eval.jsonl
```

Required JSONL format:

```json
{
  "id": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "source": "hh-rlhf"
}
```

Smoke test command:

```bash
python scripts/prepare_hh_rlhf.py \
  --dataset Anthropic/hh-rlhf \
  --calib-size 100 \
  --eval-size 200 \
  --seed 42 \
  --out-dir data/processed \
  --run-name m1_hh_rlhf_smoke
```

Checks:

```bash
python -m pytest tests/test_data.py
head data/processed/hh_rlhf_calib.jsonl
wc -l data/processed/hh_rlhf_calib.jsonl
wc -l data/processed/hh_rlhf_eval.jsonl
```

Definition of Done:

* calib/eval are disjoint;
* each record has prompt/chosen/rejected;
* no empty chosen/rejected;
* run directory saved;
* PROGRESS.md and EXPERIMENT_LOG.md updated.

Stop after M1.

---

## M2: Response-Only Logprob Computation

Goal:

Compute response-token-only length-normalized log-probability.

Implement:

```text
src/pbp/chat_format.py
src/pbp/logprobs.py
scripts/compute_logprobs.py
tests/test_logprobs.py
```

Requirements:

* prompt tokens must be masked out;
* only response tokens contribute to logprob;
* return sum_logprob, num_response_tokens, length_normalized_logprob;
* support Qwen chat template for instruct model;
* use the same final formatted prompt string for base and instruct scoring where possible.

Smoke test:

Use 5 examples and Qwen2.5-1.5B or a tiny debug model if necessary.

Command:

```bash
python scripts/compute_logprobs.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/hh_rlhf_eval.jsonl \
  --max-samples 5 \
  --out outputs/logprobs/smoke_instruct_5.jsonl \
  --dtype bfloat16 \
  --run-name m2_logprob_smoke
```

Checks:

```bash
python -m pytest tests/test_logprobs.py
head outputs/logprobs/smoke_instruct_5.jsonl
```

Definition of Done:

* masking test passes;
* num_response_tokens > 0;
* length_normalized_logprob is finite;
* run directory saved;
* logs updated.

Stop after M2.

---

## M3: Dense/Base Margin Computation

Goal:

Compute base-reference-normalized dense margins.

Implement:

```text
src/pbp/margins.py
scripts/compute_dense_margins.py
tests/test_margins.py
```

For each pair compute:

```text
ell_dense_chosen
ell_dense_rejected
ell_base_chosen
ell_base_rejected
delta_dense
```

Smoke command:

```bash
python scripts/compute_dense_margins.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-model Qwen/Qwen2.5-1.5B \
  --data data/processed/hh_rlhf_eval.jsonl \
  --max-samples 20 \
  --out outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl \
  --dtype bfloat16 \
  --run-name m3_dense_margin_smoke
```

Checks:

```bash
python -m pytest tests/test_margins.py
head outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl
```

Definition of Done:

* delta_dense is finite;
* output has one line per input example;
* dense model against itself would produce zero crossing in later tests;
* logs updated.

Stop after M3.

---

## M4: Coverage@τ Reporting

Goal:

Compute margin distribution and Coverage@τ.

Implement:

```text
src/pbp/metrics.py
scripts/report_coverage.py
tests/test_bcr_metrics.py
```

Metrics:

```text
preference_accuracy = P[delta_dense > 0]
Coverage@0
Coverage@q25
Coverage@q50
Coverage@q75
mean_delta_dense
median_delta_dense
positive_margin_quantiles
```

Smoke command:

```bash
python scripts/report_coverage.py \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl \
  --out outputs/evals/coverage_qwen2p5_1p5b_smoke.json \
  --run-name m4_coverage_smoke
```

Checks:

```bash
python -m pytest tests/test_bcr_metrics.py
cat outputs/evals/coverage_qwen2p5_1p5b_smoke.json
```

Definition of Done:

* Coverage metrics are valid probabilities;
* q25/q50/q75 are computed from positive dense margins;
* histogram or CSV distribution is saved;
* logs updated.

Stop after M4.

---

## M5: Mask-Based Coupled FFN Pruning

Goal:

Implement mask-based coupled SwiGLU intermediate-neuron pruning for Qwen2.5.

Implement:

```text
src/pbp/ffn_units.py
src/pbp/pruning.py
scripts/apply_mask_pruning.py
tests/test_pruning_shapes.py
```

MVP may use mask-based pruning first.

The mask must zero the coupled intermediate neuron:

```text
z_j = silu(gate_j) * up_j
```

Pruning neuron j means z_j is zeroed before down_proj.

Smoke command:

```bash
python scripts/apply_mask_pruning.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --method random \
  --ratio 0.10 \
  --out outputs/pruned_models/qwen2p5_1p5b_random_mask_10p \
  --run-name m5_random_mask_10p_smoke
```

Checks:

* model loads;
* generation still works;
* exactly requested fraction of FFN intermediate neurons are masked;
* no shape errors.

Stop after M5.

---

## M6: Basic Pruning Baselines

Goal:

Implement random, magnitude, activation pruning.

Implement:

```text
scripts/score_pruning_importance.py
src/pbp/scoring.py
```

Methods:

```text
random
magnitude
activation
```

Smoke commands:

```bash
python scripts/score_pruning_importance.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/hh_rlhf_calib.jsonl \
  --method activation \
  --max-samples 50 \
  --out outputs/scores/qwen2p5_1p5b_activation_smoke.json \
  --run-name m6_activation_score_smoke
```

Definition of Done:

* scores exist for all FFN intermediate neurons;
* scores are finite;
* selected pruning masks differ across methods;
* logs updated.

Stop after M6.

---

## M7: BCR Evaluation for Pruned Models

Goal:

Evaluate pruned model margins and BCR.

Implement:

```text
scripts/evaluate_bcr.py
```

Inputs:

```text
dense_margins.jsonl
pruned model or mask config
base logprobs or base model
eval data
```

Metrics:

```text
Coverage@τ
BCR@τ
preference_accuracy_dense
preference_accuracy_pruned
mean_margin_drop
```

Important:

Coverage@τ is always based on dense margins, not pruned margins.

Smoke command:

```bash
python scripts/evaluate_bcr.py \
  --model outputs/pruned_models/qwen2p5_1p5b_random_mask_10p \
  --base-model Qwen/Qwen2.5-1.5B \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_smoke.jsonl \
  --data data/processed/hh_rlhf_eval.jsonl \
  --max-samples 20 \
  --out outputs/evals/bcr_random_10p_smoke.json \
  --run-name m7_bcr_random_10p_smoke
```

Definition of Done:

* BCR for dense model vs dense model is zero;
* random pruning has finite BCR;
* margin drop is computed;
* logs updated.

Stop after M7.

---

## M8: Boundary-Aware Taylor Scoring

Goal:

Implement margin-drop-aware Taylor score.

Score:

```text
I_boundary(g) = E[w(x) * max(0, z_g * ∂Δ/∂z_g)]
```

Implement:

```text
boundary_taylor_drop
boundary_taylor_weighted
boundary_taylor_abs
general_taylor
```

Requirements:

* use calibration pairs;
* compute dense margin thresholds;
* default τ_calib = q25;
* only use examples with Δdense > τ_calib;
* support micro-batching;
* save score statistics.

Smoke command:

```bash
python scripts/score_pruning_importance.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-model Qwen/Qwen2.5-1.5B \
  --data data/processed/hh_rlhf_calib.jsonl \
  --method boundary_taylor_weighted \
  --max-samples 20 \
  --tau-mode q25 \
  --out outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted_smoke.json \
  --run-name m8_boundary_taylor_smoke
```

Definition of Done:

* scores are finite;
* not all scores are zero;
* method selects a different pruning set than activation pruning;
* logs updated.

Stop after M8.

---

## M9: First Pilot Table

Goal:

Run a small pilot on Qwen2.5-1.5B.

Matrix:

```text
methods:
- random
- magnitude
- activation
- boundary_taylor_weighted

ratios:
- 0.10
- 0.20

eval samples:
- 1000 HH-RLHF pairs
```

Output table:

```text
model | method | ratio | coverage@0 | coverage@q25 | bcr@0 | bcr@q25 | pref_acc | mean_margin_drop
```

Implement:

```text
scripts/summarize_results.py
```

Definition of Done:

* CSV table exists;
* each run has a run directory;
* EXPERIMENT_LOG.md includes every command;
* RUNS.md has all rows;
* failures are recorded in KNOWN_ISSUES.md;
* no hidden manual steps.

Stop after M9.

---

# 6. Reporting Format After Each Milestone

After completing a milestone, report exactly:

````markdown
## Milestone Report: M?

### Status
passed / failed / blocked

### What was implemented
- ...

### Files created/modified
- ...

### Commands run
```bash
...
````

### Tests

```bash
...
```

### Output artifacts

* ...

### Key metrics

```json
{}
```

### Issues

* ...

### Next proposed milestone

M?

```

Do not continue until approved.

---

# 7. First Instruction

Start with M0 only.

Read `PROJECT_SPEC.md`.

Create the repository skeleton and tracking files.

Do not load models yet.

Do not preprocess HH-RLHF yet.

Do not implement pruning yet.
```
