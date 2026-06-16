# PROJECT_SPEC.md

# Project: Preference Boundary Crossing under Structured Pruning

## 0. Project Goal

This repository implements an MVP experiment for the following research question:

> Under mild coupled SwiGLU FFN intermediate-neuron structured pruning of Qwen2.5-Instruct models, can general capability remain matched while base-reference-normalized preference boundaries cross, and can margin-drop-aware Taylor pruning reduce Boundary Crossing Rate without post-pruning recovery?

The project is a **diagnostic + pruning criterion** study.

It is **not**:

* a post-pruning DPO/LoRA repair project;
* a general safety-only evaluation project;
* a mechanism/circuit discovery project;
* a full paper-writing project.

The main output should be executable code, reproducible experiment commands, metric tables, and saved artifacts.

---

## 1. Core Definitions

### 1.1 Models

For each model size:

```text
πθ = Qwen2.5-Instruct dense or pruned model
πref = Qwen2.5-Base model of the same size
```

Pilot models:

```text
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-1.5B

Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-3B
```

Main model:

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-7B
```

The base model is only used as a log-probability reference. It is not pruned.

---

### 1.2 Preference Margin

For each preference pair:

```text
x: prompt
y+: chosen response
y-: rejected response
```

Define length-normalized response-token log-probability:

```text
ℓθ(y|x) = (1 / |y|) log πθ(y | x)
```

Only response tokens should contribute to the log-probability. Prompt tokens must be masked out.

Define base-reference-normalized preference margin:

```text
Δθ(x) =
[ℓθ(y+|x) - ℓbase(y+|x)]
-
[ℓθ(y-|x) - ℓbase(y-|x)]
```

The dense margin is:

```text
Δdense(x)
```

The pruned margin is:

```text
Δpruned(x)
```

---

### 1.3 Coverage

Because HH-RLHF is an external human preference dataset, Qwen2.5-Instruct may not agree with all chosen/rejected pairs.

Therefore, report Coverage@τ:

```text
Coverage@τ = P[Δdense(x) > τ]
```

This measures how many external preference pairs are consistent with the dense instruct model.

---

### 1.4 Boundary Crossing Rate

Boundary Crossing Rate is computed only on dense-model-consistent pairs:

```text
BCR@τ = P[Δdense(x) > τ and Δpruned(x) < 0]
```

Equivalently:

```text
Dτ = {(x, y+, y-) : Δdense(x) > τ}

BCR@τ =
(1 / |Dτ|)
Σ_{i in Dτ} 1[Δpruned(x_i) < 0]
```

Thresholds:

```text
τ ∈ {0, q25, q50, q75}
```

where `q25`, `q50`, and `q75` are quantiles of the positive dense margin distribution.

Primary threshold:

```text
τ = q25
```

---

## 2. Main Hypothesis

The main hypothesis is:

> Mild structured pruning can keep standard general benchmarks nearly matched, while still increasing pairwise preference boundary crossing.

The method hypothesis is:

> Margin-drop-aware Taylor pruning can reduce BCR compared with utility-only pruning under the same pruning ratio and similar general utility, without post-pruning recovery.

---

## 3. Data

### 3.1 Main Preference Dataset

Use:

```text
Anthropic/hh-rlhf
```

Fields should be converted into:

```text
prompt
chosen
rejected
```

Use:

```text
2k-5k calibration pairs for pruning importance scoring
10k held-out pairs for BCR evaluation
```

The calibration split and evaluation split must be disjoint.

Save processed files as JSONL:

```text
data/processed/hh_rlhf_calib.jsonl
data/processed/hh_rlhf_eval.jsonl
```

Each line:

```json
{
  "id": "...",
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "source": "hh-rlhf"
}
```

---

## 4. Pruning Target

Only prune coupled SwiGLU FFN intermediate neurons.

For a SwiGLU FFN:

```text
FFN(x) = W_down( silu(W_gate x) ⊙ W_up x )
```

The intermediate neuron index `j` corresponds to:

```text
W_gate output channel j
W_up output channel j
W_down input channel j
```

Pruning neuron `j` means removing all three coupled components together.

Main pruning ratios:

```text
10%
20%
```

Stress test:

```text
30%
```

No post-pruning recovery in main experiments.

---

## 5. Pruning Methods

Implement the following methods.

### 5.1 Magnitude Pruning

Score each coupled FFN intermediate neuron by the magnitude of its associated weights.

Possible score:

```text
score_j =
||W_gate[j, :]||_2
+ ||W_up[j, :]||_2
+ ||W_down[:, j]||_2
```

Prune lowest-score neurons globally or layer-wise.

Implement both if easy, but MVP may start with global pruning.

---

### 5.2 Activation Pruning

Run calibration data through the dense instruct model.

For each intermediate neuron:

```text
z_j = silu(W_gate x)_j ⊙ W_up(x)_j
```

Score:

```text
score_j = E[|z_j|]
```

Prune lowest-score neurons.

---

### 5.3 General Taylor Pruning

Use a general language-modeling loss on chosen responses or mixed instruction data.

For each intermediate neuron:

```text
score_j = E[|z_j * ∂L_general/∂z_j|]
```

Prune lowest-score neurons.

---

### 5.4 Boundary-Aware Margin-Drop Taylor Pruning

This is the proposed method.

For each pair:

```text
Δθ(x) =
[ℓθ(y+|x) - ℓbase(y+|x)]
-
[ℓθ(y-|x) - ℓbase(y-|x)]
```

For each coupled FFN intermediate neuron `g`, compute:

```text
drop_g(x) = z_g * ∂Δθ(x)/∂z_g
```

Use sign-aware positive margin-drop score:

```text
I_boundary(g) =
E[ w(x) * max(0, z_g * ∂Δθ(x)/∂z_g) ]
```

Recommended default:

```text
w(x) = 1 / (epsilon + Δdense(x))
```

Use only pairs where:

```text
Δdense(x) > τ_calib
```

Default:

```text
τ_calib = q25
```

To avoid exploding weights near zero, implement clipping:

```text
w(x) = clip(1 / (epsilon + Δdense(x)), 0, w_max)
```

Configurable defaults:

```text
epsilon = 1e-4
w_max = 10.0
```

Ablations:

```text
Boundary-Taylor-Abs:
E[ |z_g * ∂Δ/∂z_g| ]

Boundary-Taylor-Drop:
E[ max(0, z_g * ∂Δ/∂z_g) ]

Boundary-Taylor-Weighted:
E[ w(x) * max(0, z_g * ∂Δ/∂z_g) ]
```

Primary method:

```text
Boundary-Taylor-Weighted
```

---

## 6. Metrics

### 6.1 Preference Boundary Metrics

Compute on held-out HH-RLHF eval pairs:

```text
Coverage@0
Coverage@q25
Coverage@q50
Coverage@q75

BCR@0
BCR@q25
BCR@q50
BCR@q75

Preference accuracy:
P[Δθ(x) > 0]

Mean dense margin
Mean pruned margin
Mean margin drop:
E[Δdense(x) - Δpruned(x)]
```

Primary metric:

```text
BCR@q25
```

---

### 6.2 General Utility Metrics

For MVP, implement at least:

```text
perplexity on WikiText or C4 subset
ARC
HellaSwag
MMLU subset
GSM8K subset
```

It is acceptable to use `lm-evaluation-harness` if easier.

The key analysis is not absolute benchmark improvement. The key analysis is matched utility:

```text
At similar MMLU/GSM8K/ARC/HellaSwag/perplexity, does Boundary-Taylor have lower BCR than General Taylor or Activation pruning?
```

---

### 6.3 Optional Secondary Safety Metrics

Do not block MVP on these.

Possible later additions:

```text
harmful refusal rate
benign false refusal rate
OR-Bench subset
AdvBench subset
HarmBench subset
```

These are secondary, not the main claim.

---

## 7. Repository Structure

Create the following repository structure:

```text
preference-boundary-pruning/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    model_qwen2p5_1p5b.yaml
    model_qwen2p5_3b.yaml
    model_qwen2p5_7b.yaml
    pruning_10p.yaml
    pruning_20p.yaml
    pruning_30p.yaml
    eval_hh_rlhf.yaml
  data/
    raw/
    processed/
  scripts/
    prepare_hh_rlhf.py
    compute_base_logprobs.py
    compute_dense_margins.py
    score_pruning_importance.py
    apply_ffn_pruning.py
    evaluate_bcr.py
    evaluate_general.py
    summarize_results.py
  src/
    pbp/
      __init__.py
      data.py
      chat_format.py
      logprobs.py
      margins.py
      hooks.py
      ffn_units.py
      scoring.py
      pruning.py
      metrics.py
      eval_general.py
      io.py
      utils.py
  outputs/
    logprobs/
    margins/
    scores/
    pruned_models/
    evals/
    tables/
  tests/
    test_logprobs.py
    test_margins.py
    test_pruning_shapes.py
    test_bcr_metrics.py
```

---

## 8. Implementation Requirements

### 8.1 Log-Probability Computation

Implement response-only log-probability.

Given:

```text
prompt
response
```

Construct:

```text
full_text = formatted_prompt + response
```

Tokenize full text.

Create a loss mask where:

```text
prompt tokens = 0
response tokens = 1
```

Compute next-token log-prob only over response tokens.

Return:

```json
{
  "sum_logprob": ...,
  "num_response_tokens": ...,
  "length_normalized_logprob": ...
}
```

Important:

* Use the same formatted prompt for instruct and base model where possible.
* Do not include prompt tokens in the response log-prob.
* Save base log-probs offline to avoid recomputation.

---

### 8.2 Chat Formatting

Implement a consistent formatting function.

For Qwen2.5-Instruct, use the tokenizer chat template if available.

For base model, use the same final formatted prompt string, not a different instruction format.

The goal is to compare response log-probs under the same textual context.

---

### 8.3 Dense Margin Computation

For each eval pair, compute:

```text
ℓdense(y+|x)
ℓdense(y-|x)
ℓbase(y+|x)
ℓbase(y-|x)
Δdense(x)
```

Save as JSONL:

```text
outputs/margins/dense_margins_MODEL_DATA.jsonl
```

Each line:

```json
{
  "id": "...",
  "ell_dense_chosen": ...,
  "ell_dense_rejected": ...,
  "ell_base_chosen": ...,
  "ell_base_rejected": ...,
  "delta_dense": ...
}
```

---

### 8.4 Pruned Margin Computation

For each pruned model, compute:

```text
ℓpruned(y+|x)
ℓpruned(y-|x)
Δpruned(x)
```

Use already cached base log-probs.

Save as:

```text
outputs/margins/pruned_margins_MODEL_METHOD_RATIO.jsonl
```

---

### 8.5 Coupled FFN Unit Discovery

Implement model inspection for Qwen2.5 decoder layers.

For each layer, identify:

```text
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

A coupled neuron index `j` is valid if:

```text
gate_proj.out_features == up_proj.out_features == down_proj.in_features
```

Store metadata:

```json
{
  "layer": 0,
  "unit_index": 123,
  "gate_shape": "...",
  "up_shape": "...",
  "down_shape": "..."
}
```

---

### 8.6 Applying FFN Pruning

Given scores for all coupled FFN intermediate neurons:

1. Select lowest-score neurons according to pruning ratio.
2. Physically remove:

   * rows from `gate_proj.weight`;
   * rows from `up_proj.weight`;
   * columns from `down_proj.weight`.
3. Update module dimensions.
4. Save pruned model and tokenizer.

The pruned model must be loadable by `AutoModelForCausalLM.from_pretrained`.

If physical dimension-changing save is too hard for MVP, implement mask-based pruning first, but clearly mark it as masked structured pruning. Physical pruning is preferred.

---

## 9. Experiment Commands

The agent should generate commands like the following.

### 9.1 Prepare Data

```bash
python scripts/prepare_hh_rlhf.py \
  --dataset Anthropic/hh-rlhf \
  --calib-size 5000 \
  --eval-size 10000 \
  --seed 42 \
  --out-dir data/processed
```

---

### 9.2 Compute Base Log-Probs

```bash
python scripts/compute_base_logprobs.py \
  --base-model Qwen/Qwen2.5-1.5B \
  --data data/processed/hh_rlhf_eval.jsonl \
  --out outputs/logprobs/base_qwen2p5_1p5b_eval.jsonl \
  --dtype bfloat16
```

---

### 9.3 Compute Dense Margins

```bash
python scripts/compute_dense_margins.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-logprobs outputs/logprobs/base_qwen2p5_1p5b_eval.jsonl \
  --data data/processed/hh_rlhf_eval.jsonl \
  --out outputs/margins/dense_qwen2p5_1p5b_eval.jsonl \
  --dtype bfloat16
```

---

### 9.4 Score Importance

```bash
python scripts/score_pruning_importance.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --base-model Qwen/Qwen2.5-1.5B \
  --data data/processed/hh_rlhf_calib.jsonl \
  --method boundary_taylor_weighted \
  --tau-mode q25 \
  --out outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted.json \
  --dtype bfloat16
```

Baseline example:

```bash
python scripts/score_pruning_importance.py \
  --instruct-model Qwen/Qwen2.5-1.5B-Instruct \
  --data data/processed/hh_rlhf_calib.jsonl \
  --method activation \
  --out outputs/scores/qwen2p5_1p5b_activation.json \
  --dtype bfloat16
```

---

### 9.5 Apply Pruning

```bash
python scripts/apply_ffn_pruning.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --scores outputs/scores/qwen2p5_1p5b_boundary_taylor_weighted.json \
  --ratio 0.10 \
  --out outputs/pruned_models/qwen2p5_1p5b_boundary_taylor_10p
```

---

### 9.6 Evaluate BCR

```bash
python scripts/evaluate_bcr.py \
  --pruned-model outputs/pruned_models/qwen2p5_1p5b_boundary_taylor_10p \
  --base-logprobs outputs/logprobs/base_qwen2p5_1p5b_eval.jsonl \
  --dense-margins outputs/margins/dense_qwen2p5_1p5b_eval.jsonl \
  --data data/processed/hh_rlhf_eval.jsonl \
  --out outputs/evals/bcr_qwen2p5_1p5b_boundary_taylor_10p.json \
  --dtype bfloat16
```

---

### 9.7 Evaluate General Utility

```bash
python scripts/evaluate_general.py \
  --model outputs/pruned_models/qwen2p5_1p5b_boundary_taylor_10p \
  --tasks arc_easy,arc_challenge,hellaswag,mmlu,gsm8k \
  --out outputs/evals/general_qwen2p5_1p5b_boundary_taylor_10p.json
```

---

### 9.8 Summarize Results

```bash
python scripts/summarize_results.py \
  --eval-dir outputs/evals \
  --out outputs/tables/main_results.csv
```

---

## 10. Experiment Matrix

For pilot:

```text
Model:
- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct

Ratios:
- 0.10
- 0.20
- 0.30

Methods:
- magnitude
- activation
- general_taylor
- boundary_taylor_drop
- boundary_taylor_weighted
```

Main model:

```text
Model:
- Qwen2.5-7B-Instruct

Ratios:
- 0.10
- 0.20
- 0.30 stress test

Methods:
- activation
- general_taylor
- boundary_taylor_weighted
```

---

## 11. Expected Tables

### 11.1 Main BCR Table

```text
model | method | ratio | coverage@0 | coverage@q25 | bcr@0 | bcr@q25 | bcr@q50 | bcr@q75 | pref_acc | mean_margin_drop
```

### 11.2 Matched Utility Table

```text
model | method | ratio | mmlu | gsm8k | arc_c | hellaswag | ppl | bcr@q25
```

### 11.3 Ablation Table

```text
model | ratio | scoring_method | bcr@q25 | mean_margin_drop | mmlu | ppl
```

---

## 12. Sanity Checks

Before large experiments, verify:

1. Dense margins are not all near zero.
2. Coverage@0 is reasonably high.
3. Base and instruct tokenization are compatible enough for response log-prob comparison.
4. Response-only log-prob masking is correct.
5. BCR for dense model against itself is zero.
6. Random pruning increases damage more than magnitude or Taylor pruning.
7. Physical pruned models can be loaded and generate text.
8. Perplexity does not explode at 10% pruning.
9. Boundary-aware scores are not all zero.
10. Boundary-aware pruning selects a different set of neurons than activation pruning.

---

## 13. Failure Cases and Debugging

### Failure: Coverage is too low

If Coverage@0 is very low, HH-RLHF may not align well with the dense Qwen model.

Actions:

* filter data more carefully;
* use only helpfulness or harmlessness subset;
* try UltraFeedback as secondary data;
* report that the study is on dense-model-consistent external preference pairs.

### Failure: BCR does not increase under 10%/20% pruning

Actions:

* test 30% stress ratio;
* compare multiple pruning baselines;
* check if pruning implementation actually removes active neurons;
* check margin drop even if sign crossing is rare;
* report Margin Drop as secondary primary evidence.

### Failure: General utility drops too much

Actions:

* reduce pruning ratio;
* use layer-wise pruning instead of global if global is too destructive;
* use activation/general Taylor baseline;
* keep main claim at 10%/20%.

### Failure: Physical pruning breaks model loading

Actions:

* implement mask-based structured pruning as a fallback;
* save pruning masks and evaluate masked model;
* physical pruning can be implemented later.

### Failure: Boundary-aware score is too expensive

Actions:

* reduce calibration pairs to 500-1000 for pilot;
* accumulate gradients layer-by-layer;
* use micro-batches;
* use only selected layers for pilot;
* cache base log-probs.

---

## 14. Agent Instructions

When implementing this project:

1. Do not change the research question.
2. Do not introduce DPO/LoRA recovery into main experiments.
3. Do not replace BCR with generic safety/refusal metrics.
4. Do not use raw sequence log-prob as the primary margin.
5. Always use response-token-only length-normalized log-prob.
6. Always report Coverage@τ together with BCR@τ.
7. Prioritize Qwen2.5-1.5B pilot before 3B or 7B.
8. Build the pipeline in small testable modules.
9. Add unit tests for log-prob masking, margin computation, and BCR.
10. Produce runnable scripts and command examples before running large jobs.

---

## 15. First Milestone

The first milestone is not pruning.

The first milestone is:

```text
Load Qwen2.5-1.5B-Instruct and Qwen2.5-1.5B
Prepare 1k HH-RLHF pairs
Compute:
- ℓdense(y+)
- ℓdense(y-)
- ℓbase(y+)
- ℓbase(y-)
- Δdense
Report:
- Coverage@0/q25/q50/q75
- preference accuracy
- margin distribution histogram
```

Only after this works should pruning code be implemented.

---

## 16. Second Milestone

Implement mask-based coupled FFN pruning for Qwen2.5-1.5B-Instruct.

Run:

```text
10% random pruning
10% magnitude pruning
10% activation pruning
```

Evaluate:

```text
BCR@τ
mean margin drop
perplexity subset
```

Only after this works should Taylor and boundary-aware pruning be implemented.

---

## 17. Third Milestone

Implement boundary-aware Taylor scoring.

Compare:

```text
activation pruning
general Taylor pruning
boundary Taylor pruning
```

at:

```text
10%
20%
```

on:

```text
Qwen2.5-1.5B-Instruct
```

This is the first real MVP result.