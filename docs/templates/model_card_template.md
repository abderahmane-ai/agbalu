# AƔBALU Model Card Authoring Instructions & Template

This document defines the official standard, tone guidelines, and template structure for authoring Hugging Face model cards across the AƔBALU project.

---

## Part 1: Authoring Guidelines & Quality Standards

### 1. Tone and Voice
- **Objective and Professional**: Write in clear, publication-grade technical English matching top AI research labs and professional Hugging Face repositories.
- **Strictly Avoid AI-Sounding Rhetorical Prose**:
  - ❌ *Do not use dramatic rhetorical openers:* "Three things worth reading carefully", "The gap between the probe and the fine-tune is the task", "Nothing moves on the canonical path", "Four things the headline table does not say on its own".
  - ❌ *Do not use conversational or clipped idioms:* "load-bearing rather than tidy", "not decoration", "is the replacement", "what each of them did nobody shared".
  - ❌ *Do not use argumentative or defensive declarations:* "Adequacy is unmeasured, and will stay unmeasured", "The unclear bulk is the redistribution risk — stated here rather than left for someone to discover".
  - ✅ *State facts directly:* "Evaluated on the 1,500-sentence held-out test split, reaching 90.27% accuracy and 0.9026 macro-F1."
  - ✅ *Explain technical mechanisms plainly:* "Character normalisation standardises input before tokenization, preventing subword fragmentation caused by homoglyph substitutions."

### 2. Parameter Counts (Zero Confusion)
- **State Headline and Exact Counts**: Always provide the exact parameter count integer in the Architecture table and introductory summary (e.g., `31,123,840` parameters for Masinissa-31M; `315,479,720` for Fadhma-300M).
- **Avoid Ambiguous Shorthands**: For models with vocabulary trimming (such as Amrouche, where trimmed vocabulary reduces total parameters from ~1.37B to 1,161,745,408), avoid misleading rounded titles; clearly document both base and trimmed counts in the Architecture table.
- **Dual-Tower Models**: For models with separate text and vision/audio towers, report both tower-specific and total parameters (e.g. text tower: `4,205,751,296`, total: `4,539,265,536`).

### 3. Hyperparameters & Technical Reproducibility
Every model card must document the exact training hyperparameters:
- **Optimizer**: Exact optimizer name (`AdamW`, `LAMB`), $\beta_1, \beta_2$, $\epsilon$, and weight decay.
- **Schedule**: Warmup steps/percentage and decay schedule (e.g., linear warmup followed by cosine decay).
- **Learning Rates**: Peak learning rates, decay minimums, and differential rates if head and backbone differ.
- **Batching**: Global effective batch size, micro-batch size, gradient accumulation steps, and sequence length.
- **Precision & Compute**: Hardware used (e.g. H100 MIG, NVIDIA A10G), mixed precision (`bfloat16`, `float32`), steps, total tokens consumed, throughput, and final validation loss.
- **Context Windows**: Explicit context window length (training and maximum inference capacity).

### 4. Evaluation Standards
- **Standardized Splits**: Report exact metrics on verified, held-out test splits. State test set support per class.
- **Fair Baseline Comparisons**: All baselines must be evaluated on the identical test split under the same evaluation protocol.
- **Ablations & Probes**: When comparing fine-tunes against frozen probes, clearly define the probe architecture (e.g. single linear layer, learnable `ScalarMix` pooling).

### 5. Decontamination & Integrity
- Detail the exact decontamination filtering applied (e.g., 4-word, 8-word, or 13-word n-gram window).
- Note positive-control leak injection testing confirming detector sensitivity.

### 6. "The Name" (Cultural Homage)
- Every model in AƔBALU is named in homage to a historical, literary, or cultural figure whose life reflects the model's function.
- **DO NOT cut or shorten "The name" sections.** Preserve the complete, dignified historical and biographical tribute.

### 7. Licensing Transparency
- Distinguish between the **model weights license** (e.g. Apache-2.0) and the **training corpus redistribution composition** (percentages of unclear, permissive, share-alike, and non-commercial source texts).

---

## Part 2: Markdown Model Card Template

```markdown
---
language:
- kab
license: apache-2.0
base_model: {base_model_repo_if_applicable}
library_name: transformers
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- {task-tags}
- low-resource
pipeline_tag: {pipeline_tag}
metrics:
- {primary_metric_1}
- {primary_metric_2}
model-index:
- name: {Model-Name}
  results:
  - task:
      type: {task_type}
      name: {Task Description}
    dataset:
      type: {dataset_repo}
      name: {Dataset split description}
    metrics:
    - type: {primary_metric_1}
      value: {metric_value_1}
      name: {Metric Name 1}
    - type: {primary_metric_2}
      value: {metric_value_2}
      name: {Metric Name 2}
---

# {Model-Name}

A {parameter_count}-parameter ({exact_integer_parameters}) {model_description} for **Kabyle** (Taqbaylit, `kab`, Latin script).

On the held-out test split of [`{Dataset}`](https://huggingface.co/datasets/{org}/{dataset}) ({support} {units}), it achieves **{primary_score_1}** and **{primary_score_2}**.

{Key highlight sentence regarding robustness, calibration, or benchmark performance without rhetorical flourishes.}

## Results

Evaluated on the {test_split_description} ({support_breakdown}).

| System | Configuration | {Metric 1} | {Metric 2} |
|---|---|---|---|
| Baseline 1 | {description} | {score} | {score} |
| Baseline 2 | {description} | {score} | {score} |
| **{Model-Name}** | **{shipped_setting}** | **{score}** | **{score}** |

{Objective commentary explaining the results, baseline gaps, and performance across subcategories or classes.}

### Detailed Performance Breakdown

| Category / Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| Class A | {support} | {prec} | {rec} | **{f1}** |
| Class B | {support} | {prec} | {rec} | **{f1}** |

{Analysis of error modes and confusion distributions.}

## Intended Use

- **Primary applications**: {Concise bullet points of primary in-scope applications}.
- **Downstream integration**: {Integration into pipelines, e.g. ASR post-processing, retrieval, triage}.

**Not suitable for**:
- {Safety-critical automated decisions without human oversight}.
- {Unsupported dialects, scripts, or languages}.

## Usage

```python
from transformers import AutoModelFor{Task}, AutoTokenizer

model_id = "agbalu/{Model-Name}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelFor{Task}.from_pretrained(model_id, trust_remote_code=True)

# Standard inference example:
inputs = tokenizer("Azul fell-awen, amek i tellam?", return_tensors="pt")
outputs = model(**inputs)
```

## Architecture

| Parameter | Specification |
|---|---|
| **Parameters, trained** | {exact_integer} |
| **Parameters, published** | {exact_integer} |
| **Backbone** | {layers} layers, {hidden} hidden dimension, {heads} attention heads, {intermediate} intermediate |
| **Context Window** | {context_length} tokens |
| **Position Embeddings** | {positional_embedding_type} |
| **Vocabulary** | {vocabulary_size} tokens ({tokenizer_type}) |
| **Precision** | {BF16 / FP32} |

{Description of pooling, specialized classification heads, or dropped pretraining weights.}

## Training Recipe

| Parameter | Specification |
|---|---|
| **Training Data** | {Dataset name and split size} |
| **Optimizer** | {Optimizer name}, $\beta=({b1}, {b2})$, $\epsilon={eps}$, weight decay {wd} |
| **Learning Rate** | {lr} with {warmup_pct}% linear warmup and {decay_type} decay |
| **Effective Batch Size** | {batch_size} sequences ({tokens_per_step} tokens/step) |
| **Training Steps** | {step_count} steps ({tokens_consumed} tokens consumed) |
| **Validation Loss** | {val_loss} ({perplexity_or_metric}) |
| **Hardware** | {Hardware configuration} |

## Decontamination

{Exact description of n-gram filtering windows, overlap metrics with evaluation benchmarks (e.g. FLORES+, SIB-200), and positive-control verification results.}

## Limitations

- **Dialect & Script Scope**: {Supported orthography, Latin vs Tifinagh vs Arabizi}.
- **Domain Coverage**: {Document length limits, genre bounds}.
- **Acoustic / Prosodic Absence** (if NLP): {Features outside the text modality}.

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | {size} | Published model weights ({exact_parameters} parameters) |
| `config.json` | {size} | Architecture hyperparameters and metadata |
| `tokenizer.json` | {size} | Subword tokenizer configuration |

## Reproduction

{Precise checkpoint step, source snapshot SHA-256 hash, random seed, and deterministic evaluation protocol instructions.}

## The Name

**{Figure Name}** ({Historical Dates}) was {Biographical and historical tribute explaining why their life's work represents this model's function}.

{Preserve full homage text without truncation or summarization.}

The naming is homage and implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_{model_slug}_2026,
  title  = {{Model-Name}: {Short descriptive subtitle}},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/{Model-Name}},
  note   = {{Parameter count}; {headline metric}}
}
```

## Licence

**Apache-2.0** on weights and code.

The training text composition over the pretraining corpus is:
| Redistribution | Share |
|---|---|
| **Unclear** | **34.9%** |
| Permissive | 32.0% |
| Share-alike | 31.0% |
| Non-commercial | 2.0% |

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle and Amazigh corpus and model collection.
```
