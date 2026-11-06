# AƔBALU Dataset Card Authoring Instructions & Template

This document defines the official standard, tone guidelines, and template structure for authoring Hugging Face dataset cards across the AƔBALU project.

---

## Part 1: Authoring Guidelines & Quality Standards

### 1. Tone and Voice
- **Empirical and Precise**: Document datasets with objective academic rigor. Treat the dataset as a scientific artifact.
- **Strictly Avoid Argumentative or Stylized Prose**:
  - ❌ *Do not use combative or dramatic commentary:* "The defect is in the published reference and has never been revised upstream", "Why this benchmark exists", "A model evaluated only in its training domain has not been evaluated", "The floor is not zero, and that is why it is measured rather than assumed".
  - ❌ *Do not use rhetorical colloquialisms:* "The distribution is a property of the sources, not of Kabyle", "treat this as a wide verbal paradigm with a thin nominal layer", "Nothing here is human-reviewed at scale".
  - ✅ *State curation facts and measurements directly:* "The dataset contains 326 sentences where character substitutions present in the upstream benchmark have been corrected to standard Latin orthography."
  - ✅ *Explain baseline floors factually:* "Because the imperative singular serves as the citation form for many Kabyle verbs, copying the lemma unchanged achieves a baseline of 4.07% exact match."

### 2. Naming Convention
- **Descriptive Dataset Names**: In AƔBALU, models carry historical homage names (*Masinissa*, *Amrouche*, *Fadhma*), while datasets carry **descriptive names** prefixed by `Kab` (e.g., `KabBench`, `KabLex`, `KabSentiment`, `KabPunct`, `KabInflect`, `KabStandard`, `KabTifinagh`, `KabLiterary`, `KabG2P`).
- Datasets are tools that researchers search for by functionality; keep names clear, searchable, and consistent.

### 3. Schema & Field Definitions
- Every column/field must be documented in a markdown table specifying:
  - `Field`: Column name.
  - `Type`: Data type (`string`, `int32`, `list[str]`, etc.).
  - `Description`: Exact semantic meaning, formatting rules, and conventions (e.g., Universal Dependencies conventions, placeholder markers like `_` for empty features).

### 4. Split Structure & Partitioning Invariants
- Provide a clear summary table listing all configurations and splits (`train`, `dev`, `test`, `ood`), record counts, word counts, and character counts.
- **Document Leakage Prevention**: Clearly describe partitioning boundaries designed to prevent data leakage:
  - **Lemma-disjoint partitioning** (in morphological benchmarks to prevent paradigm memorization).
  - **Book-boundary / document-boundary splits** (in literary or long-form corpora).
  - **Speaker-disjoint splits** (in speech-aligned corpora).

### 5. Orthography and Normalisation
- Document the exact normalisation pipeline applied (e.g. AƔBALU reference normaliser `1.3.0+rules1.0.0`, 81 rules).
- Disclose how contrastive phonemic characters are handled (`ɣ`, `ɛ`, `ḥ`, `ḍ`, `ṣ`, `ṭ`, `ẓ`, `ṛ`, `č`, `ǧ`, `ţ`). If a specific phonemic distinction is unmarked in the source data (such as emphatic `ṛ` in verbal conjugators), state this transparently.

### 6. Baselines & Evaluation Floors
- Always provide empirical baseline floors:
  - **Trivial copy baseline** (e.g., copying the input unchanged in orthography standardisation or inflection).
  - **Majority class baseline** (in classification tasks).
  - **Dual-condition scoring** (raw vs. normalized scoring in machine translation).

### 7. Licensing Transparency
- Document the dataset release license (e.g. CC-BY-4.0, Apache-2.0, CC0-1.0).
- Provide a composition table detailing the upstream sources, their respective licenses, and any filtering applied to exclude restrictive or viral share-alike terms.

---

## Part 2: Markdown Dataset Card Template

```markdown
---
license: {license_identifier}
language:
  - kab
language_bcp47:
  - kab-Latn
multilinguality:
  - {monolingual / multilingual}
size_categories:
  - {size_category_e.g._100K<n<1M}
task_categories:
  - {primary_task_category}
task_ids:
  - {specific_task_id}
pretty_name: {DatasetName} — {Descriptive subtitle}
tags:
  - kabyle
  - taqbaylit
  - berber
  - amazigh
  - low-resource
  - {domain-specific-tags}
configs:
  - config_name: default
    data_files:
      - split: train
        path: {format}/train.{ext}
      - split: dev
        path: {format}/dev.{ext}
      - split: test
        path: {format}/test.{ext}
---

# {DatasetName}

A {description_of_dataset} for **Kabyle** (Taqbaylit, `kab`, Latin script), from the [AƔBALU](https://huggingface.co/agbalu) project.

**{Total_records} {units}** across {number_of_configurations_or_splits}, designed for {primary_use_cases}.

```python
from datasets import load_dataset

ds = load_dataset("agbalu/{DatasetName}")
print(ds["train"][0])
```

## Splits

| Split | Config | Records | Words | Description |
|---|---|---:|---:|---|
| `train` | `default` | {count} | {words} | Training split |
| `dev` | `default` | {count} | {words} | Validation split |
| `test` | `default` | {count} | {words} | Evaluation split |
| **Total** | | **{total}** | **{total_words}** | |

{Explanation of partitioning methodology, stratification, and leakage prevention invariants.}

## Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique record identifier (`{prefix}_{split}_{idx}`) |
| `{primary_text_field}` | string | Normalized text string in canonical Kabyle Latin |
| `{label_or_target_field}` | {type} | Supervised target, class label, or annotation |
| `source` | string | Provenance identifier of originating source |

## Data Curation

### Source Text and Extraction
{Description of source corpora, retrieval timestamps, volume of raw candidates, and initial filtering criteria (word length bounds, URL/digit/handle stripping).}

### Annotation / Normalisation Pipeline
{Details on cross-lingual teacher projection, phonetic alignment, or rule-based processing. State confidence gates, thresholds, and candidate rejection percentages.}

### Orthography Standards
All text is normalized under the AƔBALU reference normaliser (`1.3.0+rules1.0.0`, 81 rules), preserving contrastive phonemic consonants (`ɣ`, `ɛ`, `ḥ`, `ḍ`, `ṣ`, `ṭ`, `ẓ`, `ṛ`, `č`, `ǧ`, `ţ`) and repairing systematic Greek homoglyph substitutions (`ε` U+03B5 $\to$ `ɛ` U+025B).

## Baseline Benchmarks & Evaluation Floors

Evaluated on the held-out `test` split:

| Task / Setting | System / Baseline | Primary Metric | Secondary Metric |
|---|---|---|---|
| Trivial Baseline | Copy input unchanged | {score} | {score} |
| Standard Baseline | Linear probe / standard model | {score} | {score} |
| State of the Art | {Published AƔBALU Model} | **{score}** | **{score}** |

{Objective explanation of baseline floors and metric interpretations.}

## Usage

```python
from datasets import load_dataset

# Load complete dataset or specific split:
dataset = load_dataset("agbalu/{DatasetName}", split="train")

# Inspect sample:
print(dataset[0])

# Example task-specific usage helper:
```

## Composition & Provenance

| Source ID | Type | Records | License |
|---|---|---:|---|
| `{source_1}` | {type} | {count} | {license} |
| `{source_2}` | {type} | {count} | {license} |

{Description of licensing filters applied to exclude restrictive or viral share-alike terms.}

## Known Limitations

- **Domain Scope**: {Corpus domain boundaries, e.g. formal written prose vs informal social text}.
- **Dialect & Script Coverage**: {Covered varieties and scripts}.
- **Annotation Source**: {Human annotation vs teacher projection vs rule-based synthesis}.

## Citation

```bibtex
@misc{agbalu_{dataset_slug},
  title  = {{DatasetName}: {Descriptive subtitle}},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/{DatasetName}}
}
```

Please also cite the underlying source datasets: {upstream_citation_references}.

## Licence

**{License_Type}**. {Details on attribution, share-alike, or commercial redistribution rights}.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle and Amazigh corpus and model collection.
```
