---
license: apache-2.0
language:
- kab
language_bcp47:
- kab-Latn
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
task_categories:
- token-classification
- text-classification
- text-retrieval
- other
task_ids:
- part-of-speech
- sentiment-analysis
pretty_name: KabEncBench — Kabyle Encoder & Representation Benchmark
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- low-resource
- benchmark
- evaluation
- representation-learning
- probing
configs:
- config_name: syntax
  data_files:
  - split: train
    path: syntax/train.jsonl
  - split: dev
    path: syntax/dev.jsonl
  - split: test
    path: syntax/test.jsonl
- config_name: morph
  data_files:
  - split: train
    path: morph/train.jsonl
  - split: dev
    path: morph/dev.jsonl
  - split: test
    path: morph/test.jsonl
- config_name: ner
  data_files:
  - split: train
    path: ner/train.jsonl
  - split: dev
    path: ner/dev.jsonl
  - split: test
    path: ner/test.jsonl
- config_name: sent
  data_files:
  - split: train
    path: sent/train.jsonl
  - split: dev
    path: sent/dev.jsonl
  - split: test
    path: sent/test.jsonl
- config_name: topic
  data_files:
  - split: train
    path: topic/train.jsonl
  - split: dev
    path: topic/dev.jsonl
  - split: test
    path: topic/test.jsonl
- config_name: punct
  data_files:
  - split: train
    path: punct/train.jsonl
  - split: dev
    path: punct/dev.jsonl
  - split: test
    path: punct/test.jsonl
  - split: ood
    path: punct/ood.jsonl
- config_name: diag_clitic
  data_files:
  - split: train
    path: diag_clitic/train.jsonl
  - split: dev
    path: diag_clitic/dev.jsonl
  - split: test
    path: diag_clitic/test.jsonl
- config_name: diag_para
  data_files:
  - split: test
    path: diag_para/test.jsonl
- config_name: diag_spell
  data_files:
  - split: test
    path: diag_spell/test.jsonl
---

# KabEncBench: The Kabyle Representation Learning Benchmark

**`KabEncBench`** is the standard multi-task evaluation suite for contextual text representations and neural encoders in **Kabyle** (Taqbaylit, `kab`, Latin script).

Spanning **100,609 annotated items** across **six core linguistic tasks** and **three diagnostic probes**, `KabEncBench` isolates what pretrained language models understand about Kabyle phonology, morphology, syntax, semantics, and discourse—without allowing deep task heads to mask representational failure.

Every task is evaluated under **Track A: Frozen Scale-Normalized Probe**, scoring models through a unified Bradley–Terry Elo tournament with 2,000 paired cluster-bootstraps anchored at `FacebookAI/xlm-roberta-base` = 1,000.

```python
from datasets import load_dataset

# Load any of the 6 core tasks or 3 diagnostic suites
syntax = load_dataset("agbalu/KabEncBench", "syntax")
morph = load_dataset("agbalu/KabEncBench", "morph")
sentiment = load_dataset("agbalu/KabEncBench", "sent")
spell_diag = load_dataset("agbalu/KabEncBench", "diag_spell")

# Example inspection: Universal Dependencies syntax
print(syntax["test"][0]["tokens"])
print(syntax["test"][0]["upos"])
```

---

## 1. Official Bradley–Terry Leaderboard

Ratings derived from 2,000 paired cluster-bootstraps across all task test instances, anchored at `FacebookAI/xlm-roberta-base` = 1,000.0:

| Rank | Model | Parameters | Elo Rating (95% CI) | Tier | Mean Probe Score |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | [**`agbalu/Masinissa-73M`**](https://huggingface.co/agbalu/Masinissa-73M) | **73M** | **1,953.7 [1,534.7, 3,379.1]** | **Tier 1** | **85.61%** |
| **2** | [**`agbalu/Masinissa-31M`**](https://huggingface.co/agbalu/Masinissa-31M) | **31M** | **1,708.4 [1,371.1, 2,886.2]** | **Tier 2** | **83.78%** |
| 3 | `UBC-NLP/serengeti-E250` | 278M | 1,200.3 [953.6, 1,480.1] | Tier 3 | 78.76% |
| 4 | `Davlan/afro-xlmr-base-76L_script` | 278M | 1,151.6 [919.5, 1,451.0] | Tier 3 | 77.74% |
| 5 | `UBC-NLP/serengeti-E110` | 110M | 1,148.2 [918.5, 1,421.2] | Tier 3 | 78.27% |
| 6 | `jhu-clsp/mmBERT-base` | 278M | 1,141.7 [906.4, 1,435.3] | Tier 3 | 77.36% |
| 7 | `jhu-clsp/mmBERT-small` | 135M | 1,039.1 [804.2, 1,282.5] | Tier 3 | 76.17% |
| 8 | `google-bert/bert-base-multilingual-cased` | 178M | 1,012.6 [780.9, 1,266.8] | Tier 4 | 77.02% |
| 9 | `FacebookAI/xlm-roberta-base` *(Anchor)* | 278M | 1,000.0 [1,000.0, 1,000.0] | Tier 4 | 76.87% |

### Task-by-Task Performance Matrix (%)

| Model | UPOS (`Syntax`) | Aspect (`Morph`) | Micro-F1 (`NER`) | Sentiment | SIB-200 (`Topic`) | Punctuation | TaPaCo (`Paraphrase`) | Minimal Pairs (`Spell`) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`agbalu/Masinissa-73M`** | **80.59** | **57.91** | **88.24** | **93.83** | **64.87** | **55.42** | **64.38** | **60.90** |
| **`agbalu/Masinissa-31M`** | **78.88** | **54.19** | **87.74** | **92.06** | **59.28** | **52.21** | **58.78** | 18.70 |
| `serengeti-E250` | 76.98 | 49.13 | 83.98 | 85.33 | 43.72 | 45.54 | 42.55 | 6.30 |
| `serengeti-E110` | 76.24 | 48.83 | 84.91 | 85.65 | 39.64 | 43.11 | 43.86 | 4.70 |
| `mmBERT-base` | 75.52 | 48.44 | 84.01 | 85.64 | 37.96 | 39.98 | 48.63 | 53.30 |
| `mmBERT-small` | 75.72 | 45.84 | 81.79 | 83.17 | 34.94 | 39.91 | 48.98 | 61.00 |
| `xlm-roberta-base` | 75.44 | 44.52 | 79.71 | 84.75 | 38.21 | 39.12 | 42.13 | 57.40 |
| `afro-xlmr-base` | 75.32 | 47.93 | 83.04 | 85.75 | 38.41 | 38.56 | 48.54 | **62.50** |
| `bert-base-multilingual` | 73.21 | 46.79 | 80.62 | 84.35 | 40.84 | 39.01 | 51.77 | 27.00 |

---

## 2. Benchmark Architecture & Task Inventory

`KabEncBench` balances fundamental linguistic competencies across two tracks:

### Core Probing Tasks (Track A)

1. **`syntax` (Universal Dependencies UPOS & Syntax):**  
   Gold `UD_Kabyle-ADPT` treebank split (865 train, 220 dev, 845 test). Evaluates syntactic word-class identification over 16 universal POS tags. 29.5% of Kabyle words sit inside multiword tokens (MWTs); word representations are aligned via the initial subtoken.
2. **`morph` (Factored Verbal Aspect & Mood):**  
   25,129 verbal predicates (21,160 train, 1,323 dev, 2,646 test) covering Aorist, Preterite, Intensive Preterite, and Participle forms. Morphology is factored to isolate aspectual semantics from lexical root identity.
3. **`ner` (Named Entity Recognition):**  
   2,900 curated Kabyle sentences (2,000 train, 300 dev, 600 test) annotated for `PER`, `LOC`, `ORG`, and `MISC` entity spans under standard BIO tagging.
4. **`sent` (Sentence Sentiment Classification):**  
   7,050 sentences (6,000 train, 450 dev, 600 test) balanced across 3 polarity classes (`negative`, `neutral`, `positive`). Curated with English-Kabyle confidence verification.
5. **`topic` (SIB-200 7-Way Topic Categorization):**  
   1,004 sentences (701 train, 99 dev, 204 test) across 7 news topics (science/tech, sports, politics, etc.). Text repaired to canonical Kabyle Latin orthography, removing homoglyph corruptions present in public crawl datasets.
6. **`punct` (Punctuation & Case Restoration):**  
   42,246 sequences (30,000 train, 5,052 dev, 4,694 test, 2,500 OOD). Evaluates contextual boundary detection across comma, period, question mark, and colon placement.

### Diagnostic Probes (Track B)

7. **`diag_clitic` (Clitic Host Attachment):**  
   3,241 sequences (1,471 train, 384 dev, 1,386 test). Diagnoses whether representations accurately identify pronominal and directional clitic boundaries (`d/n` particles, direct/indirect object clitics) following prepositions or verb stems.
8. **`diag_para` (TaPaCo Zero-Shot Paraphrase Retrieval):**  
   15,919 sentences across 5,551 semantic clusters drawn from TaPaCo-kab. Backbones pool sentence embeddings via mean-pooling without fine-tuning; evaluated via Mean Average Precision at R (mAP@R) and Mean Reciprocal Rank (MRR@10).
9. **`diag_spell` (Minimal Pairs Orthographic & Morphological Sensitivity):**  
   1,000 controlled minimal sentence pairs testing four structural linguistic phenomena:
   * `SPIRANT_TH`: Spirantized dental fricative contrast (`ṯ` vs. standard Latin `t`).
   * `EMPHATIC_DROP`: Loss of phonemic pharyngealization (`ḍ, ḥ, ṛ, ṣ, ṭ, ẓ` vs. plain consonants).
   * `CONSTRUCT_ANNEX`: Annexed construct state vocalic alternation (e.g. *axxam* $\to$ *wexxam*).
   * `CLITIC_ORDER`: Pre-verbal clitic climbing vs. ungrammatical post-verbal attachment in negative/relative clauses.

---

## 3. Evaluation Protocol

To prevent overfitting, hyperparameter hacking, or prompt engineering from distorting model capabilities, `KabEncBench` enforces strict evaluation invariants:

### The Frozen Scale-Normalized Probe
* **Frozen Backbone:** The transformer weights $\theta$ receive zero gradient updates ($\nabla_\theta \mathcal{L} = 0$).
* **ScalarMix Layer Weighting:** Rather than arbitrarily evaluating the final layer, representations are computed via a learnable softmax over all $L+1$ layer representations (including embedding layer 0):
  $$\mathbf{h}_i = \sum_{l=0}^L \gamma_l \mathbf{h}_i^{(l)}, \quad \gamma_l = \frac{e^{\alpha_l}}{\sum_k e^{\alpha_k}}$$
* **Linear Probe Head:** A single linear layer projects pooled word/sentence representations to class logits:
  $$\hat{\mathbf{y}} = \mathbf{W} \mathbf{h} + \mathbf{b}$$
* **Standardized Optimizer & Hyperparameter Grid:** Probes are optimized with AdamW across a fixed grid of 5 learning rates ($[10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}, 10^{-2}]$). For each learning rate, 5 random seeds are trained for 20 epochs. The learning rate achieving highest dev score is selected, and evaluated once on the test split.

### Bradley–Terry Elo Tournament
For every bootstrap iteration $b \in \{1, \dots, 2000\}$, test instances are resampled with replacement within task clusters. Pairwise win/loss/tie outcomes are compiled across all model pairs, and maximum-likelihood latent skill parameters $\mu_m$ are estimated under the Bradley–Terry logistic model:
$$P(\text{Model } A > \text{Model } B) = \frac{1}{1 + 10^{(\mu_B - \mu_A) / 400}}$$
Ratings are anchored at $\mu_{\text{XLM-RoBERTa-base}} \equiv 1,000.0$.

---

## 4. Decontamination & Scientific Rigor

* **4-Word Verbatim Decontamination:** The test splits of all six core tasks and three diagnostics were scanned against the 34.9M token pretraining corpus of Masinissa. Any pretraining document sharing a 4-word verbatim sequence with an evaluation test sentence was scrubbed prior to pretraining.
* **Positive Control Verification:** A synthetic corpus containing known leak injections was evaluated through the decontamination harness to mathematically verify that the detector catches corruptions down to single homoglyph alterations.

---

## 5. Schema & Configuration Definitions

### `syntax`
```json
{
  "id": "kabsyntax_test_0042",
  "sent_id": "A104",
  "text": "Yerna-d yiwen n urgaz ɣer dinna.",
  "tokens": ["Yerna-d", "yiwen", "n", "urgaz", "ɣer", "dinna", "."],
  "upos": ["VERB", "NUM", "ADP", "NOUN", "ADP", "ADV", "PUNCT"],
  "head": [0, 1, 4, 2, 6, 1, 1],
  "deprel": ["root", "nsubj", "case", "nmod", "case", "advmod", "punct"]
}
```

### `diag_spell`
```json
{
  "id": "diag_spell_0112",
  "category": "CONSTRUCT_ANNEX",
  "good": "Iwala wexxam-nni seg lebɛid.",
  "bad": "Iwala axxam-nni seg lebɛid.",
  "lemma": "axxam"
}
```

---

## 6. Citation

```bibtex
@dataset{agbalu_kabencbench_2026,
  title  = {KabEncBench: A Multi-Task Representation Learning Benchmark for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabEncBench},
  note   = {Covers 6 core tasks, 3 diagnostics, 100,609 annotated instances across Kabyle linguistics}
}
```
