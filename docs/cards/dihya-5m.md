---
language:
- kab
- shi
- rif
- taq
- tzm
- shy
license: apache-2.0
library_name: transformers
tags:
- berber
- amazigh
- kabyle
- tashelhit
- tarifit
- tamasheq
- tamazight
- shawiya
- language-identification
- conformal-prediction
- low-resource
pipeline_tag: text-classification
metrics:
- accuracy
- f1
model-index:
- name: Dihya-5M
  results:
  - task:
      type: text-classification
      name: Berber language identification, six Latin-script varieties plus rejection
    dataset:
      type: berber-lid-heldout
      name: AƔBALU Berber LID held-out set (1,050 sentences, 150 per class)
    metrics:
    - type: accuracy
      value: 0.8552
      name: Accuracy over 7 classes
    - type: f1
      value: 0.8562
      name: Macro-F1 over 7 classes
---

# Dihya-5M

> [!NOTE]
> **Multilingual & Multiscript Successor Available: [`agbalu/Dihya-22M`](https://huggingface.co/agbalu/Dihya-22M).**
> Dihya-22M expands coverage from 7 Latin-script classes to 22 classes across 5 scripts (adding Neo-Tifinagh for all Amazigh varieties, Algerian and Moroccan Darija in Arabic script and Arabizi, MSA, and high-resource languages). It increases 7-class accuracy from 85.52% to **91.90%**, raises Kabyle precision from 0.626 to **0.900**, and significantly reduces sibling leakage.

A 5.09M-parameter byte-level Conv-Transformer designed to classify six Latin-script Amazigh varieties and an explicit `NOT_AMAZIGH` rejection class: Kabyle (`kab_Latn`), Tashelhit (`shi_Latn`), Tarifit (`rif_Latn`), Tamasheq (`taq_Latn`), Central Atlas Tamazight (`tzm_Latn`), and Shawiya (`shy_Latn`).

Evaluated on a source-stratified held-out test set of 1,050 sentences (150 per class), Dihya-5M achieves **85.52% accuracy and 0.8562 macro-F1** across all 7 classes. Scored strictly on the six Berber varieties, it reaches **84.44% accuracy**, compared to **41.78% for GlotLID** and **28.33% for NLLB `lid218e`** on the identical test split.

Dihya-5M incorporates temperature scaling and split-conformal calibration, outputting calibrated prediction sets to manage classification uncertainty on shared vocabulary.

## The Task

Northern and Saharan Berber varieties share significant core vocabulary (*aman* "water", *argaz* "man", *awal* "word") alongside common Latin orthographic conventions. Distinguishing features are predominantly morphosyntactic, clitic, and phonotactic.

Because standard multilingual language identification tools (such as GlotLID and NLLB `lid218e`) do not support Tarifit, Central Atlas Tamazight, or Shawiya, public web-mined datasets frequently misclassify sibling text as Kabyle:

| Source Variety | Misclassified as Kabyle by GlotLID | Misclassified as Kabyle by NLLB `lid218e` | Misclassified as Kabyle by Dihya-5M |
|---|---|---|---|
| **Tashelhit** | 17.3% | 79.3% | **2.0%** |
| **Tarifit** | 90.0% | 80.7% | **6.7%** |
| **Tamasheq** | 0.0% | 6.7% | **7.3%** |
| **Central Atlas Tamazight** | 66.7% | 73.3% | **16.0%** |
| **Shawiya** | 96.0% | 91.3% | **19.3%** |

Dihya-5M provides explicit classification across all six varieties, mitigating sibling text leakage into downstream monolingual corpora.

## Evaluation Results

### Comparison Against Public Identifiers

Evaluated across the 900 Amazigh sentences of the held-out test split (150 sentences per variety):

| System | Accuracy (6 Amazigh classes) | Classes in Vocabulary | Macro-F1 (Supported Classes) |
|---|---|---|---|
| **Dihya-5M** | **84.44%** | **6 of 6** | **0.8493** |
| GlotLID | 41.78% | 3 of 6 | 0.6837 |
| NLLB `lid218e` | 28.33% | 2 of 6 | 0.6022 |

### Seven-Class Detailed Metrics

Complete 7-class performance on the 1,050-sentence held-out test split:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `NOT_AMAZIGH` | 150 | 0.949 | 0.987 | **0.967** |
| `taq_Latn` (Tamasheq) | 150 | 1.000 | 0.847 | **0.917** |
| `rif_Latn` (Tarifit) | 150 | 0.916 | 0.873 | **0.894** |
| `shi_Latn` (Tashelhit) | 150 | 0.811 | 0.973 | **0.885** |
| `shy_Latn` (Shawiya) | 150 | 0.936 | 0.687 | **0.792** |
| `tzm_Latn` (Central Atlas Tamazight) | 150 | 0.952 | 0.660 | **0.780** |
| `kab_Latn` (Kabyle) | 150 | 0.626 | 0.960 | **0.758** |

- **Kabyle Precision:** Evaluated under an adversarial 1:6 class distribution (150 Kabyle against 900 sibling sentences), precision is 0.626 with 0.960 recall. In balanced or Kabyle-majority corpora, precision scales accordingly.
- **Under-Resourced Classes:** Central Atlas Tamazight and Shawiya achieve 0.660 and 0.687 recall respectively, representing the first published benchmarked metrics for both varieties.

### Domain-Stratified Performance

Evaluated across 22 `(language, domain)` cells in `held-out-report.json`:
- **Median cell recall:** 0.921
- **Cells $\ge 0.90$ recall:** 13 of 22
- **Cells $\ge 0.70$ recall:** 19 of 22
- **Cells at 1.00 recall:** 5 (Scripture texts across Kabyle, Tarifit, Tashelhit; Tashelhit everyday text; non-Amazigh Latin text).

Challenging cells with limited training support:
- `tzm_Latn / interface` (support 61): 0.410 recall (formulaic software strings with cross-dialect lexical overlap).
- `shy_Latn / everyday` (support 74): 0.676 recall.
- `shy_Latn / interface` (support 3): 0.000 recall.

### Conformal Prediction Calibration

Split-conformal prediction sets calibrated at 99.0% target coverage:

| Metric | Value |
|---|---|
| **Calibration Set** | 2,047 validation sentences |
| **Empirical Evaluation Set** | 2,046 held-out validation sentences |
| **Target Coverage** | 99.0% |
| **Empirical Coverage** | **98.78%** |
| **Quantile Threshold ($\hat{q}$)** | 0.9979 |
| **Mean Prediction Set Size** | **1.098 labels** |

## Intended Use

- **Corpus Filtering:** Identifying and sorting Amazigh web crawls, literary texts, and OCR scans into dialect-specific subsets.
- **Non-Berber Rejection:** Filtering out non-Amazigh text via the `NOT_AMAZIGH` class.
- **Ambiguity Routing:** Using calibrated prediction sets to route multi-label cases to human review or multi-dialect pipelines.
- **Out of Scope:** Tifinagh and Arabic-script text (supported in [`Dihya-22M`](https://huggingface.co/agbalu/Dihya-22M)); sub-dialectal differentiation within a single variety.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "agbalu/Dihya-5M"
model = AutoModelForSequenceClassification.from_pretrained(REPO, trust_remote_code=True)

sentences = [
    "Azul fell-awen, amek tettilim ass-a?",
    "Mayemmi ur teccid ticemrarin nnecm a gma?",
    "Bonjour tout le monde, comment allez-vous?",
]

for row in model.identify(sentences):
    print(row["language"], round(row["confidence"], 4), row["prediction_set"])
# kab_Latn 0.9959 ('kab_Latn', 'shy_Latn')
# rif_Latn 0.9998 ('rif_Latn',)
# NOT_AMAZIGH 0.9997 ('NOT_AMAZIGH',)
```

Batched inference using the byte tokenizer:

```python
tokenizer = AutoTokenizer.from_pretrained(REPO)
batch = tokenizer(["Azul fell-awen"], return_tensors="pt", padding=True, truncation=True)
logits = model(**batch).logits
```

## Architecture

| Parameter | Specification |
|---|---|
| **Total Parameters (Trained)** | 5,187,712 |
| **Total Parameters (Published)** | 5,089,408 (excluding training projection head) |
| **Encoder Layers** | 6 Transformer layers |
| **Hidden / Feed-Forward Dim** | 256 / 704 (SwiGLU) |
| **Attention Heads** | 8 heads (head dimension 32, RoPE embeddings) |
| **Convolutional Stem** | 3 parallel depthwise-separable 1D convolutions (kernels 3, 5, 7) |
| **Pooling** | Attentive pooling (learned per-position scoring) |
| **Classification Head** | Cosine classifier (scale factor 24.0) |
| **Input Representation** | 258-entry UTF-8 byte vocabulary (256 bytes, `[PAD]`, `[UNK]`) |
| **Maximum Context** | 256 bytes |

A byte-level representation ensures that multi-byte UTF-8 characters (`ɣ`, `ɛ`, `ḥ`, `ḍ`, `ṣ`, `ṭ`, `ẓ`, `ṛ`, `ṯ`) are processed without out-of-vocabulary tokenization issues across minority dialects.

## Training Recipe

| Parameter | Specification |
|---|---|
| **Training Data** | 77,756 sentences across 20 `(language, domain)` cells |
| **Validation Data** | 4,093 sentences |
| **Schedule** | 8 epochs (5,176 total steps; batch size 12 cells $\times$ 10 examples) |
| **Checkpoint** | Step 2,898 (`best.pt`, selected on validation macro-F1 and worst-group recall) |

**Training Objectives:**
- **LDAM Loss:** Label-Distribution-Aware Margin loss to balance representations across majority and long-tail classes (e.g. 24,123 Tashelhit sentences vs. 447 Shawiya sentences).
- **Group-DRO:** Distributionally robust optimization over `(language, domain)` groups to optimize performance on the worst-performing domain cell.
- **Supervised Contrastive Learning:** Memory bank of 2,048 embeddings with hard negatives sampled from different languages within the same domain.

**Checkpoint Validation Progression:**

| Checkpoint | Accuracy | Macro-F1 | Worst-Group Recall |
|---|---|---|---|
| **Step 2,898 (Published)** | 97.51% | 0.9363 | **0.8125** |
| Step 5,176 (Final Step) | 97.87% | 0.9401 | 0.6875 |

Step 2,898 was selected because it maximized worst-group recall (0.8125 vs 0.6875) while preserving macro-F1.

## Limitations

- **Lexical Overlap:** Short sentences composed entirely of shared pan-Berber roots without morphological markers produce multi-label prediction sets.
- **Adversarial Class Priors:** Precision on `kab_Latn` is 0.626 when evaluated on a 1:6 imbalanced test set; on Kabyle-dominant input, precision is significantly higher.
- **Script Support:** Latin script only. For Tifinagh and Arabic script Amazigh varieties, use [`agbalu/Dihya-22M`](https://huggingface.co/agbalu/Dihya-22M).
- **Input Granularity:** Trained on sentence-level text; single isolated words lack sufficient context for reliable classification.

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | 20.4 MB | Model weights (5,089,408 parameters) |
| `config.json` | < 5 KB | Architecture configuration, label maps, and conformal thresholds |
| `modeling_dihya.py`, `configuration_dihya.py` | < 25 KB | Transformers modeling and configuration code |
| `tokenizer.json` | < 5 KB | 258-entry byte tokenizer definition |
| `calibration.json` | < 5 KB | Complete split-conformal calibration quantiles |
| `held-out-report.json` | < 20 KB | Full evaluation report by class and domain cell |

## Reproduction

Trained for 8 epochs with LDAM loss, Group-DRO, and supervised contrastive learning using AdamW on multi-domain Amazigh splits. Evaluated against GlotLID and NLLB `lid218e` on identical test splits. All reported metrics reproduce deterministically from the published checkpoint.

## The name

**Dihya** (also called *al-Kāhina*, died c. 703 CE) was an Amazigh military leader from the Aurès
Mountains of what is now eastern Algeria — the Shawiya homeland — who led the Berber resistance to
the Umayyad conquest of Ifriqiya. Much of what is written about her comes from sources composed
well after her lifetime and is contested; the Aurès and the resistance are the parts that are not.

The naming is homage and implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_dihya_2026,
  title  = {Dihya-5M: Language Identification Across the Berber Dialect Continuum},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Dihya-5M},
  note   = {5.09M parameters; 85.52% accuracy over seven classes on 1,050 held-out sentences}
}
```

## Licence

- **Weights & Code:** Apache-2.0.
- **Training Data:** Sourced under respective open licenses cataloged in `resources/sibling_registry.yaml`.
