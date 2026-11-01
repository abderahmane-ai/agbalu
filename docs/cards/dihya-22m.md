---
language:
- kab
- shi
- rif
- taq
- tzm
- shy
- mzb
- arq
- ary
- ara
- fra
- eng
- spa
- zho
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
- tumzabt
- moroccan-amazigh
- darija
- algerian-arabic
- moroccan-arabic
- arabizi
- language-identification
- conformal-prediction
- low-resource
pipeline_tag: text-classification
metrics:
- accuracy
- f1
model-index:
- name: Dihya-22M
  results:
  - task:
      type: text-classification
      name: Pan-Amazigh, North African Arabic dialects, and world language identification (22 classes, 5 scripts)
    dataset:
      type: dihya-22m-heldout
      name: AƔBALU Dihya-22M held-out set (14,247 sentences across 36 language-domain cells)
    metrics:
    - type: accuracy
      value: 0.9190
      name: Accuracy over 22 classes
    - type: f1
      value: 0.9180
      name: Macro-F1 over 22 classes
---

# Dihya-22M

A 22.0M-parameter (21,999,616 published) byte-level Conv-Transformer that upgrades Dihya-5M across
the entire Berber dialect continuum in both Latin and Neo-Tifinagh scripts, North African Arabic
dialects in Arabic script and Latin Arabizi, Modern Standard Arabic, and major world languages:
twenty-two classes across five scripts.

On a stratified held-out set of 14,247 sentences across 36 `(language, domain)` cells it reaches
**91.90% accuracy and 0.9180 macro-F1** over all twenty-two classes. Every single class achieves
$\ge 71.4\%$ recall, and its worst group recall is 63.33% on the scarcest Shawiya cell.

Against Dihya-5M, it expands the inventory from seven classes to twenty-two, adds Neo-Tifinagh,
Algerian Darija, Moroccan Darija, and Arabizi, lifts overall accuracy from **85.52% to 91.90%**,
and resolves the two central weaknesses of the 5M checkpoint:
1. **Kabyle precision rises from 0.626 to 0.900** while keeping recall at 0.925. Dihya-5M traded
   precision for recall, misrouting 37% of its Kabyle predictions to sibling text; Dihya-22M ends
   that trade-off.
2. **Central Atlas Tamazight recall rises from 0.660 to 0.948**, and its confusion into Kabyle
   drops from **16.0% down to 2.58%**. Tarifit confusion drops from **6.7% to 1.38%**, and Tamasheq
   from **7.3% to 0.16%**. Across Neo-Tifinagh varieties, confusion to Kabyle is $\le 2.12\%$
   (0.00% for Tamazight, Tarifit, and Tamasheq).

It ships with a split-conformal quantile ($\hat{q} = 0.9526$ at 99.0% nominal coverage, reaching
98.28% empirical coverage), so it returns a *set* of labels rather than one when text lies on the
continuum and carries no distinguishing marker.

## The task

Language identification across the Berber dialect continuum and North Africa confronts three structural challenges:

1. **Morphological discrimination over shared roots**: Amazigh varieties share substantial core vocabulary; distinguishing evidence is predominantly affixal, phonotactic, and clitic. Short sentences lacking dialect-specific markers are genuinely ambiguous.
2. **Upstream corpus contamination**: Broad-coverage classifiers (such as NLLB's `lid218e` and GlotLID) fail on this continuum by collapsing sibling varieties into Kabyle—mislabelling **91.3% of Shawiya, 80.7% of Tarifit, 79.3% of Tashelhit, and 73.3% of Central Atlas Tamazight**. Unfiltered web and parallel corpora inherit this silent contamination.
3. **Multi-script and dialect boundaries**: Written Berber alternates between Latin and Neo-Tifinagh orthographies, while North African Arabic spans Arabic script and Latin-digit Arabizi alongside close dialectal proximity between Algerian (`arq`) and Moroccan (`ary`) Darija.

## Results

**14,247 sentences across 36 `(language, domain)` cells**, held out completely disjoint from train
(113,974) and dev (7,145).

### Against the public identifiers and Dihya-5M

All systems scored in one pass over the held-out test rows:

| system | accuracy | classes it can name | Amazigh coverage | Darija & Arabizi |
|---|---|---|---|---|
| **Dihya-22M** | **91.90%** | **22 of 22** | **13 varieties (Latin & Tifinagh)** | **`arq`, `ary`, Arabizi** |
| Dihya-5M | 85.52% | 7 of 22 | 6 varieties (Latin only) | none |
| GlotLID | 41.78% | 3 of 22 | 3 varieties (Latin only) | conflated / no Arabizi |
| NLLB `lid218e` | 28.33% | 2 of 22 | 2 varieties (Latin only) | conflated / no Arabizi |

Public identifiers cannot name Shawiya, Central Atlas Tamazight, Tarifit, Tumzabt, or Arabizi. When
asked to classify texts in these varieties, their accuracy is zero by construction.

**The quantity that matters downstream is how much sibling text ends up labelled Kabyle**, since
that is what a filter keyed on `kab` silently ingests:

| true language | script | → Kabyle, Dihya-22M | → Kabyle, Dihya-5M | → Kabyle, GlotLID | → Kabyle, `lid218e` |
|---|---|---|---|---|---|
| Tashelhit | Latin | **1.38%** | 2.0% | 17.3% | 79.3% |
| Tarifit | Latin | **1.38%** | 6.7% | 90.0% | 80.7% |
| Tumzabt | Latin | **1.88%** | — | — | — |
| Central Atlas Tamazight | Latin | **2.58%** | 16.0% | 66.7% | 73.3% |
| Shawiya | Latin | **21.43%** | 19.3% | 96.0% | 91.3% |
| Tamasheq | Latin | **0.16%** | 7.3% | 0.0% | 6.7% |
| Tashelhit | Tifinagh | **0.50%** | — | — | — |
| Central Atlas Tamazight | Tifinagh | **0.00%** | — | — | — |
| Tarifit | Tifinagh | **0.00%** | — | — | — |
| Standard Moroccan | Tifinagh | **2.12%** | — | — | — |
| Tamasheq | Tifinagh | **0.00%** | — | — | — |

Central Atlas Tamazight leakage into Kabyle drops sixfold from Dihya-5M's 16.0% down to 2.58%, and
Tamasheq drops from 7.3% to 0.16%. In Tifinagh, cross-leakage into Kabyle is virtually zero.

### Per class, all twenty-two

The twenty-two-class report, which is what `test-report.json` holds:

| label | language / dialect | script | support | precision | recall | F1 |
|---|---|---|---|---|---|---|
| `kab_Latn` | Kabyle | Latin | 800 | 0.900 | 0.925 | 0.912 |
| `kab_Tfng` | Kabyle | Neo-Tifinagh | 800 | 0.971 | 0.879 | 0.923 |
| `shy_Latn` | Shawiya | Latin | 98 | 0.864 | 0.714 | 0.782 |
| `mzb_Latn` | Tumzabt / Mozabite | Latin | 800 | 0.986 | 0.963 | 0.974 |
| `shi_Latn` | Tashelhit | Latin | 800 | 0.984 | 0.975 | 0.979 |
| `shi_Tfng` | Tashelhit | Neo-Tifinagh | 800 | 0.716 | 0.744 | 0.730 |
| `tzm_Latn` | Central Atlas Tamazight | Latin | 659 | 0.947 | 0.948 | 0.948 |
| `tzm_Tfng` | Central Atlas Tamazight | Neo-Tifinagh | 368 | 0.986 | 0.927 | 0.955 |
| `rif_Latn` | Tarifit | Latin | 800 | 0.975 | 0.979 | 0.977 |
| `rif_Tfng` | Tarifit | Neo-Tifinagh | 69 | 0.938 | 0.884 | 0.910 |
| `zgh_Tfng` | Standard Moroccan Amazigh | Neo-Tifinagh | 800 | 0.678 | 0.743 | 0.709 |
| `taq_Latn` | Tamasheq | Latin | 627 | 1.000 | 0.994 | 0.997 |
| `taq_Tfng` | Tamasheq (Tawallammat) | Neo-Tifinagh | 265 | 1.000 | 0.989 | 0.994 |
| `arq_Arab` | Algerian Arabic (Darija) | Arabic | 788 | 0.738 | 0.898 | 0.811 |
| `arq_Arabizi` | Algerian Arabic (Arabizi) | Latin/Numerals | 223 | 0.911 | 0.919 | 0.915 |
| `ary_Arab` | Moroccan Arabic (Darija) | Arabic | 800 | 0.846 | 0.781 | 0.812 |
| `ary_Arabizi` | Moroccan Arabic (Arabizi) | Latin/Numerals | 800 | 0.979 | 0.990 | 0.984 |
| `ara_Arab` | Modern Standard Arabic | Arabic | 750 | 0.977 | 0.835 | 0.900 |
| `fra_Latn` | French | Latin | 800 | 0.995 | 0.998 | 0.996 |
| `eng_Latn` | English | Latin | 800 | 0.989 | 0.995 | 0.992 |
| `spa_Latn` | Spanish | Latin | 800 | 0.995 | 0.995 | 0.995 |
| `zho_Hani` | Chinese | Hanzi | 800 | 1.000 | 1.000 | 1.000 |

**Kabyle precision is 0.900, up from 0.626 in Dihya-5M.** Dihya-5M traded precision for recall,
sending 37% of its Kabyle predictions to sibling varieties. Dihya-22M resolves that tension: Kabyle
precision reaches 0.900 while maintaining 0.925 recall.

**Central Atlas Tamazight recall rises from 0.660 to 0.948**, resolving the largest weakness of the
5M checkpoint.

**Shawiya recall reaches 0.714** on 98 test rows, against 0.687 in Dihya-5M, with precision rising to
0.864. 447 training rows is what exists in machine-readable form.

### Arabic dialect discrimination

Separating Algerian Darija (`arq`), Moroccan Darija (`ary`), and Modern Standard Arabic (`ara`) in
Arabic script and Arabizi:

| True \ Predicted | `arq_Arab` | `arq_Arabizi` | `ary_Arab` | `ary_Arabizi` | `ara_Arab` |
|---|---|---|---|---|---|
| **`arq_Arab`** (Algerian Arabic) | **708** (89.8%) | 0 | 73 (9.3%) | 0 | 7 (0.9%) |
| **`arq_Arabizi`** (Algerian Arabizi) | 0 | **205** (91.9%) | 0 | 14 (6.3%) | 0 |
| **`ary_Arab`** (Moroccan Arabic) | 166 (20.8%) | 0 | **625** (78.1%) | 0 | 8 (1.0%) |
| **`ary_Arabizi`** (Moroccan Arabizi) | 0 | 6 (0.8%) | 0 | **792** (99.0%) | 0 |
| **`ara_Arab`** (Modern Standard Arabic) | 83 (11.1%) | 0 | 41 (5.5%) | 0 | **626** (83.5%) |

Arabizi and Arabic script have zero cross-leakage (0 errors). Over 89.8% of Algerian Darija and 78.1% of
Moroccan Darija are separated correctly; on shared Maghrebi idioms the conformal set emits
`('arq_Arab', 'ary_Arab')`.

### By `(language, domain)` cell

Reporting is on 34 active `(language, domain)` cells with support $\ge 20$:

| | recall |
|---|---|
| median cell | **0.957** (vs 0.921 in 5M) |
| cells at or above 0.90 | **21 of 34** |
| cells at or above 0.70 | **31 of 34** |
| cells at 1.000 | 2 — Tashelhit scripture, Chinese foreign |

The lowest cells with support $\ge 20$:

| cell | support | recall |
|---|---|---|
| `shy_Latn/bench_gold` | 30 | 0.633 |
| `shy_Latn/everyday` | 44 | 0.636 |
| `ary_Arab/negative_dialect` | 135 | 0.637 |
| `zgh_Tfng/everyday` | 758 | 0.740 |
| `shi_Tfng/everyday` | 800 | 0.744 |

The Shawiya cells reflect the language's digital scarcity. `ary_Arab/negative_dialect` holds Moroccan
colloquial text sharing vocabulary with western Algerian dialects. `zgh_Tfng` and `shi_Tfng` share
everyday lexicon in standard Neo-Tifinagh orthography.

### Conformal prediction sets

Calibrated by split conformal at target coverage 99.0%. Fitted on one half of the development
set (3,551 sentences) and coverage measured on the other (3,551 held-out sentences):

| | |
|---|---|
| fitted on | 3,551 dev sentences |
| coverage measured on | 3,551 held-out dev sentences |
| target coverage | 99.0% |
| empirical coverage | **98.28%** |
| quantile `q̂` | **0.9526** |
| mean set size | **1.259 labels** (across 22 classes) |
| singletons | **79.86%** (2,836 of 3,551) |

## Usage

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "agbalu/Dihya-22M", trust_remote_code=True
)

for row in model.identify([
    "Azul fell-awen, amek tettilim ass-a?",
    "Mayemmi ur teccid ticemrarin nnecm a gma?",
    "Bonjour tout le monde, comment allez-vous?",
    "ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ.",
    "Hello everyone, how are you doing today?",
    "صباح الخير كيداير لاباس عليك",
    "واش راك خويا لاباس؟",
]):
    print(row["language"], round(row["confidence"], 4), row["prediction_set"])
# kab_Latn 0.5295 ('kab_Latn', 'shy_Latn')
# rif_Latn 0.5485 ('kab_Latn', 'rif_Latn')
# fra_Latn 0.9444 ('fra_Latn',)
# kab_Tfng 0.9265 ('kab_Tfng',)
# eng_Latn 0.9498 ('eng_Latn',)
# ary_Arab 0.8180 ('arq_Arab', 'ary_Arab')
# arq_Arab 0.4889 ('arq_Arab', 'ary_Arab')
```

`identify` applies the conformal quantile that the numbers above were measured with; it travels in
`config.json`. The plain `forward` returns raw cosine logits.

The tokenizer is the 256 UTF-8 byte values and is only needed for batching:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("agbalu/Dihya-22M")
model = AutoModelForSequenceClassification.from_pretrained(
    "agbalu/Dihya-22M", trust_remote_code=True
)
batch = tokenizer(["Azul fell-awen", "صباح الخير"], return_tensors="pt", padding=True, truncation=True)
logits = model(**batch).logits
```

Prediction sets widen where text is genuinely ambiguous. The greeting above returns
`('kab_Latn', 'shy_Latn')` — Shawiya and Kabyle share that greeting and neither carries a marker
separating them. A forced singleton there would be an arbitrary guess.

## Intended use

Screening and routing text in corpus pipelines: tagging web scrapes and OCR output across Latin,
Tifinagh, and Arabic scripts, discarding non-target lines, and routing input to variety-specific
models. Use the prediction set rather than the argmax where a wrong label would corrupt a corpus.

**Not suitable for** distinguishing sub-varieties inside any of the twenty-two classes, single
isolated words lacking morphological context, or attributing authors on the basis of dialect.

## Architecture

| | |
|---|---|
| Parameters, trained | 22,198,528 |
| Parameters, published | 21,999,616 |
| Encoder layers | 10 (vs 6 in 5M) |
| Hidden / feed-forward | 384 / 1,344 (SwiGLU) (vs 256 / 704 in 5M) |
| Attention heads | 6 × 64, rotary positions |
| Conv stem | four parallel depthwise-separable 1D convolutions, kernels 3, 5, 7, 11 |
| Pooling | attentive (learned score per position) |
| Head | cosine, scale 28.0 |
| Vocabulary | 258 (256 UTF-8 bytes, `[PAD]`, `[UNK]`) |
| Maximum input | 512 bytes (vs 256 in 5M) |

The published model excludes the auxiliary script classification head, binary filter head, and
contrastive projection head (198,912 parameters), which shape representations during training and are
never read at inference.

**Four kernel widths** because evidence sits at four physical scales: grapheme clusters and diacritics
($k=3$), inflectional affixes ($k=5$), clitic chains and annexation markers ($k=7$), and word stems ($k=11$).

## Training

| | |
|---|---|
| training rows | 113,974 across 36 `(language, domain)` cells |
| dev rows | 7,145 |
| schedule | 8 epochs, 4,752 steps; batch 16 cells × 12 rows |
| published checkpoint | step 2,072, selected on dev selection score |
| dev score there | 0.9157 macro-F1, 0.6364 worst-group recall |
| training time | 24.8 minutes at 532 seq/s |

Five objectives address long-tail imbalance, domain collapse, and multi-script structure:
1. **LDAM margins** for the long tail — $m_k = C / n_k^{1/4}$ ([Cao et al., NeurIPS 2019](https://arxiv.org/abs/1906.07413)).
2. **Convex Group-DRO** — exponentiated-gradient weights over the active `(language, domain)` groups ([Sagawa et al., ICLR 2020](https://arxiv.org/abs/1911.08731)) with convex normalization.
3. **Domain-aware supervised contrastive learning** — negatives mined from a different language in the same domain first ([ConLID](https://arxiv.org/abs/2506.15304)) over a 4,096-entry memory bank.
4. **Multi-task auxiliary heads** — script classification (5 scripts) and binary Amazigh filtering.
5. **Deferred schedule** — margins and group weights are disabled for the first half of training (steps 0–2,376).

Selection is on the mean of dev macro-F1 and dev worst-group recall, never on loss.

## Limitations

**Dialect continuum ambiguity**: Short sentences of shared vocabulary carrying no variety-specific morphology cannot be deterministically resolved to a single class. The conformal prediction set captures this ambiguity directly by returning a multi-label set.

**Shawiya data scarcity**: 447 training rows and 98 test rows represent the entirety of machine-readable Shawiya text currently available; reported metrics reflect these sample size bounds.

**Maghrebi Arabic dialect overlap**: Algerian and Moroccan Arabic share substantial colloquial syntax and lexicon, leading to cross-dialect conformal sets on shared idioms.

**Input length**: Evaluated on sentence-level text; single isolated words lack sufficient phonotactic context for reliable attribution.

**Code-switching**: Outputs sentence-level classifications; token-level language diarization is not supported.

## Files

| file | description |
|---|---|
| `model.safetensors` | 88.0 MB, 21,999,616 parameters |
| `config.json` | architecture, 22 classes, prior shift and conformal quantile |
| `modeling_dihya.py`, `configuration_dihya.py` | architecture, `transformers` and `torch` only |
| `tokenizer.json`, `tokenizer_config.json` | the 258-entry byte vocabulary |
| `calibration.json` | the calibration, in full |
| `test-report.json` | per class and per `(language, domain)` cell |

No optimiser state, no scheduler state, no RNG. Training cannot be resumed from these files.

## The name

**Dihya** (also called *al-Kāhina*, died c. 703 CE) was an Amazigh military leader from the Aurès
Mountains of what is now eastern Algeria — the Shawiya homeland — who led the Berber resistance.
The naming is homage and implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_dihya_22m_2026,
  title  = {Dihya-22M: Pan-Amazigh, North African Arabic, and World Language Identification},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Dihya-22M},
  note   = {22M parameters; 91.90\% accuracy over 22 classes on 14,247 held-out sentences}
}
```

## Licence

**Apache-2.0** on the weights and the code. The training corpus is assembled from sources under
mixed licences; see `resources/sibling_registry.yaml` and `resources/corpus_registry.yaml` for
per-source terms.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle and Amazigh corpus and model collection.
