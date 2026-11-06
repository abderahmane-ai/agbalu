---
license: cc-by-4.0
language:
  - kab
language_bcp47:
  - kab-Latn
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
task_ids:
  - sentiment-analysis
pretty_name: KabSentiment — Kabyle 3-class sentiment benchmark
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - low-resource
  - evaluation
  - sentiment-analysis
  - text-classification
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.parquet
      - split: dev
        path: dev.parquet
      - split: test
        path: test.parquet
---

# KabSentiment

A 3-class sentiment benchmark for Kabyle (Taqbaylit, `kab`, Latin script), from the [AƔBALU](https://huggingface.co/agbalu) project.

**15,000 sentences** drawn from human-written Kabyle text, labelled via cross-lingual teacher projection and balanced across negative, neutral, and positive classes.

```python
from datasets import load_dataset

ds = load_dataset("agbalu/KabSentiment")
```

## Splits

| Split | Sentences | Negative | Neutral | Positive |
|---|---|---|---|---|
| `train` | 12,000 | 4,007 | 4,000 | 3,993 |
| `dev` | 1,500 | 472 | 510 | 518 |
| `test` | 1,500 | 521 | 490 | 489 |
| **Total** | **15,000** | **5,000** | **5,000** | **5,000** |

The dataset contains exactly 5,000 sentences per class across the entire 15,000-sentence collection. Splits were partitioned via corpus-wide stratified shuffling, resulting in minor class variations within individual splits (majority class baselines are 34.5% on `dev` and 34.7% on `test`).

## Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique record ID (`kab_sent_{split}_{idx}`) |
| `text_kab` | string | Normalized Kabyle sentence |
| `label` | int | `0` negative, `1` neutral, `2` positive |
| `label_name` | string | Label string (`negative`, `neutral`, `positive`) |
| `confidence_score` | float | Classifier prediction probability for assigned class (≥ 0.80) |
| `source` | string | Originating provenance marker |

## Usage

```python
from datasets import load_dataset

train = load_dataset("agbalu/KabSentiment", split="train")
dev   = load_dataset("agbalu/KabSentiment", split="dev")
test  = load_dataset("agbalu/KabSentiment", split="test")

print(train[0])
# {'id': 'kab_sent_train_00000', 'text_kab': 'Ifukal-nwen ur sɛin ara lsas.',
#  'text_en': 'Your argument is unfounded.', 'label': 0, 'label_name': 'negative',
#  'confidence_score': 0.838, 'source': 'tatoeba_roberta_labeled'}
```

`label` encodes sentiment as `0` (negative), `1` (neutral), or `2` (positive), with `label_name` providing string identifiers. `text_en` provides the English parallel sentence from which the sentiment label was projected, included to enable auditability and cross-lingual analysis. `confidence_score` records the English teacher model's prediction probability.

## Curation

### Source Text

Sentences are human-written Kabyle text from the Tatoeba export (`tatoeba_kab_eng_2026-08-05`, 140,324 sentences). The export was deduplicated on the Kabyle side, filtered to sentence lengths between 4 and 25 words, and stripped of URLs, isolated digits, handle mentions, and currency symbols, yielding **109,723 candidate sentences**.

### Cross-Lingual Label Projection

Labels were projected from the aligned English source sentences using [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) ([Loureiro et al., TimeLMs, 2022](https://arxiv.org/abs/2202.03829)).

Predictions were retained only when teacher model confidence was $\ge 0.80$, filtering out 57% of candidates to eliminate ambiguous instances.

Candidate distribution following confidence thresholding:

| Class | Retained Candidates | Subsampled for Release |
|---|---|---|
| Negative | 8,189 | 5,000 |
| Neutral | 31,119 | 5,000 |
| Positive | 8,027 | 5,000 |

Each class was subsampled to exactly 5,000 sentences using seed 42.

### Orthography

All text is normalized under the AƔBALU reference normaliser (`1.3.0+rules1.0.0`, 81 rules), preserving all standard phonemic diacritics (`ɣ`, `ɛ`, `ḥ`, `ḍ`, `ṣ`, `ṭ`, `ẓ`, `ṛ`, `č`, `ǧ`).

## Baseline Benchmarks

Evaluated on `test.parquet` (1,500 sentences: 521 negative, 490 neutral, 489 positive):

| System / Model | Setting | Accuracy | Macro-F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---|---|---|---|---|
| **Masinissa-31M** | Linear probe, frozen encoder | 77.53% | 0.7764 | 0.7389 | 0.8298 | 0.7604 |
| **Masinissa-31M** | Full fine-tune | **88.80%** | **0.8880** | **0.8831** | **0.9111** | **0.8697** |
| [**Masinissa-Sentiment**](https://huggingface.co/agbalu/Masinissa-Sentiment) | Multi-seed average + calibrated | **90.27%** | **0.9026** | **0.9010** | **0.9160** | **0.8910** |

The frozen encoder linear probe evaluates representations by fitting a single linear layer over mean-pooled token states (77.53% accuracy). Full fine-tuning unfreezes encoder parameters with a projection bottleneck, reaching 88.80% accuracy. Both models select checkpoints on the validation split via macro-F1 before evaluation on the unseen test split.

### Benchmark Context

Prior sentiment resources for Kabyle were limited to binary classification without neutral labels, and frequently suffered from character loss where sub-dot diacritics were stripped during legacy font conversions. `KabSentiment` provides a balanced three-class dataset with canonical, verified Latin orthography.

## Known Limitations

- **Projected annotations**: Labels originate from an English teacher classifier via parallel sentence pairs rather than native human sentiment tagging in Kabyle.
- **Syntactic nuance**: While literal sentiment transfers reliably, culturally specific idioms or ironic expressions may not be fully captured by cross-lingual projection.
- **Shared sentence in train and dev**: One sentence (`Ulac ǧahennama yugaren ta.`) appears as both `kab_sent_train_06003` and `kab_sent_dev_00065` due to distinct English source sentences translating to the identical Kabyle string. Train and test share zero strings.
- **Register**: The text represents written standard Kabyle in Latin script and does not cover dialectal spoken transcripts or heavy code-switching with French.

## Citation

```bibtex
@misc{agbalu_kabsentiment,
  title  = {KabSentiment: a 3-class Kabyle sentiment benchmark},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabSentiment}
}
```

Please also cite the Tatoeba project for the source sentences, and Cardiff NLP for the teacher model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).

## Licence

**CC-BY-4.0.** The source Tatoeba sentences are CC-BY 2.0 FR; the compiled dataset is distributed under CC-BY-4.0 with attribution.
