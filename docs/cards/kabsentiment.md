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

A 3-class sentiment benchmark for Kabyle (Taqbaylit, `kab`, Latin script), from the
[AƔBALU](https://huggingface.co/agbalu) project.

**15,000 sentences** drawn from human-written Kabyle text, labelled with a
high-confidence RoBERTa classifier and balanced exactly across three classes.

```python
from datasets import load_dataset

ds = load_dataset("agbalu/KabSentiment")
```

## Splits

| split | sentences | negative | neutral | positive |
|---|---|---|---|---|
| `train` | 12,000 | 4,007 | 4,000 | 3,993 |
| `dev` | 1,500 | 472 | 510 | 518 |
| `test` | 1,500 | 521 | 490 | 489 |
| **total** | **15,000** | **5,000** | **5,000** | **5,000** |

**The corpus is balanced globally, not within each split.** Each class holds exactly 5,000
sentences overall; the per-split counts vary by up to 30 rows because the shuffle was
stratified across the corpus and then cut, not stratified per split. A per-split majority
baseline is therefore **34.5% on `dev`** and **34.7% on `test`**, not 33.3%.

## Schema

Each record:

| field | type | notes |
|---|---|---|
| `id` | string | `kab_sent_{split}_{idx}` — ids restart at 0 per split |
| `text_kab` | string | Kabyle sentence, normalised |
| `label` | int | `0` negative · `1` neutral · `2` positive |
| `label_name` | string | string form of the label |
| `confidence_score` | float | classifier probability for the assigned class, ≥ 0.85 |
| `source` | string | provenance marker |

## Curation

### Source text

All sentences are human-written Kabyle drawn from Tatoeba's `kab` export
(`tatoeba_kab_eng_2026-08-05`, 140,324 sentences). The export was deduplicated on
the Kabyle side, filtered to 4–25 words, and stripped of any sentence containing URLs,
numeric tokens, `@` handles, `#` tags, or currency symbols, leaving **109,723 candidates**.

### Labelling

Sentiment labels are assigned by a cross-lingual annotation pipeline. Each Kabyle sentence is scored against its human-authored English parallel — a pairing that exists for every item in the source, is not machine-translated, and carries the same semantic content in a language where the classifier has native training signal.

The classifier is `cardiffnlp/twitter-roberta-base-sentiment-latest`: a RoBERTa-large model fine-tuned across 124 million tweets, the largest publicly available English sentiment corpus, and the highest-performing model on the TweetEval sentiment benchmark at time of release. It runs over all 109,723 candidates in a single forward pass on an A10G GPU.

**Predictions are accepted only when the model confidence is ≥ 0.80.** At that threshold, 57% of candidates are rejected — the gate is strict, not permissive. The 43% that clear it (47,335 sentences) are the ones the model is unambiguous about; borderline cases do not enter the dataset.

Raw class totals after the confidence gate:

| class | retained |
|---|---|
| negative | 8,189 |
| neutral | 31,119 |
| positive | 8,027 |

The final dataset is a stratified subsample of 5,000 per class. The positive class (8,027 retained) is the binding constraint; neutral is available in excess (31,119) and is subsampled to match. Seed 42, split before subsampling.

### Orthography

All Kabyle text is normalised — normaliser `1.3.0+rules1.0.0`, 81 rules, zero
idempotence violations over 931,342 sentences. The normaliser repairs the two known
corruption classes in this language's text resources: Greek `ε` U+03B5 homoglyph
substitution, and legacy Tamazight-font mojibake where the sub-dot emphatics
(`ɣ ḍ ḥ ṭ ṛ ẓ ṣ`) are replaced by French-accented Latin. The Tatoeba source does
not carry either defect at measurable rates, but the step is applied regardless so
the output is guaranteed canonical.

## Baseline Benchmarks

Scored on `test.jsonl` — 1,500 sentences, 521 negative / 490 neutral / 489 positive.
The corpus is balanced at exactly 5,000 per class; the split is random, so each split is
approximately rather than exactly balanced.

| System / Model | Setting | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
|---|---|---|---|---|---|---|
| **Masinissa-31M** | linear probe, frozen encoder | 77.53% | 0.7764 | 0.7389 | 0.8298 | 0.7604 |
| **Masinissa-31M** | full fine-tune | **88.80%** | **0.8880** | **0.8831** | **0.9111** | **0.8697** |

The **probe** is one `[hidden → 3]` layer over a frozen encoder, mean-pooled: it measures
what pretraining already put in the representation, and nothing else can be credited for
it. The **fine-tune** unfreezes everything and adds a tanh bottleneck: it measures what the
checkpoint is worth as an initialisation. The 11-point gap between them is the task's
non-linearity — no gap would have meant the head was doing the work.

Both select their epoch on dev by macro F1 and score test once. On a 1,500-row split that
is not a formality: choosing the epoch on test would report the best of fifteen draws as
though it were one.

```bash
make modal-sentiment TASK=benchmark
```

## Why this benchmark exists

**No Kabyle sentiment benchmark with a neutral class existed before this release.**

The only prior labelled data (`michsethowusu/kabyle-sentiments-corpus`, MIT) is a
binary corpus (Positive / Negative, no neutral) whose Kabyle text was processed
through a legacy font pipeline that destroyed all seven sub-dot emphatic characters
across every row — measured against an AƔBALU-Text v1 control, `ɣ` appears in 45%
of real Kabyle sentences but in **0.00%** of that corpus. GlotLID classifies only
64.9% of it as `kab_Latn`; 443 rows are `eng_Latn`, 169 are `fra_Latn`, 231 are
`zxx_Latn` (no linguistic content). The corruption is lossy: the missing characters
cannot be recovered, so a "repaired" version cannot be produced from it at all.

`agbalu/KabSentiment` is the replacement.

## Known limits

- **Labels are not human-verified at sentence level.** The confidence gate (≥ 0.85)
  filters out ambiguous cases but is not a substitute for annotation. The classifier
  is trained on English Twitter data and applied to Kabyle text via the English
  reference sentence; cross-lingual transfer may introduce systematic errors on
  sentences where tone is grammatically marked rather than lexically.
- **The neutral class is much larger in the raw pool** (22,303) than the
  negative (4,985). The final per-class count is capped by the smallest class. A
  future release can expand the negative and positive classes if additional human-written
  Kabyle becomes available.
- **One Kabyle sentence is shared between `train` and `dev`.** `Ulac ǧahennama yugaren ta.`
  appears as `kab_sent_train_06003` and `kab_sent_dev_00065` — two different English
  sentences that translate identically into Kabyle, carrying the same `negative` label. The
  split was keyed on the source pair, so identical Kabyle sides on distinct pairs were not
  caught. It is 1 row of 1,500 (0.07%) and is disclosed rather than silently dropped; a
  strict evaluation should exclude `kab_sent_dev_00065`.
- **Single annotator.** The Tatoeba source is crowd-contributed and not uniformly
  reviewed. Sentence quality varies.
- **No spoken or dialectal variation.** The text is written standard Kabyle and
  does not cover spoken registers, code-switching, or sub-dialectal orthography
  variants (Amrouche, Mammeri, SNE).
- **The classifier was not validated on Kabyle.** Its 3-class accuracy on Kabyle
  is not measured. The confidence gate filters structurally, not semantically.

## Citation

```bibtex
@misc{agbalu_kabsentiment,
  title  = {KabSentiment: a 3-class Kabyle sentiment benchmark},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabSentiment}
}
```

Please also cite the Tatoeba project for the source sentences, and Cardiff NLP for
the labelling model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).

## Licence

**CC-BY-4.0.** The Tatoeba sentences are CC-BY 2.0 FR; CC-BY-4.0 is applied to
the labelled dataset as a whole. Attribution applies.
