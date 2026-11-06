---
license: cc-by-4.0
language:
  - kab
language_bcp47:
  - kab-Latn
size_categories:
  - 100K<n<1M
task_categories:
  - token-classification
  - text-generation
task_ids:
  - lemmatization
pretty_name: KabInflect — Kabyle morphological inflection and analysis benchmark
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - low-resource
  - morphology
  - inflection
  - lemmatization
configs:
  - config_name: inflection
    data_files:
      - split: train
        path: inflection/train.parquet
      - split: dev
        path: inflection/dev.parquet
      - split: test
        path: inflection/test.parquet
  - config_name: analysis
    data_files:
      - split: train
        path: analysis/train.parquet
      - split: dev
        path: analysis/dev.parquet
      - split: test
        path: analysis/test.parquet
  - config_name: paradigms
    data_files:
      - split: train
        path: paradigms/train.parquet
---

# KabInflect

A morphological inflection and analysis benchmark for Kabyle (Taqbaylit, `kab`, Latin script), from the [AƔBALU](https://huggingface.co/agbalu) project.

**336,151 inflected verb form entries** across **13,226 unique verb lemmas**, partitioned into lemma-disjoint splits (0 paradigm leakage), plus **6,198 complete verb conjugation tables**.

```python
from datasets import load_dataset

inflect = load_dataset("agbalu/KabInflect", "inflection")
analysis = load_dataset("agbalu/KabInflect", "analysis")
paradigms = load_dataset("agbalu/KabInflect", "paradigms", split="train")
```

## `inflection`

Morphological Inflection (Seq2Seq Morphological Generation): `(lemma, feats) → form`.

336,151 total entries across 13,226 verb lemmas.

| Split | Lemmas | Inflected forms |
|---|---|---|
| `train` | 10,580 | 270,026 |
| `dev` | 1,323 | 33,060 |
| `test` | 1,323 | 33,065 |
| **Total** | **13,226** | **336,151** |

### Schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique record ID (`kab_inflect_{split}_{idx}`) |
| `lemma` | string | Base verb citation form (`infinitif`) |
| `feats` | string | Universal Dependencies morphological features (e.g. `Aspect=Perf\|Gender=Masc\|Number=Sing\|Person=3`) |
| `tense_raw` | string | Source tense descriptor (e.g. `prétérit`, `aoriste intensif`) |
| `person_raw` | string | Source person descriptor (e.g. `3s_m`, `1p`) |
| `form` | string | Normalized inflected surface form |

## `analysis`

Morphological Analysis and Lemmatisation: `form → (lemma, feats)`.

336,151 entries indexed by inflected surface form, using the same lemma-disjoint split boundaries.

## `paradigms`

6,198 complete verb paradigms with French translations, irregularity flags (`is_irregular`, `is_derived`), pattern classifications, and principal aspect stems (`imperative`, `aorist`, `preterite`, `negative_preterite`, `aorist_participle`, `preterite_participle`, `negative_preterite_participle`, `intensive_forms`).

## Usage

```python
from datasets import load_dataset

inflection = load_dataset("agbalu/KabInflect", "inflection", split="train")
print(inflection[0])
# {'id': 'kab_inflect_train_000000', 'lemma': 'ttemnunnuḍ',
#  'feats': 'Aspect=Imp|VerbForm=Part', 'tense_raw': 'participe aoriste intensif',
#  'person_raw': 'participe', 'form': 'ittemnunnuḍen'}

analysis = load_dataset("agbalu/KabInflect", "analysis", split="test")
paradigms = load_dataset("agbalu/KabInflect", "paradigms", split="train")
```

The inflection task maps `(lemma, feats) → form`, while morphological analysis performs the reverse mapping `form → (lemma, feats)`.

## Lemma-Disjoint Partitioning

To prevent data leakage, dataset splits are partitioned strictly at the **verb lemma level** rather than splitting individual surface forms randomly:
- Entries are grouped by base verb lemma (`infinitif`).
- Splitting is performed across lemmas using seed 42.
- **0% lemma overlap**: All 1,323 verb lemmas in `test` (33,065 forms) are completely unseen in the training split, requiring models to generalize morphological inflection rules to novel lexemes rather than memorizing attested paradigms.

## Orthography

All Kabyle text is normalized under the AƔBALU reference normaliser (`1.3.0+rules1.0.0`, 81 rules).

In the underlying source verb resource, the emphatic `ṛ` (U+1E5B) is not distinguished from plain `r` (occurring 0 times across 270,026 training forms, whereas other emphatics like `ḍ`, `ḥ`, `ṣ`, `ṭ`, `ẓ` are fully marked). Consequently, the `r` vs. `ṛ` phonemic distinction is neutralized in this dataset. Models trained exclusively on KabInflect will not generate emphatic `ṛ`; when evaluating on corpora where `ṛ` is marked, references should be normalized accordingly to prevent spurious orthographic mismatches.

## Evaluation Protocol & Baseline Floor

The benchmark evaluates morphological generation (`inflection`) and parsing (`analysis`) on **1,323 held-out test verb lemmas (33,065 unseen test forms)**.

| Task | System / Baseline | Exact match | CER |
|---|---|---|---|
| Inflection | Copy lemma unchanged | 4.07% | 30.73% |

Because the imperative singular serves as the citation form for many Kabyle verbs, a trivial baseline copying the lemma unchanged achieves 4.07% exact match (and 30.73% CER) across the test split.

## Known Limitations

- **Unmarked `ṛ`**: Emphatic `ṛ` is written as plain `r` throughout the upstream verbal database, neutralizing this contrast.
- **Deduplication**: 8,594 redundant duplicate entries present in raw source tables were removed during curation (reducing 344,745 to 336,151 rows).
- **Verbal domain**: Focuses on Kabyle verbal morphology; nominal state alternations are provided in `agbalu/KabLex`.
- **Gloss language**: Translations and definitions within paradigm tables are in French.

## Citation

```bibtex
@misc{agbalu_kabinflect,
  title  = {KabInflect: a Kabyle morphological inflection and analysis benchmark},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabInflect}
}
```

Please also cite `boffire/kabyle-verbs` for the upstream data collection.

## Licence

**CC-BY-4.0.** Free to share, modify, and build upon with proper attribution.
