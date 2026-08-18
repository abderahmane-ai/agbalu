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

A morphological inflection and analysis benchmark for Kabyle (Taqbaylit, `kab`, Latin script), from the
[AƔBALU](https://huggingface.co/agbalu) project.

**336,151 inflected verb form entries** across **13,226 unique verb lemmas**, partitioned into
paradigmatically sealed splits (0 paradigm leakage), plus **6,198 complete verb conjugation tables**.

```python
from datasets import load_dataset

inflect = load_dataset("agbalu/KabInflect", "inflection")
analysis = load_dataset("agbalu/KabInflect", "analysis")
paradigms = load_dataset("agbalu/KabInflect", "paradigms", split="train")
```

## `inflection`

Morphological Inflection (Seq2Seq Morphological Generation): `(lemma, feats)` → `form`.

336,151 total entries across 13,226 verb lemmas.

| split | lemmas | inflected forms |
|---|---|---|
| `train` | 10,580 | 270,026 |
| `dev` | 1,323 | 33,060 |
| `test` | 1,323 | 33,065 |
| **total** | **13,226** | **336,151** |

### Schema

| field | type | description |
|---|---|---|
| `id` | string | `kab_inflect_{split}_{idx}` |
| `lemma` | string | Base verb infinitif (canonical) |
| `feats` | string | Universal Dependencies (UD) FEATS (e.g. `Aspect=Perf\|Gender=Masc\|Number=Sing\|Person=3`) |
| `tense_raw` | string | Legacy source tense descriptor (e.g. `prétérit`, `aoriste intensif`) |
| `person_raw` | string | Legacy source person descriptor (e.g. `3s_m`, `1p`) |
| `form` | string | Inflected surface form, normalised; see Orthography — emphatic `ṛ` is unmarked |

## `analysis`

Morphological Analysis and Lemmatisation: `form` → `(lemma, feats)`.

Same 336,151 entries, indexed by inflected surface form.

## `paradigms`

6,198 full verb paradigms with French translations, irregularity indicators (`is_irregular`, `is_derived`), pattern verbs, and principal aspect stems (`imperative`, `aorist`, `preterite`, `negative_preterite`, `aorist_participle`, `preterite_participle`, `negative_preterite_participle`, `intensive_forms`).

## Paradigmatically Sealed Partitioning (Zero Paradigm Leakage)

**Upstream datasets randomly split individual inflected forms across train and test.** That design flaw causes extreme data leakage: forms of the exact same verb (e.g. `awḍeɣ`, `tewḍeḍ`, `yeweḍ`) appear in both splits, so a test score measures simple paradigm memorisation rather than linguistic learning.

`agbalu/KabInflect` fixes this:
- Entries are grouped strictly by **verb lemma** (`infinitif`).
- Splitting is performed at the **lemma level** with seed 42.
- **0% lemma overlap**: all 1,323 verbs in `test` (33,065 forms) are completely unseen during training. A system must learn the underlying rules of Kabyle morphology to succeed.

## Orthography

All Kabyle text is normalised under normaliser `1.3.0+rules1.0.0` (81 rules). The source
text carries 0.00% homoglyph error and 0.00% legacy-font mojibake.

🔴 **The emphatic `ṛ` is not represented, and this is a property of the source.** Measured
over all 270,026 training forms, `ṛ` U+1E5B occurs **0 times**, while `ḍ` appears in 14.43%
of forms, `ḥ` in 12.14%, `ṣ` in 3.36%, `ẓ` in 2.73% and `ṭ` in 4.37%. The upstream
conjugation resource writes plain `r` throughout — its lemmas are `ruḥ`, `ṣber` and `ɛreḍ`,
not `ṛuḥ`, `ṣbeṛ` and `ɛṛeḍ`.

The normaliser did not remove it; there was never anything to preserve. **The `ṛ`/`r`
contrast is therefore neutralised in this dataset**, and it is *not* neutralised in
[`agbalu/KabLex`](https://huggingface.co/datasets/agbalu/KabLex), which carries both `ṛuḥ`
and `ruḥ`. A model trained here will not produce emphatic `ṛ`, and scoring its output
against text that marks it will count every such form wrong.

Restoring it would mean deciding, for each `r`, whether the verb takes the emphatic — a
lexical question this dataset has no evidence for. It is disclosed rather than guessed.

## Evaluation Protocol & Baseline Floor

The benchmark evaluates morphological generation (`inflection`) and parsing (`analysis`) on **1,323 held-out test verb lemmas (33,065 unseen test forms)**.

| task | system | exact match | CER |
|---|---|---|---|
| inflection | copy the lemma unchanged | 4.07% | 30.73% |

```bash
make bench TASK=inflect
```

**The floor is not zero, and that is why it is measured rather than assumed.** Kabyle's
imperative singular *is* the citation form for most verbs, so a copy is correct in one cell
of nearly every paradigm — 4.07% of 33,065 test forms. A headline exact match read against
zero would be flattered by exactly that much.

No system has yet been scored on the `analysis` direction.

Because the splits are sealed by lemma, a model cannot reach the floor by memorising
paradigms: no test verb appears in any training cell.

## Known Limits

- **No emphatic `ṛ`.** 0 occurrences in 270,026 forms: the upstream conjugator writes `ruḥ`
  and `ṣber` rather than `ṛuḥ` and `ṣbeṛ`, so the distinction was never in the source to
  preserve. It is the most consequential limit here and it is confined to one letter — `ḍ`
  appears in 14.43% of forms and `ḥ` in 12.14%. Score on matched orthography, or normalise
  both sides; a model trained here and evaluated against `ṛ`-marking text loses every such
  form for a reason that is not the model's.
- **Every form appears exactly once.** The raw shards repeat rows verbatim — one verb's
  intensive stems can generate the same cell twice — so **8,594 identical rows were dropped**
  (344,745 → 336,151) and no form is silently reweighted during training. Earlier revisions of
  this dataset carried them; this one does not, and the baseline floor below was recomputed
  after the rebuild rather than carried over.
- **Verb-focused.** Covers Kabyle verbal morphology (the largest morphological paradigm family in the language); nominal annexed-state paradigms are covered separately in `agbalu/KabLex`.
- **Gloss language.** Paradigm translations are French, reflecting the source dictionary.
- **Standard orthography.** Uses standard Kabyle Latin script; does not cover dialectal spellings or Tifinagh (see `agbalu/KabTifinagh` for script conversion).

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
