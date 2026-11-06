---
license: cc-by-sa-4.0
language:
  - kab
language_bcp47:
  - kab-Latn
multilinguality:
  - multilingual
size_categories:
  - 1K<n<10K
task_categories:
  - translation
  - text-classification
task_ids:
  - language-identification
pretty_name: KabBench — Kabyle evaluation suite
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - low-resource
  - evaluation
  - machine-translation
  - language-identification
configs:
  - config_name: mt
    data_files:
      - split: dev
        path: mt/dev.jsonl
      - split: devtest
        path: mt/devtest.jsonl
  - config_name: lid
    data_files:
      - split: test
        path: lid/test.jsonl
---

# KabBench

Evaluation datasets for Kabyle (Taqbaylit, `kab`, Latin script), from the [AƔBALU](https://huggingface.co/agbalu) project.

The dataset contains two configurations:
- **`mt`**: An orthographically corrected reference for machine translation evaluation, resolving homoglyph corruption present in 16.2% of sentences in the public FLORES+ Kabyle test set.
- **`lid`**: A balanced six-way Amazigh language discrimination benchmark evaluating separation between Kabyle and its sibling varieties (`kab`, `shi`, `rif`, `taq`, `tzm`, `shy`).

```python
from datasets import load_dataset

mt = load_dataset("agbalu/KabBench", "mt", split="devtest")
lid = load_dataset("agbalu/KabBench", "lid", split="test")
```

## `mt` — Corrected Kabyle MT Reference

2,009 sentences: 997 `dev` and 1,012 `devtest`.

**326 sentences (16.2%)** in the public FLORES+ reference contain systematic homoglyph substitutions (predominantly Greek `ε` U+03B5 substituted for Latin `ɛ` U+025B, alongside related Berber Latin encoding errors). Because existing benchmark scores were measured against uncorrected references, models producing canonically spelled Kabyle receive lower automated BLEU/chrF++ scores unless references are repaired or dual-normalized.

| Field | Type | Description |
|---|---|---|
| `id` | int | Sentence identifier within split (restarts at 0 per split; key on `(split, id)`) |
| `split` | string | `dev` or `devtest` |
| `text` | string | Normalized, orthographically repaired Kabyle sentence |
| `corrected` | bool | Whether the sentence was modified from the original FLORES+ reference |

On the uncorrected FLORES+ `devtest`, approximately 2.71 BLEU is mathematically unattainable by a system producing standard Kabyle orthography due to these reference substitutions. Evaluating models against this repaired reference isolates translation quality from orthographic encoding errors.

All text is normalized using the AƔBALU reference normaliser (`1.3.0+rules1.0.0`, 81 rules).

### Scoring Protocol

Report results under two conditions:
1. **Raw evaluation**: Hypotheses scored directly against the dataset text.
2. **Normalized evaluation**: Both hypotheses and references normalized prior to scoring.

The gap between these scores isolates orthographic mismatches from semantic translation differences. On standard baselines, this gap ranges from 0.04 to 0.19 chrF++ exclusively in the into-Kabyle direction.

chrF++ uses word n-grams up to order 2 (`nw:2`).

### Results

chrF++ / BLEU on `devtest` under the normalized evaluation condition:

| System | kab→eng | eng→kab | kab→fra | fra→kab |
|---|---|---|---|---|
| NLLB-200-distilled-600M | 39.70 | 27.51 | 37.30 | 26.35 |
| NLLB-200-distilled-1.3B | 44.60 | 31.53 | 42.00 | 30.21 |
| OPUS-MT | 32.26 | 24.61 | 31.67 | 25.03 |
| [**agbalu/Amrouche-1.3B**](https://huggingface.co/agbalu/Amrouche-1.3B) | **46.25** | **36.34** | **45.10** | **34.43** |

BLEU for Amrouche-1.3B: 25.29 / **10.86** / 22.10 / 8.39. Amrouche-1.3B exhibits an orthography gap of 0.00 across all four translation directions.

## `lid` — Berber Sibling Discrimination

1,500 sentences, **250 each** of `kab_Latn`, `shi_Latn`, `rif_Latn`, `taq_Latn`, `tzm_Latn`, and `shy_Latn`, all in Latin script, sampled with seed 20260809.

| Field | Type | Description |
|---|---|---|
| `text` | string | Input sentence |
| `language` | string | Ground-truth BCP-47 / NLLB language tag |

Kabyle is frequently conflated with Tashelhit, Tarifit, Central Atlas Tamazight, Tamasheq, and Shawiya in public web crawls. This benchmark provides a standardized test set to quantify cross-variety confusion.

### Baseline Identifier Performance

Proportion of each sibling language misclassified as `kab_Latn` by widely used language identifiers:

| System | Accuracy | Macro-F1 (supported classes) | shi | rif | taq | tzm |
|---|---|---|---|---|---|---|
| GlotLID | 43.80% | 0.7490 (3 of 6) | 14.8% | **94.0%** | 3.2% | **89.2%** |
| NLLB `lid218e` | 25.27% | 0.5545 (2 of 6) | **87.2%** | **93.2%** | **37.2%** | **95.2%** |

Macro-F1 is evaluated across the subset of classes supported by each identifier. Broad-coverage classifiers exhibit substantial cross-language confusion on closely related Amazigh varieties, with NLLB `lid218e` misclassifying 87–95% of sentences from three sibling languages as Kabyle, and 37.2% of Tamasheq (a variety included in its label set).

## Usage

```python
from datasets import load_dataset

# Machine translation reference (Kabyle side)
mt = load_dataset("agbalu/KabBench", "mt", split="devtest")
print(mt[0])
# {'id': 0, 'split': 'devtest', 'corrected': True,
#  'text': '"Tura nesɛa tiɣeṛdayin n 4 n wagguren di leɛmeṛ-nsent ur nesɛi aṭṭan n skeṛ...'}

# Filter to sentences where homoglyph corrections were applied:
repaired = mt.filter(lambda row: row["corrected"])
print(f"Corrected sentences: {len(repaired)}")

# Language identification benchmark
lid = load_dataset("agbalu/KabBench", "lid", split="test")
print(lid[0])
# {'text': 'D acu kan, xas akk-n ttseth\'in s tutlayt-nsen, ...', 'language': 'kab_Latn'}
```

The `mt` configuration contains only the Kabyle target sentences, indexed by `(split, id)`. To evaluate translation from or into English, French, or other languages, join on `(split, id)` against the corresponding language splits in FLORES+.

## Provenance and Limitations

- **MT configuration**: Derived from FLORES+ `kab_Latn` and distributed under **CC-BY-SA-4.0**. Corrections and normalizations were applied by the AƔBALU project.
- **LID configuration**: Kabyle text is sampled from AƔBALU-Text v1; sibling language texts are sampled from GlotCC-V1 (CC0), FLORES+ (CC-BY-SA-4.0), and Shawiya Wiktionary (CC-BY-SA-4.0).
- **Identifier label limitations**: Public baseline identifiers lack class labels for Tarifit, Central Atlas Tamazight, and Shawiya, resulting in high false-positive rates when filtering Kabyle text.
- **Decontamination**: Overlap between AƔBALU pretraining corpora and FLORES+ / SIB-200 was verified to be zero via string matching and positive-control injection.

## Citation

```bibtex
@misc{agbalu_kabbench,
  title  = {KabBench: evaluation data for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabBench}
}
```

Please also cite FLORES+, GlotCC-V1, and Universal Dependencies when using derivative data.

## Licence

**CC-BY-SA-4.0**, inherited from FLORES+ and Wiktionary. Attribution and share-alike terms apply.
