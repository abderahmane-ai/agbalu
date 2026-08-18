---
license: cc-by-2.0
language:
  - kab
language_bcp47:
  - kab-Latn
  - kab-Tfng
size_categories:
  - 100K<n<1M
task_categories:
  - translation
  - text-generation
pretty_name: KabTifinagh — Kabyle Latin <-> Tifinagh transliteration and vowel restoration benchmark
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - low-resource
  - tifinagh
  - transliteration
  - script-conversion
configs:
  - config_name: script_conversion
    data_files:
      - split: train
        path: script_conversion/train.parquet
      - split: dev
        path: script_conversion/dev.parquet
      - split: test
        path: script_conversion/test.parquet
  - config_name: trilingual_en
    data_files:
      - split: train
        path: trilingual_en/train.parquet
      - split: dev
        path: trilingual_en/dev.parquet
      - split: test
        path: trilingual_en/test.parquet
  - config_name: trilingual_fr
    data_files:
      - split: train
        path: trilingual_fr/train.parquet
      - split: dev
        path: trilingual_fr/dev.parquet
      - split: test
        path: trilingual_fr/test.parquet
---

# KabTifinagh

A standardized bidirectional script transliteration and schwa (`e`) vowel restoration benchmark for Kabyle (Taqbaylit, `kab`, Latin & Tifinagh scripts), created by the [AƔBALU](https://huggingface.co/agbalu) project.

`KabTifinagh` normalises, repairs, deduplicates, and structures **497,944 parallel sentence entries** matching Neo-Tifinagh (`kab_Tfng`) to canonical Kabyle Latin (`kab_Latn`), alongside **123,852 English** and **205,637 French** trilingual sentence alignments.

```python
from datasets import load_dataset

script = load_dataset("agbalu/KabTifinagh", "script_conversion")
trilingual_en = load_dataset("agbalu/KabTifinagh", "trilingual_en")
trilingual_fr = load_dataset("agbalu/KabTifinagh", "trilingual_fr")
```

## `script_conversion`

Bidirectional Transliteration and Vowel Restoration: `text_tfng` ↔ `text_latn`.

497,944 total parallel sentences, partitioned with seed 42 into 0-leakage splits.

| split | parallel sentences |
|---|---|
| `train` | 398,355 |
| `dev` | 49,794 |
| `test` | 49,795 |
| **total** | **497,944** |

### Schema

| field | type | description |
|---|---|---|
| `id` | string | Unique record ID (`kab_tfng_{split}_{idx}`) |
| `text_latn` | string | Canonical Kabyle Latin sentence (100% normalized) |
| `text_tfng` | string | Neo-Tifinagh Kabyle sentence |

## `trilingual_en` & `trilingual_fr`

Parallel Kabyle Latin, Kabyle Tifinagh, and English / French translations for cross-lingual multimodal and multi-script AI research.

- **`trilingual_en`**: 123,852 sentence triples (`id`, `text_latn`, `text_tfng`, `text_en`).
- **`trilingual_fr`**: 205,637 sentence triples (`id`, `text_latn`, `text_tfng`, `text_fr`).

### Trilingual Schema

| field | type | description |
|---|---|---|
| `id` | string | Unique record ID (`kab_tfng_en_{split}_{idx}` / `kab_tfng_fr_{split}_{idx}`) |
| `text_latn` | string | Canonical Kabyle Latin sentence |
| `text_tfng` | string | Neo-Tifinagh Kabyle sentence |
| `text_en` / `text_fr` | string | English or French translation |

## The Vowel Restoration & Script Conversion Challenge

Tifinagh orthography presents a key sequence-to-sequence learning task due to structural orthographic asymmetries between the two scripts:

1. **Schwa (`e`) Omission**: Standard Tifinagh orthography omits schwa vowels (`e`).
   - Latin: `Tecfiḍ fell-i ?` → Tifinagh: `ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?`
   - While forward mapping (Latin → Tifinagh) is straightforward, reverse transliteration (Tifinagh → Latin) requires predicting where omitted schwas belong based on Kabyle phonotactics and context.
2. **Morphological Boundaries**: Clitic hyphens (`fell-i`, `yefka-yas`) are represented contiguously in Tifinagh (`ⴼⵍⵍⵉ`).
3. **Vowel & Consonant Mappings**: Standard Tifinagh codepoints map `o/u` to `ⵓ` and `b/p` to `ⴱ`.

A deterministic rule-based character mapper provides a baseline floor, but sequence-to-sequence models (character-level transformers or fine-tuned LLMs) trained on `agbalu/KabTifinagh` learn contextual phonotactic rules to reconstruct missing schwas (`e`) and clitic boundaries with high fidelity.

## Evaluation protocol, and the floor

Scored on the held-out test split by `agbalu.bench.tifinagh`, references case-folded
because the model is defined over a case-folded alphabet.

| direction | system | sentences | exact match | CER | schwa F1 |
|---|---|---|---|---|---|
| Tifinagh → Latin | character table | 49,795 | 1.02% | 13.49% | 1.72% |
| Latin → Tifinagh | character table | 49,795 | 1.02% | 14.83% | — |
| **Tifinagh → Latin** | [**Juba-27M**](https://huggingface.co/agbalu/Juba-27M), greedy | 5,000 | **94.22%** | **0.33%** | **93.51%** |

```bash
make bench TASK=tifinagh                 # the table, over the whole split
make tifinagh TASK=evaluate LIMIT=5000   # the model, free-running, no GPU needed
```

**Schwa F1 here is positional.** A hypothesis with the right *number* of `e` in the wrong
places scores 100% under a count and 0% under this metric; the scorer strips every `e`,
compares the consonant skeletons, and matches the vowels by their index into that skeleton.

**The model's number is free-running**: it is fed its own output, which is what a caller
gets. A teacher-forced pass, which hands the decoder the gold prefix at every position,
scores higher and does not describe inference.

## Orthography & Engineering

All Kabyle Latin text is normalised under AƔBALU normaliser `1.3.0+rules1.0.0` (81 rules), preserving sub-dot emphatic distinctions (`ɣ ḍ ḥ ṭ ṛ ẓ ṣ`). The text carries **0.00% homoglyph error** (all Greek `ε` homoglyphs repaired).

## Known Limits

- **Script Standard.** Uses IRCAM Neo-Tifinagh codepoints; does not cover historical Tuareg Tifinagh variations.
- **Prose Domain.** Drawn primarily from general prose and Tatoeba parallel sentences.

## Citation

```bibtex
@misc{agbalu_kabtifinagh,
  title  = {KabTifinagh: a Kabyle script transliteration and vowel restoration benchmark},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabTifinagh}
}
```

Upstream data sourced from `abdelhaqueidali/kab-latn-tfng`.

## Licence

**CC-BY-2.0.** Free to share, modify, and build upon with proper attribution.
