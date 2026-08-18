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

Evaluation data for Kabyle (Taqbaylit, `kab`, Latin script), from the
[AƔBALU](https://huggingface.co/agbalu) project.

Two configs. **`mt`** is a repaired Kabyle reference for machine translation — the public one
is 16.2% corrupt. **`lid`** is a balanced six-language set for telling Kabyle apart from its
Berber siblings, which the identifiers in common use cannot do.

```python
from datasets import load_dataset

mt = load_dataset("agbalu/KabBench", "mt", split="devtest")
lid = load_dataset("agbalu/KabBench", "lid", split="test")
```

## `mt` — a corrected Kabyle reference

2,009 sentences: 997 `dev` and 1,012 `devtest`.

**326 of them — 16.2% — carried a corrupted spelling and were repaired here.** The defect is
homoglyph substitution: Greek `ε` U+03B5 written for Latin `ɛ` U+025B, and similar
confusions across the Berber Latin letters. It is in the published reference, it has never
been revised upstream, and every Kabyle BLEU or chrF++ score ever reported is measured
against it.

| field | |
|---|---|
| `id` | sentence id **within its split** — ids restart at 0 per split, so key on `(split, id)` |
| `split` | `dev` or `devtest` |
| `text` | the Kabyle sentence, repaired where it needed repairing |
| `corrected` | whether this row differs from the published reference |

The practical consequence is that **a system spelling Kabyle correctly is penalised by the
uncorrected reference**, and the penalty is not small: on the uncorrected `devtest`, 2.71
BLEU is unreachable by a system whose output is orthographically perfect. Scores against
this file and scores against the original are not comparable.

Text is normalised to a single documented orthography — normaliser `1.3.0+rules1.0.0`, 81
rules, zero idempotence violations over 931,342 sentences.

### Scoring protocol

Report **both** conditions: raw, and with both sides normalised. The gap between them
measures how much of a score is orthography rather than translation. On the baselines below
the gap is 0.04–0.19 chrF++ and appears **only on into-Kabyle directions**.

chrF++ is chrF with word n-grams to order 2. `nw:0` is a different metric under the same
name — check the signature.

### Results

chrF++ / BLEU on `devtest`, normalised condition.

| system | kab→eng | eng→kab | kab→fra | fra→kab |
|---|---|---|---|---|
| NLLB-200-distilled-600M | 39.70 | 27.51 | 37.30 | 26.35 |
| NLLB-200-distilled-1.3B | 44.60 | 31.53 | 42.00 | 30.21 |
| OPUS-MT | 32.26 | 24.61 | 31.67 | 25.03 |
| [**agbalu/Amrouche-1.3B**](https://huggingface.co/agbalu/Amrouche-1.3B) | **46.25** | **36.34** | **45.10** | **34.43** |

BLEU for Amrouche: 25.29 / **10.86** / 22.10 / 8.39. As a control, this harness reproduces
NLLB's own published 6.2 BLEU for Kabyle at 6.02 on the 600M.

**Amrouche's two-condition gap is 0.00 in all four directions** — it already spells Kabyle
canonically, where both NLLB baselines do not.

## `lid` — Kabyle against its siblings

1,500 sentences, **250 each** of `kab_Latn`, `shi_Latn`, `rif_Latn`, `taq_Latn`, `tzm_Latn`
and `shy_Latn`, all Latin script, sampled with seed 20260809.

| field | |
|---|---|
| `text` | one sentence |
| `language` | its true label, an NLLB-style code |

Kabyle is routinely confused with Tashelhit, Tarifit, Central Atlas Tamazight, Tamasheq and
Shawiya, and many corpora labelled "Tamazight" silently mix them. This set makes the
confusion measurable.

**The share of each sibling that two widely used identifiers label `kab_Latn`:**

| | accuracy | macro-F1 (nameable) | shi | rif | taq | tzm |
|---|---|---|---|---|---|---|
| GlotLID | 43.80% | 0.7490 (3 of 6) | 14.8% | **94.0%** | 3.2% | **89.2%** |
| NLLB `lid218e` | 25.27% | 0.5545 (2 of 6) | **87.2%** | **93.2%** | **37.2%** | **95.2%** |

Macro-F1 is reported only over the languages a system can actually name; a system with no
label for Tarifit cannot be scored on it, and scoring it anyway flatters the total.

**`lid218e` labels 87–95% of three sibling languages as Kabyle — and 37.2% of Tamasheq,
which it *can* name, so that is genuine confusion rather than a missing label.** It is also
the identifier that gated the bitext mining behind most published "Kabyle" parallel data.

## Provenance and limits

- The `mt` config derives from FLORES+ `kab_Latn` and inherits **CC-BY-SA-4.0**. Repairs are
  ours; the sentences are not.
- The `lid` config draws Kabyle from AƔBALU-Text v1 and the siblings from GlotCC-V1 (CC0),
  FLORES+ (CC-BY-SA-4.0) and Shawiya Wiktionary (CC-BY-SA-4.0).
- **Sibling contamination is bounded, not cleared.** Neither identifier can name Tarifit,
  Central Atlas Tamazight or Shawiya, so a `kab_Latn` label cannot exclude them. What *is*
  excluded is Tashelhit and Tamasheq, which both systems can name.
- **`zgh` is out of scope.** No Latin-script Standard Moroccan Tamazight was found in any
  source surveyed, and Tifinagh separates from Kabyle on the encoding alone — including it
  would measure the script.
- Decontamination of AƔBALU-Text v1 against FLORES+ and SIB-200 measured **zero, with a
  positive control**.

## Citation

```bibtex
@misc{agbalu_kabbench,
  title  = {KabBench: evaluation data for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabBench}
}
```

Please also cite FLORES+, GlotCC-V1 and Universal Dependencies where you use data derived
from them.

## Licence

**CC-BY-SA-4.0**, inherited from FLORES+ and Wiktionary. Attribution and share-alike apply.
