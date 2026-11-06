---
license: apache-2.0
language:
  - kab
size_categories:
  - 100K<n<1M
task_categories:
  - translation
pretty_name: KabStandard — informal Kabyle to canonical Kabyle Latin orthography standardisation
tags:
  - kabyle
  - taqbaylit
  - berber
  - amazigh
  - low-resource
  - orthography
  - keyboard-normalisation
  - arabizi
  - synthetic
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: dev
        path: dev.jsonl
      - split: test
        path: test.jsonl
---

# KabStandard

A 497,944-pair parallel dataset for **Kabyle orthography standardisation** — mapping informal, French-keyboard, and Arabizi Kabyle text to canonical Kabyle Latin orthography. Derived from the Latin side of [`agbalu/KabTifinagh`](https://huggingface.co/datasets/agbalu/KabTifinagh) using a deterministic probabilistic corruption engine that simulates keyboard conventions used on mobile devices and social platforms.

Used to train [`agbalu/Boulifa-48M`](https://huggingface.co/agbalu/Boulifa-48M), which reaches **97.39% character accuracy** and **85.70% sentence exact match** on held-out test pairs, compared to an 89.70% character accuracy baseline for copying the input unchanged.

```python
from datasets import load_dataset

ds = load_dataset("agbalu/KabStandard")
# DatasetDict({'train': Dataset(448149), 'dev': Dataset(24897), 'test': Dataset(24898)})
```

## Splits

497,944 total pairs, partitioned with seed 42 into 0-leakage splits.

| Split | Pairs |
|---|---:|
| `train` | 448,149 |
| `dev` | 24,897 |
| `test` | 24,898 |
| **Total** | **497,944** |

## Schema

| Field | Type | Description |
|---|---|---|
| `source` | string | Informal input (French-keyboard, Arabizi, or identity) |
| `target` | string | Canonical Kabyle Latin (normalized) |

## Usage

```python
from datasets import load_dataset

train = load_dataset("agbalu/KabStandard", split="train")
print(train[0])
# {'source': 'Ssubbetd negh ad naligh.', 'target': 'Ṣṣubbet-d neɣ ad n-aliɣ.'}
```

The task maps `source → target`. Because corruption is applied probabilistically at the character level, a baseline copying the input unchanged retains 89.70% character accuracy. Evaluation should compare model performance against this baseline floor.

## Construction

Source sentences are drawn from the `text_latn` column of `agbalu/KabTifinagh` (all splits combined) and normalized under the AƔBALU reference normaliser (`1.3.0+rules1.0.0`). Each sentence generates exactly one pair at seed 42, making the dataset fully reproducible.

- **Identity pairs (15%)**: `IDENTITY_RATE = 0.15`. 15% of sentences are preserved unchanged (`source == target`), training models to avoid modifying already canonical orthography.
- **Corrupted pairs (85%)**: The remaining 85% receive stochastic transformations applied independently per character:

### Phoneme substitutions (`PROB_SUBSTITUTION = 0.90`)

| Canonical | Informal Variants | Probabilities |
|---|---|---|
| `ɣ` / `Ɣ` | `gh` / `g` / `3` / `8` | 0.75 / 0.10 / 0.08 / 0.07 |
| `x` / `X` | `kh` / `k` / `5` | 0.85 / 0.10 / 0.05 |
| `c` / `C` | `ch` / `c` / `sh` | 0.75 / 0.20 / 0.05 |
| `č` / `Č` | `tch` / `ch` / `tc` | 0.70 / 0.20 / 0.10 |
| `ğ` / `Ğ` | `dj` / `j` / `g` | 0.80 / 0.15 / 0.05 |
| `ḍ` / `Ḍ` | `dh` / `d` | 0.75 / 0.25 |
| `ṭ` / `Ṭ` | `th` / `t` | 0.70 / 0.30 |
| `ṣ` / `Ṣ` | `s` / `ss` | 0.75 / 0.25 |
| `ẓ` / `Ẓ` | `z` / `zz` | 0.80 / 0.20 |
| `ṛ` / `Ṛ` | `r` / `rr` | 0.90 / 0.10 |
| `ḥ` / `Ḥ` | `h` / `7` / `hh` | 0.70 / 0.25 / 0.05 |
| `ɛ` / `Ɛ` | `e` / `a` / `3` / `'` | 0.35 / 0.30 / 0.25 / 0.10 |

### Vowel digraph (`PROB_DIGRAPH_OU = 0.45`)

`u` → `ou` (French convention for /u/) with probability 0.45; `U` → `Ou` with probability 0.45.

### Clitic hyphen omission (`PROB_CLITIC_DROP = 0.50`)

For sentences containing `-`, hyphens are either replaced with spaces (`d-yeffeɣ` → `d yeffegh`) or removed (`d-yeffeɣ` → `dyeffegh`), each with probability 0.50.

### Preposition contraction (`PROB_PREP_SHORTEN = 0.25`)

`deg ` → `g `, `seg ` → `s ` with probability 0.25 at word boundaries.

## Examples

```
source: "achimi ur d-thekhedmedh ara tamazight g l'ecole?"
target: "acimi ur d-tḥexedmeḍ ara tamaziɣt deg lɛecule?"

source: "3emmi l7adj yerza-d 5ir d lbaraka s wuzzal"
target: "Ɛemmi lḥadj yerza-d xir d lbaraka s wuzzal"

source: "Azul fell-awen, amek i telliḍ taṣebḥit-a?"
target: "Azul fell-awen, amek i telliḍ taṣebḥit-a?"
```

## Evaluation

Evaluated across 1,000 held-out test pairs drawn with seed 4711 under greedy autoregressive decoding:

| System | Character Accuracy | Character Error Rate | Exact Match |
|---|---|---|---|
| **Boulifa-48M** | **97.39%** | **2.61%** | **85.70%** |
| Copy input unchanged (baseline) | 89.70% | 10.30% | — |

The baseline floor reflects that most characters in a typical corrupted sentence remain unchanged, resulting in an 89.70% character accuracy floor for an identity copy. Boulifa-48M removes approximately 75% of character-level corruption, achieving 97.39% accuracy.

Evaluation measures recovery of canonical orthography from synthetically corrupted text matching the defined probabilistic distribution; real-world informal typing exhibits additional idiosyncratic orthographic variance.

## Known Limitations

- **Synthetic generation**: All `source` strings are synthetically generated from rules; while designed to reflect common keyboard practices, they do not constitute natural keystroke logs.
- **Single corruption per sentence**: Each canonical sentence generates one paired corrupted instance.
- **Target normalization**: Targets correspond to output from the reference normaliser.

## Reproduction

The dataset is regenerated deterministically at seed 42 from `agbalu/KabTifinagh` using the standardized orthography rules and probabilistic corruption engine.

## Citation

```bibtex
@misc{agbalu_kabstandard,
  title  = {KabStandard: a synthetic parallel corpus for Kabyle orthography standardisation},
  author = {AGBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabStandard}
}
```

Derived from [`agbalu/KabTifinagh`](https://huggingface.co/datasets/agbalu/KabTifinagh).

## Licence

**Apache-2.0.** Derived from `agbalu/KabTifinagh` (CC-BY-2.0).

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection.
