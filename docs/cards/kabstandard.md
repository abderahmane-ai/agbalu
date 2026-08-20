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

A 497,944-pair parallel dataset for **Kabyle orthography standardisation** — mapping informal,
French-keyboard and Arabizi Kabyle text to canonical Kabyle Latin orthography. Derived from
the Latin side of [`agbalu/KabTifinagh`](https://huggingface.co/datasets/agbalu/KabTifinagh)
by a deterministic seeded probabilistic corruption pass that simulates the keyboard strategies
Kabyle speakers use on phones and social media.

Used to train [`agbalu/Boulifa-48M`](https://huggingface.co/agbalu/Boulifa-48M), which reaches
**97.39% character accuracy** on held-out test pairs under greedy free-running decoding,
against **89.70%** for leaving the input untouched.

```python
from datasets import load_dataset

ds = load_dataset("agbalu/KabStandard")
# DatasetDict({'train': Dataset(448149), 'dev': Dataset(24897), 'test': Dataset(24898)})
```

## Splits

497,944 total pairs, partitioned at seed 42 into 0-leakage splits.

| split | pairs |
|---|---:|
| `train` | 448,149 |
| `dev` | 24,897 |
| `test` | 24,898 |
| **total** | **497,944** |

## Schema

| field | type | description |
|---|---|---|
| `source` | string | Informal input (French-keyboard, Arabizi, or identity) |
| `target` | string | Canonical Kabyle Latin (normalised, unmodified) |

## Usage

```python
from datasets import load_dataset

train = load_dataset("agbalu/KabStandard", split="train")
train[0]
# {'source': 'Ssubbetd negh ad naligh.', 'target': 'Ṣṣubbet-d neɣ ad n-aliɣ.'}
```

`source` is the corrupted spelling and `target` the canonical one — the task is
`source → target`, and the reverse direction is what the corruption pass already does.

**Score against the do-nothing floor.** These pairs are corrupted probabilistically, so most
characters arrive already correct and copying the input scores 89.70% character accuracy. Any
number reported without that floor beside it is unreadable.

## Construction

Source sentences are the `text_latn` column of `agbalu/KabTifinagh` (all three splits
combined), normalised under AƔBALU normaliser `1.3.0+rules1.0.0`. Each sentence generates
exactly one pair at seed 42 — the dataset is fully reproducible from the source corpus alone.

**Identity pairs (15%).** `IDENTITY_RATE = 0.15`. One in seven sentences is left unchanged
(`source == target`), teaching any model trained on this data not to edit already-canonical
text.

**Corrupted pairs (85%).** The remaining 85% are passed through a probabilistic corruption
pass that applies the following transformations stochastically and independently per character:

### Phoneme substitutions (`PROB_SUBSTITUTION = 0.90`)

| Canonical | Informal variants | Probabilities |
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

`u` → `ou` (French convention for /u/) with probability 0.45; `U` → `Ou` with the same
probability.

### Clitic hyphen omission (`PROB_CLITIC_DROP = 0.50`)

If the sentence contains `-`, with probability 0.50: replace all hyphens with a space
(`d-yeffeɣ` → `d yeffegh`) or delete them (`d-yeffeɣ` → `dyeffegh`), each with probability
0.50.

### Preposition contraction (`PROB_PREP_SHORTEN = 0.25`)

`deg ` → `g `, `seg ` → `s ` (word-boundary anchored), with probability 0.25.

## Examples

```
source: "achimi ur d-thekhedmedh ara tamazight g l'ecole?"
target: "acimi ur d-tḥexedmeḍ ara tamaziɣt deg lɛecule?"

source: "3emmi l7adj yerza-d 5ir d lbaraka s wuzzal"
target: "Ɛemmi lḥadj yerza-d xir d lbaraka s wuzzal"

source: "Azul fell-awen, amek i telliḍ taṣebḥit-a?"
target: "Azul fell-awen, amek i telliḍ taṣebḥit-a?"
```

The third row is an identity pair (`source == target`).

## Evaluation

1,000 pairs drawn at seed 4711 from the held-out test split, greedy and free-running.

| system | character accuracy | character error rate | exact match |
|---|---|---|---|
| **Boulifa-48M** | **97.39%** | **2.61%** | **85.70%** |
| leave the input untouched | 89.70% | 10.30% | — |

```bash
make standardise TASK=evaluate LIMIT=1000
```

**The floor is in the table because these pairs are corrupted probabilistically**, so most
characters in most sentences arrive already correct and a system that does nothing scores
89.70%. Read the model as removing 75% of the character error, not as near-perfect.

**The evaluation pairs are synthetic.** The figure measures the round-trip — can the model
recover the canonical target from a plausibly corrupted source? It cannot be read as accuracy
on arbitrary human typing, only on the corruption distribution defined here.

## Known Limits

- **Synthetic only.** Every `source` string was generated by a rule. No human typed any of
  these inputs. The distribution approximates real typing but is not a sample of it.
- **One variant per sentence.** Each canonical sentence generates exactly one corrupted
  source. A model has not seen the same sentence under multiple corruption strategies.
- **No adequacy judgement.** The `target` strings are the normaliser's output. No human
  verification of the canonical form of any source sentence exists.
- **Sibling language contamination.** The source sentences come from `agbalu/KabTifinagh`,
  which carries the same contamination bound from its upstream sources: LID systems cannot
  reliably distinguish Kabyle from Tarifit, Central Atlas Tamazight or Shawiya.

## Reproduction

```bash
make modal-boulifa TASK=prepare   # writes train/dev/test.jsonl and commits to the volume
```

The dataset is regenerated deterministically at seed 42 from `agbalu/KabTifinagh`. No GPU
required.

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

**Apache-2.0.** Derived from `agbalu/KabTifinagh` (CC-BY-2.0); a permissive grant on this
derived dataset does not relicense the upstream corpus. Read `agbalu/KabTifinagh`'s licence
before redistributing derivatives of the training corpus.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection.
