---
license: cc0-1.0
language:
  - kab
language_bcp47:
  - kab-Latn
size_categories:
  - 10K<n<100K
task_categories:
  - token-classification
task_ids:
  - lemmatization
pretty_name: KabG2P — Kabyle grapheme-to-phoneme pronunciation dictionary
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - low-resource
  - grapheme-to-phoneme
  - g2p
  - pronunciation
  - phonetics
  - ipa
  - text-to-speech
  - automatic-speech-recognition
configs:
  - config_name: default
    data_files:
      - split: train
        path: default/train.jsonl
---

# KabG2P

A grapheme-to-phoneme pronunciation dictionary for Kabyle (Taqbaylit, `kab`, Latin script),
from the [AƔBALU](https://huggingface.co/agbalu) project.

**25,634 Kabyle word–IPA pairs** recovered by aligning 292,921 tokens across 59,462
sentence pairs at a 99.53% alignment rate, with a **0% ambiguity rate** across the entire
vocabulary. Every attested word has exactly one IPA reading. It is the phonetics layer
underlying [`agbalu/Matoub-TTS`](https://huggingface.co/agbalu) and
[`agbalu/Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M), and the reference
target for any Kabyle G2P model.

```python
from datasets import load_dataset

ds = load_dataset("agbalu/KabG2P", split="train")

print(ds[4])
# {
#   "word":     "ababat",
#   "ipa":      "æβæβæθ",
#   "variants": [{"ipa": "æβæβæθ", "count": 10}],
#   "repaired": False
# }
```

## Data

| split | rows | source tokens |
|---|---|---|
| `train` | 25,634 | 292,921 |

Eight entries whose headword falls outside the Kabyle writing system (`3d`, `androïd`,
`mp3`, `muḥ€nd`, `rosé`, `supermarché`, `xelleṣ̣`, `ṭeyyeb‟`) are present in the source
lexicon with correct IPA but are excluded from this release; G2P benchmarking on non-Kabyle
words is not meaningful.

## Schema

| field | type | description |
|---|---|---|
| `word` | `str` | Canonical Kabyle Latin spelling, NFC-normalised |
| `ipa` | `str` | IPA transcription — the majority pronunciation across aligned tokens |
| `variants` | `list[dict]` | Frequency-sorted alternative readings: `[{"ipa": "...", "count": N}, ...]` |
| `repaired` | `bool` | `true` where this project restored a character the upstream source dropped |

## IPA Phoneme Inventory

42 phoneme symbols in the source; 38 appear in 3 or more entries. The full inventory of
symbols observed across 292,921 aligned tokens:

| category | symbols |
|---|---|
| Vowels | `æ` `ɑ` `ə` `i` `u` |
| Stops (plain) | `b` `d` `ɡ` `k` `t` `p` |
| Stops (emphatic) | `dˤ` |
| Fricatives (plain) | `β` `ð` `ʝ` `ç` `θ` `f` `s` `z` `ʃ` `ʒ` `x` `χ` `ʁ` `ħ` `ʕ` `h` |
| Fricatives (emphatic) | `ðˤ` |
| Affricates | `t͡ʃ` |
| Nasals | `m` `n` `ɲ` `ŋ` |
| Liquids | `r` `l` |
| Glides | `j` `w` |
| Length mark | `ː` |

## Phonological Rules

Three conditioned rules are applied and verified against the aligned data:

**1. Spirantization by gemination.**
Singleton `b d g k t ḍ` → fricatives `β ð ʝ ç θ ðˤ`; geminates → stops `b d ɡ k t dˤ`.
Count agreement is exact for `t`, `b`, and `k` across 292,921 tokens; off by one for `g`.

**2. Vowel backing beside emphatics and uvulars.**
`a` → `ɑ` when adjacent to `ḍ ṣ ṭ ẓ ṛ q ɣ x`. The rule is **92.44% accurate** against
an 87.19% always-`æ` baseline — +5.25 pp.

**3. `i`/`u` laxing is not applied.**
The only recoverable conditioning environment (closed syllable) scored 75.59% against a
74.99% baseline — below the threshold for inclusion. Attested words carry their verified
allophones from this dictionary; out-of-vocabulary words receive the tense vowel.

## Benchmark Baseline

The AƔBALU rule-based G2P system achieves **78.18% exact-match rate** on this dictionary
without any dictionary lookup — purely from spelling rules. This is the floor any learned
G2P model must exceed, and the dictionary itself is the training target and ceiling.

## Repaired Entries

The upstream generator drops any character it has no rule for, silently producing a shorter
IPA string. This release repairs **199 entries**, flagged by `repaired: true`:

| character | affected words | repair |
|---|---|---|
| `o` | 128 | `bob` → `ββ` repaired to `βoβ` |
| `ţ` U+0163 | 73 | `aţan` → `ææn` repaired to `ætːæn` (geminate, from corpus evidence) |
| `é`, `ï` | 3 | left as-is (outside the writing system) |

🔴 **`ţ` is attested Kabyle.** It is the Dallet-tradition spelling for the spirantised `t`,
occurring 21,058 times in AƔBALU-Text v1 and in 1,407 word types. It is repaired to the
geminate on corpus evidence. No audio confirms this mapping — `ţ` does not appear in
Common Voice Kabyle — and the repair is disclosed rather than hidden.

## Usage

```python
from datasets import load_dataset

g2p = load_dataset("agbalu/KabG2P", split="train")
g2p[0]
# {'word': 'a', 'ipa': 'æ', 'variants': [{'ipa': 'æ', 'count': 2644}], 'repaired': False}

lookup = {row["word"]: row["ipa"] for row in g2p}
lookup["azul"]
```

**No word takes two transcriptions.** `variants` records how often each spelling of the
pronunciation was seen while aligning, and in all 25,634 entries the winner is the only
entry — so a plain dictionary is a faithful representation, not a lossy one.

`repaired` marks the 199 entries where the upstream generator dropped a character it had no
rule for and this build restored it. Filter on it to see exactly which:

```python
g2p.filter(lambda row: row["repaired"])
```

## Orthography

All headwords are normalised to canonical Kabyle Latin script under normaliser
`1.3.0+rules1.0.0` (81 rules). 199 entries carried Greek homoglyphs (`ε γ Σ Γ Ԑ`) in the
upstream source and were corrected to canonical `ɛ ɣ Ɛ Ɣ Ɛ`. The `repaired` field records
every affected entry.

## Source

| field | value |
|---|---|
| Upstream | `boffire/kabyle-g2p-training-data` |
| Upstream licence | **CC0-1.0** |
| Sentences aligned | 59,462 (59,185 successfully aligned) |
| Alignment rate | 99.53% |
| Ambiguity rate | 0.00% |
| Normaliser version | `1.3.0+rules1.0.0` |
| Repaired entries | 199 |

## Citation

```bibtex
@misc{agbalu_kabg2p,
  title  = {KabG2P: a Kabyle grapheme-to-phoneme pronunciation dictionary},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabG2P}
}
```

Please also cite the upstream source: `boffire/kabyle-g2p-training-data`.

## Licence

**CC0-1.0.** The upstream source (`boffire/kabyle-g2p-training-data`) is CC0-1.0. The
repairs and normalisation applied by this project are released under the same licence.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection.
