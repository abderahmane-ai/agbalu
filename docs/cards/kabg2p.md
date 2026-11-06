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

A grapheme-to-phoneme pronunciation dictionary for Kabyle (Taqbaylit, `kab`, Latin script), from the [AƔBALU](https://huggingface.co/agbalu) project.

**25,634 Kabyle word–IPA pairs** recovered by aligning 292,921 tokens across 59,462 sentence pairs at a 99.53% alignment rate, with a **0% ambiguity rate** across the vocabulary (each attested word maps to a single canonical IPA transcription). It provides the phonetic foundation for [`agbalu/Matoub-82M`](https://huggingface.co/agbalu/Matoub-82M) and [`agbalu/Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M), and serves as a reference target for Kabyle G2P modeling.

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

| Split | Rows | Source Tokens |
|---|---|---|
| `train` | 25,634 | 292,921 |

Eight non-Kabyle loanwords or malformed entries containing punctuation or digits (`3d`, `androïd`, `mp3`, `muḥ€nd`, `rosé`, `supermarché`, `xelleṣ̣`, `ṭeyyeb‟`) were excluded from the benchmark release.

## Schema

| Field | Type | Description |
|---|---|---|
| `word` | `str` | Canonical Kabyle Latin spelling (NFC-normalized) |
| `ipa` | `str` | Canonical IPA transcription (majority reading across aligned tokens) |
| `variants` | `list[dict]` | Attested pronunciations with token frequencies: `[{"ipa": "...", "count": N}, ...]` |
| `repaired` | `bool` | Whether character omission in the upstream source was repaired |

## IPA Phoneme Inventory

42 phoneme symbols in the source inventory (38 appear in 3 or more entries):

| Category | Symbols |
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

Three conditioned phonological rules were evaluated against the aligned token data:

1. **Spirantization conditioned by gemination**:
   Singleton stops `b d g k t ḍ` surface as fricatives `β ð ʝ ç θ ðˤ`, while geminates surface as stops `b d ɡ k t dˤ`.
2. **Vowel backing adjacent to emphatics and uvulars**:
   `a` backs to `ɑ` when adjacent to `ḍ ṣ ṭ ẓ ṛ q ɣ x`. The rule achieves **92.44% accuracy** against an 87.19% default baseline (+5.25 pp).
3. **High vowel laxing**:
   Laxing for `i`/`u` in closed syllables scored 75.59% against a 74.99% baseline; attested words retain observed allophones, while out-of-vocabulary words use tense vowels.

## Benchmark Baseline

The AƔBALU rule-based G2P model reaches a **78.18% exact-match rate** on this dictionary using orthographic phonotactic rules without dictionary lookup.

## Repaired Entries

The upstream generator omitted characters lacking explicit phonetic mappings. This release repairs **199 entries**, flagged by `repaired: true`:

| Character | Words | Repair Applied |
|---|---|---|
| `o` | 128 | `bob` → `ββ` repaired to `βoβ` |
| `ţ` (U+0163) | 73 | `aţan` → `ææn` repaired to `ætːæn` (geminate, based on corpus evidence) |
| `é`, `ï` | 3 | Preserved unchanged (loanwords outside Kabyle phonology) |

The character `ţ` represents Dallet-tradition orthography for the spirantised/tense `t`, attested 21,058 times across 1,407 word types in the text corpus. Based on corpus co-occurrence with `tt`, missing IPA mappings for `ţ` were mapped to geminate stops.

## Usage

```python
from datasets import load_dataset

g2p = load_dataset("agbalu/KabG2P", split="train")
print(g2p[0])
# {'word': 'a', 'ipa': 'æ', 'variants': [{'ipa': 'æ', 'count': 2644}], 'repaired': False}

lookup = {row["word"]: row["ipa"] for row in g2p}
print(lookup["azul"])
```

Each word maps to a single canonical IPA transcription (0% ambiguity rate). `variants` records frequency-ranked pronunciations observed during alignment.

`repaired` flags the 199 entries where dropped characters were restored:

```python
repaired_entries = g2p.filter(lambda row: row["repaired"])
print(f"Repaired entries: {len(repaired_entries)}")
```

## Orthography

All headwords are normalized to canonical Kabyle Latin script under the AƔBALU reference normaliser (`1.3.0+rules1.0.0`, 81 rules).

## Source Metadata

| Field | Value |
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

**CC0-1.0.** The upstream dataset is CC0-1.0; repairs and normalization are released under the same terms.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection.
