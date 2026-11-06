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
task_ids:
  - lemmatization
  - part-of-speech
pretty_name: KabLex — Kabyle lexicon and pronunciation dictionary
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - low-resource
  - lexicon
  - morphology
  - grapheme-to-phoneme
configs:
  - config_name: lexicon
    data_files:
      - split: train
        path: lexicon/train.jsonl
  - config_name: pronunciations
    data_files:
      - split: train
        path: pronunciations/train.jsonl
---

# KabLex

A lexical layer for Kabyle (Taqbaylit, `kab`, Latin script), from the [AƔBALU](https://huggingface.co/agbalu) project.

**366,892 lexical entries** merged from three permissively licensed sources and normalised to one orthography, plus **25,642 word–pronunciation pairs** aligned from sentence-level grapheme-to-phoneme data.

```python
from datasets import load_dataset

lex = load_dataset("agbalu/KabLex", "lexicon", split="train")
ipa = load_dataset("agbalu/KabLex", "pronunciations", split="train")
```

## `lexicon`

366,892 entries — 234,936 distinct surface forms over 17,090 lemmas, 2,408 of them multiword expressions.

| Field | Type | Description |
|---|---|---|
| `form` | string | Normalized surface form |
| `lemma` | string | Base lemma (where available in source) |
| `upos` | string | Universal Dependencies POS tag |
| `feats` | string | Morphological features in UD format (`_` indicates no features; non-null) |
| `glosses` | list[dict] | Translations as `{"lang": "...", "text": "..."}` (available for 9,658 entries) |
| `source` | string | Originating source registry ID |
| `licence` | string | Upstream source license |

| POS | Entries |
|---|---|
| `VERB` | 345,057 |
| Unlabelled | 15,357 |
| `NOUN` | 5,603 |
| `PROPN` | 287 |
| `PART` | 249 |
| `ADJ`, `ADV`, `INTJ`, others | 246 |

The part-of-speech distribution reflects the underlying source datasets: 94% of entries are inflected verb forms drawn from verb inflection tables. Users should treat this dataset primarily as an extensive verbal morphological resource with an accompanying nominal and toponymic layer.

### Composition

| Source | Entries | License |
|---|---|---|
| `hf.boffire.kabyle-verbs` | 342,257 | CC-BY-4.0 |
| `hf.boffire.hunspell-kab` | 20,967 | MIT |
| `hf.agurzil.tafsut-maths-lexicon` | 3,668 | CC-BY-4.0 |

### Licensing Filter

From the raw 395,834 assembled entries, **28,942 entries were excluded by automated licensing filters** to ensure a strictly permissive release:
- **8,554 entries** licensed under ODbL were excluded to prevent share-alike requirements from restricting downstream model development.
- **20,388 entries** with unresolved licensing metadata were excluded.

The published dataset is distributed under CC-BY-4.0.

## `pronunciations`

25,642 distinct words with IPA transcriptions, aligned from 59,185 sentence-level pairs (99.53% alignment rate) across a 42-phoneme inventory.

| Field | Type | Description |
|---|---|---|
| `word` | string | Normalized orthographic word |
| `ipa` | string | Primary IPA pronunciation |
| `variants` | list[dict] | Frequency-ranked alternative readings: `[{"ipa": "...", "count": N}, ...]` |
| `repaired` | bool | Whether character dropping in the upstream generator was corrected |

This dataset provides orthographic-to-phonetic transliterations rather than continuous acoustic transcriptions. Characteristics across the 292,921 aligned tokens include:
- **Deterministic mapping**: Each word spelling maps to a single canonical IPA representation (0.0% ambiguity).
- **Schwa representation**: Phonetic `ə` maps directly to orthographic `e` across 100.00% of entries; this dictionary does not model context-sensitive epenthetic schwa insertion (for context-dependent schwa restoration from consonant skeletons, refer to [`agbalu/Juba-27M`](https://huggingface.co/agbalu/Juba-27M)).
- **Gemination**: Length markers (`ː`) correspond directly to doubled consonant graphemes in 100.00% of entries.
- **Word-isolated forms**: Sandhi, cross-word spirantization, and assimilation across word boundaries are outside the scope of single-word dictionary entries.

This resource is intended as a lexical pronunciation dictionary and G2P training target.

### Repaired Characters

The upstream generation pipeline dropped characters lacking explicit conversion rules. This release repairs **199 entries**, flagged by `repaired: true`:

| Character | Words | Original | Repaired |
|---|---|---|---|
| `o` | 128 | `bob` → `ββ` | `βoβ` |
| `ţ` (U+0163) | 73 | `aţan` → `ææn` | `ætːæn` |
| `é`, `ï` | 3 | `rosé` → `rs` | Unchanged (loanword foreign letters) |

The letter `ţ` (U+0163) represents Dallet-tradition orthography for the spirantised / tense `t`, attested 21,058 times in the AƔBALU text corpus. In 43% of word types carrying `ţ`, a corresponding `tt` form exists, while 34% have a `t` form (compared to 0.7% for `ṭ`). Based on this distribution, missing pronunciations for `ţ` were mapped to the geminate stop `ætːæn`.

Eight foreign loanwords or malformed entries (`3d`, `mp3`, `androïd`, `rosé`, `supermarché`, `muḥ€nd`, `xelleṣ̣`, `ṭeyyeb‟`) containing digits or stray punctuation were preserved verbatim without speculative correction. Repairs were restricted strictly to entries where source rules truncated character strings.

Source: `boffire/kabyle-g2p-training-data`, **CC0-1.0**.

## Usage

```python
from datasets import load_dataset

lexicon = load_dataset("agbalu/KabLex", "lexicon", split="train")
print(lexicon[0])
# {'form': 'asider', 'lemma': None, 'upos': None, 'feats': 'Domain=Math',
#  'glosses': [{'lang': 'fra', 'text': 'abaissement'}],
#  'source': 'hf.agurzil.tafsut-maths-lexicon', 'licence': 'cc-by-4.0'}

pronunciations = load_dataset("agbalu/KabLex", "pronunciations", split="train")
print(pronunciations[0])
# {'word': 'a', 'ipa': 'æ', ...}
```

Filtering by license tier:

```python
permissive = lexicon.filter(lambda row: row["licence"].startswith("cc-by-4"))
```

## Orthography

All text is normalized under the AƔBALU reference normaliser (`1.3.0+rules1.0.0`, 81 rules).

Text normalization standardizes character encoding across the dataset. In raw public Kabyle corpora, between 2.60% and 3.19% of rows exhibit Greek `ε` (U+03B5) substitutions for Latin `ɛ` (U+025B), which fragments tokenizer subwords if uncorrected. The normalization pipeline preserves all standard Berber Latin letters (`ɣ`, `ɛ`, `ḥ`, `ḍ`, `ṣ`, `ṭ`, `ẓ`, `ṛ`, `č`, `ǧ`, `ţ`).

## Known Limitations

- **Source imbalance**: The lexicon is predominantly composed of verbal inflections rather than a balanced distribution of parts of speech.
- **UPOS coverage**: Part-of-speech tags are missing for 15,357 entries and reflect upstream source annotations where present.
- **Loanwords in pronunciation data**: Includes common loanwords and foreign names occurring in source text (`amikruskop`, `armstrong`, `android`).
- **Glosses**: Lexical definitions are primarily in French, reflecting original lexicographical sources.

## Citation

```bibtex
@misc{agbalu_kablex,
  title  = {KabLex: a Kabyle lexicon and pronunciation dictionary},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabLex}
}
```

Please also cite the upstream sources listed under Composition.

## Licence

**CC-BY-4.0.** Published entries derive from CC-BY-4.0 and MIT sources. The pronunciation configuration is CC0-1.0 upstream. Each row includes source licensing metadata in the `licence` column.
