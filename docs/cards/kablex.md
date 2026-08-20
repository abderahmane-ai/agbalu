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

A lexical layer for Kabyle (Taqbaylit, `kab`, Latin script), from the
[AƔBALU](https://huggingface.co/agbalu) project.

**366,892 lexical entries** merged from three permissively licensed sources and normalised to
one orthography, plus **25,642 word–pronunciation pairs** aligned from sentence-level
grapheme-to-phoneme data.

```python
from datasets import load_dataset

lex = load_dataset("agbalu/KabLex", "lexicon", split="train")
ipa = load_dataset("agbalu/KabLex", "pronunciations", split="train")
```

## `lexicon`

366,892 entries — 234,936 distinct surface forms over 17,090 lemmas, 2,408 of them multiword.

| field | |
|---|---|
| `form` | the surface form, normalised |
| `lemma` | its lemma, where the source gives one |
| `upos` | Universal POS tag, where the source gives one |
| `feats` | morphological or domain features, UD-style. **`"_"` is UD's placeholder for "no features"**, on 25,858 entries; the other 341,034 carry a real value, and the column is never null |
| `glosses` | list of `{lang, text}` — 9,658 entries carry at least one |
| `source` | the registry id this entry came from |
| `licence` | that source's licence |

| POS | entries |
|---|---|
| `VERB` | 345,057 |
| unlabelled | 15,357 |
| `NOUN` | 5,603 |
| `PROPN` | 287 |
| `PART` | 249 |
| `ADJ`, `ADV`, `INTJ`, others | 246 |

**The distribution is a property of the sources, not of Kabyle.** 94% of the entries are
verb forms because the largest contributing source is an inflected verb list. Treat this as
a wide verbal paradigm resource with a thin nominal layer, not as a balanced dictionary.

### Composition

| source | entries | licence |
|---|---|---|
| `hf.boffire.kabyle-verbs` | 342,257 | CC-BY-4.0 |
| `hf.boffire.hunspell-kab` | 20,967 | MIT |
| `hf.agurzil.tafsut-maths-lexicon` | 3,668 | CC-BY-4.0 |

### What was excluded, and why

The built lexicon holds 395,834 entries. **28,942 are not published here**, and the cut is
made by code from each source's licence rather than by hand:

- **8,554 entries** from an **ODbL** toponym source. ODbL §4.4 obliges a publicly used
  derivative database to be offered under ODbL too, so including them would impose
  share-alike on this whole release. ODbL is share-alike, not permissive — a distinction
  that is easy to get wrong and that changes what a downstream user is allowed to do.
- **20,388 entries** whose upstream licence could not be established. An unresolved licence
  is not a permissive one. That source is independently available on the Hub.

Anyone who wants the full set can rebuild it from the registry; the exclusions are
mechanical and reproducible, not editorial.

## `pronunciations`

25,642 distinct words with an IPA reading, aligned from 59,185 sentence-level pairs
(99.53% alignment rate) over a 42-phoneme inventory.

| field | |
|---|---|
| `word` | the orthographic word, normalised |
| `ipa` | its reading |
| `variants` | list of `{ipa, count}` — every attested reading with its frequency |
| `repaired` | true where this project restored a character the source dropped (§ below) |

🔴 **This is a transliteration, not a phonetic transcription, and the distinction is
load-bearing.** Measured over all 292,921 aligned tokens:

- No word ever takes two readings — **ambiguity is 0.0%** — and that holds even when
  conditioned on the preceding word. The reading is a deterministic function of the
  spelling, which is a property of the generator, not evidence that Kabyle has no
  heterophonic homographs.
- **`count(ə)` equals `count(<e>)` in 100.00% of entries.** The schwa is orthographic `e`
  relabelled. **This data does not restore schwa**, and Kabyle schwa epenthesis is
  context-sensitive. For that, use
  [`agbalu/Juba-27M`](https://huggingface.co/agbalu/Juba-27M), which scores 93.51%
  positional schwa F1.
- **`ː` corresponds to a doubled letter in 100.00%** of the 10,465 entries carrying it.
- There is no cross-word phonology. Kabyle spirantization and assimilation cross word
  boundaries; a per-word table cannot express that.

Use it as a pronunciation dictionary and as a G2P training target. Do not use it as
evidence about Kabyle phonetics.

### Repaired characters

🔴 **The upstream generator emits nothing for a character it has no rule for, silently.**
This release repairs **199 entries**, flagged by the `repaired` field:

| character | words | before | after |
|---|---|---|---|
| `o` | 128 | `bob` → `ββ` | `βoβ` |
| **`ţ`** U+0163 | 73 | `aţan` → `ææn` | `ætːæn` |
| `é`, `ï` | 3 | `rosé` → `rs` | left as-is, see below |

**`ţ` is real Kabyle, not a foreign letter.** It is the Dallet-tradition notation for the
spirantised/tense t, attested 21,058 times in this project's corpus; of 1,407 word types
carrying it, 43% have a `tt` counterpart and 34% a `t` one, against **0.7%** for `ṭ`. It is
repaired to the geminate on that evidence. 🟡 No audio confirms this mapping — `ţ` occurs
zero times in Common Voice Kabyle.

Eight entries are **left exactly as the source had them** and named in
`agbalu-pronunciations-v1.stats.json`: `3d`, `mp3`, `androïd`, `rosé`, `supermarché`,
`muḥ€nd`, `xelleṣ̣`, `ṭeyyeb‟`. Each carries a digit, a currency sign, a stray combining
mark or a curly quote — outside the writing system, so repairing them would mean guessing.

Repairs are made only where the source reading is *shorter* than the reference table's.
Sound entries are untouched, because the table does not model `i`/`u` laxing and
regenerating one would replace an attested allophone with an approximation.

Source: `boffire/kabyle-g2p-training-data`, **CC0-1.0**. The repair is this project's.

## Usage

```python
from datasets import load_dataset

lexicon = load_dataset("agbalu/KabLex", "lexicon", split="train")
lexicon[0]
# {'form': 'asider', 'lemma': None, 'upos': None, 'feats': 'Domain=Math',
#  'glosses': [{'lang': 'fra', 'text': 'abaissement'}],
#  'source': 'hf.agurzil.tafsut-maths-lexicon', 'licence': 'cc-by-4.0'}

pronunciations = load_dataset("agbalu/KabLex", "pronunciations", split="train")
pronunciations[0]
# {'word': 'a', 'ipa': 'æ', ...}
```

**`licence` is a column because the tiers are not uniform.** Cut on it before redistributing:

```python
permissive = lexicon.filter(lambda row: row["licence"].startswith("cc-by-4"))
```

**`_` in `feats` is UD's own placeholder for "no features"**, and `None` in `lemma` or `upos`
means the source supplied none. Neither is missing data to be cleaned.

## Orthography

Everything is normalised to one documented orthography — normaliser `1.3.0+rules1.0.0`, 81
rules, zero idempotence violations over 931,342 sentences.

This matters more for Kabyle than the phrase usually suggests. Two seed corpora in this
project carry **2.60% and 3.19% homoglyph-corrupted rows** — Greek `ε` U+03B5 written for
Latin `ɛ` U+025B — and published Kabyle tokenizer vocabularies have that corruption baked
into their merges, at a measured **+17.8–21.3% token cost**.

Kabyle legitimately uses `ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`, and the emphatic and spirantised
distinctions carry meaning. Do not "clean" them away.

## Known limits

- **It is a union, not a balanced dictionary.** The POS distribution above is what the three
  sources contain, not a designed sample. Read it before drawing a rate from this file.
- **`upos` is absent for 15,357 entries** and is the source's tag where present, not a
  reviewed annotation.
- **The pronunciation set covers loanwords and proper nouns** — `amikruskop` ("microscope"),
  `armstrong`, `android` — because it is aligned from running text and those occur in it.
  These are also where the source's dropped-character defect concentrated, so most of the
  199 repaired entries are loanwords.
  **Do not filter this config by word length or character inventory.** The shortest entries
  are the most frequent words in the language: `ad` occurs 12,024 times, `d` 18,862, `i`
  11,141, `n` 6,001.
- **One upstream defect is patched here and flagged**: the G2P source deletes characters it
  has no rule for, and the `repaired` field marks every entry this release restored.
- Two upstream defects are known and *not* patched: one source maps `ţ`→`ṭ` against corpus
  evidence, and another tags `i` as `isem`. Both are owed as upstream reports.
- **Glosses are mostly French**, reflecting the sources.
- Nothing here is human-reviewed at scale. It is a merged, normalised, provenance-carrying
  union of existing resources — which is exactly what did not exist before, and is not the
  same thing as a curated dictionary.

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

**CC-BY-4.0.** Every published entry is CC-BY-4.0 or MIT; CC-BY-4.0 is the more restrictive
of the two and governs the release. The pronunciation config is CC0-1.0 upstream. Each row
carries its own `licence` field, so a stricter subset can be cut by code.
