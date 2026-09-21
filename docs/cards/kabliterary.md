---
license: cc-by-sa-4.0
language:
  - kab
language_bcp47:
  - kab-Latn
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
  - fill-mask
  - translation
pretty_name: KabLiterary — Kabyle literary translations & classical prose corpus
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - literature
  - classical-prose
  - translations
  - long-form
  - parallel-corpus
  - books
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.parquet
      - split: dev
        path: dev.parquet
      - split: test
        path: test.parquet
---

# KabLiterary

A 28,907-paragraph, 657,713-word corpus of classical world literature in canonical Kabyle Latin
orthography, from the [AƔBALU](https://huggingface.co/agbalu) project.

Eleven works — translated or originally composed in Kabyle — spanning epic poetry, gothic fiction,
philosophical prose, folktales, military strategy, and 19th-century novels. The corpus is designed
for language modelling, vocabulary probing, fine-tuning, and literary translation benchmarking
where long-form, high-register Kabyle text is required.

```python
from datasets import load_dataset

ds = load_dataset("agbalu/KabLiterary")
# DatasetDict({
#     'train': Dataset(27619),
#     'dev': Dataset(352),
#     'test': Dataset(936)
# })
```

## Splits

Split boundaries are at the book level. No paragraph from a dev or test book appears in `train`.

| split | books | paragraphs | words | characters | purpose |
|---|---|---:|---:|---:|---|
| `train` | 9 books | 27,619 | 630,454 | 3,837,504 | Pre-training, instruction tuning, MLM |
| `dev` | *The Prophet* (*Nnbi*) | 352 | 4,243 | 24,240 | Hyperparameter tuning, dev loss |
| `test` | *Alice in Wonderland* (*Tirmit n Alice*) | 936 | 23,016 | 138,756 | Held-out perplexity, generation eval |
| **total** | **11 books** | **28,907** | **657,713** | **4,000,500** | Full literary corpus |

## Corpus Inventory

| Book | Kabyle Title | Author | Year | Source | Genre | Split | Paragraphs | Words |
|---|---|---|---:|---|---|---|---:|---:|
| *The Count of Monte Cristo* | *Lkont n Monte-Cristo* | Alexandre Dumas | 1844 | French | Adventure fiction | `train` | 4,362 | 106,000 |
| *Les Misérables* | *Imeɣban* | Victor Hugo | 1862 | French | Historical fiction | `train` | 2,862 | 96,315 |
| *The Confessions* | *Isteɛṛufen n Uɣustin Aqeddis* | Augustine of Hippo | 397 | English (Latin) | Theology / Philosophy | `train` | 1,336 | 91,652 |
| *Dracula* | *Dracula, Amudir n Yiḍ* | Bram Stoker | 1897 | English | Gothic horror | `train` | 2,239 | 133,380 |
| *The Arabian Nights* | *Alf Yiḍ d Yiḍ* | Anonymous / A. Lang | 1898 | English (Arabic) | Folklore tales | `train` | 1,737 | 85,944 |
| *The Odyssey* | *Asikel n Ulises* | Homer / S. Butler | -800 | English (Greek) | Epic poetry | `train` | 14,427 | 101,584 |
| *The Art of War* | *Taẓuri n Ṭṭraḍ* | Sun Tzu / L. Giles | 1910 | English (Chinese) | Military philosophy | `train` | 410 | 9,803 |
| *The Fables of Bidpai* | *Timucuha n Bidpai* | Bidpai / Ibn al-Muqaffa | 1884 | English (Sanskrit) | Fables & folklore | `train` | 106 | 5,013 |
| *Poems of Si Mohand* | *Isefra n Si Mohand u Mhand* | Si Mohand u Mhand | 1904 | Kabyle (original) | Classical poetry | `train` | 140 | 763 |
| *The Prophet* | *Nnbi* | Kahlil Gibran | 1923 | English | Poetic philosophy | `dev` | 352 | 4,243 |
| *Alice in Wonderland* | *Tirmit n Alice deg Tmurt n Lɛeǧayeb* | Lewis Carroll | 1865 | English | Children's fiction | `test` | 936 | 23,016 |

*Isefra n Si Mohand u Mhand* is the single original Kabyle work — oral poetry transcribed by
Mouloud Mammeri and not derived from any source language. All other works are machine-translated
from the source editions listed above.

## Schema

Every record shares a fixed rectangular schema:

| field | type | description |
|---|---|---|
| `id` | string | Unique record identifier (e.g. `kabliterary_train_00001`) |
| `book_id` | string | Canonical book slug (e.g. `le-comte-de-monte-cristo`, `alice-in-wonderland`) |
| `book_title` | string | Standard title in English or original language |
| `book_title_kab` | string | Title in Kabyle Latin orthography |
| `author` | string | Author name |
| `year` | int32 | Original publication year (negative for BC dates) |
| `genre` | string | Literary genre classification |
| `source_language` | string | ISO 639-3 code of the source language (`fra`, `eng`, `kab`) |
| `target_language` | string | ISO 639-3 code of the target language (`kab`) |
| `chapter` | string | Chapter or section title |
| `paragraph_id` | int32 | 1-indexed paragraph counter within the work |
| `text` | string | Kabyle literary text in canonical Latin orthography |
| `source_text` | string | Aligned source text where 1-to-1 line parallel; empty string elsewhere |
| `word_count` | int32 | Word count of the Kabyle text |
| `char_count` | int32 | Character count of the Kabyle text |

`source_text` is populated only for works where line-level alignment was preserved during
translation — *The Art of War*, *The Prophet*, and *Isefra n Si Mohand u Mhand*. For the
longer prose novels, paragraphs were batched during translation and `source_text` is empty.

## Usage

### Default splits

```python
from datasets import load_dataset

dataset = load_dataset("agbalu/KabLiterary")
print(dataset["test"][4]["text"])
# 'Tella Alice tebda teɛya aṭas seg akken teqqim ɣer tama n uletma-s...'
```

### Filtering by work or genre

```python
from datasets import load_dataset

train = load_dataset("agbalu/KabLiterary", split="train")

# French 19th-century novels only
french_novels = train.filter(
    lambda row: row["book_id"] in ("le-comte-de-monte-cristo", "les-miserables")
)
print(f"{len(french_novels):,} paragraphs, {sum(french_novels['word_count']):,} words")

# Records with 1-to-1 source alignment
parallel = train.filter(lambda row: len(row["source_text"]) > 0)
print(f"{len(parallel):,} line-level parallel records")
```

### Full plain-text books

All 11 works are included in the repository's `books/` directory:

```bash
git clone https://huggingface.co/datasets/agbalu/KabLiterary
ls KabLiterary/books/
cat KabLiterary/books/the-prophet.txt
```

## Curation

### Translation

The ten translated works were produced by NLLB-based machine translation and subjected to a
multi-stage quality pass before inclusion:

1. **Repetition pruning.** 276+ beam-search generation loops were detected and removed across
   all ten texts. No tail-degenerate paragraph survives in any split.
2. **Heading localisation.** Residual French chapter headings in *Les Misérables* and other
   works were translated into standard Kabyle Latin (e.g. *L'évêque travaille* →
   *Aselway ixeddem*, *L'Alouette* → *Tiwiziwt*).
3. **Source-language artifact removal.** Any paragraph retaining a title or quotation in
   English or French was re-translated or manually corrected to Kabyle.

### Orthography

All text is normalised to AƔBALU canonical Kabyle Latin orthography (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ`).
A census across all 28,907 records confirmed:

- zero Greek epsilon homoglyphs (`ε` U+03B5)
- zero soft hyphens (`\u00ad`)
- NFC normalisation throughout

HTML entities (`&quot;` and variants) were repaired to standard punctuation during the quality
pass.

## Known Limitations

- **Machine-translated syntax.** Ten of eleven works are machine-translated. The text is
  fluent and orthographically correct, but phrase-level structure carries influence from the
  French or English source. No human post-editing was performed beyond the artifact and heading
  corrections described above.
- **No 1-to-1 paragraph alignment in prose novels.** Longer works (*Monte-Cristo*,
  *Les Misérables*, *Dracula*, *The Arabian Nights*) were translated with multi-sentence
  batching. Paragraph boundaries do not map to individual source sentences.
- **Archaic and specialised vocabulary.** Works such as *Confessions of Saint Augustine* and
  *The Odyssey* require theological and classical vocabulary that has no established Kabyle
  equivalent. The translations use loan-adapted morphological frames; the decisions are
  internally consistent but not reviewed against a Kabyle lexicographic standard.
- **Single source edition per work.** Each translation was produced from one source edition.
  Variant readings, abridgements, or multi-edition differences were not considered.

## Reproduction

```bash
make release-kabliterary
```

The dataset is assembled by `tools.build_kabliterary` from the verified texts in
`artifacts/translations/` and `data/documents/`.

## Citation

```bibtex
@misc{agbalu_kabliterary,
  title  = {KabLiterary: A 657K-Word Corpus of Classical Literary Translations in Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabLiterary}
}
```

## Licence

**CC-BY-SA-4.0.** Source texts are public domain (Project Gutenberg and digitised archives).
Translations and the assembled dataset are produced by the AƔBALU project under CC-BY-SA-4.0.
Attribution is required; derivative datasets must carry the same licence.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle NLP corpus and model suite.
