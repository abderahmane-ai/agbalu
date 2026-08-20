---
license: apache-2.0
language:
  - kab
language_bcp47:
  - kab-Latn
size_categories:
  - 1M<n<10M
task_categories:
  - token-classification
task_ids:
  - part-of-speech
pretty_name: KabPunct — Kabyle punctuation and casing restoration corpus
tags:
  - kabyle
  - taqbaylit
  - berber
  - tamazight
  - low-resource
  - punctuation-restoration
  - truecasing
  - asr-post-processing
configs:
  - config_name: default
    data_files:
      - split: train
        path: default/train.jsonl
      - split: dev
        path: default/dev.jsonl
      - split: test
        path: default/test.jsonl
  - config_name: ood
    data_files:
      - split: ood
        path: ood/ood.jsonl
---

# KabPunct

A punctuation and capitalisation restoration corpus for Kabyle (Taqbaylit, `kab`, Latin
script), from the [AƔBALU](https://huggingface.co/agbalu) project.

**1,318,707 word-labelled sentences** drawn from the full AƔBALU-Text v1 corpus and
speaker-disjoint Common Voice Kabyle splits. Every sentence is broken into lowercased ASR
tokens with two parallel label sequences: which punctuation mark follows each word, and how
the word is capitalised. It is the training and evaluation corpus for
[`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M), which restores punctuation
at **0.793 macro-F1** on the held-out `test` split.

It is, as far as we can establish, **the first labelled punctuation restoration corpus for
Kabyle or any Berber language.**

```python
from datasets import load_dataset

ds = load_dataset("agbalu/KabPunct")

row = ds["train"][0]
# {
#   "words":       ["tecfiḍ", "fell-i"],
#   "punctuation": ["NONE",   "QUESTION"],
#   "case":        ["UPPER_INIT", "LOWER"],
#   "source":      "hf.abdelhaqueidali.kab-latn-tfng"
# }

def restore(row):
    mark_map = {"NONE": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?", "COLON": ":"}
    parts = []
    for word, punct, case in zip(row["words"], row["punctuation"], row["case"]):
        w = word[:1].upper() + word[1:] if case == "UPPER_INIT" else word
        parts.append(w + mark_map[punct])
    return " ".join(parts)

print(restore(row))
# "Tecfiḍ fell-i?"
```

## Splits

| split | config | rows | words | description |
|---|---|---|---|---|
| `train` | `default` | 1,262,922 | 12,101,835 | Text corpus + Common Voice train |
| `dev` | `default` | 5,597 | 29,280 | Common Voice dev, decontaminated |
| `test` | `default` | 5,160 | 26,969 | Common Voice test, decontaminated |
| `ood` | `ood` | 45,028 | 1,001,370 | Long-form prose (HCA), held out of training entirely |

## Schema

| field | type | description |
|---|---|---|
| `words` | `list[str]` | Lowercased, unpunctuated word tokens — the form an ASR system emits |
| `punctuation` | `list[str]` | Per-word punctuation label: `NONE`, `COMMA`, `PERIOD`, `QUESTION`, or `COLON` |
| `case` | `list[str]` | Per-word casing label: `LOWER` or `UPPER_INIT` |
| `source` | `str` | Provenance tag: which corpus this sentence came from |

Every list is the same length (one entry per word). Labels align to words, not subwords:
`words[i]` takes mark `punctuation[i]` and initial casing `case[i]`.

## Label Scheme

### Punctuation

| label | character | support in `test` |
|---|---|---|
| `NONE` | — | 20,904 |
| `PERIOD` | `.` | 4,357 |
| `QUESTION` | `?` | 861 |
| `COMMA` | `,` | 804 |
| `COLON` | `:` | 43 |

`!` and `;` fold into `PERIOD`. Together they account for 0.41% of tokens, and the corpus
contains identical sentences under both marks — the distinction is not recoverable from text
alone, and collapsing them costs less than carrying a class that cannot be learned.

### Casing

| label | meaning |
|---|---|
| `LOWER` | Word begins lowercase |
| `UPPER_INIT` | First character is uppercase (sentence-initial or proper noun) |

`ALL_CAPS` was trialled as a third class. It occurs once in dev and three times in test —
too few to learn without spurious generalisation. The model that carried it emitted fourteen
all-capitals words for the three real ones. Restoring an acronym to full capitals is out
of scope, and the class is absent.

## Usage

Labels are **per word**, three parallel lists of equal length.

```python
from datasets import load_dataset

train = load_dataset("agbalu/KabPunct", split="train")
train[0]
# {'words': ['tecfiḍ', 'fell-i'],
#  'punctuation': ['NONE', 'QUESTION'],
#  'case': ['UPPER_INIT', 'LOWER'],
#  'source': 'hf.abdelhaqueidali.kab-latn-tfng'}

ood = load_dataset("agbalu/KabPunct", "ood", split="ood")   # multi-sentence records
```

Reconstruct the surface form by applying `case` then appending `punctuation`:

```python
MARK = {"NONE": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?", "COLON": ":"}

def render(row):
    return " ".join(
        (word.capitalize() if case == "UPPER_INIT" else word) + MARK[mark]
        for word, mark, case in zip(row["words"], row["punctuation"], row["case"])
    )

render(train[0])   # 'Tecfiḍ fell-i?'
```

**Score on `ood` as well as `test`, or the number is a training-domain number.** `test` is
Common Voice, where every record is one sentence; `ood` carries 1.75 sentence-final marks per
record and is where a model that learned "one period, at the end" shows it.

## Decontamination

**58.2% of Common Voice Kabyle transcripts also appear in AƔBALU-Text v1**, because the
majority are Tatoeba sentences and Tatoeba is in the corpus. Every clip whose text is found
in the text corpus is removed from `dev` and `test`. This simultaneously decontaminates the
`Masinissa-31M` encoder used by `Belaid-31M`, which was pretrained on the same file. The
rows in the `Belaid-31M` evaluation are unseen by both the punctuation heads and the backbone.

A transcript without a sentence-final mark is dropped rather than labelled `NONE`. Those
transcripts carry neither punctuation nor capitals — transcriber habit, not Kabyle grammar.

`opus.nllb-kab` (871,663 records) is excluded from training entirely. Its record boundaries
were chosen by a bitext miner rather than a writer; a record can end mid-sentence, and
sentence-final punctuation is the predicted label.

## Out-of-Domain Split (`ood`)

The `ood` split is `hf.imsidag.kabyle-corpus-hca` — 45,028 records of continuous long-form
Kabyle prose, held out of training entirely. Common Voice records are one sentence each;
HCA records average 22 words across several sentences.

The `Belaid-31M` card reports this split explicitly alongside `test`. A model evaluated only
in its training domain has not been evaluated.

## Composition

| source id | type | train rows |
|---|---|---|
| `hf.abdelhaqueidali.kab-latn-tfng` | text corpus | 647,039 |
| `hf.imsidag.kabyle-corpus-ummto` | text corpus | 136,799 |
| `hf.imsidag.kabyle-raw-text` | text corpus | 121,338 |
| `hf.imsidag.kabyle-corpus-ubouira` | text corpus | 76,336 |
| `hf.boffire.kab-en-toponyms-sentences` | text corpus | 30,584 |
| `speech.train` | Common Voice 22.0 train | 151,100 |
| others (28 sources) | text corpus | 99,726 |
| `opus.nllb-kab` | **excluded** (miner-cut boundaries) | — |

The `ood` config contains `hf.imsidag.kabyle-corpus-hca` exclusively (45,028 rows).

## Reproduction

```bash
make punctuation TASK=corpus
```

Reads AƔBALU-Text v1 and the Common Voice speech splits, applies decontamination, and writes
the four files to `data/processed/punctuation/`.

## Citation

```bibtex
@misc{agbalu_kabpunct,
  title  = {KabPunct: a Kabyle punctuation and casing restoration corpus},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/datasets/agbalu/KabPunct}
}
```

## Licence

**Apache-2.0** for the label annotations and the dataset release. The underlying sentences
retain their upstream licences:

- Common Voice train/dev/test clips: **CC0-1.0** (Mozilla Foundation).
- Text corpus sentences: **mixed** — see the composition table above. Every share-alike source
  (`opus.nllb-kab`, ODbL-licensed rows) was excluded before publication. No row in any
  published split imposes a redistribution obligation on downstream users.
- The label sequences (`punctuation`, `case`) are this project's annotation and are
  **Apache-2.0**.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection.
