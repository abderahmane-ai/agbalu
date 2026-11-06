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

A punctuation and capitalisation restoration corpus for Kabyle (Taqbaylit, `kab`, Latin script), from the [AƔBALU](https://huggingface.co/agbalu) project.

**1,318,707 word-labelled sentences** drawn from the AƔBALU-Text v1 corpus and speaker-disjoint Common Voice Kabyle splits. Every sentence is tokenized into lowercased ASR words with two parallel label sequences: the trailing punctuation mark following each word, and the word's capitalization state. It serves as the training and evaluation corpus for [`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) (**0.793 macro-F1** on the held-out `test` split).

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

| Split | Config | Rows | Words | Description |
|---|---|---|---|---|
| `train` | `default` | 1,262,922 | 12,101,835 | Text corpus + Common Voice train |
| `dev` | `default` | 5,597 | 29,280 | Common Voice dev, decontaminated |
| `test` | `default` | 5,160 | 26,969 | Common Voice test, decontaminated |
| `ood` | `ood` | 45,028 | 1,001,370 | Continuous prose (HCA), held out of training |

## Schema

| Field | Type | Description |
|---|---|---|
| `words` | `list[str]` | Lowercased, unpunctuated word tokens (raw ASR format) |
| `punctuation` | `list[str]` | Per-word trailing punctuation mark: `NONE`, `COMMA`, `PERIOD`, `QUESTION`, `COLON` |
| `case` | `list[str]` | Per-word casing label: `LOWER` or `UPPER_INIT` |
| `source` | `str` | Provenance source identifier |

Labels align to whitespace-delimited words: `words[i]` takes trailing mark `punctuation[i]` and capitalization `case[i]`.

## Label Scheme

### Punctuation

| Label | Mark | Support in `test` |
|---|---|---|
| `NONE` | — | 20,904 |
| `PERIOD` | `.` | 4,357 |
| `QUESTION` | `?` | 861 |
| `COMMA` | `,` | 804 |
| `COLON` | `:` | 43 |

`!` and `;` fold into `PERIOD` during training, as together they account for 0.41% of tokens and represent terminal pauses.

### Casing

| Label | Meaning |
|---|---|
| `LOWER` | Word begins lowercase |
| `UPPER_INIT` | Initial character is uppercase (sentence-initial or proper noun) |

`ALL_CAPS` is excluded from the single-utterance configuration because acronyms occur too infrequently in isolated Common Voice sentences (1 instance in dev, 3 in test) for robust sentence-level supervision. Full-discourse acronym restoration is addressed in [`agbalu/Belaid-75M`](https://huggingface.co/agbalu/Belaid-75M).

## Usage

```python
from datasets import load_dataset

train = load_dataset("agbalu/KabPunct", split="train")
print(train[0])
# {'words': ['tecfiḍ', 'fell-i'],
#  'punctuation': ['NONE', 'QUESTION'],
#  'case': ['UPPER_INIT', 'LOWER'],
#  'source': 'hf.abdelhaqueidali.kab-latn-tfng'}

ood = load_dataset("agbalu/KabPunct", "ood", split="ood")
```

Surface text reconstruction helper:

```python
MARK = {"NONE": "", "COMMA": ",", "PERIOD": ".", "QUESTION": "?", "COLON": ":"}

def render(row):
    return " ".join(
        (word.capitalize() if case == "UPPER_INIT" else word) + MARK[mark]
        for word, mark, case in zip(row["words"], row["punctuation"], row["case"])
    )

print(render(train[0]))  # 'Tecfiḍ fell-i?'
```

Evaluating on both `test` and `ood` provides a comprehensive assessment: `test` consists of single-utterance Common Voice sentences, while `ood` contains continuous multi-sentence literary prose averaging 1.75 sentence-final marks per record.

## Decontamination

58.2% of Common Voice Kabyle transcripts appear in the general web crawl via Tatoeba. All utterances sharing strings with the training corpus were removed from `dev` and `test` splits.

Transcripts lacking sentence-final punctuation were removed from the corpus to prevent unpunctuated transcription artifacts from biasing label statistics.

`opus.nllb-kab` (871,663 records) was excluded from training entirely because automated bitext-mined segment boundaries frequently truncate sentences mid-clause.

## Out-of-Domain Split (`ood`)

The `ood` split comprises 45,028 continuous prose paragraphs from `hf.imsidag.kabyle-corpus-hca`, held out completely from training. While Common Voice utterances contain a single sentence per record, HCA records average 22 words across multiple clauses and sentences.

The `Belaid-31M` evaluation reports metrics across both `test` (in-domain single utterances) and `ood` (out-of-domain continuous prose).

## Composition

| Source ID | Type | Train Rows |
|---|---|---|
| `hf.abdelhaqueidali.kab-latn-tfng` | Text corpus | 647,039 |
| `hf.imsidag.kabyle-corpus-ummto` | Text corpus | 136,799 |
| `hf.imsidag.kabyle-raw-text` | Text corpus | 121,338 |
| `hf.imsidag.kabyle-corpus-ubouira` | Text corpus | 76,336 |
| `hf.boffire.kab-en-toponyms-sentences` | Text corpus | 30,584 |
| `speech.train` | Common Voice 22.0 train | 151,100 |
| Others (28 sources) | Text corpus | 99,726 |
| `opus.nllb-kab` | Excluded (automated bitext boundaries) | — |

The `ood` split contains `hf.imsidag.kabyle-corpus-hca` exclusively (45,028 rows).

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

**Apache-2.0** for dataset annotations and release format. Underlying text retains source licensing:
- Common Voice clips: **CC0-1.0** (Mozilla Foundation).
- Text corpus sentences: Mixed permissive sources (all share-alike and ODbL sources excluded prior to publication).
- Label sequences (`punctuation`, `case`): **Apache-2.0**.
