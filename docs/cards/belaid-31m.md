---
language:
- kab
license: apache-2.0
base_model: agbalu/Masinissa-31M
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- punctuation-restoration
- truecasing
- token-classification
- asr-post-processing
- low-resource
pipeline_tag: token-classification
library_name: transformers
metrics:
- f1
model-index:
- name: Belaid-31M
  results:
  - task:
      type: token-classification
      name: Punctuation and casing restoration
    dataset:
      type: fsicoli/common_voice_22_0
      name: Common Voice 22.0 Kabyle, decontaminated test split (5,160 sentences)
    metrics:
    - type: f1
      value: 0.793
      name: Macro-F1 over the four marks and NONE
    - type: f1
      value: 0.933
      name: F1, capitalisation of non-initial proper nouns
    - type: accuracy
      value: 0.863
      name: Exact match, marks and casing, whole sentence
  - task:
      type: token-classification
      name: Punctuation and casing restoration, out of domain
    dataset:
      type: imsidag/kabyle-corpus-hca
      name: Long-form Kabyle prose held out of training (45,028 records)
    metrics:
    - type: f1
      value: 0.635
      name: Macro-F1 over the four marks and NONE
---

# Belaid-31M

> [!TIP]
> **Utterance vs. Discourse Restorers:** Belaid-31M is optimized for single-utterance speech restoration (128 subwords, based on [`Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M)), achieving **86.3% whole-sentence exact match** and **0.793 macro-F1**. For paragraph-length or continuous prose, [`Belaid-75M`](https://huggingface.co/agbalu/Belaid-75M) provides a 1,024-token context window based on [`Masinissa-73M`](https://huggingface.co/agbalu/Masinissa-73M).

A 31.4M-parameter punctuation and capitalization restoration model for **Kabyle** (Taqbaylit, `kab_Latn`), fine-tuned from [`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M).

Belaid-31M processes unpunctuated and uncased text—such as raw ASR transcripts from [`agbalu/Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M)—and predicts sentence punctuation marks and word capitalization per token. It operates strictly via token classification without modifying underlying character spellings.

## Evaluation Results

### Common Voice In-Domain Test (5,160 sentences, 26,969 words)

Scored on decontaminated Common Voice 22.0 test sentences against a standard heuristic baseline (capitalizing initial words and appending a terminal period):

| Metric | Baseline | Belaid-31M |
|---|---|---|
| **Macro-F1 over marks** | 0.227 | **0.793** |
| **Sentence-final mark accuracy** | 0.837 | **0.970** |
| **Exact match (marks + casing, whole sentence)** | 0.592 | **0.863** |
| **Non-initial proper noun capitalization F1** | 0.000 | **0.933** |

Per-class performance on punctuation marks:

| Label | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `NONE` | 20,904 | 20,897 | 0.989 | 0.988 | **0.988** |
| `COMMA` | 804 | 843 | 0.680 | 0.713 | **0.696** |
| `PERIOD` | 4,357 | 4,351 | 0.978 | 0.976 | **0.977** |
| `QUESTION` | 861 | 838 | 0.911 | 0.886 | **0.898** |
| `COLON` | 43 | 40 | 0.625 | 0.581 | **0.602** |

Kabyle lexical question markers (`acu`, `anida`, `amek`, `melmi`) enable high question mark precision and recall (0.898 F1). Non-initial proper nouns reach 0.933 F1.

### Out-of-Domain Evaluation (45,028 long-form records, 1,000,928 words)

Evaluated on held-out long-form prose (`imsidag/kabyle-corpus-hca`):

| Metric | Baseline | Belaid-31M |
|---|---|---|
| **Macro-F1 over marks** | 0.168 | **0.635** |
| **Sentence-final mark accuracy** | 0.925 | **0.955** |
| **Non-initial proper noun capitalization F1** | 0.000 | **0.674** |
| **`COMMA` F1** | 0.000 | **0.580** |
| **`COLON` F1** | 0.000 | **0.721** |

## Intended Use

- **ASR Post-Processing:** Restoring standard punctuation and capitalization on raw acoustic transcripts from [`Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M).
- **Text Normalization:** Re-casing and punctuating web-scraped or lowercased Kabyle corpora.
- **Input Scope:** Single-utterance inputs up to 128 subwords. For multi-sentence paragraphs, use [`Belaid-75M`](https://huggingface.co/agbalu/Belaid-75M).
- **Out of Scope:** Structural document segmentation or paragraph re-breaking.

## Usage

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO = "agbalu/Belaid-31M"
model = AutoModelForTokenClassification.from_pretrained(REPO, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

sentences = [
    "ur zmireɣ ara ad d-aseɣ assa",
    "anida i tedduḍ a yelli",
]

restored = model.restore(sentences, tokenizer)
print(restored)
# Output: ['Ur zmireɣ ara ad d-aseɣ assa.', 'Anida i tedduḍ a yelli?']
```

The `restore()` helper aligns subword predictions back to whole surface words, applying capitalization to word characters and attaching punctuation marks.

## Label Scheme

- **Punctuation (5 classes):** `NONE`, `COMMA`, `PERIOD`, `QUESTION`, `COLON`. Exclamation marks and semicolons are mapped to `PERIOD` due to sparse training support.
- **Casing (2 classes):** `LOWER`, `UPPER_INIT` (capitalized first letter).

## Architecture

| Parameter | Value |
|---|---|
| **Backbone** | [`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M) |
| **Total Parameters** | 31,423,751 (299,911 across two classification heads) |
| **Layers / Hidden / Heads** | 12 layers / 384 hidden / 6 attention heads |
| **Intermediate Dimension** | 1,280 (gated GLU) |
| **Positional Embeddings** | Log-bucketed relative positions (32 buckets, max window 512) |
| **Context Window** | 128 subwords |
| **Vocabulary** | 16,000 SentencePiece Unigram ([`agbalu/Mammeri-Tok`](https://huggingface.co/agbalu/Mammeri-Tok)) |

Two dedicated linear heads independently predict punctuation marks and word capitalization over the encoder hidden states.

## Training Recipe

| Parameter | Value |
|---|---|
| **Training Examples** | 1,262,922 sentences |
| **Epochs** | 3 epochs |
| **Checkpoint Selection** | Step 11,097 (selected based on peak validation macro-F1: 0.8041 marks, 0.9368 casing) |
| **Decontamination** | Excluded all Common Voice sentences overlapping with pretraining corpora to ensure strict isolation |

## Limitations

- **Utterance Scope:** Optimized for single-sentence inputs. In continuous multi-sentence discourse, use [`Belaid-75M`](https://huggingface.co/agbalu/Belaid-75M).
- **Comma Ambiguity:** Comma placement reflects empirical patterns in corpus texts; precision is 0.680 on in-domain test data.
- **Acoustic Information:** Operates purely on text features; audio pauses or pitch contours are not accessible to the model.

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | ~125.7 MB | Model weights (115 tensors, 31,423,751 parameters) |
| `config.json` | < 2 KB | Model configuration |
| `modeling_belaid.py`, `configuration_belaid.py` | < 25 KB | Transformers modeling and configuration code |
| `tokenizer.json`, `tokenizer_config.json` | ~1.0 MB | Fast tokenizer configuration |
| `agbalu-tok-base-16k.model` | 260 KB | Raw SentencePiece unigram model |

## Reproduction

Fine-tuned from `agbalu/Masinissa-31M` for 3 epochs with linear learning rate warmup and decay. Evaluation compares the model and the heuristic baseline over identical test splits. All reported metrics reproduce deterministically from the published checkpoint.

## The name

**Belaïd At Ali** (1909–1950) is the foundational figure of written Kabyle prose literature. Born in the
village of Taourirt Amrane in the Ouacif region, he was commissioned in the 1940s to record traditional oral
tales. Rather than merely transcribing them, he composed original short stories, essays, and autobiographical
narratives in a language that had carried neither. *Les Cahiers de Belaïd*, published posthumously in 1964
by the Fichier de Documentation Berbère, is the first extensive body of Kabyle prose in Latin characters.

Writing prose where none existed meant deciding where a spoken pause becomes a punctuation mark, where a
clause breaks, where a sentence ends, and what takes a capital — with no written precedent to follow. That
exact set of syntactic and orthographic decisions is this model's objective, and it is why the model
carries his name in homage to his pioneering contribution to Kabyle writing.

## Citation

```bibtex
@software{agbalu_belaid_2026,
  title  = {Belaid-31M: Punctuation and Casing Restoration for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Belaid-31M},
  note   = {Two token-classification heads over Masinissa-31M; macro-F1 0.793 over marks}
}
```

## Licence

- **Weights & Code:** Apache-2.0.
- **Training Corpus:** Derived from AƔBALU-Text v1.
