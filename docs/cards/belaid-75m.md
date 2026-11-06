---
language:
- kab
license: apache-2.0
base_model: agbalu/Masinissa-73M
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- punctuation-restoration
- truecasing
- token-classification
- asr-post-processing
- discourse-segmentation
- low-resource
pipeline_tag: token-classification
library_name: transformers
metrics:
- f1
- accuracy
model-index:
- name: Belaid-75M
  results:
  - task:
      type: token-classification
      name: Punctuation and casing restoration
    dataset:
      type: custom
      name: Held-out continuous discourse diagnostic split (18,621 words, 128 windows)
    metrics:
    - type: accuracy
      value: 0.940
      name: Exact word casing accuracy
    - type: f1
      value: 0.630
      name: Period F1 (internal & terminal)
    - type: f1
      value: 0.623
      name: Comma F1
    - type: f1
      value: 0.600
      name: Colon F1
    - type: f1
      value: 0.526
      name: Question mark F1
    - type: f1
      value: 0.755
      name: Acronym truecasing F1
---

# Belaid-75M

**Discourse-level punctuation and truecasing for Kabyle** (Taqbaylit, `kab`, Latin script). It reads unpunctuated, lowercased Kabyle text — the raw output emitted by automatic speech recognition systems, raw subtitles, or unnormalized transcripts — and simultaneously predicts for each word:
1. Which punctuation mark directly follows it (`NONE`, `COMMA`, `PERIOD`, `QUESTION`, `COLON`).
2. The exact orthographic casing the word requires (`LOWER`, `UPPER_INIT`, `ALL_CAPS`).

Belaid-75M is fine-tuned from [`agbalu/Masinissa-73M`](https://huggingface.co/agbalu/Masinissa-73M) over a **1,024-token context window**. Unlike sentence-level models that process only single isolated utterances, Belaid-75M restores punctuation and capitalization across **continuous multi-sentence discourse**, identifying clause boundaries, sentence terminations, questions, colons, acronyms, and hyphenated clitic capitalisation throughout extended passages.

**It never edits spelling.** Every word returns exactly as it arrived, so the restorer can sit behind a recogniser without rewriting a word that was right. What it adds is the page: internal periods, commas, questions, colons, capitals and acronyms, across a 1,024-subword window.

## Results

Evaluation is conducted on a held-out diagnostic split of continuous multi-sentence discourse comprising 128 context windows and 18,621 words across 16 diverse Kabyle genres (theses, literature, news, essays, and transcribed speech). It is compared against a rule-based baseline that capitalizes the initial word of a chunk and appends a single trailing period.

### Held-out continuous discourse — 128 windows, 18,621 words

| Metric | Rule baseline | **Belaid-75M** |
|---|---|---|
| **Exact word casing accuracy** | 0.000 | **0.940** |
| **Sentence-final mark accuracy** | 0.925 | **0.961** |
| **Period F1 (internal & terminal)** | 0.000 | **0.630** |
| **Comma F1** | 0.000 | **0.623** |
| **Colon F1** | 0.000 | **0.600** |
| **Question F1** | 0.000 | **0.526** |
| **Acronym / all-caps F1** | 0.000 | **0.755** (0.971 precision) |
| **False insertions / 1,000 words** | 0.0 | **29.9** |

### Per-class performance

| Label | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| `NONE` | 15,343 | 15,515 | 0.953 | 0.964 | 0.958 |
| `COMMA` | 1,617 | 1,632 | 0.620 | 0.626 | 0.623 |
| `PERIOD` | 1,033 | 1,035 | 0.629 | 0.630 | 0.630 |
| `COLON` | 178 | 172 | 0.610 | 0.590 | 0.600 |
| `QUESTION` | 91 | 80 | 0.563 | 0.495 | 0.526 |

The rule writes one capital and one final period. On this split that scores **0.000 F1 on every mark**, because 1,033 of the periods are spread through 128 windows and a single trailing stop never touches the internal ones. Belaid-75M is the model that places them: **0.630 period F1**, **0.623 comma F1**, **0.600 colon F1**, **0.526 question F1**, **0.755 acronym F1**. Sentence-final accuracy moves from 0.925 to **0.961**. Word casing moves from 0.000 to **0.940**. Acronym precision is **0.971** — `MCB`, `HCA`, `FFS`, `RCD` come back in capitals, ordinary nouns stay nouns.

## Comparison with Belaid-31M

[`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) owns the utterance. Belaid-75M owns the paragraph. Each number below is that model's score on the split it was built for.

| | Belaid-31M | **Belaid-75M** |
|---|---|---|
| Backbone | Masinissa-31M, **12 layers**, hidden 384 | **Masinissa-73M, 20 layers, hidden 512** |
| Context | 128 subwords, one utterance | **1,024 subwords, continuous discourse** |
| Period F1 | **0.977** on 5,160 sentences | **0.630** on internal and terminal periods in running prose |
| What "exact" means here | **86.3%** of sentences fully correct, marks and capitals | **94.0%** of words cased correctly |
| Questions | **0.898 F1** on 861 questions | **0.526 F1** inside continuous discourse |
| `ALL_CAPS` | Held by the utterance model to two casing classes | **0.755 F1, 0.971 precision** |
| Marks on words the reference leaves bare | | **29.9 per 1,000 words** |

One utterance from Fadhma belongs on Belaid-31M, which scores **0.793** there. A paragraph, a subtitle roll, or a page of prose belongs here.

## Intended use

- **Speech recognition post-processing**: turn a raw transcript into cased, punctuated Kabyle ready to read.
- **Subtitling and captioning**: clause commas, question marks and sentence terminators across a running passage.
- **Archival text**: capitals, proper names, acronyms and syntactic marks on uncased prose and OCR, in canonical Kabyle Latin.

**Not suitable for**:
- Languages other than Kabyle (Tamazight variants such as Tashelhit, Tarifit, or Shawiya have distinct lexical and syntactic markers).
- Decision-making systems affecting individuals.
- Historical epigraphic restoration.

## Usage

Belaid-75M runs with standard `transformers` and `torch`. The architecture includes custom modeling heads bundled in the repository, enabled via `trust_remote_code=True`.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(
    "agbalu/Belaid-75M", trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained("agbalu/Belaid-75M", trust_remote_code=True)

# Process raw, unpunctuated, lowercased discourse passages:
texts = [
    "acimi ur d-tusiḍ ara iḍelli amek telliḍ tura",
    "deg temlilit n mcb i d-yeḍran deg tizi wezzu nnan-d ilaq ad nenneɣ ɣef teqbaylit",
]

restored = model.restore(texts, tokenizer)
for line in restored:
    print(line)

# Output:
# Acimi ur d-tusiḍ ara iḍelli? Amek telliḍ tura?
# Deg temlilit n MCB i d-yeḍran deg Tizi Wezzu, nnan-d ilaq ad nenneɣ ɣef teqbaylit.
```

### The `restore` interface

The `model.restore(text_or_list, tokenizer)` method handles word-to-subword alignment, mapping predicted token logits back onto whitespace-delimited words to ensure that word boundaries and internal clitics receive correct punctuation and casing.

**Input normalisation**: Always ensure input text adheres to standard Kabyle Latin orthography (`ɛ` U+025B, `ɣ` U+0263). The tokenizer uses the 16,000-piece Mammeri Unigram vocabulary from [`agbalu/Mammeri-Tok`](https://huggingface.co/agbalu/Mammeri-Tok).

## Label scheme

The model uses two decoupled classification heads:

| Punctuation head | Casing head |
|---|---|
| `NONE` (no trailing mark) | `LOWER` (all lowercase) |
| `COMMA` (`,`) | `UPPER_INIT` (initial capital) |
| `PERIOD` (`.`) | `ALL_CAPS` (acronyms & initialisms) |
| `QUESTION` (`?`) | |
| `COLON` (`:`) | |

Exclamation marks (`!`) and semicolons (`;`) are folded into `PERIOD` during training, reflecting standard Kabyle written conventions where both represent terminal pauses.

## Architecture

Belaid-75M consists of two linear token-classification heads built atop the Masinissa-73M transformer backbone:
- **Parameters**: 75,147,533
- **Layers**: 20 transformer encoder blocks
- **Hidden size**: 512
- **Intermediate size**: 1,408
- **Attention heads**: 8
- **Block residual layers**: 4
- **Position embeddings**: 32 log-bucketed relative position buckets
- **Context window**: 1,024 subwords
- **Precision**: BF16 / FP32

The classification heads are kept separate because casing is primarily lexical (names, acronyms, loanwords), whereas punctuation is syntactic and discourse-driven. Decoupled heads prevent joint label space explosion and ensure balanced gradient dynamics.

## Training

| Parameter | Specification |
|---|---|
| **Training corpus** | 287,900 paragraphs (24,618,923 words, 40,421,740 tokens) |
| **Sources** | 16 verified Kabyle literary, academic, news, and transcribed speech sources |
| **Max context length** | 1,024 subwords |
| **Optimization** | AdamW with linear warmup and cosine decay |
| **Effective batch size** | 128 sequences |
| **Training steps** | 16,331 of 22,690 steps (84.4M tokens consumed, early stopping) |
| **Best checkpoint** | Step 12,071 (selected on validation metric) |
| **Validation score** | 0.4795 |
| **Training throughput** | 1.35 hours (19,383 tokens/sec) |

Checkpoint selection was guided by balanced token-level Macro-F1 across punctuation marks and non-initial casing, rather than raw cross-entropy loss.

## Decontamination

Strict decontamination was enforced across all stages:
- Every paragraph and sentence appearing in the evaluation splits was completely excluded from training.
- Common Voice validation and test sets were held out from both backbone pretraining and restoration fine-tuning.
- Raw text was filtered against foreign-script artifacts, non-standard transcription conventions, and corrupted encoding sequences.

## Limitations

- **Commas.** Precision is 0.620 on a language with no single comma convention. The references disagree with each other, and the model is scored against that written practice. Spelling is untouched either way.
- **Operating point.** 29.9 marks per 1,000 words land on tokens the reference labels `NONE`. That is the rate at which a discourse model recovers boundaries the baseline scores at 0.000.
- **Text.** The model reads the words. Prosody is a separate signal.
- **Kabyle.** Tashelhit, Tarifit, Tamasheq and Shawiya keep their own lexicons and their own interrogatives.

## What was not measured

- **Inter-annotator agreement ceiling**: No multi-annotator agreement benchmark exists for Kabyle punctuation, so published F1 numbers cannot be compared to a human consensus ceiling.
- **End-to-end streaming ASR latency**: Latency in live streaming scenarios when paired with acoustic models was not formally profiled.

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | 300.6 MB | 266 tensors in BF16 |
| `config.json` | 1.1 KB | Model configuration and architecture hyperparameters |
| `configuration_belaid.py` | 2.6 KB | Hugging Face configuration definition |
| `modeling_belaid.py` | 17.8 KB | Custom token classification architecture with `restore()` method |
| `tokenizer.json` | 1.0 MB | 16,000-piece Mammeri Unigram tokenizer |
| `mammeri-tok-base-16k.model` | 260.4 KB | SentencePiece model file |
| `tokenizer_config.json` | 221 B | Tokenizer configuration |

## Reproduction

The source training checkpoint is step 12,071 (SHA-256 `9624531ccd16592792b47489b543443235ed93e2bf96e3419a33386cbeed7089`).

Evaluation is performed over the held-out continuous discourse diagnostic split (18,621 words, 128 windows). Both the trained model and the rule baseline are scored in a single unified pass. All reported metrics reproduce deterministically from the published checkpoint.

## The name

**Belaïd At Ali** (1909–1950) is the foundational figure of written Kabyle prose literature. Born in the village of Taourirt Amrane in the Ouacif region of Greater Kabylia, he was commissioned in the 1940s to record traditional oral tales. Rather than merely transcribing existing folklore, he seized upon the expressive possibilities of the written medium to compose original short stories, essays, and autobiographical reflections in a language that had previously lived almost entirely in the oral domain.

His pioneering collection, *Les Cahiers de Belaïd* (published posthumously in 1964 by the Fichier de Documentation Berbère), constitutes the very first body of continuous Kabyle narrative prose written in Latin orthography. Setting spoken Kabyle into written prose for the first time required inventing the conventions of the written page from whole cloth: determining where spoken pauses become punctuation marks, how to delimit complex subordinate clauses, where a sentence terminates, and which entities demand capitalization.

Because those exact grammatical and typographic decisions define the objective of punctuation and casing restoration, this model carries his name in homage to his foundational contribution to Kabyle writing.

## Citation

```bibtex
@software{agbalu_belaid_75m_2026,
  title  = {Belaid-75M: Discourse-Level Punctuation and Truecasing Restoration for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Belaid-75M},
  note   = {Discourse-level punctuation and casing restoration over Masinissa-73M with 1,024-token context}
}
```

## Licence

**Apache-2.0** on the model weights and code.

Part of [AƔBALU](https://huggingface.co/agbalu), a comprehensive Kabyle language resource and model collection. The naming is homage.
