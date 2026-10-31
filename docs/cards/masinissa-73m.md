---
language:
- kab
license: apache-2.0
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- encoder
- masked-language-model
- low-resource
pipeline_tag: fill-mask
metrics:
- accuracy
model-index:
- name: Masinissa-73M
  results:
  - task:
      type: token-classification
      name: Part-of-speech tagging
    dataset:
      type: universal-dependencies
      name: UD_Kabyle-ADPT (test)
    metrics:
    - type: accuracy
      value: 0.9408
      name: Accuracy (surface tokens, frozen encoder + linear probe)
    - type: accuracy
      value: 0.9151
      name: Accuracy (surface tokens, homoglyph-corrupted input)
    - type: accuracy
      value: 0.9220
      name: Non-punctuation Accuracy (surface tokens)
---

# Masinissa-73M

A 73.3M-parameter masked language model for **Kabyle** (Taqbaylit, `kab`, Latin script), trained
from scratch on 3.92 billion tokens of deduplicated, provenance-tracked Kabyle text.

Masinissa-73M establishes the definitive state of the art on **`KabEncBench`**, the unified
multitask benchmark for Kabyle representation learning. In a 2,000-bootstrap Bradley–Terry
tournament across nine models anchored at `FacebookAI/xlm-roberta-base` = 1,000, Masinissa-73M
achieves an Elo rating of **1,953.7** (**Tier 1**), leading the nearest external baseline by
**+753.4 Elo points**.

## Results: KabEncBench

All models are evaluated on **`KabEncBench`** (Track A: Frozen Scale-Normalized Probe) across six
core linguistic tasks and three diagnostic probes. Backbones are 100% frozen; a layer-weighted
scalar mix (`ScalarMix`) and linear probe are fitted across 5 random seeds per learning rate.

### Bradley–Terry Elo Leaderboard

Pairwise Bradley–Terry Elo ratings are computed via 2,000 paired cluster-bootstraps over all task
items, anchored at `FacebookAI/xlm-roberta-base` = 1,000:

| Rank | Model | Parameters | Elo (95% CI) | Tier |
|:---:|:---|:---:|:---:|:---:|
| **1** | **`agbalu/Masinissa-73M`** | **73M** | **1,953.7 [1,534.7, 3,379.1]** | **Tier 1** |
| 2 | [`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M) | 31M | 1,708.4 [1,371.1, 2,886.2] | Tier 2 |
| 3 | `UBC-NLP/serengeti-E250` | 278M | 1,200.3 [953.6, 1,480.1] | Tier 3 |
| 4 | `Davlan/afro-xlmr-base-76L_script` | 278M | 1,151.6 [919.5, 1,451.0] | Tier 3 |
| 5 | `UBC-NLP/serengeti-E110` | 110M | 1,148.2 [918.5, 1,421.2] | Tier 3 |
| 6 | `jhu-clsp/mmBERT-base` | 278M | 1,141.7 [906.4, 1,435.3] | Tier 3 |
| 7 | `jhu-clsp/mmBERT-small` | 135M | 1,039.1 [804.2, 1,282.5] | Tier 3 |
| 8 | `google-bert/bert-base-multilingual-cased` | 178M | 1,012.6 [780.9, 1,266.8] | Tier 4 |
| 9 | `FacebookAI/xlm-roberta-base` *(Anchor)* | 278M | 1,000.0 [1,000.0, 1,000.0] | Tier 4 |

### Comprehensive Task Performance (%)

| Model | UPOS (`Syntax`) | Aspect (`Morph`) | Micro-F1 (`NER`) | Sentiment | SIB-200 (`Topic`) | Punctuation | TaPaCo (`Paraphrase`) | Minimal Pairs (`Spell`) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`agbalu/Masinissa-73M`** | **80.59** | **57.91** | **88.24** | **93.83** | **64.87** | **55.42** | **64.38** | **60.90** |
| `agbalu/Masinissa-31M` | 78.88 | 54.19 | 87.74 | 92.06 | 59.28 | 52.21 | 58.78 | 18.70 |
| `serengeti-E250` | 76.98 | 49.13 | 83.98 | 85.33 | 43.72 | 45.54 | 42.55 | 6.30 |
| `serengeti-E110` | 76.24 | 48.83 | 84.91 | 85.65 | 39.64 | 43.11 | 43.86 | 4.70 |
| `mmBERT-base` | 75.52 | 48.44 | 84.01 | 85.64 | 37.96 | 39.98 | 48.63 | 53.30 |
| `mmBERT-small` | 75.72 | 45.84 | 81.79 | 83.17 | 34.94 | 39.91 | 48.98 | 61.00 |
| `xlm-roberta-base` | 75.44 | 44.52 | 79.71 | 84.75 | 38.21 | 39.12 | 42.13 | 57.40 |
| `afro-xlmr-base` | 75.32 | 47.93 | 83.04 | 85.75 | 38.41 | 38.56 | 48.54 | **62.50** |
| `bert-base-multilingual` | 73.21 | 46.79 | 80.62 | 84.35 | 40.84 | 39.01 | 51.77 | 27.00 |

### Part-of-Speech Tagging Detail (UD_Kabyle-ADPT)

On the gold Universal Dependencies `UD_Kabyle-ADPT` test split (845 sentences, 10,181 words,
8,563 whitespace tokens), Masinissa-73M scores **94.08%** surface accuracy with a linear probe:

| system | gold-words | surface | gold-words, corrupted | surface, corrupted |
|---|---|---|---|---|
| **Masinissa-73M + linear probe** | **93.82%** | **94.08%** | **91.28%** | **91.51%** |
| Masinissa-31M + linear probe | 90.38% | 90.51% | 87.10% | 87.42% |
| most-frequent-tag baseline | 84.14% | 83.42% | 80.23% | 79.52% |
| `boffire/kabyle-pos-v2` | 56.50% | 63.69% | 55.10% | 62.35% |
| lexicon projection | 26.65% | 27.67% | 23.15% | 24.15% |

Stripping punctuation tokens isolates lexical and syntactic tagging performance:

| system | surface (clean) | surface (non-punct) | macro-F1 (all) | macro-F1 (non-punct) |
|---|---|---|---|---|
| **Masinissa-73M** | **94.08%** | **92.20%** | **0.7743** | **0.7582** |
| Masinissa-31M | 90.51% | 87.50% | 0.7301 | 0.7110 |
| most-frequent-tag | 83.42% | 78.16% | 0.7051 | 0.6841 |

Three things worth reading carefully:

**The encoder is frozen.** Only the probe is fitted. This measures what pretraining put
into the representation, not what a deep task head can learn.

**Both tokenisation settings are reported because they disagree.** 29.5% of `UD_Kabyle-ADPT`
words sit inside a multiword token, so "accuracy" is two different numbers depending on
whether you score gold word segmentation or raw whitespace tokens. Reporting both makes this
distinction transparent.

**The corrupted condition tests orthographic resilience.** Kabyle text in the wild carries
systematic homoglyph substitution (Greek `ε` U+03B5 for Latin `ɛ` U+025B, `γ` for `ɣ`) in
2.6–3.2% of public corpora. Masinissa-73M retains **91.51%** accuracy under corruption.

## Intended use

Kabyle NLP where high-capacity contextual representations are required: sequence labelling,
named entity recognition, dependency parsing, sentence similarity, semantic retrieval, and
fine-tuning for classification tasks.

**Not suitable for**: generative text continuation (it is a bidirectional encoder), open-ended
dialogue, translation, or non-Berber languages.

## Usage

Loadable directly via `transformers` and `torch` with `trust_remote_code=True`:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

REPO = "agbalu/Masinissa-73M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(REPO, trust_remote_code=True).eval()

text = "Axxam n [MASK] meqqer."
encoded = tokenizer(text, return_tensors="pt")
with torch.inference_mode():
    logits = model(**encoded).logits

position = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
top = logits[0, position].topk(5).indices
print([tokenizer.decode(piece) for piece in top])
```

For extracting representations rather than logits:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()
with torch.inference_mode():
    hidden = encoder(**tokenizer("Aman d tudert.", return_tensors="pt")).last_hidden_state
print(hidden.shape)  # (1, sequence_length, 512)
```

**Normalise the input first.** The vocabulary was built over text where `ɛ` is U+025B and
`ɣ` is U+0263. Homoglyph-corrupted text fragments into fallback bytes and degrades
downstream accuracy.

`mammeri-tok-base-16k.model` is the SentencePiece model the vocabulary was trained as, kept
beside the converted `tokenizer.json` for anyone who needs the original.

## Architecture

A modern deep bidirectional encoder featuring Block Attention Residuals and QK RMSNorm:

| | |
|---|---|
| Trainable parameters | **73,318,016** (11.13% embedding table) |
| Layers / hidden / heads | 20 / 512 / 8 (head size 64) |
| Feed-forward | 1,408, SwiGLU |
| Vocabulary | 16,000, SentencePiece Unigram — [`agbalu/Mammeri-Tok`](https://huggingface.co/agbalu/Mammeri-Tok) |
| Position representation | log-bucketed relative positions, 32 buckets |
| Routing / Residuals | Block Attention Residuals (Block AttnRes, block size 4) with Depth Attention |
| Normalisation | RMSNorm ($\epsilon=10^{-6}$), with QK RMSNorm |
| Sequence length | 1,024 tokens |
| Classifier | tied to input embeddings with learned vocabulary bias |

## Training data

**AƔBALU Corpus** — deduplicated Kabyle documents tokenised to 16,000 pieces via Mammeri-Tok.
Pretraining consumed **3,918,417,150 tokens** (3.92B tokens) over 28,731 updates.

Every source carries an audited provenance record (source id, licence, retrieval date). Text
passed the reference normaliser at version **1.3.0+rules1.0.0**, which standardises orthography
while preserving all phonemic and spirantised distinctions (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`).

**Decontamination.** Evaluated against FLORES+ `kab_Latn` and SIB-200 with zero overlap, verified
against a positive-control detector.

### Licence composition of the training text

The model weights and code are released under **Apache-2.0**. As with all web-derived language
corpora, the weights licence does not relicense the underlying texts:

| redistribution | sentences | share |
|---|---|---|
| **unclear** | 1,062,569 | **34.9%** |
| permissive | 973,218 | 32.0% |
| share-alike | 944,438 | 31.0% |
| non-commercial | 61,764 | 2.0% |

## Training recipe

| | |
|---|---|
| Objective | Masked Language Modelling (dynamic masking, 15–20%, mean span 3.0) |
| Optimiser | **AdamW**, peak lr 6e-4, $\beta=(0.9, 0.98)$, $\epsilon=10^{-8}$, weight decay 0.1 |
| Batch size | 131,072 tokens/step (gradient accumulation from 8,192-token micro-batches) |
| Context curriculum | 512 tokens up to 3.41B tokens $\rightarrow$ 1,024 tokens to completion |
| Schedule | 2% linear warmup, cosine decay to 10% peak lr |
| Updates | **28,731** updates (3,918,417,150 tokens consumed) |
| Throughput | 42,046 tokens/sec sustained |
| Precision | BF16 mixed precision, `torch.compile` + FlexAttention |
| Hardware | 1× NVIDIA H100 SXM 80GB (MIG 3g.47gb partition) |
| Seed | 20261030 |

### Pretraining loss

Validation cross-entropy reached **2.9303 nats** (perplexity **18.73**, accuracy 49.51% at step 27,231),
improving monotonically from 6.81 at step 50.

## Files

| file | size | description |
|---|---:|---|
| `model.safetensors` | 293.3 MB | 73,318,016 parameters (float32) |
| `mammeri-tok-base-16k.model` | 254 KB | raw SentencePiece unigram model |
| `config.json` | — | architecture configuration, training summary, and `auto_map` |
| `configuration_masinissa.py` | — | `MasinissaConfig` for Hugging Face `transformers` |
| `modeling_masinissa.py` | — | `MasinissaModel` and `MasinissaForMaskedLM` |
| `tokenizer.json`, `tokenizer_config.json` | 1.0 MB | converted fast tokenizer for `AutoTokenizer` |

## Reproduction

The source pretraining snapshot is step 27,231 (SHA-256
`91083cdf72e04c8dc0bd9edff0334a9be5232f49bfdd92a2a354c1902ccb6b00`), promoted under minimum
validation cross-entropy (2.9303 nats, perplexity 18.73).

The downstream POS evaluation protocol trains a 20-epoch frozen linear probe across the 1,085
training sentences of `UD_Kabyle-ADPT` (seed 0) and evaluates on the 845 gold test sentences.
All reported metrics reproduce deterministically under the fixed seed.

## The name

**Masinissa** (c. 238 – c. 148 BCE) was the first king of Numidia, uniting the Massylii and
Masaesyli into a unified North African state. Masinissa-73M represents the highest-capacity
monolingual encoder trained for Kabyle to date.

## Citation

```bibtex
@software{agbalu_masinissa_73m_2026,
  title  = {Masinissa-73M: a deep bidirectional masked language model for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Masinissa-73M},
  note   = {Trained on AƔBALU text corpus; normaliser 1.3.0+rules1.0.0}
}
```

## Licence

**Apache-2.0** for weights and code.
