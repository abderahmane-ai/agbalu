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
- name: Masinissa-31M
  results:
  - task:
      type: token-classification
      name: Part-of-speech tagging
    dataset:
      type: universal-dependencies
      name: UD_Kabyle-ADPT (test)
    metrics:
    - type: accuracy
      value: 0.9051
      name: Accuracy (surface tokens, frozen encoder + linear probe)
    - type: accuracy
      value: 0.8742
      name: Accuracy (surface tokens, homoglyph-corrupted input)
---

# Masinissa-31M

> [!TIP]
> **Higher-capacity successor available:** [`agbalu/Masinissa-73M`](https://huggingface.co/agbalu/Masinissa-73M) scales the encoder to 73.3M parameters and 20 layers, achieving 1,953.7 Elo on the `KabEncBench` benchmark suite (+3.6% Syntax, +8.8% Morphology, +21.2% Topic Classification). Masinissa-31M remains the recommended lightweight encoder for edge, mobile, and latency-critical CPU deployments.

A 31.1M-parameter bidirectional masked language model for **Kabyle** (Taqbaylit, `kab_Latn`), trained from scratch on 3.04M deduplicated and orthographically normalized sentences.

Masinissa-31M produces high-quality contextual representations for Kabyle NLP. Evaluated on the gold Universal Dependencies benchmark (`UD_Kabyle-ADPT`) using a frozen encoder and a single linear probe, it achieves **90.51% part-of-speech accuracy** (0.7301 macro-F1), significantly outperforming baseline systems while maintaining high robustness against character-level orthographic corruptions.

## Evaluation Results

### Part-of-Speech Tagging (`UD_Kabyle-ADPT`)

Evaluated on the gold `UD_Kabyle-ADPT` test split (845 sentences, 10,181 words, 8,563 surface tokens):

| Model / Baseline | Gold Words (Clean) | Surface Tokens (Clean) | Gold Words (Corrupted) | Surface Tokens (Corrupted) |
|---|---|---|---|---|
| **Masinissa-31M + linear probe** | **90.39%** | **90.51%** | **87.10%** | **87.42%** |
| Most-frequent-tag baseline | 84.14% | 83.42% | 80.23% | 79.52% |
| `boffire/kabyle-pos-v2` | 56.50% | 63.69% | 55.10% | 62.35% |
| Lexicon projection | 26.65% | 27.67% | 23.15% | 24.15% |

- **Frozen Probe Evaluation:** The encoder backbone is completely frozen; only a single linear classification layer is trained, isolating the quality of the pretrained representation.
- **Tokenization Alignment:** Both gold word segmentation and raw whitespace surface tokens are reported, reflecting the presence of multiword tokens in 29.5% of `UD_Kabyle-ADPT` words.
- **Orthographic Robustness:** Evaluated under common homoglyph substitutions found in public web corpora (Greek `ε` U+03B5 for Latin `ɛ` U+025B, and `γ` for `ɣ`), retaining 87.42% surface accuracy.

### Multitask Benchmark: KabEncBench

Evaluated on `KabEncBench` (Track A: Frozen Scale-Normalized Probe) across 6 core tasks and 3 diagnostic probes. Bradley–Terry Elo ratings use 2,000 paired bootstraps anchored at `FacebookAI/xlm-roberta-base` = 1,000:

| Rank | Model | Parameters | Elo (95% CI) | Tier |
|:---:|:---|:---:|:---:|:---:|
| 1 | [`agbalu/Masinissa-73M`](https://huggingface.co/agbalu/Masinissa-73M) | 73M | **1,953.7 [1,534.7, 3,379.1]** | **Tier 1** |
| **2** | **`agbalu/Masinissa-31M`** | **31M** | **1,708.4 [1,371.1, 2,886.2]** | **Tier 2** |
| 3 | `UBC-NLP/serengeti-E250` | 278M | 1,200.3 [953.6, 1,480.1] | Tier 3 |
| 4 | `Davlan/afro-xlmr-base-76L_script` | 278M | 1,151.6 [919.5, 1,451.0] | Tier 3 |
| 5 | `UBC-NLP/serengeti-E110` | 110M | 1,148.2 [918.5, 1,421.2] | Tier 3 |
| 6 | `jhu-clsp/mmBERT-base` | 278M | 1,141.7 [906.4, 1,435.3] | Tier 3 |
| 7 | `jhu-clsp/mmBERT-small` | 135M | 1,039.1 [804.2, 1,282.5] | Tier 3 |
| 8 | `google-bert/bert-base-multilingual-cased` | 178M | 1,012.6 [780.9, 1,266.8] | Tier 4 |
| 9 | `FacebookAI/xlm-roberta-base` *(Anchor)* | 278M | 1,000.0 [1,000.0, 1,000.0] | Tier 4 |

#### Task Breakdown (%)

| Model | UPOS (`Syntax`) | Aspect (`Morph`) | Micro-F1 (`NER`) | Sentiment | SIB-200 (`Topic`) | Punctuation | TaPaCo (`Paraphrase`) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Masinissa-73M` | **80.59** | **57.91** | **88.24** | **93.83** | **64.87** | **55.42** | **64.38** |
| **`Masinissa-31M`** | **78.88** | **54.19** | **87.74** | **92.06** | **59.28** | **52.21** | **58.78** |
| `serengeti-E250` | 76.98 | 49.13 | 83.98 | 85.33 | 43.72 | 45.54 | 42.55 |
| `serengeti-E110` | 76.24 | 48.83 | 84.91 | 85.65 | 39.64 | 43.11 | 43.86 |
| `afro-xlmr-base` | 75.32 | 47.93 | 83.04 | 85.75 | 38.41 | 38.56 | 48.54 |
| `mmBERT-base` | 75.52 | 48.44 | 84.01 | 85.64 | 37.96 | 39.98 | 48.63 |
| `xlm-roberta-base` | 75.44 | 44.52 | 79.71 | 84.75 | 38.21 | 39.12 | 42.13 |

## Intended Use

- **Dense Representations:** Token-level and sentence-level feature extraction for sequence classification, named entity recognition, information retrieval, and clustering.
- **Masked Token Prediction:** Direct inference on masked spans (`[MASK]`).
- **Downstream Fine-Tuning:** Pretrained initialization for downstream classifiers, including [`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) (punctuation restoration) and [`Masinissa-Sentiment`](https://huggingface.co/agbalu/Masinissa-Sentiment).
- **Out of Scope:** Autoregressive text generation, open-ended dialog, or machine translation.

## Usage

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

REPO = "agbalu/Masinissa-31M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(REPO, trust_remote_code=True).eval()

text = "Axxam n [MASK] meqqer."
encoded = tokenizer(text, return_tensors="pt")
with torch.inference_mode():
    logits = model(**encoded).logits

position = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0, 0]
top_tokens = logits[0, position].topk(5).indices
print([tokenizer.decode(idx) for idx in top_tokens])
```

Extracting hidden representations:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()
with torch.inference_mode():
    hidden = encoder(**tokenizer("Aman d tudert.", return_tensors="pt")).last_hidden_state
print(hidden.shape)  # torch.Size([1, sequence_length, 384])
```

## Architecture

| Parameter | Value |
|---|---|
| **Architecture** | LTG-BERT / GPT-BERT bidirectional encoder |
| **Trainable Parameters** | 31,123,840 (19.74% in embedding matrix) |
| **Layers / Hidden / Heads** | 12 layers / 384 hidden / 6 attention heads (head dim 64) |
| **Feed-Forward Dimension** | 1,280 (gated GLU) |
| **Vocabulary** | 16,000 SentencePiece Unigram ([`agbalu/Mammeri-Tok`](https://huggingface.co/agbalu/Mammeri-Tok)) |
| **Position Representation** | Log-bucketed relative positions (32 buckets, max window 512) |
| **Training Sequence Length** | 128 tokens |
| **Tied Weights** | Output classification head tied to input embeddings |

## Training Recipe

| Parameter | Value |
|---|---|
| **Objective** | Masked LM (GPT-BERT 15:16 masked-to-causal hybrid) |
| **Masking Strategy** | Inverse schedule (0.30 → 0.15), spans up to 3 tokens (10% random, 10% keep) |
| **Optimizer** | LAMB (peak lr 1.2e-2, $\beta=(0.9, 0.98)$, weight decay 0.1) |
| **Batch Size** | 8,192 sequences/step = 1,048,576 tokens/step (gradient accumulation) |
| **Schedule** | 1.6% linear warmup, cosine decay, 1.6% cooldown |
| **Training Steps** | 4,500 steps (4.72B total tokens seen) |
| **Loss** | Final validation loss 3.1957 (perplexity 24.43) |

Validation loss decreased monotonically across training:

| Step | 250 | 1,000 | 2,000 | 3,000 | 4,000 | 4,500 |
|---|---|---|---|---|---|---|
| **Validation Loss** | 4.3962 | 3.6162 | 3.4177 | 3.2969 | 3.2122 | **3.1957** |

## Training Data & Methodology

- **Corpus:** AƔBALU-Text v1 (3,041,989 deduplicated Kabyle sentences from 42 sources; 70.18M training tokens, 359k validation tokens).
- **Text Normalization:** Reference normalizer v1.3.0 applied across all sources, repairing character substitutions while preserving phonemic Kabyle characters (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`).
- **Decontamination:** Zero overlap verified against FLORES+ `kab_Latn` and SIB-200.

## Limitations

- **Context Window:** Pretrained with a 128-token sequence length and a 512-token relative positional window. For longer documents or tasks requiring broader context, [`agbalu/Masinissa-73M`](https://huggingface.co/agbalu/Masinissa-73M) (1,024 context) is recommended.
- **Morphology:** The subword vocabulary does not explicitly factor annexed-state stem alternations (e.g. `axxam` → `wexxam`), memorizing both surface forms.
- **Corpus Scaling Limits:** Trained for 67 epochs over 70.18M unique tokens, approaching the point of diminishing returns for fixed-corpus pretraining without additional unique source text.
- **Script Support:** Latin-script Kabyle orthography only; does not support Tifinagh or Arabic scripts.

## Models Fine-Tuned from Masinissa-31M

- [`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M): Restores punctuation and capitalization on Kabyle ASR transcripts (two token-classification heads, 0.793 macro-F1 over punctuation marks).
- [`agbalu/Masinissa-Sentiment`](https://huggingface.co/agbalu/Masinissa-Sentiment): Classifies Kabyle sentiment across three polarity classes (negative, neutral, positive) using calibrated multi-sample dropout (90.27% accuracy on held-out test splits).

## Files

| File | Size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 124.5 MB | `dca2a960c06a4bb14a5dd1d33557b8b4537230807a32d0a4debbee7fda44fbad` |
| `mammeri-tok-base-16k.model` | 260 KB | `c8094fccd936d2e8954809bd9cf45331679e9550d2c7bbabe5c38c3bf365dc4e` |
| `config.json` | < 5 KB | Architecture configuration and validation curves |
| `configuration_masinissa.py` | < 10 KB | Configuration class for Hugging Face Transformers |
| `modeling_masinissa.py` | < 30 KB | PyTorch model implementation |
| `tokenizer.json`, `tokenizer_config.json` | 1.0 MB | Fast tokenizer configuration |

## Reproduction

Pretrained from scratch with LAMB (lr 1.2e-2, 4,500 steps). The downstream POS evaluation protocol trains a 20-epoch frozen linear probe across `UD_Kabyle-ADPT` train split (seed 0) and evaluates on test. All metrics reproduce deterministically from the published checkpoint.

## The name

**Masinissa** (r. 202–148 BCE) was the first king of a united Numidia, who brought the eastern
and western tribes into one kingdom. The encoder is the unifying representation of this
project — 42 noisy sources rendered into one shared space — so it carries his name.

His grandson Jugurtha, who fought Rome from 111 to 104 BCE and was never taken in battle, is
reserved for the generative model. The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_masinissa_2026,
  title  = {Masinissa-31M: A Masked Language Model for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Masinissa-31M},
  note   = {Trained on AƔBALU-Text v1; normaliser 1.3.0+rules1.0.0}
}
```

## Licence

- **Weights and Code:** Apache-2.0.
- **Training Corpus Composition:** 34.9% unclear, 32.0% permissive, 31.0% share-alike, 2.0% non-commercial.
