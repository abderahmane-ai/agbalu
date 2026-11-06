---
language:
- kab
- en
- fr
license: apache-2.0
base_model: intfloat/multilingual-e5-base
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- sentence-transformers
- sentence-similarity
- feature-extraction
- matryoshka
- low-resource
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- recall@1
- recall@5
model-index:
- name: SiMohand-278M
  results:
  - task:
      type: sentence-similarity
      name: Bitext retrieval
    dataset:
      type: agbalu_embed_dev
      name: AƔBALU held-out cluster-sealed bitext pairs (500 pairs, 18 sources)
    metrics:
    - type: recall@1
      value: 97.0
      name: Recall@1 (500-way)
    - type: recall@5
      value: 99.8
      name: Recall@5 (500-way)
    - type: mrr
      value: 0.9833
      name: MRR (500-way)
---

# SiMohand-278M

A 278M-parameter sentence embedding model for **Kabyle** (Taqbaylit, `kab_Latn`), fine-tuned from [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) on 511,549 decontaminated parallel sentence pairs using Matryoshka Representation Learning (MRL).

SiMohand-278M provides dense semantic representations for cross-lingual and monolingual Kabyle text retrieval. Evaluated on a 500-way bitext retrieval benchmark against hard in-domain distractors, it achieves **Recall@1 = 97.0%** (an improvement of +33.2 percentage points over the base `multilingual-e5-base` backbone) and an MRR of **0.9833**. Its 64-dimensional Matryoshka slice retains **97.0% Recall@1**, enabling a 12× reduction in vector index storage with minimal degradation in retrieval precision.

## Evaluation Results

### 500-Way Bitext Retrieval

Evaluated on 500 held-out sentence pairs across 18 sources against 499 in-domain distractors:

| Model | Recall@1 | Recall@5 | MRR | Mean Positive Cosine | Mean Negative Cosine | Cosine Margin | Isotropy Mean |
|---|---|---|---|---|---|---|---|
| **SiMohand-278M** | **97.0%** | **99.8%** | **0.9833** | 0.7913 | 0.0013 | **+0.790** | **0.0042** |
| `multilingual-e5-base` (backbone) | 63.8% | 77.6% | 0.7063 | 0.8547 | 0.7680 | +0.087 | 0.8040 |
| `LaBSE` (Google baseline) | 22.2% | 30.8% | 0.2787 | 0.2730 | 0.0529 | +0.220 | 0.1754 |

- **Anisotropy Resolution:** Unmodified `multilingual-e5-base` exhibits representation collapse on Kabyle (mean cosine of 0.8040 on random unaligned pairs) because its base XLM-RoBERTa tokenizer maps key Kabyle characters (`Ɛ`, `Ɣ`, `Ǧ`, `Ẓ`, `ẓ`) to `<unk>`. SiMohand restores geometric isotropy (mean cosine 0.0042) through targeted vocabulary expansion and contrastive tuning.
- **Distractor Hardness:** Distractors comprise in-domain Kabyle sentences and their parallel translations, testing semantic discrimination within the target domain.

### Matryoshka Dimension Sweep

SiMohand-278M was trained with nested prefix losses across 5 vector dimensions:

| Dimensions | Recall@1 | Recall@5 | MRR | Cosine Margin |
|---|---|---|---|---|
| **768** (Full vector) | 97.0% | 99.8% | 0.9833 | +0.790 |
| **512** | 97.0% | 99.8% | 0.9830 | +0.794 |
| **256** | 97.2% | 99.8% | 0.9838 | +0.802 |
| **128** | 97.2% | 99.8% | 0.9836 | +0.814 |
| **64** (12× compressed) | **97.0%** | **99.6%** | **0.9819** | **+0.826** |

Truncating embeddings to 64 dimensions preserves 97.0% Recall@1 while providing a 12× reduction in vector index memory overhead.

## Intended Use

- **Semantic Search & RAG:** Vector retrieval and passage ranking for Kabyle, English, and French queries.
- **Cross-Lingual Matching:** Aligning Kabyle text with English or French translations without intermediate translation steps.
- **Sentence Clustering:** Unsupervised grouping and deduplication of Kabyle text corpora.
- **Out of Scope:** Autoregressive text generation; classification of languages outside the training distribution.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("agbalu/SiMohand-278M")

queries = [
    "Yebɣa ad yeǧǧ axxam.",
    "Taqbaylit d tutlayt tayemmat nneɣ.",
]

passages = [
    "Ira ad yeǧǧ axxam.",                   # Kabyle paraphrase
    "He wants to leave the house.",          # English translation
    "Tutlayt nneɣ n tyemmat d Taqbaylit.",   # Kabyle paraphrase
    "Yuli s adrar deg tegrest.",             # Unrelated distractor
]

q_emb = model.encode(queries, normalize_embeddings=True)
p_emb = model.encode(passages, normalize_embeddings=True)

# Similarity matrix (queries x passages)
print(q_emb @ p_emb.T)
```

### Matryoshka Slicing (64 Dimensions)

```python
import torch

def slice_and_normalize(embeddings, dims=64):
    tensor = torch.tensor(embeddings[:, :dims])
    return torch.nn.functional.normalize(tensor, p=2, dim=1).numpy()

q64 = slice_and_normalize(model.encode(queries))
p64 = slice_and_normalize(model.encode(passages))
print(q64 @ p64.T)
```

When slicing Matryoshka representations, slice before L2 normalization to ensure valid unit vectors.

## Architecture

| Parameter | Specification |
|---|---|
| **Base Model** | [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) |
| **Total Parameters** | 278,043,648 |
| **Layers / Hidden / Heads** | 12 layers / 768 hidden dimension / 12 attention heads |
| **Pooling Mode** | Mean pooling followed by L2 normalization |
| **Vocabulary Size** | 250,007 tokens (expanded from 250,002) |
| **Maximum Context** | 128 tokens |

### Vocabulary Expansion

Five missing Kabyle characters were added to the base XLM-RoBERTa SentencePiece vocabulary, initialized with donor embeddings from phonetically related tokens:

| Character | Unicode | Description | Donor Token Initializer |
|---|---|---|---|
| `Ɛ` | U+0190 | Capital pharyngeal fricative | `ɛ` |
| `Ɣ` | U+0194 | Capital velar fricative | `ɣ` |
| `Ǧ` | U+01E6 | Capital affricate | `ǧ` |
| `Ẓ` | U+1E92 | Capital emphatic sibilant | `ẓ` |
| `ẓ` | U+1E93 | Lowercase emphatic sibilant | Mean of `z` and `ṣ` |

This expansion prevents key consonant distinctions from mapping to `<unk>`.

## Training Recipe

| Parameter | Specification |
|---|---|
| **Training Pairs** | 511,549 bitext pairs (291,318 kab–eng, 220,231 kab–fra) across 18 sources |
| **Validation Pairs** | 500 cluster-sealed pairs |
| **Loss Function** | `MatryoshkaLoss` wrapping `MultipleNegativesRankingLoss` (scale 20.0, slices: [768, 512, 256, 128, 64]) |
| **Optimizer** | AdamW fused (lr 2e-5, weight decay 0.01) |
| **Batching** | `ClusterAwareBatchSampler` (batch size 64; guarantees at most 1 item per semantic cluster per batch) |
| **Schedule** | 10% linear warmup, cosine decay over 3 epochs (23,979 steps) |
| **Precision** | bfloat16 mixed precision |
| **Convergence** | Final training loss 0.1194; final validation isotropy mean cosine 0.0042 |

## Decontamination

Validation pairs were sampled using cluster-based partitioning (`split_clusters`, seed 42) prior to training. No semantic cluster or sentence pair spans across train and validation splits. Training sets were additionally decontaminated against FLORES+ and SIB-200 benchmarks via NFKD character fingerprinting.

## Limitations

- **Bitext Alignment Focus:** Trained primarily on Kabyle-to-English and Kabyle-to-French parallel sentence pairs; Kabyle-to-Kabyle monolingual semantic similarity is supported via cross-lingual transfer rather than direct monolingual paraphrase supervision.
- **Sentence-Level Focus:** Maximum sequence length is 128 tokens; long paragraphs or multi-page documents should be chunked into sentences before embedding.
- **Dialect Scope:** Optimized for standard Latin-script Kabyle orthography.

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | 1.04 GB | Model weights (278M parameters, 250,007-token vocabulary) |
| `tokenizer.json` | 16.3 MB | Fast tokenizer with 5 added Kabyle character tokens |
| `config.json` | < 2 KB | Model configuration and architecture specifications |
| `sentence_bert_config.json`, `modules.json` | < 1 KB | Sentence-transformers pipeline configuration |
| `1_Pooling/config.json`, `2_Normalize/config.json` | < 1 KB | Mean pooling and L2 normalization layer configurations |

## Reproduction

Fine-tuned from `intfloat/multilingual-e5-base` across 511,549 pairs with Matryoshka Multiple Negatives Ranking Loss (scale 20.0, lr 2e-5, batch size 64). Evaluated on the 500-pair held-out bitext retrieval set against `multilingual-e5-base` and LaBSE. All reported metrics reproduce deterministically from the published checkpoint.

## The name

**Si Mohand ou Mhand** (c. 1848–1905) is the greatest poet of the Kabyle oral tradition —
*ameddaḥ* by vocation, carrying the whole grammar of feeling in verse. He composed in *asefru*,
the short lyric form, and never wrote down a word. Everything that survives was recovered from
memory by those who had heard him, recalled line by line.

A model that retrieves meaning by fidelity of embedding — that must hold the semantic content of
a sentence in a fixed-dimensional vector and recover it against 499 alternatives — does, in its
technical way, what Si Mohand's listeners did: carry a representation that survives at distance.

His verse circulates without a canonical text. Its persistence is the persistence of
representation. The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_simohand_2026,
  title  = {SiMohand-278M: Kabyle Sentence Transformer with Matryoshka Representation Learning},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/SiMohand-278M},
  note   = {Fine-tuned from intfloat/multilingual-e5-base on 511,549 bitext pairs}
}
```

## Licence

- **Weights & Code:** Apache-2.0.
- **Training Data:** Sourced from public parallel corpora maintaining respective upstream licenses.
