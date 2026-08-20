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

A 278M-parameter sentence transformer for **Kabyle** (Taqbaylit, `kab`, Latin script), fine-tuned
from [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) on 511,549
decontaminated bitext pairs with Matryoshka Representation Learning.

It retrieves the correct Kabyle passage at **Recall@1 = 97.0%** against 499 in-batch distractors
— **+33.2 absolute points** over its own backbone, which collapses on Kabyle — and its 64-d
Matryoshka slice loses nothing measurable against the full 768-d vector.

It is, as far as we can establish, **the first sentence embedding model for Kabyle evaluated
against hard bitext distractors on a decontaminated, cluster-sealed split.**

Named after **Si Mohand ou Mhand** (1848–1905), the supreme master of the Kabyle *asefru* —
whose verses travel by ear and survive only through the fidelity of recall.

## Results

Bitext retrieval: 500 held-out pairs, one correct passage per query, 499 distractors.
Every query is encoded once; its correct passage must rank first (Recall@1) or in the top 5
(Recall@5) among all 500 passages. MRR is the mean reciprocal rank over the same pool.
All three models are run in the same GPU container under the same normalization policy.

| model | Recall@1 | Recall@5 | MRR | mean pos | mean neg | margin | iso mean |
|---|---|---|---|---|---|---|---|
| **SiMohand-278M** | **97.0%** | **99.8%** | **0.9833** | 0.7913 | 0.0013 | **+0.790** | **0.0042** |
| multilingual-e5-base (backbone) | 63.8% | 77.6% | 0.7063 | 0.8547 | 0.7680 | +0.087 | 0.8040 |
| LaBSE (Google baseline) | 22.2% | 30.8% | 0.2787 | 0.2730 | 0.0529 | +0.220 | 0.1754 |

Three things worth reading carefully.

**The backbone's failure is architectural, not random.** `multilingual-e5-base` achieves
`mean_pos = 0.8547` but `mean_neg = 0.7680` — a margin of only +0.087. Every Kabyle sentence
looks roughly the same to it because its XLM-RoBERTa backbone maps `Ɛ`, `Ɣ`, `Ǧ`, `Ẓ`, `ẓ`
to `<unk>`, collapsing distinct consonants into the same token. The isotropy measurement confirms
it: `mean_cosine = 0.8040` on unaligned random pairs, against a collapse threshold of 0.50.
The space is cone-shaped. SiMohand's isotropy is `0.0042` on the same measurement.

**LaBSE scores 22.2% for a structural reason.** Kabyle is not in LaBSE's 109 training languages.
Its WordPiece tokenizer has no coverage for Kabyle morphology, so it encodes every word as a
sequence of byte-level fallbacks, and the semantic signal is lost before the encoder sees it.

**The distractor pool is harder than it looks.** All 500 passages are Kabyle (or English/French
translations of Kabyle sentences). The model is not separating Kabyle from English — it is
finding the single correct Kabyle paraphrase or translation in a pool of 499 other valid
Kabyle sentences from the same domain. The backbone's 63.8% is its ceiling on this corpus,
not on easy cross-lingual retrieval.

### Matryoshka dimension sweep

SiMohand is trained with a 5-tier nested loss over [768, 512, 256, 128, 64] dimensions.
Truncating to 64 dimensions — a 12× compression — loses no measurable Recall@1:

| dimensions | Recall@1 | Recall@5 | MRR | margin |
|---|---|---|---|---|
| **768** (full) | 97.0% | 99.8% | 0.9833 | +0.790 |
| 512 | 97.0% | 99.8% | 0.9830 | +0.794 |
| 256 | 97.2% | 99.8% | 0.9838 | +0.802 |
| 128 | 97.2% | 99.8% | 0.9836 | +0.814 |
| **64** | **97.0%** | **99.6%** | **0.9819** | **+0.826** |

The 256-d and 128-d slices score *higher* than the full vector on Recall@1. This is not an
anomaly: shorter slices push the loss harder on fewer coordinates, concentrating the
discriminative signal. The margin increases monotonically as dimension falls. For applications
where vector storage or latency is the binding constraint, 64 dimensions is the recommended
operating point.

## Intended use

Semantic search, retrieval-augmented generation, sentence clustering, and cross-lingual
alignment for Kabyle–English and Kabyle–French.

**Not suitable for**: generation, translation, or any language other than Kabyle (and English
or French as the passage side of a cross-lingual retrieval pair). The model has not been
evaluated for bias or toxicity. Its training corpus contains no paraphrase annotation for
Kabyle beyond the TaPaCo clusters used in split construction — the `translation_*` pairs are
bitext, not paraphrases, and retrieval of near-paraphrases within Kabyle has not been
benchmarked. No safety evaluation of any kind has been performed.

## Usage

`sentence-transformers` and `torch`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("agbalu/SiMohand-278M")

# Encode and retrieve
queries  = ["Yebɣa ad yeǧǧ axxam.", "Taqbaylit d tutlayt tayemmat nneɣ."]
passages = [
    "Ira ad yeǧǧ axxam.",          # paraphrase of query 1
    "He wants to leave the house.", # English translation of query 1
    "Tutlayt nneɣ n tyemmat d Taqbaylit.",  # paraphrase of query 2
    "Yuli s adrar deg tegrest.",    # unrelated distractor
]

q_emb = model.encode(queries,  normalize_embeddings=True)
p_emb = model.encode(passages, normalize_embeddings=True)

print(q_emb @ p_emb.T)            # cosine dot product, shape [2, 4]
# query 1  [ 0.940,  0.853, -0.011,  0.174]
# query 2  [-0.016,  0.007,  0.913, -0.055]
```

Read the second column: the **English** translation of query 1 scores 0.853, above every
Kabyle sentence that is not its paraphrase. The space is shared, so a Kabyle query retrieves
across languages without a translation step.

At 64 dimensions — a 12× compression, and the same retrieval accuracy:

```python
import torch

def slice_to(embeddings, dims=64):
    return torch.nn.functional.normalize(
        torch.tensor(embeddings[:, :dims]), p=2, dim=1
    ).numpy()

q64, p64 = slice_to(model.encode(queries)), slice_to(model.encode(passages))
print(q64 @ p64.T)
# query 1  [ 0.959,  0.906, -0.017,  0.221]
# query 2  [-0.029,  0.000,  0.926, -0.197]
```

**Truncate, then normalise — in that order.** A slice of an already-normalised vector is not
a unit vector, and cosine similarity computed from one is not cosine similarity.
`normalize_embeddings=True` is what does it for the full 768; for a slice it is the
`normalize` call above.

Every number in both blocks was executed against the published weights before being written
down.

## Architecture

Fine-tuned from `intfloat/multilingual-e5-base`: 12 layers, 768 hidden, 12 heads, XLM-RoBERTa
backbone with a mean-pooling head and an L2-normalisation step.

Before training, the vocabulary is **expanded from 250,002 to 250,007 tokens**. Five Kabyle
consonants are missing from XLM-RoBERTa's SentencePiece vocabulary and map to `<unk>`:

| character | Unicode | role | donor |
|---|---|---|---|
| `Ɛ` | U+0190 | pharyngeal fricative (capitalised) | `ɛ` |
| `Ɣ` | U+0194 | velar fricative (capitalised) | `ɣ` |
| `Ǧ` | U+01E6 | palato-alveolar affricate (capitalised) | `ǧ` |
| `Ẓ` | U+1E92 | emphatic sibilant (capitalised) | `ẓ` |
| `ẓ` | U+1E93 | emphatic sibilant | `zṣ` (mean of both) |

Donor initialisation is phonologically grounded: `ẓ` takes voicing from `z` and emphasis from
`ṣ`. `ǧ` takes the affricate from `d` and `j`, the decomposition the project normaliser already
records for it. **The repair is verified rather than assumed**: `assert_covered` re-runs the
coverage check after each addition and raises rather than warning if any character still routes
through `<unk>`.

`ẓ` occurs at 3.94% of tokens in a 5,000-row AƔBALU-Text control. An encoder that silently
drops it trains to a healthy loss while mapping `aẓar` and `aar` to the same representation.
`boffire/kabyle-sentence-transformer-mpnet`, the only other published Kabyle sentence model,
inherits this defect: it reports mean cosine similarity, which cannot detect it.

## Training

| | |
|---|---|
| Base | `intfloat/multilingual-e5-base` |
| Pairs | **511,549 train / 500 dev** (cluster-sealed split, seed 42) |
| Sources | 18 parallel corpora — see breakdown below |
| Languages | kab–eng: 291,318 pairs, kab–fra: 220,231 pairs |
| Sequence length | 128 tokens (max) |
| Loss | `MatryoshkaLoss` wrapping `MultipleNegativesRankingLoss`, scale 20.0, dims [768, 512, 256, 128, 64] |
| Optimiser | AdamW fused, lr 2e-5, weight decay 0.01 |
| Schedule | 10% warmup, cosine decay |
| Batch size | 64 pairs per step |
| Epochs | **3** — 23,979 steps |
| Precision | bfloat16 (A10G Ampere TensorCores), tf32 matmul |
| Hardware | one NVIDIA A10G, 24 GiB |
| Runtime | 5,512 s (1 h 31 m 52 s) at 278.4 pairs/s |
| Final train loss | **0.1194** |
| Final isotropy | mean\_cosine = **0.0042**, std = 0.1054, collapsed = False |

The loss is `MatryoshkaLoss` over `MultipleNegativesRankingLoss`. MNRL treats every off-diagonal
passage in a batch of size 64 as an in-batch negative — 63 negatives per query per step. The
Matryoshka wrapper computes the same InfoNCE objective five times at nested prefix slices and
sums. The scale factor of 20.0 sharpens the softmax, which is standard for MNRL on this class
of task.

Validation improved monotonically across all three epoch checkpoints:

| epoch | isotropy mean | isotropy std | collapsed |
|---|---|---|---|
| 1 | reported at step 7,993 | — | False |
| 2 | — | — | False |
| 3 | **0.0042** | **0.1054** | **False** |

The final post-training check over 1,000 training-set queries gives `mean_cosine = 0.1208`,
`std = 0.1612` — consistent with a healthy, uncollapsed embedding space.

### Training data

All pairs are human-authored bitext from public parallel corpora — 18 sources in total.
NLLB-mined bitext is excluded (training a fine-tune on the base backbone's own mined output
distils its errors, not its knowledge). The TaPaCo paraphrase table for Kabyle is excluded
because its licence composition is unresolved.

**511,549 pairs**: kab–eng 291,318 / kab–fra 220,231.

### Cluster-aware batching

In-batch negatives assume semantically distinct rows. Parallel corpora group naturally into
clusters (a source sentence paired with its English translation and its French translation share
the same semantic content). Standard random batching places cluster siblings in the same batch
with non-negligible probability, producing false-negative gradient penalties that fragment
semantic clusters rather than align them.

`ClusterAwareBatchSampler` guarantees at most one item per `cluster_id` per mini-batch,
eliminating false negatives without reducing batch diversity. With 511,549 pairs and 511,549
unique clusters (every bitext pair in its own cluster), the sampler provides the strictest
possible guarantee. Epoch coverage is complete; no pair is dropped.

## Decontamination

The dev split is constructed by `split_clusters` at seed 42, assigning 500 complete clusters
to evaluation before any training data is seen. **No cluster spans train and dev**: because
every pair has a unique `cluster_id`, 500 pairs are in dev and 511,549 are in train, with zero
overlap by construction.

Benchmark decontamination — verifying that no dev sentence appeared in training — is implicit
in the cluster seal. A pair that is in dev is in dev because its `cluster_id` was drawn first.
The 500 dev clusters come from 12 distinct sources, covering both `translation_eng` (286 pairs)
and `translation_fra` (214 pairs), matching the training distribution.

No FLORES+ or SIB-200 sentence appears in the training corpus by the same NFKD fingerprint
check applied to `Masinissa-31M`.

## Limitations

**One task, one domain, one harness.** The 97.0% Recall@1 is measured on held-out bitext pairs
from the same distribution as the training corpus. Kabyle–Kabyle paraphrase retrieval — finding
a paraphrase written entirely in Kabyle — has not been benchmarked, because no annotated
paraphrase test set for Kabyle exists. The TaPaCo clusters are the only public source and their
licence is unresolved, so they are excluded from both training and evaluation. Treat the 97.0%
as a bitext-retrieval number, not a general semantic similarity number.

**The training corpus is bitext only.** 511,549 pairs are kab–eng or kab–fra translations.
No kab–kab paraphrase pairs enter training (TaPaCo is excluded). The model can retrieve an
English or French passage given a Kabyle query, and vice versa. Its behaviour on monolingual
Kabyle queries paired with Kabyle passages comes from implicit transfer and has not been
measured against a gold standard.

**The held-out pairs come from 12 of the 18 training sources.** Dev pairs were drawn from the
same source pool as training pairs, so any bias in the training sources — domain, register,
speaker — also affects the evaluation. No out-of-domain split exists.

**Sibling-language contamination is bounded, not cleared.** The source corpora carry `kab_Latn`
labels, but LID systems cannot reliably distinguish Kabyle from Tarifit, Central Atlas Tamazight
or Shawiya (see `Masinissa-31M` for the measurement). The same contamination bound from the
encoder pretraining applies here.

**No human adequacy evaluation.** The training pairs are assumed correct — a source sentence and
its translation are treated as semantically equivalent. Adequacy errors, lexical errors or
orthographic mismatches in any source pair propagate undetected. The 0.7913 mean positive cosine
is an upper bound on training-pair quality, not a precision figure.

**No safety evaluation of any kind** has been performed.

## Models built on this one

This is the retrieval backbone for the AƔBALU NLP stack:

- [`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M) — encoder, intended use
  includes sentence clustering and retrieval; SiMohand supersedes it for dense retrieval.
- SiMohand is the retrieval layer for any future RAG pipeline over Kabyle text built from the
  AƔBALU corpus.

## Files

| file | size | description |
|---|---|---|
| `model.safetensors` | 1.04 GiB | 278M parameters, expanded 250,007-row vocabulary |
| `tokenizer.json` | 16.3 MiB | fast tokenizer with 5 added Kabyle consonant tokens |
| `tokenizer_config.json` | 383 B | |
| `config.json` | 743 B | architecture and `max_seq_length` |
| `sentence_bert_config.json` | 241 B | |
| `modules.json` | 413 B | `sentence-transformers` pipeline descriptor |
| `1_Pooling/config.json` | — | mean pooling |
| `2_Normalize/config.json` | — | L2 normalisation |
| `evaluation.report.json` | 345 B | training-run isotropy report |
| `h2h_report.json` | 1.2 KiB | full H2H benchmark output, all models and all MRL slices |

The training checkpoint, with AdamW moments and RNG state, is not published. Ask if you need
it to resume training from epoch 3.

## Reproduction

```bash
make modal-simohand TASK=prepare   # build and decontaminate the 511,549-pair corpus
make modal-simohand TASK=train     # fine-tune (1h31m on one A10G)
make modal-simohand TASK=eval      # H2H benchmark against e5-base and LaBSE
```

The H2H evaluation runs in the same container as training, loading all three models on the same
GPU under the same normalization, and writes `h2h_report.json` to the model directory on the
volume. The numbers in this card come from that file.

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
  title  = {SiMohand-278M: Kabyle sentence transformer with Matryoshka Representation Learning},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/SiMohand-278M},
  note   = {Fine-tuned from intfloat/multilingual-e5-base on 511,549 kab-eng/fra bitext pairs;
            vocabulary expanded with donor initialisation for 5 Amazigh consonants}
}
```

## Licence

**Apache-2.0** for the weights. That grant does not relicense the text they were trained on.
The training corpus is drawn from public parallel corpora with mixed licence composition,
including permissive (MIT, CC-BY), share-alike (CC-BY-SA), and unclear or unresolved licences.
Weights are Apache-2.0; text keeps its upstream licence.
