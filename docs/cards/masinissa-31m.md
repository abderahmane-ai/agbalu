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

A 31M-parameter masked language model for **Kabyle** (Taqbaylit, `kab`, Latin script), trained
from scratch on the cleanest Kabyle corpus assembled to date.

Frozen, with only a linear probe fitted on top, it scores **90.51%** on gold Universal
Dependencies part-of-speech tags — against **83.42%** for a most-frequent-tag baseline and
**63.69%** for the previously published Kabyle tagger. It is, as far as we can establish, the
first Kabyle model to beat the statistical floor on gold annotation.

Under systematic homoglyph corruption of its input it still scores **87.42%**, above every
baseline's *clean* number.

## Results

Gold `UD_Kabyle-ADPT` test split: 845 sentences, 10,181 words, 8,563 whitespace tokens.
All four systems scored by one harness, under four conditions.

| system | gold-words | surface | gold-words, corrupted | surface, corrupted |
|---|---|---|---|---|
| **Masinissa-31M + linear probe** | **90.39%** | **90.51%** | **87.10%** | **87.42%** |
| most-frequent-tag baseline | 84.14% | 83.42% | 80.23% | 79.52% |
| `boffire/kabyle-pos-v2` | 56.50% | 63.69% | 55.10% | 62.35% |
| lexicon projection | 26.65% | 27.67% | 23.15% | 24.15% |

Macro-F1 over 15 tags, surface/canonical: **0.7301** for the probe, 0.7051 for the baseline,
0.4957 for `kabyle-pos-v2`.

Three things worth reading carefully:

**The encoder is frozen.** Only the linear head is fitted. This measures what pretraining put
into the representation, not what a task head can learn.

**Both tokenisation settings are reported because they disagree.** 29.5% of `UD_Kabyle-ADPT`
words sit inside a multiword token, so "accuracy" is two different numbers depending on
whether you score gold word segmentation or raw whitespace tokens. Reporting one is a choice
that should be visible.

**The corrupted condition is not noise injection for its own sake.** Kabyle text in the wild
carries systematic homoglyph substitution — Greek `ε` U+03B5 for Latin `ɛ` U+025B, `γ` for
`ɣ` — in 2.6–3.2% of rows of the largest public sources. The corrupted condition applies
exactly that substitution. The 3.1-point drop, from a baseline that loses 3.9, is the
corpus-level orthographic repair showing up as downstream robustness.

## Intended use

Kabyle NLP where a sentence or token representation is needed: sequence labelling, token
classification, sentence similarity, clustering, retrieval, and as an initialisation for
task-specific fine-tuning. It fills `[MASK]` tokens directly.

**Not suitable for**: generation of any kind (it is a bidirectional encoder), translation,
any language other than Kabyle, or any decision about a person. It has not been evaluated for
bias, toxicity, or factuality, and its training corpus has not been audited for offensive
content.

## Usage

`transformers` and `torch`, nothing else. LTG-BERT is not one of the library's own
architectures, so the modelling code travels in this repository and `trust_remote_code=True`
is what loads it.

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
top = logits[0, position].topk(5).indices
print([tokenizer.decode(piece) for piece in top])
```

For representations rather than predictions, `AutoModel` returns the encoder alone:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()
with torch.inference_mode():
    hidden = encoder(**tokenizer("Aman d tudert.", return_tensors="pt")).last_hidden_state
hidden.shape  # (1, tokens, 384)
```

**Normalise the input first.** The vocabulary was built over text where `ɛ` is U+025B and
never Greek epsilon; homoglyph-corrupted text fragments into byte pieces and the
representation degrades accordingly. This is not hypothetical — it is 2.6–3.2% of the
seed corpora this model was trained from.

`agbalu-tok-base-16k.model` is the SentencePiece model the vocabulary was trained as, kept
beside the converted `tokenizer.json` for anyone who needs the original. The two are id for
id, which the release staging checks before publishing.

## Architecture

An LTG-BERT / GPT-BERT encoder, reimplemented from
[`ltgoslo/gpt-bert`](https://github.com/ltgoslo/gpt-bert) — the BabyLM Challenge winner —
with its published recipe.

| | |
|---|---|
| Trainable parameters | **31,123,840** (19.74% of them the embedding table) |
| Layers / hidden / heads | 12 / 384 / 6 (head size 64) |
| Feed-forward | 1,280, gated (GLU) |
| Vocabulary | 16,000, SentencePiece Unigram — [`agbalu/Mammeri-Tok`](https://huggingface.co/agbalu/Mammeri-Tok) |
| Positions | log-bucketed **relative** positions, 32 buckets; max 512 |
| Sequence length (training) | 128 |
| Classifier | tied to the input embedding |
| Non-parameter buffers | 3,145,728 (the relative-position index) |

Relative position buckets rather than RoPE, and a tied classifier, both following the
reference implementation.

## Training data

**AƔBALU-Text v1** — 3,041,989 deduplicated Kabyle sentences from 42 sources, tokenised to
**70,184,279** training tokens and **359,340** validation tokens (0.5% held out).

Every source carries a provenance record: source id, licence, retrieval date. Every sentence
passed a reference normaliser at version **1.3.0+rules1.0.0**, which repairs the homoglyph
corruption described above while preserving the letters Kabyle actually uses
(`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`) and the emphatic and spirantised distinctions that carry meaning.

**Decontamination.** Measured against FLORES+ `kab_Latn` and SIB-200 at **zero overlap**,
with a positive control confirming the detector fires when contamination is present. Both
benchmarks derive from Wikipedia, which is in the corpus, so this was checked rather than
assumed.

### Licence composition of the training text

The weights are Apache-2.0. **That grant does not relicense the text they were trained on**,
and a third of that text has no licence anyone could resolve. The composition is published so
you can make your own judgement:

| redistribution | sentences | share |
|---|---|---|
| **unclear** | 1,062,569 | **34.9%** |
| permissive | 973,218 | 32.0% |
| share-alike | 944,438 | 31.0% |
| non-commercial | 61,764 | 2.0% |

`unclear` is not a permissive category — it is the absence of a resolvable licence. It is
29.9% sources declaring `other`, plus 5.0% declaring `cc`, which is not a licence at all
because no variant is given. One source contributing 711 sentences has unsettled copyright.

A permissive-only rebuild is possible by code from the 973,218 permissive sentences. If you
need one, open an issue.

## Training recipe

| | |
|---|---|
| Objective | masked LM with a 15:16 masked-to-causal hybrid (GPT-BERT) |
| Masking | **inverse schedule, 0.30 → 0.15**; spans up to 3; 10% random, 10% keep |
| Optimiser | **LAMB**, lr 1.2e-2, β (0.9, 0.98), ε 1e-8, weight decay 0.1 |
| Batch | 8,192 sequences per step = **1,048,576 tokens/step** (gradient accumulation) |
| Schedule | 1.6% warmup, cosine, 1.6% cooldown |
| Steps | **4,500** — 4,718,592,000 tokens, **67.2 epochs** |
| Regularisation | z-loss 1e-4, dropout 0.1, gradient clip 2.0 |
| Precision | bf16 mixed, `torch.compile` (measured **1.79×**: 70,200 → 125,700 tok/s) |
| Hardware | one NVIDIA A10, 24 GiB |
| Seed | 20260807 |

Masking is *inverse* — harder early, standard late — following the reference recipe, and
independently arrived at by mmBERT. LAMB at 1.2e-2 does not transfer to AdamW; the learning
rate is specific to the optimiser.

**Final validation loss 3.1957** (perplexity **24.43**), monotonically improving across all
18 evaluations:

| step | 250 | 1,000 | 2,000 | 3,000 | 4,000 | 4,500 |
|---|---|---|---|---|---|---|
| val loss | 4.3962 | 3.6162 | 3.4177 | 3.2969 | 3.2122 | **3.1957** |

## Limitations

**It is over-trained on unique tokens, and this is the binding constraint.** 67 epochs over
70.18M tokens, against a literature threshold of roughly 4 repetitions before returns
collapse ([Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264)). The run shows it
directly: between steps 3,000 and 3,250 training loss fell 0.22 while validation loss fell
0.027. More steps will not help, and the model is not too large either — LTG-BERT runs 0.77
parameters per token against this model's 0.443. **The constraint is unique Kabyle text, and
there is very little more of it.**

**It cannot represent the annexed state.** Kabyle marks a word-initial state alternation
(`axxam` "house" → `wexxam` in the annexed state). The vocabulary memorises both forms whole
at every size tested from 4k to 48k, in both initialisation arms — 0 of 15 test pairs share a
stem. This is a measured refutation of a design hypothesis, not an oversight, and it means
morphological structure of this kind has to come from the objective or an explicit
morphological layer rather than from segmentation.

**Sibling-language contamination is bounded but not cleared.** Kabyle has closely related
neighbours — Tashelhit, Tarifit, Central Atlas Tamazight, Tamasheq, Shawiya — and many
"Tamazight"-labelled datasets silently mix them. Corpus sources were language-identified, but
neither GlotLID nor NLLB's `lid218e` can *name* Tarifit, Central Atlas Tamazight or Shawiya,
so a `kab_Latn` label cannot exclude them. Measured on a balanced set, NLLB's identifier
labels 87–95% of Tashelhit, Tarifit and Central Atlas Tamazight as Kabyle. What is excluded
is Tashelhit and Tamasheq, which both systems can name.

**Not evaluated beyond POS.** One downstream task, one treebank, 845 test sentences. Treat
everything else as unmeasured.

**No safety evaluation of any kind** has been performed.

## Models fine-tuned from this one

[`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) restores punctuation and
capitalisation on Kabyle ASR output — two token-classification heads over this encoder, at
0.793 macro-F1 over marks against a rule baseline's 0.227.

## Files

| file | size | SHA-256 |
|---|---:|---|
| `model.safetensors` | 124.5 MB | `dca2a960c06a4bb14a5dd1d33557b8b4537230807a32d0a4debbee7fda44fbad` |
| `agbalu-tok-base-16k.model` | 260 KB | `c8094fccd936d2e8954809bd9cf45331679e9550d2c7bbabe5c38c3bf365dc4e` |
| `config.json` | — | architecture, training summary, the full validation curve, and the `auto_map` |
| `configuration_masinissa.py`, `modeling_masinissa.py` | — | the architecture in code, importing only `torch` and `transformers` |
| `tokenizer.json`, `tokenizer_config.json` | 1.0 MB | the same 16,000 pieces, as a fast tokenizer `AutoTokenizer` can read |

The weights file is exactly 31,123,840 float32 parameters. Two things are deliberately *not*
in it: the classifier decoder, which is tied to the embedding table and re-tied on load; and
twelve identical 512×512 relative-position index tables, which the module rebuilds
byte-identically in `__init__` and which would otherwise add 25.2 MB — 17% of the download —
of derived data. The exporter asserts both before writing.

The training checkpoint, with LAMB's optimizer moments and RNG state, is 398 MB and is not
published. Ask if you need it to resume training.

## Reproduction

The source checkpoint is `best.pt`, SHA-256
`b37b77ca22f0ed8e52dfcc52307cd09e4536d71670917051eba824ebf0c7a743`, and it loads into the
architecture above with zero missing and zero unexpected keys.

The POS numbers reproduce exactly. Head initialisation is seeded per-layer with an explicit
generator, so two fits of the same frozen checkpoint are bit-identical. Seeding only the
batch order is not enough: it leaves the reported accuracy moving by up to 0.5 points per
refit.

```
python -m agbalu.bench.cli pos --systems encoder neural lexicon baseline
```

## The name

**Masinissa** (r. 202–148 BCE) was the first king of a united Numidia, who brought the eastern
and western tribes into one kingdom. The encoder is the unifying representation of this
project — 42 noisy sources rendered into one shared space — so it carries his name.

His grandson Jugurtha, who fought Rome from 111 to 104 BCE and was never taken in battle, is
reserved for the generative model. The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_masinissa_2026,
  title  = {Masinissa-31M: a masked language model for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Masinissa-31M},
  note   = {Trained on AƔBALU-Text v1; normaliser 1.3.0+rules1.0.0}
}
```

## Licence

**Apache-2.0** for the weights and code. Read the licence composition of the training text
above before redistributing derivatives — a permissive grant on the weights makes no claim
about the underlying text, 34.9% of which has no resolvable licence.
