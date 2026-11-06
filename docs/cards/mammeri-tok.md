---
language:
- kab
license: apache-2.0
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- tokenizer
- sentencepiece
- unigram
- low-resource
---

# Mammeri-Tok

Ten SentencePiece Unigram vocabularies for **Kabyle** (Taqbaylit, `kab`, Latin script), trained on 3,041,989 orthographically normalised sentences across five vocabulary sizes (8k, 12k, 16k, 24k, and 32k) and two initialization arms (`base` and `seeded`).

The **base-16k** vocabulary is used by [`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M), [`agbalu/Masinissa-73M`](https://huggingface.co/agbalu/Masinissa-73M), and downstream models. The remaining nine configurations are published to document the empirical evaluation across vocabulary scale and lexical initialization.

## Initialization arms and morphology

- **Lexicon seeding**: The `seeded` arm initialises the Unigram candidate set using a 395,834-entry Kabyle lexicon (258,221 inflected forms across 32,914 lemmas, augmented with annexed-state rules and Hunspell affixes). Evaluated against the standard unseeded `base` arm across all five sizes, lexicon seeding reduces tokenised sequence length by only **0.17%** at 16k (665,935 tokens versus 667,068 across 344,648 words) and does not improve morphological segmentation metrics.
- **Annexed state representation**: Kabyle marks word-initial grammatical state through vowel alternation (e.g., free state *axxam*, "house", becomes annexed state *wexxam*). Across all vocabulary sizes and both arms, **0 of 15 diagnostic annexed-state pairs share a stem**, with Viterbi segmentation preferring full-word pieces by approximately 9 nats. This indicates that standard statistical subword segmenters memorize alternations as distinct units rather than factoring out underlying stems.
- **Clitic attachment**: Clitic fixtures remain consistently atomic across configurations, with **72 of 72** diagnostic clitic forms preserved across all sizes.

## Results

Evaluated on 30,000 held-out sentences (344,648 words, 2,028,636 characters). `Fertility` measures average subwords per word (lower indicates higher compression).

| Vocabulary | Fertility | Tokens/char | Whole-word pieces | Annexed state | Clitics | Embedding params @ 384 |
|---|---:|---:|---:|:---:|:---:|---:|
| base-8k | 2.0915 | 0.3553 | 53.0% | 0/15 | 72/72 | 3,072,000 |
| base-12k | 1.9949 | 0.3389 | 55.8% | 0/15 | 72/72 | 4,608,000 |
| **base-16k** | **1.9355** | **0.3288** | **57.8%** | 0/15 | 72/72 | **6,144,000** |
| base-24k | 1.8621 | 0.3163 | 59.8% | 0/15 | 72/72 | 9,216,000 |
| base-32k | 1.8163 | 0.3086 | 60.7% | 0/15 | 72/72 | 12,288,000 |
| seeded-8k | 2.0895 | 0.3550 | 52.9% | 0/15 | 72/72 | 3,072,000 |
| seeded-12k | 1.9925 | 0.3385 | 55.9% | 0/15 | 72/72 | 4,608,000 |
| seeded-16k | 1.9322 | 0.3283 | 57.9% | 0/15 | 72/72 | 6,144,000 |
| seeded-24k | 1.8602 | 0.3160 | 59.9% | 0/15 | 72/72 | 9,216,000 |
| seeded-32k | 1.8146 | 0.3083 | 60.8% | 0/15 | 72/72 | 12,288,000 |

Round-trip decoding reconstruction error is 0% across all configurations. Byte-fallback pieces account for 0.65–0.74% of token occurrences.

### Vocabulary selection rationale

The 16k vocabulary was selected for the Masinissa encoder family. Increasing vocabulary size from 8k to 16k reduces token count by 7.5%, whereas the subsequent doubling from 16k to 32k yields only an additional 6.2% reduction while doubling embedding table memory. At hidden dimension 384, a 16k embedding table accounts for 19.7% of total parameters (6.14M of 31.1M), compared to 33.0% (12.29M of 37.3M) at 32k.

## Build parameters

| Parameter | Value |
|---|---|
| Algorithm | SentencePiece **Unigram**, `byte_fallback=True` |
| Normalisation rule | `identity` (corpus pre-normalised) |
| Character coverage | 0.9995 |
| Input sample | 2,000,000 sampled sentences |
| Random seed | 20260807 |
| Normaliser version | **1.3.0+rules1.0.0** |
| Tokenizer version | 1.0.0 |

The normalisation rule is set to `identity` because text is pre-normalised by the AƔBALU pipeline. Default SentencePiece NMT normalisation erroneously folds distinct Kabyle characters, such as mapping `ţ` to `ṭ`. `required_chars` is explicitly defined over the Kabyle alphabet plus `-` and `'` to prevent foreign-script symbols from consuming vocabulary slots.

## Training corpus

Trained on a 2,000,000-sentence sample of **AƔBALU-Text v1** (3,041,989 deduplicated sentences across 42 provenance-tracked sources). All text is pre-processed with the reference normaliser (version 1.3.0+rules1.0.0). The normaliser standardises homoglyph errors common in public text (e.g. Greek `ε` U+03B5 for Latin `ɛ` U+025B, and `γ` for `ɣ`), which affect 2.6–3.2% of lines in raw web data, while preserving phonemically contrastive spirantised and emphatic consonants (`ɣ`, `ɛ`, `ḥ`, `ḍ`, `ṣ`, `ṭ`, `ẓ`, `ṛ`, `č`, `ǧ`, `ţ`).

Without normalisation, homoglyph inconsistencies fragment subwords during vocabulary construction, increasing token sequences by +17.8% to +21.3% on clean Kabyle text.

### Licence composition of the training text

Apache-2.0 covers these vocabulary files. The redistribution distribution of the underlying corpus is:

| Redistribution | Sentences | Share |
|---|---|---|
| **Unclear** | 1,062,569 | **34.9%** |
| Permissive | 973,218 | 32.0% |
| Share-alike | 944,438 | 31.0% |
| Non-commercial | 61,764 | 2.0% |

## Intended use

- Subword tokenization for Kabyle language modeling and NLP applications.
- **base-16k** is the default configuration, used by [`Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M) and [`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M).

**Not suitable for**:
- Tifinagh script (these vocabularies are Latin-script only).
- Non-Kabyle Berber languages with distinct phonotactics.
- General-purpose multilingual tokenization.

## Usage

```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download

model = hf_hub_download("agbalu/Mammeri-Tok", "agbalu-tok-base-16k.model")
sp = spm.SentencePieceProcessor(model_file=model)

sp.encode("Azul fell-awen, amek i tellam?", out_type=str)
# ['▁Azul', '▁fell', '-', 'awen', ',', '▁amek', '▁i', '▁tellam', '?']

sp.encode("Azul fell-awen, amek i tellam?")
# [3173, 354, 261, 549, 265, 437, 267, 6399, 296]
```

The 16k base vocabulary is also published as a `transformers` fast tokenizer inside [`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M), id-for-id with this file.

Input should be normalised to standard Kabyle Latin orthography prior to tokenization to prevent byte-fallback fragmentation.

`agbalu-tok-base-16k.model` has SHA-256 `c8094fccd936d2e8954809bd9cf45331679e9550d2c7bbabe5c38c3bf365dc4e`.

## Limitations

- **Morphological factorisation**: Statistical subword units do not align cleanly with Kabyle morphemes, particularly in cases of root-internal vowel alternation.
- **Script support**: Latin script only; Tifinagh orthography is not covered.
- **Diagnostic scope**: Annexed state and clitic evaluations use targeted diagnostic sets (15 pairs and 72 clitics).

## Files

| File | Contents |
|---|---|
| `agbalu-tok-{base,seeded}-{8,12,16,24,32}k.model` | SentencePiece binary model file |
| `agbalu-tok-*.vocab` | Vocabulary list and log-probabilities |
| `agbalu-tok-*.metadata.json` | Build specification and SHA-256 checksum |
| `sweep.json` | Complete benchmark metrics across all ten configurations |

## Reproduction

Trained with SentencePiece using Unigram segmentation and 0.9995 character coverage across the balanced Kabyle corpus sample. Evaluated on 30,000 held-out sentences for fertility, byte fallback rate, and morphological agreement. All metrics reproduce deterministically from the published model artifacts.

## The name

**Mouloud Mammeri** wrote `Tajeṛṛumt n tmaziɣt` (Maspero, Paris, 1976) — the first Berber
grammar written entirely *in* Kabyle, which meant inventing the metalanguage to describe the
language in its own words. He founded the research centre CERAM and the journal *Awal* ("the
word") in 1982. On 10 March 1980 the Algerian government cancelled his lecture on ancient
Kabyle poetry at Hasnaoua University in Tizi-Ouzou, and that cancellation began Tafsut
Imaziɣen, the Berber Spring.

The man who worked out how to write the language down gives his name to the thing that
decides how it is written down. The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_mammeri_tok_2026,
  title  = {Mammeri-Tok: Kabyle subword vocabularies},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Mammeri-Tok},
  note   = {Ten Unigram vocabularies, 8k-32k, two initialisation arms;
            normaliser 1.3.0+rules1.0.0}
}
```

## Licence

**Apache-2.0.** Check the licence composition table above before redistributing derivative models or datasets.
