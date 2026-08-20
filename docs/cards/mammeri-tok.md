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

Ten SentencePiece Unigram vocabularies for **Kabyle** (Taqbaylit, `kab`, Latin script),
trained on 3,041,989 orthographically normalised sentences: five sizes from 8k to 32k, in two
initialisation arms.

The vocabulary at **16,000, default arm** is the one
[`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M) is trained on. The other
nine are published because the sweep is the result — it settles two questions about Kabyle
segmentation that were previously argued rather than measured.

## Why ten and not one

**Lexicon-seeded initialisation is a controlled negative.** The `seeded` arm initialises the
Unigram candidate set from a 395,834-entry Kabyle lexicon — 258,221 forms over 32,914 lemmas,
plus annexed-state rules and Hunspell affixes. It was expected to help. Measured against a
matched default arm at all five sizes, it changes tokenised length by **0.17%** at 16k
(665,935 tokens against 667,068 over the same 344,648 words) and does not change the
morphological measures at all. Published so nobody spends the effort again.

**Vocabulary construction cannot represent the annexed state.** Kabyle marks a word-initial
state alternation — `axxam` "house" becomes `wexxam` in the annexed state. The design argued
that a small vocabulary would be *forced* to factor this so the stem is shared. It is false.
Across every size and both arms, **0 of 15 annexed-state pairs share a stem**; both members
are memorised whole. Viterbi prefers the whole word by 9 nats. Tested twice, refuted twice:
this has to come from the model, the objective, or an explicit morphological layer.

The one place the vocabulary does behave well is clitics: **72 of 72** clitic fixtures remain
atomic at every size.

## Results

30,000 held-out sentences, 344,648 words, 2,028,636 characters. `fertility` is tokens per
word; lower is denser.

| vocabulary | fertility | tokens/char | whole-word pieces | annexed state | clitics | embedding params @384 |
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

**Round-trip failures: 0** at every size. Byte-fallback pieces are 0.65–0.74% of tokens.

**Why 16k was chosen for the encoder.** Compression improves monotonically with size and
buys 7.5% fewer tokens from 8k to 16k, then only 6.2% more for the next doubling — while the
embedding table doubles each time. At hidden size 384, a 16k table is **19.7%** of the
resulting 31.1M-parameter encoder, against **33.0%** of the 37.3M one at 32k. 16,384 is also
what the BabyLM-winning LTG-BERT line uses at 100M words. Neither of those is a downstream
measurement, and the choice should be revisited by one.

## Build parameters

| | |
|---|---|
| Algorithm | SentencePiece **Unigram**, `byte_fallback=True` |
| Normalisation rule | `identity` — the corpus is already normaliser output |
| Character coverage | 0.9995 |
| Input sentences | 2,000,000 sampled |
| Random seed | 20260807 |
| Normaliser | **1.3.0+rules1.0.0** |
| Tokenizer version | 1.0.0 |

`normalization_rule_name="identity"` is deliberate and load-bearing. SentencePiece's default
NMT normalisation re-folds characters Kabyle needs kept apart — `ţ` above all, which one
published specification maps to `ṭ` against the evidence of the corpus. `required_chars` is
set explicitly over the Kabyle inventory plus `-` and `'`; without it, vocabulary slots leak
to scripts that are collectively under 0.5% of tokens.

## Training corpus

**AƔBALU-Text v1** — 3,041,989 deduplicated sentences over 42 provenance-tracked sources,
every one passed through the reference normaliser at 1.3.0+rules1.0.0. That normaliser repairs
the systematic homoglyph corruption in public Kabyle text (Greek `ε` U+03B5 for Latin `ɛ`
U+025B, `γ` for `ɣ`) which affects 2.6–3.2% of rows in the largest sources, while preserving
`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ` and the emphatic and spirantised distinctions that carry meaning.

This matters for a tokenizer specifically: **homoglyph corruption gets baked into vocabulary
merges**. Measured on published Kabyle tokenizers, it costs **+17.8% to +21.3% tokens** on
correctly spelled text, because the corrupted and correct spellings of the same word occupy
different pieces.

### Licence composition of the training text

Apache-2.0 covers these vocabulary files. It does not relicense the text they were built from:

| redistribution | sentences | share |
|---|---|---|
| **unclear** | 1,062,569 | **34.9%** |
| permissive | 973,218 | 32.0% |
| share-alike | 944,438 | 31.0% |
| non-commercial | 61,764 | 2.0% |

`unclear` is the absence of a resolvable licence, not a permissive one.

## Intended use

Segmenting Kabyle for a model you are training yourself. **base-16k is the default** — it is
what [`Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M) and
[`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) were built on, and the other nine
exist so a size or initialisation decision can be made against measurements rather than by
convention.

**Normalise first.** The vocabulary was trained on text where `ɛ` is U+025B and never Greek
epsilon; homoglyph-corrupted input fragments into byte pieces, and that is 2.6–3.2% of the
seed corpora this project started from.

**Not suitable for**: Tifinagh, which no vocabulary here covers; the Berber sibling languages,
which share part of the orthography and were not measured; or as a general-purpose
multilingual tokenizer — these are Kabyle vocabularies and nothing else.

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

Nine pieces for six words: `Azul`, `amek`, `i` and `tellam` are each one piece, and the
clitic chain `fell-awen` splits on its hyphen. That ratio is what the sweep below measures —
**1.9355 pieces per word** for this vocabulary over 30,000 held-out sentences.

The 16k base vocabulary is also published as a `transformers` fast tokenizer inside
[`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M), id for id with this
file. Use that one if you want `AutoTokenizer`; use this one for the other nine sizes, which
have no model of their own.

Input should already be normalised to the same orthography the vocabulary was built on;
otherwise homoglyph-corrupted text will fragment into byte pieces.

`agbalu-tok-base-16k.model` has SHA-256
`c8094fccd936d2e8954809bd9cf45331679e9550d2c7bbabe5c38c3bf365dc4e`. Each vocabulary ships with
a `.vocab` listing and a `.metadata.json` recording its build spec and checksum.

## Limitations

- **No morphological factorisation.** See the annexed state above. Do not assume subword
  boundaries correspond to Kabyle morphemes; measured, they largely do not.
- **Latin script only.** Kabyle is also written in Tifinagh, and lossless Latin↔Tifinagh
  transliteration is not possible — the mapping is genuinely many-to-one in places. These
  vocabularies do not cover Tifinagh.
- **Compression is not a claim we own.** At 48k this Unigram measures 1.754 tokens per word
  against a community Kabyle BPE's 1.542. That is what Unigram trades away. The advantage
  here is orthographic, not compressive: the vocabulary is built on repaired text, so it does
  not spend pieces on corrupted spellings.
- The sweep's morphological measures use small fixture sets — 15 annexed-state pairs, 72
  clitic fixtures. They are diagnostic, not a benchmark.

## Files

Ten vocabularies, each in three files, plus the sweep.

| file | contents |
|---|---|
| `agbalu-tok-{base,seeded}-{8,12,16,24,32}k.model` | the SentencePiece model — the only file inference needs |
| `agbalu-tok-*.vocab` | the pieces and their log-probabilities, one per line, for reading |
| `agbalu-tok-*.metadata.json` | the build spec that produced it, and its SHA-256 |
| `sweep.json` | every number in the results table above, for all ten |

Each `.metadata.json` records the corpus, the character coverage, the seed list where there
is one, and the checksum — so a vocabulary can be tied back to the build that made it rather
than trusted by filename.

## Reproduction

```bash
make tokenizer STAGE=prepare      # the corpus sample the vocabularies are trained on
make tokenizer STAGE=sweep        # all ten, from that sample
make tokenizer STAGE=evaluate     # the results table, over 30,000 held-out sentences
```

No GPU. The whole sweep is CPU work.

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

**Apache-2.0.** Read the licence composition of the training text above before redistributing
derivatives.
