# AƔBALU

*Aɣbalu* — Kabyle for **the source, the fountainhead**.

Natural language processing for Kabyle (Taqbaylit, ISO 639-3 `kab`, BCP-47 `kab-Latn`), a
Northern Berber language of Kabylia, Algeria with roughly 5–7 million speakers. The corpus
is the primary artifact; the models are built on top of it.

## Why

Kabyle has an unusual resource profile: it is **speech-rich and text-poor**. Common Voice
v26.0 holds 571.29 validated hours of it — 10th of 294 locales — while the entire clean
Kabyle web crawl is smaller than a single Tatoeba export.

The text that exists is also damaged in a specific, measurable way. Both seed corpora carry
systematic homoglyph corruption: Greek `ε` U+03B5 standing in for Latin `ɛ` U+025B in 2.60%
and 3.19% of rows. Kabyle legitimately uses `ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`, and the emphatic and
spirantised distinctions carry meaning, so the repair has to be a versioned normalisation
layer rather than a character filter.

## Published artifacts

Eighteen repositories on the Hugging Face Hub under [`agbalu`](https://huggingface.co/agbalu),
ten models and eight datasets. Every model except `Matoub-82M` loads with `transformers` and
`torch` alone, without this repository; the voice is a StyleTTS2 checkpoint and ships its own
`inference.py`.

| Model | | |
|---|---|---|
| [`Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M) | encoder | 90.51% on gold POS, against a most-frequent-tag baseline of 83.42% |
| [`SiMohand-278M`](https://huggingface.co/agbalu/SiMohand-278M) | sentence embeddings and retrieval | 97.0% Recall@1 against the backbone's 63.8, and the same 97.0 at a 12× compression |
| [`Amrouche-1.3B`](https://huggingface.co/agbalu/Amrouche-1.3B) | translation | beats NLLB-1.3B in all four directions; eng→kab 36.34 chrF++ |
| [`Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M) | speech recognition | CER 8.01 / WER 25.65 over 888 unseen speakers |
| [`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) | punctuation and casing | macro-F1 0.793 |
| [`Boulifa-48M`](https://huggingface.co/agbalu/Boulifa-48M) | orthography standardisation | informal → canonical Kabyle Latin; 97.39% character accuracy against 89.70% for leaving the input alone |
| [`Feraoun-36M`](https://huggingface.co/agbalu/Feraoun-36M) | document OCR, Latin and Tifinagh | CER 2.85% and 70.20% line exact match over 1,000 held-out lines |
| [`Juba-27M`](https://huggingface.co/agbalu/Juba-27M) | Latin ↔ Tifinagh, with schwa restoration | 94.22% sentence exact match |
| [`Mammeri-Tok`](https://huggingface.co/agbalu/Mammeri-Tok) | tokenizer | ten vocabularies, 8k–32k |
| [`Matoub-82M`](https://huggingface.co/agbalu/Matoub-82M) | speech synthesis (preview) | First neural TTS for Kabyle — 24 kHz, StyleTTS2, 42-symbol IPA inventory, male voice |

| Dataset | |
|---|---|
| [`KabBench`](https://huggingface.co/datasets/agbalu/KabBench) | 2,009 MT pairs (326 repaired) and 1,500 LID rows across six Berber languages |
| [`KabLex`](https://huggingface.co/datasets/agbalu/KabLex) | 366,892 lexicon entries and 25,642 pronunciations |
| [`KabInflect`](https://huggingface.co/datasets/agbalu/KabInflect) | 336,151 inflected forms over 6,198 paradigms |
| [`KabTifinagh`](https://huggingface.co/datasets/agbalu/KabTifinagh) | 497,944 script-conversion pairs |
| [`KabSentiment`](https://huggingface.co/datasets/agbalu/KabSentiment) | 15,000 rows, three balanced classes |
| [`KabPunct`](https://huggingface.co/datasets/agbalu/KabPunct) | 1,318,707 word-labelled sentences for punctuation and casing restoration |
| [`KabG2P`](https://huggingface.co/datasets/agbalu/KabG2P) | 25,634 word forms with IPA transcriptions |
| [`KabStandard`](https://huggingface.co/datasets/agbalu/KabStandard) | 497,944 pairs, informal → canonical Kabyle Latin (training data for Boulifa-48M) |

Weights are Apache-2.0. Text keeps its upstream licences, so each card carries the licence
composition of the data behind it, the decontamination result, and the defects that are not
inferable from the artifact.

## Scope

Kabyle means Kabyle. Tashelhit (`shi`), Central Atlas Tamazight (`tzm`), Tarifit (`rif`),
Tamasheq (`taq`) and Shawiya (`shy`) are related but distinct languages, and datasets
labelled "Tamazight" routinely mix them.

That is measured here rather than asserted. On a balanced Latin-script set, NLLB's own
language identifier labels **87.2% of Tashelhit, 93.2% of Tarifit and 95.2% of Central
Atlas Tamazight as `kab_Latn`** — and it is the identifier that gated the bitext mining
behind most published Kabyle parallel data. Sibling sources are registered in a separate
file under a schema that refuses `kab`, so one cannot enter the corpus by accident.

## Install

```bash
make install
```

Python ≥ 3.12.

## Use

`make help` lists every target. The three that matter first:

```bash
make check
```

The gate: `ruff check`, `ruff format --check`, `mypy --strict`, and the full pytest suite.
It must pass with zero findings before anything is committed.

```bash
make registry
```

Validates the corpus registry against its schema. `SIBLINGS=1` validates the Berber sibling
registry instead.

```bash
make clean
```

Deletes every tool cache and all bytecode.

A family of subcommands is one target with a variable — `make bench TASK=pos`, not
`make bench-pos` — and defaults live in each CLI's `argparse` rather than in the Makefile,
so there is one copy of each.

## Layout

```
src/agbalu/    one sub-package per pipeline stage, each with its own cli.py
modal_app/     remote GPU execution; all training runs here
tests/         unit/ and integration/, mirroring src
resources/     registries, lookup tables, schemas — versioned YAML
tools/         standalone scripts, no package
data/          raw/ (immutable) → interim/ → processed/
artifacts/     checkpoints, weights, exports
configs/       hyperparameters and environment settings
```

Data and model bytes are never committed. Provenance lives in
[`resources/corpus_registry.yaml`](resources/corpus_registry.yaml), where every source
carries its licence, size, URI and checksum; reproduction lives in code.

## Licence

Apache-2.0 for the code and the released weights — see [LICENSE](LICENSE). Data retains its
upstream licences, which the registry records per source alongside a redistribution class.
