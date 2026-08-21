---
title: README
emoji: 💧
colorFrom: blue
colorTo: green
sdk: static
pinned: false
---

# AƔBALU

**Aɣbalu** (ⴰⵖⴱⴰⵍⵓ) is Kabyle for *the source* — a spring, the place water comes from.

This organization builds NLP resources for **Kabyle** (Taqbaylit, `kab`, Latin script), a
Northern Berber language of Kabylia, Algeria, spoken by 5–7 million people. Kabyle is
unusual among low-resource languages: it has a *disproportionately large* speech resource
and a thin, noisy text resource. Its entire filtered web crawl — FineWeb-2 and HPLT v2
together — is 13.8M words, and the corpus underneath these models, deduplicated across 42
sources, is 34.9M words.

The first artifact is not a model. It is the corpus.

The speech model is where that asymmetry pays: Kabyle has 571 validated hours of audio and
roughly 70 million unique tokens of text, and transcription is what turns the abundant
resource into the scarce one.

## Models

| | | |
|---|---|---|
| [**Amrouche-1.3B**](https://huggingface.co/agbalu/Amrouche-1.3B) | translation, 1.3B | Beats NLLB-200-1.3B in **all four directions**, and by the widest margin into Kabyle: **36.34 chrF++** and **10.86 BLEU** on English→Kabyle, against 31.53 and 7.74 for the base model on the same harness |
| [**SiMohand-278M**](https://huggingface.co/agbalu/SiMohand-278M) | sentence embeddings and retrieval, 278M | **97.0% Recall@1** and **0.9833 MRR** finding the right Kabyle passage among 499 distractors — against **63.8%** for the multilingual backbone it was fine-tuned from, which maps five Kabyle consonant characters to `<unk>` and collapses the language into one cone. Its 64-dimension Matryoshka slice, a 12× compression, scores the same 97.0% |
| [**Masinissa-31M**](https://huggingface.co/agbalu/Masinissa-31M) | encoder, 31M | **90.51%** on gold UD part-of-speech tags — against 83.42% for a most-frequent-tag baseline and 63.69% for the previously published Kabyle tagger. **0.8880 macro F1** on three-class sentiment, against 0.7764 from a linear probe on the frozen encoder |
| [**Mammeri-Tok**](https://huggingface.co/agbalu/Mammeri-Tok) | tokenizer | Ten Unigram vocabularies, 8k–32k, two initialisation arms. The sweep settles two open questions about Kabyle segmentation |
| [**Juba-27M**](https://huggingface.co/agbalu/Juba-27M) | transliteration, 27M | **94.22% sentence exact match** converting Tifinagh to Kabyle Latin, against **1.16%** for the character table that was the only prior tool — because Kabyle Tifinagh omits the vowel `e`, and where it belongs has to be predicted from the consonants around it |
| [**Fadhma-300M**](https://huggingface.co/agbalu/Fadhma-300M) | speech recognition, 300M | **8.01% character error rate** and **25.65% word error rate** on 15,003 utterances from **888 speakers it has never heard** — as far as we can establish, the first Kabyle speech recognition system published with an error rate measured on a speaker-disjoint split, and the first CTC result for the language at any size |
| [**Belaid-31M**](https://huggingface.co/agbalu/Belaid-31M) | punctuation and casing, 31M | **0.793 macro-F1** over marks on 5,160 held-out sentences, against **0.227** for the rule a system without a model uses. It reads what Fadhma writes and puts back the punctuation a CTC vocabulary cannot emit |
| [**Boulifa-48M**](https://huggingface.co/agbalu/Boulifa-48M) | orthography standardisation, 48M | **97.39% character accuracy and 85.70% sentences exactly right** converting informal French-keyboard and Arabizi Kabyle into canonical Kabyle Latin, against **89.70%** character accuracy for leaving the input untouched — as far as we can establish, the first orthography standardisation model published for any Berber language. Resolves Arabizi digits, digraphs, clitic hyphens and preposition contractions in one pass, where each is ambiguous on its own |
| [**Feraoun-36M**](https://huggingface.co/agbalu/Feraoun-36M) | document OCR, 36M | **2.85% character error rate and 70.20% line exact match** over 1,000 held-out lines of printed Kabyle, reading Latin and Neo-Tifinagh from one 171-symbol character alphabet — the first OCR model trained for the language, and the first that keeps the sub-dot on `ḍ ḥ ṛ ṣ ṭ ẓ` instead of normalising it away |
| [**Matoub-82M**](https://huggingface.co/agbalu/Matoub-82M) | speech synthesis (preview), 82M | StyleTTS2 fine-tune of Kokoro-82M on 21,953 restored Common Voice clips — 24 kHz, 42-symbol IPA inventory, male voice, Apache-2.0 where the incumbent `mms-tts-kab` is non-commercial. A preview checkpoint whose Cycle-CER against that baseline is **not yet measured**; future production models will be released under their own dedicated names |

## Datasets

| | | |
|---|---|---|
| [**KabBench**](https://huggingface.co/datasets/agbalu/KabBench) | evaluation | A **repaired** Kabyle MT reference — 326 of its 2,009 sentences were corrupt in the public original and have never been fixed upstream — plus a balanced six-language set for telling Kabyle from its Berber siblings |
| [**KabLex**](https://huggingface.co/datasets/agbalu/KabLex) | lexicon | 366,892 lexical entries over 17,090 lemmas, with morphological features, and 25,642 word–pronunciation pairs in IPA |
| [**KabSentiment**](https://huggingface.co/datasets/agbalu/KabSentiment) | sentiment | 15,000 human-written Kabyle sentences, labelled by projection from their English parallels and balanced exactly across three classes — the first Kabyle sentiment set that covers *neutral*, and the first whose text survives the language's own orthography |
| [**KabInflect**](https://huggingface.co/datasets/agbalu/KabInflect) | morphology | 336,151 inflected verb form entries across 13,226 lemmas, partitioned into sealed splits (0 paradigm leakage), plus 6,198 full verb conjugation paradigms |
| [**KabTifinagh**](https://huggingface.co/datasets/agbalu/KabTifinagh) | transliteration | 497,944 parallel sentences matching Kabyle Tifinagh to canonical Kabyle Latin, plus 123,852 English and 205,637 French trilingual sentence alignments |
| [**KabPunct**](https://huggingface.co/datasets/agbalu/KabPunct) | punctuation restoration | 1,318,707 word-labelled sentences with per-token punctuation (`NONE`, `COMMA`, `PERIOD`, `QUESTION`, `COLON`) and casing (`LOWER`, `UPPER_INIT`) labels — the training and evaluation corpus for Belaid-31M, and as far as we can establish the first labelled punctuation restoration corpus for any Berber language |
| [**KabG2P**](https://huggingface.co/datasets/agbalu/KabG2P) | phonetics / G2P | 25,634 Kabyle word forms with IPA transcriptions recovered at 99.53% alignment rate from 292,921 aligned tokens — zero ambiguity, 42-symbol phonetic inventory, and per-entry spirantisation and vowel-backing annotations; the pronunciation dictionary for Matoub-82M and any downstream Kabyle TTS or ASR system |
| [**KabStandard**](https://huggingface.co/datasets/agbalu/KabStandard) | orthography standardisation | 497,944 parallel pairs mapping informal French-keyboard and Arabizi Kabyle to canonical Kabyle Latin — training and evaluation corpus for Boulifa-48M. Fully synthetic and reproducible at seed 42 from `agbalu/KabTifinagh` |

**Every Kabyle BLEU score ever published is measured against a broken reference.** 16.2% of
the Kabyle side of the standard MT benchmark carries homoglyph corruption, so a system that
spells Kabyle correctly is penalised for it. `KabBench` is that reference, repaired.

`KabLex` publishes 366,892 of 395,834 entries, licence-cut by code: excluding a share-alike
source and one whose licence could not be established, with every row carrying its own
`source` and `licence` so a stricter subset can be cut downstream.

`KabSentiment` is the only Kabyle sentiment dataset that covers neutral and that passes the
language's orthography through without destruction — the prior binary corpus carries 0.00%
retention of `ɣ`, against 45% in real Kabyle text.

`KabInflect` splits by **verb lemma, not by verb form**. Kabyle verb morphology is
templatic, so a split by row puts two forms of one verb on either side of it and measures
memorisation; splitting by lemma means no test verb's paradigm was seen in any cell. Copying
the lemma unchanged scores 4.07%, which is the floor every result is read against.

`KabTifinagh` is 497,944 sentence pairs in both scripts. Kabyle Neo-Tifinagh writes no `e`,
so converting back to Latin is not a lookup — the vowel has to be predicted. A character
table gets 1.16% of sentences exactly right; `Juba-27M`, trained on this, gets 94.22%.

## What makes these different

**The orthography is repaired, not ignored.** Public Kabyle text carries systematic homoglyph
corruption: Greek `ε` U+03B5 standing in for Latin `ɛ` U+025B, `γ` for `ɣ`, in 2.6–3.2% of
rows of the largest sources. Published Kabyle tokenizers have baked this into their merges at
a measured cost of **+17.8% to +21.3% tokens** on correctly spelled text. Every artifact here
passes a versioned 81-rule normaliser first, and every artifact is stamped with its version.

That repair shows up downstream as measurement rather than as a claim, twice. Under deliberate
homoglyph corruption of its input, Masinissa-31M still scores **87.42%** — above every
baseline's *clean* number. And Amrouche-1.3B's score is **identical** whether or not both sides
are normalised before scoring — a gap of 0.00 into Kabyle, where the untuned base model gains
0.17–0.19 — so the model has learned to spell Kabyle canonically.

**Provenance is a precondition, not a footnote.** Every sentence in the training corpus
carries a source id, a licence, and a retrieval date. A record whose origin cannot be named
does not enter. The licence composition of the training text is published on every model
card, including the uncomfortable part: **34.9% of it has no licence anyone could resolve.**
A permissive grant on weights makes no claim about the text underneath them, and saying so
is cheaper than pretending otherwise.

**Evaluation reports what it measures.** FLORES+ `kab_Latn`, the benchmark every published
Kabyle result is scored against, is **16.2% orthographically corrupt and has never been
revised upstream** — which means 2.71 BLEU on its devtest is unreachable by a system that
spells Kabyle correctly. Results here are scored twice, as-published and with both sides
normalised, and the difference is reported. A published accuracy is a claim about a dataset,
not about a task: one Kabyle model card's 94.8% is agreement with the lexicon projection that
generated its own labels; on gold annotation it is 63.69%.

**Decontamination is measured, with a positive control.** Zero overlap between the training
corpus and all 2,009 FLORES+ `kab_Latn` sentences, which are also the text SIB-200 reuses.
FLORES+ is built from Wikipedia, which is in the corpus, so this was checked rather than
assumed, and the detector is positive-controlled against injected leaks — including one
that differs only by the homoglyph corruption.

## The names

Each model is named for someone whose life was the work that model does, in two families.

| model | does | name | |
|---|---|---|---|
| tokenizer | decides how Kabyle is written down | **Mammeri** | Mouloud Mammeri wrote the first Berber grammar written *in* Kabyle (1976), inventing the metalanguage to do it. The 1980 cancellation of his lecture on Kabyle poetry began the Berber Spring |
| encoder | understands; one representation | **Masinissa** | first king of a united Numidia, r. 202–148 BCE — he brought the eastern and western tribes into one kingdom, as the encoder brings 42 sources into one representation |
| sentence embeddings | recalls Kabyle exactly | **Si Mohand** | Si Mohand ou Mhand (1848–1905), master of the Kabyle *isefra*, who wrote none of them down — the verses survive because they were heard once and recalled intact |
| translation | carries Kabyle out and back | **Amrouche** | Taos Amrouche sang exclusively in Kabyle and wrote in French — a life spent carrying one into the other |
| transliteration | translates scripts and restores text | **Juba** | King Juba II (48 BCE – 23 CE) — the ancient Numidian scholar-king, linguist, and author whose life's work was the study and preservation of North African scripts and language |
| speech recognition | hears Kabyle, writes it down | **Fadhma** | Fadhma Aït Mansour Amrouche, Taos's mother, who from 1930 began writing down the songs and tales inherited from her ancestors |
| punctuation and casing | writes Kabyle as prose on a page | **Belaïd** | Belaïd At Ali (1909–1950), founder of written Kabyle prose. Asked to set down oral tales, he composed instead — *Les Cahiers de Belaïd*, the first Kabyle prose in Latin characters, and the first text whose author had to decide where a Kabyle sentence ends |
| voice | speaks Kabyle | **Matoub** | Lounès Matoub (1956–1998) — recorded 36 albums in Kabyle at a time when the Algerian state was suppressing Berber language and culture, making the language audible to an entire generation. Assassinated on 25 June 1998, ten days before the Arabisation law he had spent years opposing took effect. His voice is inseparable from the survival of Kabyle as a spoken language in collective memory |
| orthography standardisation | writes Kabyle correctly from any input | **Boulifa** | Si Amar ou Saïd Boulifa (1865–1931) — the first grammarian to systematically codify Kabyle Latin orthography. His *Recueil de poésies kabyles* (1904) set 555 pages of Kabyle in one consistent Latin notation, solving in print what this model solves from a keyboard |
| document OCR | reads Kabyle from printed paper and books | **Feraoun** | Mouloud Feraoun (1913–1962), novelist and teacher who taught reading and transcribed the oral poetry of Si Mohand onto the printed page |
| generative | generates; sovereign | **Jugurtha** | Masinissa's grandson, who fought Rome 111–104 BCE and was never taken in battle |

Two lineages carry the design. **Masinissa and Jugurtha are grandfather and grandson**, so
the smallest model and the largest are the founding of the kingdom and the refusal to
surrender it. **Fadhma and Taos Amrouche are mother and daughter**, and they split the
oral-tradition work exactly as their two models do: the mother heard the songs and wrote them
down, which is speech recognition, and the daughter carried them into French while never
singing in anything but Kabyle, which is translation. One of Taos's albums is called *Chants
sauvés de l'oubli* — songs saved from oblivion.

Mammeri and Matoub stand alone, because what each of them did nobody shared: the grammar, and
the voice.

The naming is homage. It implies no endorsement by anyone, and none of these figures or their
families are affiliated with this work.

## Licence

Weights are **Apache-2.0**. The text they were trained on is not, and cannot be relicensed by
that grant — each model card publishes the full composition so you can judge for yourself.
