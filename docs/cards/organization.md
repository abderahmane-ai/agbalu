---
title: README
emoji: 💧
colorFrom: blue
colorTo: green
sdk: static
pinned: false
---

# AƔBALU

**Aɣbalu** (ⴰⵖⴱⴰⵍⵓ) is Kabyle for *the source* — a natural spring where water originates.

This organization develops open-source NLP and speech resources for **Kabyle** (Taqbaylit, `kab`, Latin script), a Northern Berber language spoken by 5–7 million people in Kabylia, Algeria. While public web crawls for Kabyle are relatively scarce (FineWeb-2 and HPLT v2 total approximately 13.8M words), the language possesses extensive speech resources, including over 571 validated hours in Mozilla Common Voice. The AƔBALU project builds foundational datasets, models, and benchmarks to expand Kabyle language technology from curated corpora to high-capacity foundation models.

## Models

| Model | Task & Capacity | Description |
|---|---|---|
| [**Jugurtha-2-9B**](https://huggingface.co/agbalu/Jugurtha-2-9B) | Sovereign foundation & reasoning LLM, 8.95B | Second-generation sovereign foundation model for Kabyle. Achieves **80.81% on SIB-200 topic classification**, **49.45 chrF++ / 27.86 BLEU** on FLORES+ kab→eng, and **30.80 chrF++ / 7.62 BLEU** on eng→kab with zero orthography gap (0.00). Native chain-of-thought in Taqbaylit (`<think>` blocks). Also available in [**GGUF**](https://huggingface.co/agbalu/Jugurtha-2-9B-GGUF) and [**MLX**](https://huggingface.co/agbalu/Jugurtha-2-9B-MLX-4bit) |
| [**Jugurtha-4B**](https://huggingface.co/agbalu/Jugurtha-4B) | Foundation & reasoning LLM, 4.2B | Sovereign Kabyle generative model with native step-by-step reasoning (`<think>` blocks). Continued pretraining on 800M tokens and SFT on curated Kabyle dialogues. **100.0% NIAH** at 32,768 context, **80.89% GSM8K**, **62.86% MMLU-Pro**, **54.16% IFEval**, **73.74% SIB-200** topic classification on Kabyle. Available in [**GGUF**](https://huggingface.co/agbalu/Jugurtha-4B-GGUF) and [**MLX**](https://huggingface.co/agbalu/Jugurtha-4B-MLX-4bit) |
| [**Amrouche-1.3B**](https://huggingface.co/agbalu/Amrouche-1.3B) | Machine translation, 1.3B | Vocabulary-trimmed translation model outperforming NLLB-200-1.3B across all four directions (36.34 chrF++ / 10.86 BLEU on English→Kabyle vs. 31.53 / 7.74 for the baseline) |
| [**Fadhma-300M**](https://huggingface.co/agbalu/Fadhma-300M) | Speech recognition, 300M | Acoustic CTC model achieving **8.01% character error rate** and **25.65% word error rate** on 15,003 utterances across 888 unseen speakers in Common Voice Kabyle |
| [**Belaid-75M**](https://huggingface.co/agbalu/Belaid-75M) | Discourse punctuation & truecasing, 75M | Restores punctuation and truecasing across continuous multi-sentence discourse using a 1,024-token context window over Masinissa-73M (**0.630 period F1, 0.623 comma F1, 94.0% word casing accuracy, 0.755 acronym F1**) |
| [**Belaid-31M**](https://huggingface.co/agbalu/Belaid-31M) | Utterance punctuation & casing, 31M | Sentence-level punctuation and casing restoration for ASR transcripts (**0.793 macro-F1** and **86.3% exact sentence match** across 5,160 held-out utterances) |
| [**Matoub-4B**](https://huggingface.co/agbalu/Matoub-4B) | Speech synthesis, 4.2B | Neural codec language model with an 8-codebook 24 kHz audio tokenizer fine-tuned on Common Voice Kabyle. Supports zero-shot voice cloning and cross-lingual reference conditioning. Also available in [**GGUF**](https://huggingface.co/agbalu/Matoub-4B-GGUF) |
| [**Matoub-82M**](https://huggingface.co/agbalu/Matoub-82M) | Speech synthesis, 82M | 24 kHz neural TTS model with a 42-phoneme Kabyle inventory including dedicated embeddings for `ˤ`, `ʕ`, and `ħ`. Shipped with single voice checkpoint |
| [**Boulifa-48M**](https://huggingface.co/agbalu/Boulifa-48M) | Orthography standardisation, 48M | Converts informal French-keyboard input and Arabizi into canonical Kabyle Latin (**97.39% character accuracy, 85.70% sentence exact match**) |
| [**Juba-27M**](https://huggingface.co/agbalu/Juba-27M) | Transliteration, 27M | Transliterates vowel-less Neo-Tifinagh to canonical Kabyle Latin with learned schwa placement (**94.22% sentence exact match, 96.06% schwa F1**) |
| [**Feraoun-Pro**](https://huggingface.co/agbalu/Feraoun-Pro) | Document OCR, 51M | Vision-encoder CTC document OCR with variable-width canvas up to 1,024 px and 173-character alphabet including `ţ` / `Ţ` (**7.34% Latin CER, 0.91% Neo-Tifinagh CER, 81.77% exact line match**) |
| [**Feraoun-36M**](https://huggingface.co/agbalu/Feraoun-36M) | Document OCR, 36M | Vision transformer OCR reading Latin and Neo-Tifinagh across a 171-character alphabet (**2.85% CER and 70.20% line exact match** on held-out printed lines) |
| [**SiMohand-278M**](https://huggingface.co/agbalu/SiMohand-278M) | Sentence embeddings & retrieval, 278M | Kabyle passage retrieval model fine-tuned with Matryoshka representations (**97.0% Recall@1, 0.9833 MRR**; 64-dimensional slice retains 97.0% Recall@1) |
| [**Masinissa-73M**](https://huggingface.co/agbalu/Masinissa-73M) | Flagship encoder, 73M | High-capacity masked language model leading all models on `KabEncBench` (**1,953.7 Bradley–Terry Elo**; 94.08% surface POS accuracy, 80.59% UPOS macro-F1) |
| [**Masinissa-31M**](https://huggingface.co/agbalu/Masinissa-31M) | Lightweight encoder, 31M | Efficient bidirectional encoder (**1,708.4 Bradley–Terry Elo** on `KabEncBench`, #2 overall; 90.51% gold POS surface accuracy) |
| [**Masinissa-Sentiment**](https://huggingface.co/agbalu/Masinissa-Sentiment) | Sentiment classification, 31M | Three-class sentiment classifier with split-conformal prediction (**90.27% accuracy, 0.9026 macro-F1, 94.00% conformal coverage** at 1.13 mean set size) |
| [**Dihya-22M**](https://huggingface.co/agbalu/Dihya-22M) | Pan-Berber & Maghrebi LID, 22M | Language identification across 22 classes and 5 scripts, covering Berber varieties in Latin and Neo-Tifinagh, Maghrebi Arabic dialects, MSA, and world languages (**91.90% accuracy, 0.9180 macro-F1**) |
| [**Dihya-5M**](https://huggingface.co/agbalu/Dihya-5M) | Lightweight language identification, 5M | Distinguishes six Latin-script Amazigh varieties and rejects non-Berber text (**85.52% overall accuracy, 84.44% on Berber classes**) |
| [**Mammeri-Tok**](https://huggingface.co/agbalu/Mammeri-Tok) | Tokenizer collection | Ten SentencePiece Unigram vocabularies (8k–32k) across standard and lexicon-seeded initialization arms |

## Interactive Spaces

| Space | Focus | Description |
|---|---|---|
| [**KabEncBench-Leaderboard**](https://huggingface.co/spaces/agbalu/KabEncBench-Leaderboard) | Benchmark leaderboard | Pairwise Bradley–Terry Elo tournament (2,000 bootstraps), radar chart comparisons, and win-rate matrices across 9 foundation encoders on Kabyle |
| [**Jugurtha-Chat**](https://huggingface.co/spaces/agbalu/Jugurtha-Chat) | Conversational AI interface | Interactive chat interface powered by `Jugurtha-2-9B`, supporting native chain-of-thought (`<think>` blocks) and streaming generation |

## Datasets

| Dataset | Domain | Description |
|---|---|---|
| [**KabEncBench**](https://huggingface.co/datasets/agbalu/KabEncBench) | Multi-task evaluation | 100,609 annotated instances across 6 core tasks (Syntax UPOS, Aspect Morphology, NER, Sentiment, SIB-200 Topic, Punctuation) and 3 diagnostic probes |
| [**KabBench**](https://huggingface.co/datasets/agbalu/KabBench) | Evaluation benchmark | Audited and orthographically corrected Kabyle MT reference (correcting homoglyph corruption in 326 sentences), plus a balanced six-language Amazigh discrimination set |
| [**KabLex**](https://huggingface.co/datasets/agbalu/KabLex) | Lexicon & pronunciation | 366,892 permissive lexical entries over 17,090 lemmas with morphological features, and 25,642 word–pronunciation pairs in IPA |
| [**KabInflect**](https://huggingface.co/datasets/agbalu/KabInflect) | Morphology | 336,151 inflected verb forms across 13,226 lemmas partitioned into lemma-disjoint splits, plus 6,198 complete verb conjugation paradigms |
| [**KabTifinagh**](https://huggingface.co/datasets/agbalu/KabTifinagh) | Transliteration | 497,944 parallel sentences pairing vowel-less Neo-Tifinagh with canonical Kabyle Latin, plus English and French trilingual alignments |
| [**KabStandard**](https://huggingface.co/datasets/agbalu/KabStandard) | Orthography | 497,944 parallel sentence pairs mapping informal keyboard and Arabizi text to canonical Kabyle Latin |
| [**KabPunct**](https://huggingface.co/datasets/agbalu/KabPunct) | Punctuation restoration | 1,318,707 word-labelled sentences with per-token punctuation and casing labels |
| [**KabG2P**](https://huggingface.co/datasets/agbalu/KabG2P) | Phonetics & G2P | 25,634 Kabyle word forms with validated IPA transcriptions across a 42-symbol phonetic inventory |
| [**KabSentiment**](https://huggingface.co/datasets/agbalu/KabSentiment) | Sentiment | 15,000 human-written Kabyle sentences balanced across three sentiment classes, labelled via cross-lingual teacher projection |
| [**KabLiterary**](https://huggingface.co/datasets/agbalu/KabLiterary) | Literary corpus | 28,907 paragraphs (657,713 words) of translated world literature and oral poetry in canonical Kabyle Latin with book-boundary splits |

## Technical Principles

- **Orthographic consistency**: Public Kabyle text frequently exhibits systematic homoglyph substitutions (e.g. Greek `ε` U+03B5 for Latin `ɛ` U+025B, and `γ` for `ɣ`) in 2.6–3.2% of lines in raw web data. All AƔBALU datasets and models pass through a versioned 81-rule normaliser that standardises character representations while preserving contrastive spirantised and emphatic phonemes (`ɣ`, `ɛ`, `ḥ`, `ḍ`, `ṣ`, `ṭ`, `ẓ`, `ṛ`, `č`, `ǧ`, `ţ`).
- **Provenance tracking**: Every sentence in the corpus carries traceable source identifiers, licensing metadata, and retrieval timestamps. The complete license distribution of training corpora is disclosed on each model card.
- **Audited benchmarks**: Standard benchmarks with known upstream orthographic defects (such as the 16.2% homoglyph corruption in FLORES+ `kab_Latn`) are evaluated both in original and normalized forms to isolate model translation quality from benchmark encoding errors.
- **Decontamination controls**: Training corpora are strictly decontaminated against benchmark evaluation splits (such as FLORES+ and SIB-200), verified through positive-control leak injection.
- **Standardized representation benchmarking**: On `KabEncBench`, representation quality is evaluated with frozen backbones, layer-weighted `ScalarMix` pooling, and linear probes across identical hyperparameters, using 2,000 paired cluster-bootstrap tournament rounds to establish statistical significance.

## The names

Each model is named in homage to a historical or cultural figure whose life and work reflect the function of that model.

| Model | Task | Homage | Contribution |
|---|---|---|---|
| Tokenizer | Subword vocabulary | **Mammeri** | Mouloud Mammeri wrote the first Berber grammar written *in* Kabyle (1976), inventing the metalanguage to describe the language in its own words. The 1980 cancellation of his lecture on Kabyle poetry began the Berber Spring |
| Encoder | Bidirectional representation | **Masinissa** | First king of a united Numidia (r. 202–148 BCE), who brought eastern and western territories into a single state |
| Sentence embeddings | Exact passage retrieval | **Si Mohand** | Si Mohand ou Mhand (1848–1905), master of the Kabyle *isefra*, whose oral poetry survived intact through oral recitation and memory |
| Translation | Cross-lingual transfer | **Amrouche** | Taos Amrouche sang exclusively in Kabyle and wrote in French, bridging oral Kabyle traditions and literature |
| Transliteration | Script conversion | **Juba** | King Juba II (48 BCE – 23 CE), ancient Numidian scholar-king and author who studied and documented North African languages and scripts |
| Speech recognition | Speech-to-text | **Fadhma** | Fadhma Aït Mansour Amrouche, Taos's mother, who began transcribing ancestral Kabyle songs and oral histories in the 1930s |
| Punctuation & casing | Discourse structuring | **Belaïd** | Belaïd At Ali (1909–1950), author of *Les Cahiers de Belaïd*, the first continuous narrative prose written in Kabyle Latin orthography |
| Speech synthesis | Voice generation | **Matoub** | Lounès Matoub (1956–1998), celebrated singer and poet who recorded 36 albums in Kabyle, defending the language and culture throughout his life |
| Orthography standardisation | Text normalization | **Boulifa** | Si Amar ou Saïd Boulifa (1865–1931), author of *Recueil de poésies kabyles* (1904), who established early systematic conventions for Kabyle Latin orthography |
| Document OCR | Document recognition | **Feraoun** | Mouloud Feraoun (1913–1962), novelist and educator who transcribed oral poetry onto the printed page |
| Language identification | Sibling discrimination | **Dihya** | Queen Dihya, Amazigh historical leader from the Aurès region (c. 703 CE) |
| Generative foundation | Generative reasoning | **Jugurtha** | King Jugurtha of Numidia (c. 160–104 BCE), grandson of Masinissa, who defended Numidian independence |

The model lineage reflects two historical and family relationships: **Masinissa and Jugurtha** represent the grandfather who unified Numidia and the grandson who defended its sovereignty, linking the foundational encoder and the generative model. **Fadhma and Taos Amrouche** are mother and daughter, reflecting the connection between transcribing spoken oral heritage into text (speech recognition) and translating across languages.

The naming is homage. It implies no endorsement by anyone, and none of these figures or their families are affiliated with this work.

## Licence

Model weights and code are distributed under **Apache-2.0** (unless otherwise noted, such as Matoub-4B's non-commercial research license). Training data licensing distributions are disclosed on each respective model and dataset card.
