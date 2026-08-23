# Paper Plan & Architecture

> **Title:** AƔBALU: A Sovereign Spring for Kabyle NLP  
> **Subtitle:** Audited Corpora, Forensic Evaluation, and a Full-Stack Open Ecosystem for Taqbaylit  
> **Venue Target:** LREC-COLING (Language Resource Track) / ACL Resource & Evaluation Track  

---

## 1. Executive Summary & Central Thesis

Mainstream low-resource NLP generally assumes a single axis of scarcity: *"collect more text, train larger models, report higher BLEU."* 

**AƔBALU demonstrates that this paradigm fails for Kabyle (*Taqbaylit*, `kab`, `kab-Latn`) due to three structural crises:**
1. **The Inverted Resource Asymmetry:** Kabyle is speech-rich (**571.29 validated hours** on Common Voice v26.0, ranking **10th of 294 locales globally**) but text-poor (the deduplicated clean web crawl across FineWeb-2 and HPLT-2 is only **13.8M words**, and the total multi-source deduplicated corpus is **34.9M words**). The primary engine for text generation is not web scraping, but speech transcription (ASR) and document digitisation (OCR).
2. **Systemic Measurement and Benchmark Collapse:** 
   - **16.2% of FLORES+ `kab_Latn` references** (326/2,009 sentences) are corrupted by Greek homoglyphs (`ε`, `γ`), enforcing an artificial **97.29 BLEU / 98.93 chrF++ ceiling** that penalises correctly spelled systems.
   - **NLLB's Language Identifier (`lid218e`)** misclassifies **87.2% of Tashelhit, 93.2% of Tarifit, and 95.2% of Central Atlas Tamazight as Kabyle**, corrupting the upstream pipeline that mined **90.1% of all published parallel Kabyle bitext**.
   - Prior community models (e.g. `kabyle-pos-v2`) claimed **94.8%** accuracy on self-projected silver data, but collapse to **63.69%** on the gold `UD_Kabyle-ADPT` treebank—scoring 20 points *below* a simple most-frequent-tag baseline (83.42%).
   - **The King James Bible is in training corpora on both sides** (559/800 sampled verses verbatim), disqualifying scripture-based benchmarks.
3. **Rigorous Methodological Invariance & Controlled Negatives:**
   - Establishing an 81-rule idempotent normalisation standard (1.3.0) that recovers +17.8% to +21.3% token waste across tokenizers.
   - Proving experimentally that likelihood-based subword tokenizers (Unigram/BPE) **cannot factor the Kabyle annexed state (`axxam` $\to$ `wexxam`)** regardless of vocabulary size (8k–48k) or lexicon-seeded initialisation.
   - Delivering a closed suite of **11 specialized models** and **8 decontaminated datasets** covering the complete linguistic pipeline (ASR, TTS, MT, Encoders, Dense Retrieval, Punctuation Restoration, Script Conversion, Orthography Standardisation, and OCR).

---

## 2. Paper Structure & Section Breakdown

```
Abstract
1. Introduction
   1.1 The Kabyle NLP Landscape & The Inverted Asymmetry
   1.2 The Crisis of Unverified Baselines and Corrupted Ground Truth
   1.3 Summary of Contributions
2. Orthographic and Linguistic Foundations
   2.1 Canonical Inventory and the Homoglyph Census (288M Characters)
   2.2 The ţ Resolution: Dallet Spirantisation vs. Spec Misalignment
   2.3 The 81-Rule Idempotent Normaliser & Foreign Proper Noun Guard
   2.4 The Annexed State and Transliteration Irreversibility
3. Corpus Reconstruction, Provenance, and Decontamination
   3.1 AƔBALU-Text v1: 34.9M Words, 42 Sources, and Licence Decomposition
   3.2 AƔBALU-Parallel v1: Forensic Audit of 5.56M Bitext Pairs
   3.3 AƔBALU-Speech v1: 175.5h Clean Speech and the 162 vs 888 Speaker Boundary
   3.4 Positive-Controlled Decontamination Protocol
4. Forensic Auditing of Existing Benchmarks and Pipelines
   4.1 The FLORES+ Audit: 16.2% Homoglyph Rate and the 97.29 BLEU Ceiling
   4.2 Cross-Lingual Confusion in Sibling Berber Language Identification
   4.3 The Gold Treebank Reality: Exposing the 94.8% Silver POS Illusion
   4.4 The Biblical Contamination Trap
5. Tokenization: Evidence, Scaling, and Controlled Negative Results
   5.1 Empirical Tokenizer Comparison: Unigram vs. BPE
   5.2 Lexicon-Seeded Initialisation: Refuting Annexed-State Factorisation
   5.3 Downstream Parameter-Budget Scaling
6. AƔBALU-Bench: Standardised Multi-Task Evaluation
   6.1 Two-Condition Protocol (Raw vs. Normalised)
   6.2 Machine Translation (kab↔eng, kab↔fra, arb→kab)
   6.3 Part-of-Speech Tagging & Morphological Inflection
   6.4 Script Transliteration & Schwa Restoration
   6.5 Cross-Lingual Dense Retrieval & MRL Geometry
   6.6 Speech Recognition (ASR) & Cycle-Consistent Synthesis (TTS)
   6.7 Document OCR & Punctuation Truecasing
7. Sovereign Models Suite
   7.1 Encoder & Representations: Masinissa-31M and SiMohand-278M
   7.2 Sequence Transduction: Amrouche-1.3B, Boulifa-48M, and Juba-27M
   7.3 Speech & Audio: Fadhma-300M and Matoub-82M
   7.4 Document Recognition & Restoration: Feraoun-Pro and Belaid-31M
   7.5 Generative Adaptation: Jugurtha Continued Pretraining Baseline
8. Discussion & Lessons for Low-Resource NLP
   8.1 Why Curation Trumps Crowdsourcing in the Inverted Regime
   8.2 The Necessity of Multi-Condition Evaluation
   8.3 Naming, Cultural Grounding, and Open Governance
9. Conclusion & Reproducibility Statement
References
Appendices (A: Datasheets; B: Normalisation Rules; C: Detailed Error Tables; D: Hardware & Hyperparameters)
```

---

## 3. The Master Results Matrix (The Table that Sells the Paper)

| Task | Model | Params | Primary Metric | Baseline / Floor | AƔBALU Result | Delta / Significance |
|---|---|---|---|---|---|---|
| **Syntax (POS)** | `Masinissa-31M` | 31.1M | UPOS Accuracy (Gold UD Test) | 63.69% (`kabyle-pos-v2`)<br>83.42% (Most-Freq Tag) | **90.51%** (surface)<br>**90.39%** (gold words) | **+26.82%** over published SOTA<br>**+7.09%** over statistical floor |
| **Robustness** | `Masinissa-31M` | 31.1M | UPOS Accuracy (Corrupted Input) | 62.35% (`kabyle-pos-v2`)<br>79.52% (Most-Freq Tag) | **87.42%** | **Surpasses clean baseline** under 7.0% homoglyph injection |
| **MT (eng$\to$kab)** | `Amrouche-1.3B` | 1.3B | chrF++ / BLEU (Corrected Devtest) | 31.53 / 7.74 (NLLB-1.3B) | **36.34 / 10.86** | **+4.81 chrF++ / +3.12 BLEU**; 0.00 orthography gap |
| **MT (kab$\to$eng)** | `Amrouche-1.3B` | 1.3B | chrF++ / BLEU (Corrected Devtest) | 44.60 / 23.06 (NLLB-1.3B) | **46.25 / 25.29** | **+1.65 chrF++ / +2.23 BLEU** |
| **Dense Retrieval** | `SiMohand-278M` | 278M | Recall@1 / MRR (500-way Bitext) | 63.8% / 0.7063 (multilingual-e5)<br>22.2% / 0.2787 (LaBSE) | **97.0% / 0.9833** | **+33.2% R@1**; uncollapses 0.8040 cone to 0.0042 isotropy |
| **MRL Slice (64d)** | `SiMohand-278M` | 278M | Recall@1 / MRR (12× Compressed) | 63.8% (e5 full 768d) | **97.0% / 0.9819** | 12× vector compression with 0.0% R@1 loss |
| **ASR** | `Fadhma-300M` | 315M | CER / WER (888 Unseen Speakers) | 31.88% / 9.44% (CV8 XLS-R)<br>6.2% (Meta 7B LLM-ASR) | **8.01% / 25.65%** | First speaker-disjoint Kabyle ASR; first open CTC result |
| **Punctuation** | `Belaid-31M` | 31.4M | Macro-F1 (CV22 Marks) | 0.227 (No-Model Baseline) | **0.793** (0.898 Question F1) | Restores punctuation/casing for CTC ASR without text edits |
| **Script Conversion** | `Juba-27M` | 26.9M | Exact Match / Schwa F1 | 1.02% / 1.72% (Char Table) | **94.22% / 93.51%** | Solves vowel-less Neo-Tifinagh $\to$ Latin schwa restoration |
| **Standardisation** | `Boulifa-48M` | 47.8M | Char Accuracy / Exact Match | 89.70% / 0.00% (Identity Floor) | **97.39% / 85.70%** | **75% relative error reduction** on SMS/French-keyboard text |
| **Document OCR** | `Feraoun-Pro` | 51.5M | Latin / Tifinagh CER (30k lines) | N/A (First multi-script model) | **7.34% Latin / 0.91% Tfng**<br>81.77% Line Exact Match | Sub-dot consonant preservation + ţ/Ţ slot support |
| **TTS** | `Matoub-82M` | 81.7M | Audio Quality / Cycle-CER | +3.56 Cycle-CER (`mms-tts-kab`) | **24 kHz Native Synthesis** | 42-phoneme Kabyle inventory; Apache-2.0 release |

---

## 4. Key Figures & Architectural Visualisations

1. **Figure 1: End-to-End System Pipeline:** Flowchart illustrating:
   - Multi-source raw ingestion (42 text sources, 21 parallel sources, Common Voice audio)
   - 81-rule normalisation standard & foreign proper noun guard
   - Decontamination boundary
   - Sovereign model training branches (Encoder, MT, ASR, TTS, OCR, Transliteration, Truecasing, Standardisation)
2. **Figure 2: Annexed State Tokenizer Failure:** Visual breakdown comparing tokenization of citation (`axxam`) vs. annexed (`wexxam`) states across Unigram and BPE models, contrasting mathematical likelihood preference (whole word) against morphological segmentation.
3. **Figure 3: Embedding Geometry & Isotropy:** Visualisation comparing the collapsed cone distribution of `multilingual-e5-base` (mean cosine 0.8040, missing 5 Kabyle consonants) against the isotropic hypersphere of `SiMohand-278M` (mean cosine 0.0042).
4. **Figure 4: Speaker-Disjoint Generalisation:** Visualisation of the Common Voice speech split partition (162 training speakers vs. 888 test speakers).

---

## 5. Required Action Items to Finalise Manuscript

1. **LaTeX Setup:** Prepare `paper/main.tex` and section files according to official LREC-COLING/ACL templates.
2. **LaTeX Table Exporter:** Auto-generate publication-grade LaTeX tables from project evaluation artifacts (`data/processed/bench/*.json`).
3. **Drafting Strategy:**
   - Transcribe and formalize Sections 2–6 and 8 from existing documentation.
   - Finalize Sections 1, 7, and 9 with the authoritative sovereign framing.

