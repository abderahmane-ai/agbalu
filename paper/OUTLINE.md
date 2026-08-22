# Paper plan

The submission plan for AƔBALU: venue, claim, structure, and the measurements that must
exist before it can be written. Evidence lives in `docs/`; this file says which parts of it
become the paper and what is still missing.

---

## 1. Venue and shape

**LREC-COLING, main resource track.** It is the venue that rewards what this project
actually has — a provenance-tracked corpus, a repaired benchmark, datasheets, licence tiers
and nineteen public artifacts — and where a suite paper is a paper rather than a demo. ACL
and EMNLP main tracks want one sharp claim and treat a resource release as background; that
is the wrong shape for this work and would force most of it into an appendix.

If a second paper is wanted, it is the sharp one: §2's measurement failures alone, four
pages, to an ACL/EMNLP short or Findings. Do not attempt both in one submission.

**Title.** Lead with the resource, not with a stance.

> **AƔBALU: A Provenance-Tracked Corpus, a Repaired Benchmark, and Eleven Models for Kabyle**

"Sovereign AI suite" is a position, and a reviewer cannot check a position. Every word above
is checkable, and the two words a reviewer will stop on — *provenance-tracked* and
*repaired* — are the two contributions no other Kabyle resource can claim. Alternatives, in
order: *AƔBALU: Building Kabyle NLP From the Corpus Up*; *AƔBALU: A Measured Baseline for
Kabyle NLP*.

---

## 2. The claim

Three findings carry the paper, in this order. The models are evidence for them, not the
point.

1. **Kabyle's published measurements are broken, and we measured how.** FLORES+ `kab_Latn`
   is 16.2% corrupt and has never been revised, so every published Kabyle BLEU is scored
   against text that penalises correct spelling — a perfect system scores a bounded maximum
   on it, which we compute. NLLB's own language identifier labels 87.2% of Tashelhit, 93.2%
   of Tarifit and 95.2% of Central Atlas Tamazight as `kab_Latn`, and that identifier mined
   90.1% of the parallel Kabyle data the field trains on. The King James Bible is in the
   training corpus on both sides — 559 of 800 sampled verses verbatim — so no biblical text
   can evaluate anything, which retires a commonly used Berber evaluation register.
2. **Kabyle is not uniformly low-resource, and the asymmetry decides what to build.** 571
   validated speech hours against 13.8M words of filtered web crawl and 34.9M words in the
   whole deduplicated corpus. Speech is the abundant resource and text is the scarce one,
   which inverts the usual low-resource recipe: transcription is the lever, not collection.
3. **A corpus with provenance and licence tiers is buildable, and the tiers are
   load-bearing.** 34.9% of the corpus is under an unclear licence, and `unclear` is the
   absence of a licence rather than a permissive one; a permissive-only rebuild is available
   by code at 973,218 sentences. Dedup order alone decided which licence 949,643 sentences
   were credited to.

Everything else — eleven models, eight datasets — is the demonstration that a corpus built
this way supports a full stack.

---

## 3. Structure

| § | content | source | state |
|---|---|---|---|
| 1 | Introduction: the asymmetry, and the three findings | `docs/contribution.md` | write |
| 2 | Kabyle, and what "low-resource" hides | `CLAUDE.md` §1, `docs/orthography.md` | write |
| 3 | Related work and the survey: 88 sources, 185 datasets, what went offline | `docs/corpus_landscape.md`, `docs/prior_art.md` | ✅ have |
| 4 | **The corpus**: acquisition, provenance schema, normalisation versioning, dedup order, licence tiers, decontamination with positive controls | `docs/phases/`, `resources/corpus_registry.yaml` | ✅ have |
| 5 | **The measurement failures**: the FLORES+ audit, the sibling-LID measurement, the scripture contamination | `docs/phases/phase-07-task-0-flores-audit.md`, `docs/benchmark.md` | ✅ have |
| 6 | **The benchmark**: AƔBALU-Bench protocol, sealed splits, admissibility, per-class support | `docs/benchmark.md` | ✅ have |
| 7 | **The models**, one paragraph each with its floor: encoder, MT, ASR, punctuation, script conversion, embeddings, standardisation, OCR, TTS | the cards | ✅ have, one gap |
| 8 | What was refuted: curriculum learning, vocabulary construction and the annexed state (twice), Whisper for Kabyle, F5-TTS on licence | `docs/tokenizer_design.md` §11, `docs/asr_design.md` | ✅ have |
| 9 | Limitations and ethics: the 90.1% unmeasured adequacy, sibling contamination bounded not cleared, `unclear` licences, the naming as homage | `CLAUDE.md` §6.2 | write |
| 10 | Availability: nineteen Hub repositories, the cards, the reproduction path | | ✅ have |

Every result reported must be the free-running number, and every headline must name its
floor beside it. This project has shipped a teacher-forced figure as an inference figure
twice; a paper is where that becomes permanent.

---

## 4. The table that sells it

One table, models against their floors, because a Kabyle number alone means nothing to a
reviewer who does not know the language.

| task | model | result | floor it is read against |
|---|---|---|---|
| POS | Masinissa-31M | 90.51% | 83.42% most-frequent tag; 63.69% published SOTA |
| MT eng→kab | Amrouche-1.3B | 36.34 chrF++ / 10.86 BLEU | NLLB-1.3B, same harness: 31.53 / 7.74 |
| ASR | Fadhma-300M | CER 8.01 / WER 25.65 | 888 unseen speakers; first speaker-disjoint Kabyle result |
| punctuation | Belaid-31M | 0.793 macro-F1 | 0.227 for the no-model rule |
| script conversion | Juba-27M | 94.22% exact | 1.02% for a character table |
| embeddings | SiMohand-278M | 97.0 R@1 | 63.8 for the backbone |
| standardisation | Boulifa-48M | 97.39% char accuracy | 89.70% for doing nothing |
| OCR | Feraoun | *pending* | *pending* |
| TTS | Matoub-82M | *pending* | `mms-tts-kab` +3.56 Cycle-CER |

**The two `pending` rows are what stands between this project and a submission.** Both are
measurable now and neither is expensive.

---

## 5. Blocking measurements

Nothing else gates the paper. In order of how much a reviewer will care.

1. **OCR on real pages.** Every Kabyle OCR figure in this project is measured against the
   renderer that produced its training data. `docs/ocr_design.md` tasks 1 and 2 fix it, need
   no GPU for the first, and produce the first real-page Kabyle OCR benchmark — which is a
   contribution in its own right, since the published Tifinagh work is isolated-character
   classification on 1,376 images.
2. **Matoub's Cycle-CER.** The baseline is measured and the instrument exists; the run has
   not been made. A TTS row that says "not measured" against a baseline this project itself
   established is the one weak cell in §4.
3. **A head-to-head for the two OCR models.** Falls out of 1.
4. **Task 7.3** — score `omniASR-CTC-300M-v2` and `nvidia/stt_kab_conformer_transducer_large`
   on the CV22 split beside Fadhma. NVIDIA publishes WER 18.86 against Fadhma's 25.65 **on a
   different split**, and a reviewer will find that number. Scoring it ourselves on one split
   is the difference between a limitation and a hole. **Do not carry their published figure
   into the comparison** — score it or leave it out.
5. **The two datasheets still owed** (3.8, 5.6). LREC reviewers read datasheets;
   `docs/datasheets/parallel-v1.md` is the template.

Not blocking, and not to be started for the paper: Phase 11, the cleaning ablation (8.8 has
no instance and the reason is documented), and 7.6 (a sealed split cut from already-public
text cannot back a contamination guarantee).

---

## 6. Writing order

Sections 3–6 and 8 are transcription and compression of documents that already exist and are
already verified; they are cheap and should be drafted first, while the blocking measurements
run. Sections 1, 2, 7, 9 are written last, from the finished numbers.

Do not write a results paragraph before its measurement lands. Prose written about a number
that then changes costs the tokens twice and the first version was noise.
