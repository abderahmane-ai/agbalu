---
language:
- kab
license: apache-2.0
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- ocr
- image-to-text
- tifinagh
- document-understanding
- low-resource
pipeline_tag: image-to-text
metrics:
- cer
- wer
- exact_match
model-index:
- name: Feraoun-36M
  results:
  - task:
      type: image-to-text
      name: Line-level document OCR, Kabyle Latin
    dataset:
      type: agbalu/AGBALU-Text
      name: 1,000 held-out lines, rendered
    metrics:
    - type: cer
      value: 0.0285
      name: Character error rate
    - type: wer
      value: 0.1320
      name: Word error rate
    - type: exact_match
      value: 0.7020
      name: Line exact match
---

# Feraoun-36M

A 36.3M-parameter Vision-Encoder-Decoder that reads a **line of printed Kabyle**
(Taqbaylit, `kab`) and writes it out as text — in the Latin orthography and in Neo-Tifinagh,
from one model.

It exists because the Kabyle written record is on paper. Novels, grammars, periodicals and
the archive scans that carry a century of prose are images, and no OCR system had been
trained on the language: the emphatic consonants `ḍ ḥ ṛ ṣ ṭ ẓ` are a base letter plus a dot
that general-purpose engines drop or normalise away, and dropping them changes the word.

## Results

**1,000 held-out lines, in text the model never read and typefaces it never saw**, drawn with
a fixed seed from beyond the 80,000-sentence prefix training consumed.

| | |
|---|---|
| Character error rate | **2.85%** |
| Word error rate | **13.20%** |
| Line exact match | **70.20%** |
| Diacritic F1, positional | **0.8538** (P 0.8416 / R 0.8664, over 1,355 gold glyphs) |

Reproduce it without a GPU, in about six minutes on a laptop:

```bash
make ocr TASK=evaluate LINES=1000
```

**Tifinagh is read as well as Latin.** On a second 1,000-line draw, half of it Neo-Tifinagh
from the `agbalu/KabTifinagh` **test** split, the model returns **CER 1.64%, WER 7.12%,
79.50% line exact match** (`make ocr TASK=evaluate LINES=1000 RATIO=0.5`).

**Those are lower numbers and not a better result.** Tifinagh writes no sub-dots and no
capitals, so half of that draw cannot produce the two error classes that dominate the Latin
condition, and its lines are shorter. Read it as evidence that the Tifinagh half works —
`ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ.` comes back exact — not as a lower error rate on the
same task. **The Latin row is the headline.**

Four things the headline table does not say on its own.

**The diacritic score is positional, not a count.** A hypothesis holding the right *number*
of `ḍ` in the wrong places scores 1.0 under a bag-of-characters comparison and 0.0 here: the
metric aligns hypothesis to reference by edit distance and credits a glyph only where it
lands on its own position. The distinction is not academic — it is worth several points on
this model, and the count-based figure should not be compared with this one.

**Word error rate is 4.6× the character rate, and that ratio is the behaviour.** Most
failures are one character inside an otherwise correct word — a lost space, a capital the
model supplies where the source has none — so a 2.85% character rate still fails 13.2% of
words and 29.8% of lines. `Azul, ansuf yis-m!` comes back as `Azul, Ansuf yi s-m!`: **two
character edits, three word edits of three, one line failed.**

**Every typeface in the table is one the model has never seen.** Training renders in DejaVu,
FreeFont, Liberation and Noto; the evaluation renders in Times New Roman, Times, Arial,
Helvetica, Georgia, Baskerville, Courier New and Didot. The numbers are therefore a
generalisation result across letterforms, not a memorisation one — and they are conservative
for it.

**Sub-dot accuracy depends on which of those eight it is.** In Times New Roman and Arial the
model returns `Aḍris n uḥric ɣef tɛeṛṛamt d uẓekka.` exactly. In a high-contrast display face
with hairline serifs the same line loses dots — `tɛeṛṛamt` becomes `tzeṛṛamt`, `uḥric`
becomes `uɣric`. 0.8538 is the average over all eight. On a clean 300 DPI scan in a book
face, expect better; on a photocopy of a display-set title page, expect worse.

**The training log's 2.65% CER is a different statistic and is not comparable.** It is
measured over at most sixteen batches — about 512 lines — of a split held out from the same
prefix the model trained on. It selects checkpoints; the table above is the benchmark.

## Intended use

Turning printed Kabyle into text: scanned novels, grammars, periodicals and archive
material, in the Latin orthography or in Neo-Tifinagh, from the same model and without
telling it which it is looking at.

**A line or a page, not a word.** Every input is scaled to 52 px of usable height, so a
single word blown up to that height is nothing the model has seen.

**Not suitable for**: handwriting, which is absent from the training data entirely;
multi-column layout, which `transcribe_page` reads straight across; any decision about a
person; or any language other than Kabyle. Tarifit, Tashelhit and Central Atlas Tamazight
share part of this orthography and are not evaluated here.

## Usage

`transformers`, `torch`, `pillow` and `numpy`, nothing else. The architecture is not one of
the library's own, so the modelling code travels in this repository and
`trust_remote_code=True` is what loads it.

```python
from PIL import Image
from transformers import AutoModel, AutoTokenizer

REPO = "agbalu/Feraoun-36M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()

model.transcribe([Image.open("line.png")], tokenizer)
# ['Aḍris n uḥric ɣef tɛeṛṛamt d uẓekka.']

print(model.transcribe_page(Image.open("page.png"), tokenizer))   # one line per text band
```

`transcribe` does the preprocessing as well as the decoding, and that matters: the model was
fitted on lines scaled to 52 px of usable height on a 64 × 512 canvas, so a caller who builds
the tensor themselves at another scale is measuring a different model. `prepare_line_image`
and `segment_page_into_lines` are exported from the same module if you need the pieces.

**Feed it a line, or a page — not a cropped word.** A single word blown up to 52 px is
nothing the model has seen.

Every example above was executed against the published directory before being written down,
with this repository absent from the import path. Decoding is greedy and free-running: there
is no beam search and **no key-value cache**, so every step re-reads the whole prefix, which
is what the published evaluation did and what makes the two agree. A few hundred milliseconds
per line on CPU; nothing here needs a GPU.

## Architecture

| | |
|---|---|
| Parameters | **36,291,840** |
| Encoder | DeiT, 12 layers, 384 hidden, patch 16 over a 384 × 384 field |
| Encoder initialisation | `microsoft/trocr-small-stage1` |
| Decoder | 6 layers, 6 heads, 1,536 feed-forward, GELU, pre-norm |
| Positions | sinusoidal, 256 maximum |
| Vocabulary | 171 characters — Latin, Kabyle extended and sub-dot, accented Latin, digits, punctuation, 31 Tifinagh |
| Output projection | 171 × 384, untied |
| Optimiser | AdamW, weight decay 0.01 on non-norm parameters |
| Schedule | cosine to 5% of peak, 5% warmup, peak 1.5e-4 |
| Best checkpoint | step 6,500 of epoch 3 |

**A 64 × 512 line canvas, resampled to the encoder's square input.** 52 px of usable height is
what keeps a sub-dot at four to six distinct dark pixels instead of the one or two it occupies
at 32 px. `extract_features` then interpolates that canvas to the backbone's own 384 × 384 —
576 patches plus the class and distillation tokens, **578 memory positions**. The 8:1 line
aspect ratio is lost at that interpolation, and it is the first thing to change if the decoder
is ever memory-bound.

**A character alphabet, not subwords.** A subword tokenizer completes vocabulary words where
the ink is broken, which is precisely the failure an OCR system must not have: it produces
fluent Kabyle that is not what the page says. 171 classes put the logit surface at 0.25 MB and
make the character error rate a count of this table's own symbols.

## Training data

Lines are **rendered, not scanned.** There is no page of real Kabyle print with
character-level ground truth to train against, so the corpus is text set in the serif and
sans faces of DejaVu, FreeFont, Liberation and Noto — plus Noto Sans Tifinagh, which no
distribution font package carries and which ships in the training repository — and put
through a degradation pipeline: skew up to ±2°, Gaussian defocus, sensor grain, ink bleed
and ribbon fade, photocopier exposure jitter, aged-paper tint, spine-gutter shadow, and
bleed-through from the reverse side.

**The Latin text is one source.** Training read the first 80,000 sentences of AƔBALU-Text v1,
and that prefix is **100% `hf.abdelhaqueidali.kab-latn-tfng`, CC-BY-2.0** — a single
sentence-level dataset, not the 42-source corpus. The Tifinagh half is the
`agbalu/KabTifinagh` train split. **The held-out lines in the results table come from the
same source**, further down the same file: they are text the model has not read, in a
register it has. Nothing here measures generalisation to a different kind of Kabyle prose,
and a novel's long sentences and a newspaper's headlines are both outside what was tested.

## Training recipe

| | |
|---|---|
| Corpus | 80,000 rendered lines, 50/50 Latin and Neo-Tifinagh |
| Split | 95/5, drawn at a fixed seed over the whole set rather than off the end |
| Epochs | 3 |
| Batch | 32 lines |
| Optimiser | AdamW, weight decay 0.01 on non-norm parameters |
| Schedule | cosine to 5% of peak, 5% warmup, peak 1.5e-4 |
| Gradient clipping | 1.0, checked before the step rather than after |
| Selection | best character error rate on the held-out split |
| Hardware | one A10G |

## Limitations

**`ţ` (U+0163) has no slot in the vocabulary, and it is real Kabyle.**
`docs/orthography.md` attests it 21,058 times in this project's own corpus. It encodes as
`<unk>` and cannot be produced: `Aţan yeţţaḍsa, ţ-ţaqbaylit.` comes back as
`Aṭan yeaḍsa, -aqbaylit.` in every typeface tested. Adding the letter changes the width of
the output projection, so it costs a retrain, and until then any page using that convention
is unrecoverable at those positions.

**Casing and spacing are the dominant error class**, not the diacritics. Word-initial
capitals are inserted where the source has none, and word boundaries are split or merged.

**Page layout is a projection profile, nothing more.** `transcribe_page` finds horizontal
ink bands and crops them. There is no column detection, no reading-order model and no table
or figure handling, so a two-column periodical is read straight across.

## Files

| file | contents |
|---|---|
| `model.safetensors` | 314 tensors, 36,291,840 parameters |
| `config.json` | the architecture, and the `auto_map` that points at the code below |
| `configuration_feraoun.py`, `modeling_feraoun.py` | the architecture and the preprocessing in code, importing only `torch`, `transformers`, `pillow` and `numpy` |
| `tokenizer.json`, `tokenizer_config.json` | the 171-symbol table, id for id with the one the model was trained on |
| `feraoun-v1-heldout.json` | the evaluation behind the results table: every hypothesis's conditions, the seed, and the typefaces it was rendered in |
| `export.stats.json` | what the source checkpoint held, and what the export dropped |

`pos_encoder.pe` is in `model.safetensors` although it is derived. It is a registered
buffer, so `from_pretrained` allocates it empty and fills it from the file; dropped as
derived it would come back as uninitialised memory and rotate every position by a garbage
angle without raising.

The source checkpoint holds the weights, the optimizer, the scheduler, the step, the epoch
and the best CER. `export.stats.json` records that, and records the optimizer and scheduler
as dropped — 435 MB down to 146 MB. Training cannot be resumed from the published files.

## Reproduction

```bash
make modal-upload TASK=ocr             # corpus and Tifinagh split onto the data volume
make modal-ocr TASK=smoke              # 20 steps on a real GPU before anything is paid for
make modal-ocr TASK=train RATIO=0.5 LINES=80000 EPOCHS=3 LR=1.5e-4
make modal-ocr-pull                    # the checkpoint back into artifacts/runs/
make ocr TASK=evaluate LINES=1000      # the results table, on CPU
```

## The name

**Mouloud Feraoun** (1913–1962), the schoolteacher from Tizi Hibel who wrote *Le Fils du
pauvre* and put Si Mohand's oral poetry onto the printed page — the direction this model
reverses. He was assassinated by the OAS on 15 March 1962, three days before the Évian
Accords were signed. The naming is homage and carries no endorsement.

## Citation

```bibtex
@software{agbalu_feraoun_2026,
  title  = {Feraoun-36M: dual-script document OCR for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Feraoun-36M},
  note   = {36,291,840 parameters; CER 2.85% over 1,000 held-out rendered lines}
}
```

## Licence

**Apache-2.0** on the weights and the code. The training text is CC-BY-2.0 and a permissive
grant on weights makes no claim about the text behind them, so the composition is stated
above rather than assumed. The Tifinagh half of the training corpus is `agbalu/KabTifinagh`;
read its licence before redistributing derivatives.
