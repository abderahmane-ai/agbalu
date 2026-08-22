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
- ctc
pipeline_tag: image-to-text
metrics:
- cer
- wer
- exact_match
model-index:
- name: Feraoun-Pro
  results:
  - task:
      type: image-to-text
      name: Line-level document OCR, Kabyle Latin
    dataset:
      type: agbalu/KabTifinagh
      name: 25,484 held-out rendered lines, Latin
    metrics:
    - type: cer
      value: 0.0734
      name: Character error rate
  - task:
      type: image-to-text
      name: Line-level document OCR, Kabyle Neo-Tifinagh
    dataset:
      type: agbalu/KabTifinagh
      name: 4,516 held-out rendered lines, Neo-Tifinagh
    metrics:
    - type: cer
      value: 0.0091
      name: Character error rate
---

# Feraoun-Pro

A 51M-parameter CTC recogniser that reads a **line of printed Kabyle** (Taqbaylit, `kab`)
from a pixel strip and writes it out as text — in the Berber Latin orthography and in
Neo-Tifinagh, from one checkpoint, without being told which script it is looking at, in a
single forward pass.

It exists because the Kabyle written record is on paper. Novels, grammars, periodicals and
the archive scans that carry a century of prose are images, and general-purpose OCR was
never trained on the language: the emphatic consonants `ḍ ḥ ṛ ṣ ṭ ẓ` are a base letter plus
a sub-dot that other engines drop or normalise away, and dropping the dot changes the word.
`ţ` — Boulifa's original transcription convention — has a slot in this vocabulary. It has
never had one before.

**What it adds over [`agbalu/Feraoun-36M`](https://huggingface.co/agbalu/Feraoun-36M),
which also reads both scripts.** A 173-symbol table against 171 — `ţ` and `Ţ`, Boulifa's
convention, and 33 Neo-Tifinagh codepoints against 31. A single CTC forward pass in place of
one autoregressive decoder step per character. A variable-width canvas up to 1,024 px, so a
long line is no longer squeezed to fit. 600,000 training lines against 80,000. The two
models are separate artifacts on the same task and both stay published; Results says how to
choose.

## Results

**30,000 held-out lines, rendered**, drawn at a fixed seed over the full 600,000-line pool —
85% Latin, 15% Neo-Tifinagh, matching the training mix. The model saw none of these
sentences during training.

| | Latin (`n` = 25,484) | Neo-Tifinagh (`n` = 4,516) |
|---|---|---|
| Character error rate | **7.34%** | **0.91%** |

**Line exact match over all 30,000 lines: 81.77%.**

Three things the table does not say on its own.

**`Feraoun-36M` publishes 2.85% Latin CER, 1.64% Neo-Tifinagh CER and 70.20% exact match,
and none of those is comparable with the figures above.** The two models were scored on
different held-out draws from different line sets, and neither has been run on the other's —
so the pair is two measurements, not a ranking, and this card does not turn it into one.
Choose by page: this model takes a line up to 1,024 px without squeezing it, has `ţ` in its
alphabet, and decodes in one pass; the 36M model is smaller and is the one whose font
generalisation has been tested.

**The Tifinagh number is a retention result, not a difficulty result.** Neo-Tifinagh writes
no sub-dots and no capitals and its lines are shorter, so sub-1% there is not a better score
on the same task — it is an easier task, and the figure's value is that it proves the script
survived training alongside Latin rather than fading.

**Checkpoint selection was gated on both scripts at once.** The combined error
`0.5 · Latin CER + 0.5 · Tifinagh CER` had to improve while neither script regressed past
its baseline. A checkpoint that reads Tifinagh better by forgetting Latin is not selected,
which is the failure this model's whole training design is built around.

## What the numbers cover, and what they do not

Every figure above is measured on rendered lines. What that leaves open, stated so you can
plan around it rather than discover it:

- **Real scanned pages are not in the evaluation.** Training and validation are both
  synthetic renders, as is `Feraoun-36M`'s 2.85%. Performance on Adlis book scans — genuine
  century-old print — is open for both models.
- **The head-to-head against `Feraoun-36M` has not been run**, for the reason above.
- **`ţ`/`Ţ` have vocabulary slots and no per-glyph score.** The validation pool holds too
  few `ţ` lines to report one. The slots are what make the glyph representable at all; no
  other Kabyle OCR system has them.
- **Font generalisation is not quantified.** Training and evaluation draw from the same set
  of container faces, so a typeface the model has never seen is outside what these numbers
  describe. Expect it to be the axis that moves most.

## Architecture

| | |
|---|---|
| Parameters | **51,440,589**, plus 132,036 elements of buffer — the sinusoidal position table and the stem's batch-norm statistics — for 51,572,625 tensor elements in the weights file. The training checkpoint holds 60,158,014: the difference is an 8,585,389-parameter auxiliary decoder that supplies gradients and is not exported |
| Stem | 5-layer convolutional front end, 64 px fixed height, variable width to 1,024 px |
| Encoder | 16-layer pre-norm Transformer, 512 hidden, 8 heads |
| Head | CTC over 173 classes, greedy decode |
| Input | `(1, 3, 64, W)`, pixel range `[−1, 1]`, aspect ratio preserved |
| Vocabulary | 173 classes — 4 special, Latin, Kabyle extended and sub-dot (`ḍ ḥ ṛ ṣ ṭ ẓ`), accented Latin, `ţ`/`Ţ`, digits, punctuation, 33 Neo-Tifinagh |

**No decoder steps, no beam search, no key-value cache.** A line is one forward pass:
convolutional stem, Transformer encoder, linear projection, CTC argmax. The auxiliary
autoregressive decoder existed only to keep gradients reaching the early convolutional
layers, and it is not in the published weights.

**173 classes, not subwords.** A subword tokeniser completes vocabulary words where the ink
is broken, which is precisely the failure an OCR system must not have: it produces fluent
Kabyle that is not what the page says.

## Intended use

Turning printed Kabyle into text: scanned novels, grammars, periodicals and archive
material, in the Latin orthography or in Neo-Tifinagh, from the same model.

**Not suitable for**: handwriting, absent from the training data; multi-column layout, which
`transcribe_page` reads straight across; any decision about a person; or any language other
than Kabyle. Tarifit, Tashelhit and Central Atlas Tamazight share part of this orthography
and are not evaluated here.

## Usage

`transformers`, `torch`, `pillow` and `numpy`, nothing else. The architecture is not one of
the library's own, so the modelling code travels in this repository and
`trust_remote_code=True` is what loads it.

```python
from PIL import Image
from transformers import AutoModel

model = AutoModel.from_pretrained("agbalu/Feraoun-Pro", trust_remote_code=True).eval()

print(model.transcribe(Image.open("line_crop.png")))
# 'Taqbaylit d tutlayt tayemmat nneɣ.'

print(model.transcribe(Image.open("tifinagh_line.png")))
# 'ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ.'
```

`transcribe` does the preprocessing as well as the decoding, and that matters: the model was
fitted on lines scaled to 64 px of height with aspect ratio preserved, so a caller who builds
the tensor themselves at another scale is measuring a different model.

For a full page, `transcribe_page` segments on the horizontal ink profile first. That
segmentation assumes a single column of roughly horizontal lines; it has not been evaluated
on multi-column layouts, marginalia or skewed scans.

## Training data

Lines are **rendered, not scanned.** There is no page of real Kabyle print with
character-level ground truth to train against.

| | |
|---|---|
| Latin text | AƔBALU-Text v1, not publicly released |
| Neo-Tifinagh text | [`agbalu/KabTifinagh`](https://huggingface.co/datasets/agbalu/KabTifinagh) `script_conversion` train split, CC-BY-2.0 |
| Objective | CTC, plus an auxiliary autoregressive decoder whose gradients reach the encoder and which is **not** exported |

The renderer composites text set at 24–38 pt in the faces available in the training container
onto generated paper grain, then degrades by blur, exposure, JPEG artefacting and a slight
skew. Height is fixed at 64 px; width follows the line up to 1,024 px, so glyphs are never
squashed to a square canvas.

**The degradation is deliberately modest**, and the model has not been shown genuinely hard
inputs: no bleed-through from the reverse of the page, no ink starvation, no page-edge
shadow, no show-through from facing pages. Expect real archival scans to be harder than
anything in training.

**The corpus composition is uneven.** AƔBALU-Text v1 is a 42-source mixture and the sources
are not weighted equally. The line set is a seeded draw over the full pool rather than a
prefix, which is what keeps it from being one source, but it does not flatten the source
distribution: a novel's long sentences and a newspaper's headlines are both in the pool and
both outside what a 30,000-line validation draw can fully represent.

## Training recipe

One A10G throughout, 600,000 rendered lines per stage, 570,000 train / 30,000 validation on
a fixed seed, batch 64, one epoch per stage, cosine schedule with 5% warmup.

| stage | mix | peak lr | what it does |
|---|---|---|---|
| 1 | Latin only | `2e-5` | builds the Latin recogniser and the sub-dot distinctions |
| 2 | 60% Latin / 40% Tifinagh | `2e-5` | introduces the 33 Neo-Tifinagh codepoints |
| 3 | 85% Latin / 15% Tifinagh | `5e-6` → `2.5e-7` | settles both scripts; 8,907 steps |

**The staging is the design, not a schedule that happened to be split up.** Introducing
Neo-Tifinagh at 40% costs Latin accuracy — that is where this model's 7.34% Latin CER comes
from, and it is why stage 3 drops the Tifinagh share to a rehearsal fraction and the learning
rate by 4×. The 15% rehearsal is enough to keep the Tifinagh codepoints from fading while the
Latin weights settle. Every stage's checkpoint is gated on the combined two-script error with
a per-script regression ceiling, so no stage can trade one script for the other.

## Limitations

**7.34% Latin CER on synthetic renders is not 7.34% on your documents.** The validation set
is drawn from the same rendered distribution as training. Real scanned pages have ink bleed,
page curvature, show-through and aging that the renderer approximates at modest levels.
Whether the gap is one point or ten is not known.

**Latin character accuracy is the weak axis.** It is the price of dual-script capacity in a
model this size, and it is stated in Results rather than buried here.

**Punctuation after dense glyph clusters is a consistent weakness.** In Neo-Tifinagh,
`ⴰⵣⵓⵍ ⴼⵍⵍ-ⴰⵡⴻⵏ, ⴰⵎⴻⴽ ⵜⴻⵍⵍⴰⵎ?` came back as `ⴰⵣⵓⵍ ⴼⵍⵍ,ⴰⵡⴻⵏ, ⴰⵎⴻⴽ ⵜⴻⵍⵍⴰⵎ!` — the hyphen read
as a comma and the closing `?` as `!`, both at marks that follow glyphs without a full
inter-word space.

**Whitespace boundaries are not reliable.** The validation ground truth is stripped, so
trailing-space errors introduced by the renderer's post-processing do not appear in the CER.
They will appear in your output if the downstream consumer is whitespace-sensitive.

**Page layout is a projection profile, nothing more.** `transcribe_page` finds horizontal ink
bands and crops them. There is no column detection, no reading-order model and no table or
figure handling, so a two-column periodical is read straight across.

## Files

| file | contents |
|---|---|
| `model.safetensors` | the convolutional stem, the 16-layer encoder and the CTC head — 51,440,589 parameters and 132,036 elements of buffer, 206 MB fp32. `from_pretrained` reports no missing and no unexpected keys |
| `config.json` | architecture and the `auto_map` that points at the code below |
| `configuration_feraoun_pro.py`, `modeling_feraoun_pro.py` | the architecture and the preprocessing, importing only `torch`, `transformers`, `pillow` and `numpy` |
| `tokenizer.json`, `tokenizer_config.json` | the 173-symbol table, id for id with the one the model was trained on |

Training cannot be resumed from the published files: the optimizer, scheduler, step and epoch
stay in the source checkpoint on the training volume and are not exported.

## Reproduction

```bash
make modal-ocr-pro RUN=latin      EPOCHS=1 LR=2e-5   RATIO=0.0  LINES=600000
make modal-ocr-pro RUN=dual       EPOCHS=1 LR=2e-5   RATIO=0.40 RESUME=latin
make modal-ocr-pro RUN=feraoun-pro EPOCHS=1 LR=5e-6  RATIO=0.15 RESUME=dual
make modal-ocr-pro-pull RUN=feraoun-pro
make push REPO=feraoun-pro
```

## The name

**Mouloud Feraoun** (1913–1962), the schoolteacher from Tizi Hibel who wrote *Le Fils du
pauvre* and put Si Mohand's oral poetry onto the printed page — the direction this model
reverses. He was assassinated by the OAS on 15 March 1962, three days before the Évian
Accords were signed. The naming is homage and carries no endorsement.

## Citation

```bibtex
@software{agbalu_feraoun_pro_2026,
  title  = {Feraoun-Pro: dual-script CTC document OCR for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Feraoun-Pro},
  note   = {51,572,625 parameters; Latin CER 7.34\%, Neo-Tifinagh CER 0.91\%,
            line exact match 81.77\%, over 30,000 held-out rendered lines}
}
```

## Licence

**Apache-2.0** on the weights and the code. The training text is CC-BY-2.0 on the Tifinagh
side and a mixed corpus of uneven provenance on the Latin side. A permissive grant on the
weights makes no claim about the text behind them, so the corpus composition is stated above
rather than assumed.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle NLP collection.
