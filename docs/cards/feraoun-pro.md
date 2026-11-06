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

A 51.4M-parameter CTC-based optical character recognition model for **Kabyle** (Taqbaylit, `kab`), trained to transcribe single-line document images into text across both Latin orthography and Neo-Tifinagh in a single forward pass.

Feraoun-Pro processes variable-width document strips up to 1,024 px wide without aspect-ratio distortion. It features an expanded 173-character vocabulary that includes historical Kabyle characters (`ţ`, `Ţ`), standard Latin emphatic sub-dots (`ḍ`, `ḥ`, `ṛ`, `ṣ`, `ṭ`, `ẓ`), and 33 Neo-Tifinagh characters. Evaluated on 30,000 held-out rendered lines, Feraoun-Pro achieves **81.77% overall line exact match**, with **7.34% character error rate (CER) on Latin** and **0.91% CER on Neo-Tifinagh**.

## Comparison: Feraoun-36M vs. Feraoun-Pro

- **Vocabulary:** 173 symbols (adds `ţ`/`Ţ` and 2 additional Tifinagh characters) vs. 171 symbols in Feraoun-36M.
- **Decoding Mechanism:** Single-pass non-autoregressive CTC vs. autoregressive vision-encoder-decoder.
- **Input Geometry:** Dynamic variable-width line input up to 1,024 px (preserving aspect ratio) vs. fixed 384 × 384 square interpolation.
- **Training Scale:** 600,000 lines across a 3-stage curriculum vs. 80,000 lines.

## Evaluation Results

Evaluated on 30,000 held-out rendered lines (85% Latin, 15% Neo-Tifinagh):

| Metric | Kabyle Latin (`n` = 25,484) | Kabyle Neo-Tifinagh (`n` = 4,516) | Overall (`n` = 30,000) |
|---|---|---|---|
| **Character Error Rate (CER)** | **7.34%** | **0.91%** | — |
| **Line Exact Match** | — | — | **81.77%** |

- **Dual-Script Training Dynamics:** Neo-Tifinagh text features simpler glyph geometry without sub-dots or case distinctions, reaching 0.91% CER. Checkpoint selection was jointly constrained on both scripts to prevent regression on Latin while learning Tifinagh.
- **Synthetic Benchmark Conditions:** Evaluation was performed on rendered document lines; real historical scans with physical paper aging and severe ink degradation remain to be benchmarked.

## Intended Use

- **Document Recognition:** OCR transcription of scanned books, periodicals, and archival texts in Kabyle Latin and Neo-Tifinagh.
- **Line-by-Line Processing:** Single forward-pass decoding of variable-width text strips up to 1,024 px wide (height normalized to 64 px).
- **Out of Scope:** Handwritten documents; complex multi-column layouts without pre-segmentation.

## Usage

```python
from PIL import Image
from transformers import AutoModel

REPO = "agbalu/Feraoun-Pro"
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()

# Transcribe Latin text strip
latin_img = Image.open("line_crop.png")
print(model.transcribe(latin_img))
# Output: 'Taqbaylit d tutlayt tayemmat nneɣ.'

# Transcribe Neo-Tifinagh strip
tifinagh_img = Image.open("tifinagh_line.png")
print(model.transcribe(tifinagh_img))
# Output: 'ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ.'
```

The `transcribe` function normalizes line height to 64 px and preserves horizontal aspect ratio up to 1,024 px. For full page processing, `transcribe_page` applies horizontal projection profiling to segment lines.

## Architecture

| Parameter | Specification |
|---|---|
| **Parameters (Published)** | 51,440,589 trainable parameters (51,572,625 tensor elements including positional buffers) |
| **Parameters (Training Checkpoint)** | 60,158,014 (includes 8.6M auxiliary autoregressive decoder dropped at export) |
| **Visual Stem** | 5-layer convolutional frontend (64 px height, stride preserving horizontal aspect ratio) |
| **Encoder** | 16-layer pre-norm Transformer encoder (512 hidden dimension, 8 attention heads) |
| **Output Head** | CTC linear projection over 173 classes |
| **Input Shape** | `(1, 3, 64, W)`, normalized to `[-1, 1]`, with $W \le 1,024$ px |
| **Vocabulary** | 173 tokens: 4 special tokens, standard Latin, emphatic sub-dots (`ḍ ḥ ṛ ṣ ṭ ẓ`), `ţ`/`Ţ`, accented Latin, digits, punctuation, and 33 Neo-Tifinagh characters |

## Training Recipe

Trained across a 3-stage curriculum on 600,000 rendered lines per stage (570,000 train / 30,000 validation) using AdamW:

| Stage | Script Mix | Peak LR | Focus |
|---|---|---|---|
| **Stage 1** | 100% Latin | 2e-5 | Establishes Latin character recognition and sub-dot discrimination |
| **Stage 2** | 60% Latin / 40% Tifinagh | 2e-5 | Introduces 33 Neo-Tifinagh glyphs |
| **Stage 3** | 85% Latin / 15% Tifinagh | 5e-6 → 2.5e-7 | Joint refinement and dual-script stabilization (8,907 steps) |

## Limitations

- **Dual-Script Trade-offs:** Adding Neo-Tifinagh in a shared 51M-parameter model increases Latin CER to 7.34% compared to single-script baselines.
- **Dense Glyph Punctuation:** Hyphens or punctuation marks immediately adjacent to dense letter clusters without spacing occasionally misclassify (e.g. hyphen recognized as comma).
- **Layout Requirements:** `transcribe_page` performs horizontal band segmentation and does not handle multi-column layouts, marginal notes, or non-rectilinear text.

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | 206.3 MB | Model weights (51,440,589 parameters + buffers) |
| `config.json` | < 5 KB | Architecture configuration and `auto_map` |
| `configuration_feraoun_pro.py`, `modeling_feraoun_pro.py` | < 30 KB | Model definition and image preprocessing pipeline |
| `tokenizer.json`, `tokenizer_config.json` | < 10 KB | 173-symbol character tokenizer |

## Reproduction

Trained across 3 curriculum stages (600,000 lines each) with AdamW (batch size 64, cosine decay). Evaluated on 30,000 held-out rendered text lines using Levenshtein CER and exact match metrics. All reported figures reproduce deterministically from the published checkpoint.

## The name

**Mouloud Feraoun** (1913–1962), the schoolteacher from Tizi Hibel who wrote *Le Fils du
pauvre* and put Si Mohand's oral poetry onto the printed page — the direction this model
reverses. He was assassinated by the OAS on 15 March 1962, three days before the Évian
Accords were signed. The naming is homage and carries no endorsement.

## Citation

```bibtex
@software{agbalu_feraoun_pro_2026,
  title  = {Feraoun-Pro: Dual-Script CTC Document OCR for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Feraoun-Pro},
  note   = {51,572,625 parameters; Latin CER 7.34\%, Neo-Tifinagh CER 0.91\%, line exact match 81.77\%}
}
```

## Licence

- **Weights & Code:** Apache-2.0.
- **Training Corpus Text:** CC-BY-2.0 (`agbalu/KabTifinagh`) and AƔBALU-Text v1.
