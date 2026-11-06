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

A 36.3M-parameter Vision-Encoder-Decoder model that transcribes line images of printed **Kabyle** (Taqbaylit, `kab`) into text across both Latin and Neo-Tifinagh scripts within a single model.

Feraoun-36M was developed to transcribe printed Kabyle books, periodicals, and archival documents. Standard multilingual OCR systems frequently omit or misidentify Kabyle emphatic consonants with sub-dots (`ḍ`, `ḥ`, `ṛ`, `ṣ`, `ṭ`, `ẓ`). Evaluated on 1,000 held-out rendered lines across eight unseen typefaces, Feraoun-36M achieves **2.85% character error rate (CER)**, **13.20% word error rate (WER)**, **70.20% line exact match**, and **0.8538 positional diacritic F1**.

## Results

Evaluated on 1,000 held-out rendered text lines drawn from beyond the training corpus prefix:

| Metric | Kabyle Latin (1,000 Lines) | Dual Script: Latin + Neo-Tifinagh (1,000 Lines) |
|---|---|---|
| **Character Error Rate (CER)** | **2.85%** | **1.64%** |
| **Word Error Rate (WER)** | **13.20%** | **7.12%** |
| **Line Exact Match** | **70.20%** | **79.50%** |
| **Positional Diacritic F1** | **0.8538** (P: 0.8416 / R: 0.8664) | — |

- **Positional Diacritic Scoring:** Evaluates emphatic sub-dots and diacritical marks via alignment against exact character positions rather than bag-of-character counts.
- **Font Generalization:** Evaluated across eight typefaces excluded from training (Times New Roman, Times, Arial, Helvetica, Georgia, Baskerville, Courier New, Didot). Training used DejaVu, FreeFont, Liberation, and Noto.
- **Dual-Script Capability:** Neo-Tifinagh lines (which do not employ sub-dots or case distinctions) achieve 1.64% CER and 79.50% line exact match without requiring explicit script prompts.

## Intended Use

- **Document Digitization:** Optical character recognition of scanned Kabyle books, newspapers, articles, and historical documents in Latin and Neo-Tifinagh scripts.
- **Input Granularity:** Optimized for single-line text bands scaled to 52 px usable height. Full pages can be processed via the line-segmentation helper `transcribe_page()`.
- **Out of Scope:** Handwritten manuscripts; complex multi-column layouts without pre-segmentation; languages outside Kabyle.

## Usage

```python
from PIL import Image
from transformers import AutoModel, AutoTokenizer

REPO = "agbalu/Feraoun-36M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()

# Transcribe a cropped line image
line_img = Image.open("line.png")
print(model.transcribe([line_img], tokenizer))
# Output: ['Aḍris n uḥric ɣef tɛeṛṛamt d uẓekka.']

# Transcribe a full page by automatic band segmentation
page_img = Image.open("page.png")
lines = model.transcribe_page(page_img, tokenizer)
for line in lines:
    print(line)
```

The `transcribe()` and `transcribe_page()` helpers scale line images to 52 px height on a 64 × 512 canvas, matching the preprocessing pipeline used during training and evaluation.

## Architecture

| Parameter | Value |
|---|---|
| **Total Parameters** | 36,291,840 |
| **Vision Encoder** | DeiT (12 layers, 384 hidden dimension, patch size 16 over 384 × 384 input) |
| **Encoder Backbone** | Initialized from `microsoft/trocr-small-stage1` |
| **Text Decoder** | 6 Transformer layers, 6 attention heads, 1,536 FFN (GELU) |
| **Positional Embeddings** | Sinusoidal positional embeddings (max 256 tokens) |
| **Vocabulary** | 171 character tokens (Latin, Kabyle sub-dot characters, accents, digits, punctuation, 31 Tifinagh characters) |
| **Output Head** | Untied linear projection (171 × 384) |

A character-level vocabulary is used rather than subwords to prevent hallucinated word completions over broken or low-contrast letterforms.

## Training Data & Methodology

- **Synthetic Document Rendering:** Rendered across 80,000 lines (50% Kabyle Latin, 50% Neo-Tifinagh) using open-source typefaces (DejaVu, FreeFont, Liberation, Noto, Noto Sans Tifinagh).
- **Degradation Pipeline:** Augmented with realistic document artifacts: rotation and skew ($\pm 2^\circ$), defocus blur, sensor grain, ink bleed, photocopy contrast jitter, aged paper background, and bleed-through shadows.
- **Text Sources:** Latin text from AƔBALU-Text v1 (`hf.abdelhaqueidali.kab-latn-tfng`, CC-BY-2.0); Tifinagh text from `agbalu/KabTifinagh`.

## Training Recipe

| Parameter | Value |
|---|---|
| **Dataset Size** | 80,000 rendered text lines (95% train / 5% dev) |
| **Optimizer** | AdamW (peak lr 1.5e-4, weight decay 0.01) |
| **Batch Size** | 32 lines |
| **Schedule** | 5% warmup, cosine decay to 5% peak lr |
| **Gradient Clipping** | Maximum norm 1.0 |
| **Epochs** | 3 epochs |
| **Checkpoint Selection** | Step 6,500 (selected on validation character error rate) |

## Limitations

- **Vocabulary Gaps:** The letter `ţ` (U+0163) was not included in this 171-character vocabulary set; texts containing `ţ` decode it as `<unk>` or `ṭ`.
- **Layout Complexity:** `transcribe_page` uses horizontal projection profiles to segment text lines. It does not perform column parsing or table layout analysis; multi-column documents should be cropped into single columns before processing.
- **Handwriting:** Trained entirely on printed typography; handwritten text is out of domain.

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | 145.2 MB | Model weights (314 tensors, 36,291,840 parameters) |
| `config.json` | < 5 KB | Architecture configuration and `auto_map` |
| `configuration_feraoun.py`, `modeling_feraoun.py` | < 30 KB | Vision-Encoder-Decoder implementation and image pipeline |
| `tokenizer.json`, `tokenizer_config.json` | < 10 KB | 171-symbol character tokenizer |
| `feraoun-v1-heldout.json` | < 50 KB | Evaluation predictions across test typefaces |
| `export.stats.json` | < 5 KB | Export verification and parameter summary |

## Reproduction

Trained for 3 epochs on 80,000 rendered lines with AdamW (lr 1.5e-4, batch size 32). Evaluated on 1,000 held-out lines rendered in unseen fonts using Levenshtein CER, WER, and exact line match metrics. All reported figures reproduce deterministically from the published checkpoint.

## The name

**Mouloud Feraoun** (1913–1962), the schoolteacher from Tizi Hibel who wrote *Le Fils du
pauvre* and put Si Mohand's oral poetry onto the printed page — the direction this model
reverses. He was assassinated by the OAS on 15 March 1962, three days before the Évian
Accords were signed. The naming is homage and carries no endorsement.

## Citation

```bibtex
@software{agbalu_feraoun_2026,
  title  = {Feraoun-36M: Dual-Script Document OCR for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Feraoun-36M},
  note   = {36,291,840 parameters; CER 2.85% over 1,000 held-out rendered lines}
}
```

## Licence

- **Weights & Code:** Apache-2.0.
- **Training Corpus Text:** CC-BY-2.0 (Latin source) and Apache-2.0 (`agbalu/KabTifinagh`).
