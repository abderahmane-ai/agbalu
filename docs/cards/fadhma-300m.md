---
language:
- kab
license: apache-2.0
base_model: ylacombe/omniASR_W2V_300M_SSL
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- automatic-speech-recognition
- speech-to-text
- wav2vec2
- ctc
- low-resource
pipeline_tag: automatic-speech-recognition
metrics:
- wer
- cer
model-index:
- name: Fadhma-300M
  results:
  - task:
      type: automatic-speech-recognition
      name: Kabyle speech recognition
    dataset:
      type: fsicoli/common_voice_22_0
      name: Common Voice 22.0 Kabyle test split (15,003 clips, 888 unseen speakers)
    metrics:
    - type: wer
      value: 25.65
      name: Word error rate (%, beam search with 5-gram shallow fusion)
    - type: cer
      value: 8.01
      name: Character error rate (%, beam search with 5-gram shallow fusion)
    - type: wer
      value: 30.12
      name: Word error rate (%, greedy CTC)
    - type: cer
      value: 8.53
      name: Character error rate (%, greedy CTC)
---

# Fadhma-300M

A 315M-parameter Connectionist Temporal Classification (CTC) acoustic model for **Kabyle** (Taqbaylit, `kab_Latn`), fine-tuned from Meta's Omnilingual ASR self-supervised backbone on 144.31 hours of speech and decoded with a 5-gram KenLM language model trained on 3.2M sentences.

Evaluated on the held-out Common Voice 22.0 test split across 888 unseen speakers (15,003 utterances, 16.00 hours), Fadhma-300M achieves **8.01% character error rate (CER)** and **25.65% word error rate (WER)** with shallow fusion beam search decoding, and **8.53% CER / 30.12% WER** with greedy CTC decoding. Transcripts are generated natively in canonical Kabyle orthography.

## Results

Evaluated on the Common Voice 22.0 Kabyle test split (15,003 utterances, 16.00 hours, 888 disjoint speakers):

| Decoding Method | Character Error Rate (CER) | Word Error Rate (WER) |
|---|---|---|
| **Beam search + 5-gram shallow fusion** | **8.01%** | **25.65%** |
| **Greedy CTC decoding (acoustic only)** | **8.53%** | **30.12%** |

- **Language Model Rescoring:** 5-gram shallow fusion reduces word error rate by 4.47 percentage points and character error rate by 0.52 percentage points.
- **Orthography:** Generates canonical Kabyle Latin characters (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`) directly.
- **Speaker Independence:** The test partition contains zero speaker overlap with the training or validation splits.
- **Hyphenation Impact:** Pronominal and directional clitics in Kabyle attach via hyphens (`-d`, `-awen`, `-iw`). Discrepancies between hyphen and whitespace boundaries account for a substantial share of word errors while character accuracy remains above 91.9%.

## Intended Use

- **Speech Transcription:** Automatic transcription of Kabyle voice notes, audio archives, and spoken dialogue at 16 kHz sample rates.
- **Downstream Pipelines:** Providing transcribed text for translation ([`Amrouche-1.3B`](https://huggingface.co/agbalu/Amrouche-1.3B)), punctuation and casing restoration ([`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M)), or language modeling.
- **Out of Scope:** High-noise audio without prior speech enhancement; clinical, legal, or emergency transcription without human review.

## Usage

### Greedy Decoding

```python
import librosa
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

REPO = "agbalu/Fadhma-300M"
processor = AutoProcessor.from_pretrained(REPO)
model = Wav2Vec2ForCTC.from_pretrained(REPO).eval()

waveform, _ = librosa.load("sample.wav", sr=16_000, mono=True)
inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")

with torch.inference_mode():
    logits = model(inputs.input_values).logits

predicted_ids = logits.argmax(dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
print(transcription)
```

### Beam Search with 5-gram Language Model

```python
import json
import librosa
import torch
from huggingface_hub import hf_hub_download
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ForCTC

REPO = "agbalu/Fadhma-300M"
processor = AutoProcessor.from_pretrained(REPO)
model = Wav2Vec2ForCTC.from_pretrained(REPO).eval()

with open(hf_hub_download(REPO, "vocab.json"), encoding="utf-8") as handle:
    vocabulary = json.load(handle)

labels = [token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
labels = ["" if token == "[PAD]" else " " if token in ("|", "[UNK]") else token for token in labels]

decoder = build_ctcdecoder(labels, kenlm_model_path=hf_hub_download(REPO, "5gram.klm"))

waveform, _ = librosa.load("sample.wav", sr=16_000, mono=True)
inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")

with torch.inference_mode():
    logits = model(inputs.input_values).logits[0].float().numpy()

transcript = decoder.decode(logits, beam_width=100, alpha=0.5, beta=1.5)
print(transcript)
```

**Implementation notes:**
- `pyctcdecode` requires label 0 to be the CTC blank and whitespace as word delimiter, handled by mapping `[PAD]` to empty and `|` to space.
- Cast logits with `.float()` before conversion to numpy when operating in bfloat16.
- Use `AutoProcessor` rather than a bare feature extractor to ensure input audio normalization (`do_normalize: true`) is applied.

## Architecture

| Parameter | Value |
|---|---|
| **Base Model** | [`ylacombe/omniASR_W2V_300M_SSL`](https://huggingface.co/ylacombe/omniASR_W2V_300M_SSL) |
| **Total Parameters** | 315,479,720 (311,285,416 trainable, 4,194,304 frozen frontend) |
| **Encoder Layers** | 24 Conformer/Transformer layers |
| **Hidden / Feed-Forward Dim** | 1,024 / 4,096 (GELU) |
| **Attention Heads** | 16 heads (head dimension 64) |
| **Convolutional Frontend** | 7 convolutions (stride 5,2,2,2,2,2,2; receptive field 20 ms per frame) |
| **Vocabulary** | 40 CTC tokens: `[PAD]`, `[UNK]`, `|`, hyphen, and 36 Kabyle phonemic characters |
| **Loss Function** | CTC Loss (`ctc_zero_infinity=True`) |

## Training Recipe

| Parameter | Value |
|---|---|
| **Dataset** | Mozilla Common Voice 22.0 Kabyle (CC-0) |
| **Optimizer** | AdamW (peak lr 1e-4, $\beta=(0.9, 0.98)$, weight decay 0.01) |
| **Batch Size** | Dynamic micro-batches of 160 audio-seconds, gradient accumulation 4 |
| **Schedule** | 250 warmup steps, cosine decay to 1e-5 |
| **Steps** | 6,568 optimizer steps (~8 epochs) |
| **Precision** | bfloat16 autocast |
| **Regularization** | `mask_time_prob` 0.05, `layerdrop` 0.0 |

Validation progress across training:

| Step | 2,500 | 4,500 | 5,250 | 5,750 | 6,000 | 6,568 |
|---|---|---|---|---|---|---|
| **Validation Loss** | 0.4275 | 0.3599 | 0.3572 | 0.3506 | 0.3425 | **0.3340** |
| **Greedy CER (%)** | 10.83 | 9.49 | 9.24 | 8.97 | 8.67 | **8.53** |
| **Greedy WER (%)** | 36.67 | 33.76 | 32.69 | 31.62 | 30.78 | **30.12** |

## Language Model

`5gram.klm` (186.65 MB binary trie) was constructed from AƔBALU-Text v1 (3,246,174 lines) using KenLM (`--order 5 --prune 0 1 2 3 --skip_symbols --discount_fallback`). Shallow fusion parameters are set to $\alpha=0.5, \beta=1.5$.

## Limitations

- **Speaker Distribution:** Common Voice Kabyle data has 162 training speakers against 888 in the test split; generalisation across diverse speaker characteristics remains the primary source of variance.
- **Punctuation and Casing:** CTC acoustic decoding emits unpunctuated lowercase characters. Downstream punctuation and capitalization can be restored using [`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M).
- **Acoustic Domain:** Benchmarked on clean read speech from desktop and mobile microphones; conversational or reverberant speech may exhibit higher error rates.

## Files

| File | Description | Size |
|---|---|---|
| `model.safetensors` | Acoustic model weights (424 tensors) | ~1.26 GB |
| `config.json` | Model configuration (`Wav2Vec2ForCTC`) | < 5 KB |
| `preprocessor_config.json` | Feature extractor configuration (`do_normalize: true`) | < 1 KB |
| `vocab.json`, `tokenizer_config.json`, `added_tokens.json` | 40-token CTC vocabulary | < 5 KB |
| `5gram.klm` | Binary KenLM language model | 186.65 MB |

## Reproduction

Fine-tuned from `ylacombe/omniASR_W2V_300M_SSL` across 144.31 hours of Common Voice 22.0 Kabyle using AdamW (lr 1e-4, 6,568 steps). KenLM binary built from AƔBALU-Text v1 and tuned on validation data ($\alpha=0.5, \beta=1.5$). Evaluated across the 15,003 test utterances with CER and WER. All reported metrics reproduce deterministically from the published checkpoint.

## The name

**Fadhma Aït Mansour Amrouche** (1882–1967) began, in 1930, writing down the songs and tales
she had inherited from her ancestors — hearing Kabyle and putting it on paper, which is
precisely what this model does.

Her daughter **Taos** names the translation model
[Amrouche-1.3B](https://huggingface.co/agbalu/Amrouche-1.3B), and the pair split the
oral-tradition work the way the two models do: the mother wrote the tradition down, the
daughter carried it into French while singing only in Kabyle.

The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_fadhma_2026,
  title  = {Fadhma-300M: CTC Speech Recognition for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Fadhma-300M},
  note   = {Fine-tuned from omniASR_W2V_300M_SSL on Common Voice 22.0 Kabyle; normaliser 1.3.0+rules1.0.0}
}
```

## Licence

- **Acoustic Weights & Code:** Apache-2.0.
- **Audio & Transcripts:** CC-0 (Mozilla Common Voice).
- **Language Model Corpus:** Derived from AƔBALU-Text v1.
