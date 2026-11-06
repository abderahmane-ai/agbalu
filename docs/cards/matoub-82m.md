---
language:
- kab
license: apache-2.0
library_name: transformers
base_model: hexgrad/Kokoro-82M
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- text-to-speech
- speech-synthesis
- styletts2
- low-resource
pipeline_tag: text-to-speech
---

# Matoub-82M

> [!NOTE]
> **Codec-Based Successor Available:** [`agbalu/Matoub-4B`](https://huggingface.co/agbalu/Matoub-4B) is a 4.2B-parameter autoregressive codec-language model offering cross-lingual voice cloning, selectable reference voices, and higher acoustic fidelity. Matoub-82M remains maintained for lightweight, single-speaker StyleTTS2 CPU and edge inference.

An 81.7M-parameter neural text-to-speech model for **Kabyle** (Taqbaylit, `kab`), fine-tuned from [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) on 21,953 restored Common Voice clips from a male native speaker.

Matoub-82M synthesizes 24 kHz audio reproducing the distinctive phonology of Kabyle, including gemination, spirantization, emphatic consonants, and pharyngeal fricatives. The front-end G2P maps Kabyle Latin orthography into a verified 42-symbol International Phonetic Alphabet (IPA) inventory, ensuring faithful acoustic realization.

## Baseline Benchmark

Evaluated against Meta's `mms-tts-kab` baseline using Cycle-CER (acoustic distortion measured by transcribing synthetic audio with [`agbalu/Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M) against reference prompt text):

| Model | Architecture | Sample Rate | License | Cycle-CER | Real-Audio Control CER | Delta |
|---|---|---|---|---|---|---|
| `mms-tts-kab` | VITS | 16 kHz | CC-BY-NC-4.0 | 11.89% | 8.33% | +3.56% |
| **Matoub-82M** | StyleTTS2 | 24 kHz | Apache-2.0 | *Pending run* | — | — |

- **Sample Rate & License:** Matoub-82M outputs 24 kHz audio under a permissive Apache-2.0 license, compared to 16 kHz under non-commercial terms for MMS.
- **Phoneme Coverage:** The base Kokoro vocabulary was expanded with 3 dedicated embedding rows for Kabyle pharyngeal symbols (`ˤ`, `ʕ`, `ħ`), preventing missing consonant deletions during grapheme-to-phoneme conversion.

## Intended Use

- **Speech Synthesis:** Text-to-speech generation for Kabyle accessibility tools, educational applications, and multimedia screen readers.
- **Pipeline Integration:** Terminal stage for Kabyle NLP pipelines, accepting output from transliteration ([`Juba-27M`](https://huggingface.co/agbalu/Juba-27M)), casing restoration ([`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M)), or orthography standardisation ([`Boulifa-48M`](https://huggingface.co/agbalu/Boulifa-48M)).
- **Out of Scope:** Voice cloning without consent; synthesis of languages outside Kabyle.

## Usage

```bash
pip install "transformers>=5" torch soundfile
```

```python
import soundfile as sf
from transformers import AutoModelForTextToWaveform, AutoTokenizer

REPO = "agbalu/Matoub-82M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForTextToWaveform.from_pretrained(REPO, trust_remote_code=True).eval()

text = "Azul fell-awen, amek i telliḍ taṣebḥit-a?"
inputs = tokenizer(text, return_tensors="pt")
audio = model(**inputs).waveform[0]

sf.write("output.wav", audio.numpy(), model.config.sampling_rate)
```

**Synthesis Pipeline:**
1. **Grapheme-to-Phoneme (G2P):** Converts Kabyle Latin text to phonetic IPA representations with contextual assimilation and spirantization rules.
2. **Prosody & Duration:** PL-BERT (12 layers) and an LSTM duration predictor compute phoneme durations, F0 pitch contours, and energy conditioned on the speaker style vector.
3. **Acoustic Decoding:** iSTFTNet decodes the aligned representations into 24 kHz mono waveforms.

## Architecture

| Parameter | Specification |
|---|---|
| **Total Parameters** | 81,731,256 (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`) |
| **Base Architecture** | StyleTTS2 initialized from `hexgrad/Kokoro-82M` |
| **Vocoder** | iSTFTNet (HiFi-GAN upsampling with inverse-STFT head, 20-point FFT, hop size 5) |
| **Speaker Embedding** | Single 256-dimensional style vector (128 acoustic, 128 prosodic) |
| **Language Model** | PL-BERT (12 layers) |
| **Duration Predictor** | LSTM with linear projection (max 50 frames per phoneme) |
| **Token Table** | 178 tokens (expanded to include Kabyle pharyngeal phonemes) |
| **Sample Rate** | 24,000 Hz |

Training-only modules (discriminators, WavLM loss, and diffusion samplers) were removed from the exported weights, reducing model size to 327 MB.

## Training Data

Common Voice Kabyle (restored partition): amplitude-normalized, silence-trimmed, and filtered against clipping:

| Voice Partition | Clean Clips | Total Speech Duration | Mean Clip Length |
|---|---|---|---|
| `kab_male` (Selected voice) | 14,679 | ~12.9 hours | 3.73 s |
| `kab_female` | 7,274 | ~5.0 hours | 3.11 s |
| **Total** | **21,953** | **~17.9 hours** | — |

## Training Recipe

Trained in two sequential stages:

- **Stage 1 (Multi-Speaker Acoustic Pretraining):** Jointly trained `text_encoder`, `decoder`, and duration predictor across all 20,953 training clips (batch size 4) for 6 epochs, stabilizing acoustic representation.
- **Stage 2 (Single-Speaker Adaptation):** Fine-tuned the language model (`bert`, `predictor`) on 14,174 `kab_male` clips (batch size 4) for 4 epochs (13,944 iterations), reaching a final validation loss of 0.3475.

## Limitations

- **Source Audio Bandwidth:** Common Voice recordings exhibit a physical acoustic cutoff around ~7.9 kHz due to smartphone recording hardware and lossy compression; output above this ceiling reflects silent upper harmonics.
- **Prosodic Scope:** Trained on short read-speech clips (mean duration 3.73 s); long-form paragraph prosody is synthesized sentence-by-sentence.
- **Speaker Selection:** Shipped with one male voice (`kab_male`).

## Files

| File | Size | Description |
|---|---|---|
| `model.safetensors` | ~327 MB | Exported inference weights (81,731,256 parameters) |
| `config.json` | < 5 KB | Architecture configuration and `auto_map` |
| `configuration_matoub.py`, `modeling_matoub.py`, `istftnet.py` | < 50 KB | StyleTTS2 modeling and vocoder code |
| `tokenization_matoub.py`, `tokenizer_config.json`, `vocab.json` | < 25 KB | Kabyle G2P rules and 178-token vocabulary |

## Reproduction

Trained in two stages on restored 24 kHz Common Voice Kabyle clips: Stage 1 across merged voices for 6 epochs, followed by Stage 2 adaptation on the `kab_male` voice for 4 epochs (13,944 steps) reaching validation loss 0.3475. All inference results reproduce deterministically with fixed random seeds.

## The name

**Lounès Matoub** (1956-1998) was the most celebrated Kabyle singer of the 20th century and one of the fiercest advocates for the survival of Taqbaylit. He recorded 36 albums in Kabyle at a time when the Algerian state was suppressing Berber language and culture, making the language audible to an entire generation. He was assassinated on 25 June 1998, ten days before the Arabisation law he had spent years opposing took effect.

His voice is inseparable from the survival of Kabyle as a spoken language in collective memory. Naming a Kabyle TTS model after him is not metaphor; it is acknowledgment that what this model does -- make the language heard -- is what he spent his life doing.

The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_matoub_2026,
  title  = {Matoub-82M: Neural Speech Synthesis for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Matoub-82M},
  note   = {StyleTTS2 fine-tune of Kokoro-82M on 21,953 Common Voice Kabyle clips; 24 kHz}
}
```

## Licence

- **Weights & Code:** Apache-2.0.
- **Base Model:** Kokoro-82M (Apache-2.0).
- **Training Audio:** Common Voice Kabyle (CC-0).
