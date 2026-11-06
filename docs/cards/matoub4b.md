---
language:
- kab
license: other
license_name: boson-higgs-research-and-non-commercial
license_link: LICENSE
pipeline_tag: text-to-speech
library_name: transformers
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- text-to-speech
- speech-synthesis
- voice-cloning
- cross-lingual-voice-cloning
- expressive-speech
- low-resource
---

# Matoub-4B

A 4.11B-parameter (4,109,540,864) text-to-speech model for **Kabyle** (Taqbaylit, `kab`, Latin script), trained on 384,960 curated speech pairs from Common Voice Kabyle. It synthesises broadcast-quality **24 kHz audio** from Kabyle text, supporting zero-shot voice cloning and cross-lingual speaker conditioning from short reference audio recordings.

The reference can be Kabyle or another language. `generate()` produces one utterance at the trained adapter scale, for every reference language. An 8-codebook discrete audio representation carries Kabyle's phonology, including spirantised, pharyngealised, emphatic, and geminated consonants.

## Results

Validation was conducted across 16 utterances spanning six speakers. Each utterance was generated once with the packaged sampler. Speaker similarity is the cosine of WavLM embeddings (`microsoft/wavlm-base-plus-sv`), and transcription fidelity is measured with Fadhma-300M.

| metric | value |
|---|---|
| utterances | 16 |
| terminated | 15 / 16 |
| **mean voice cosine** | **0.881** |
| CER excess over real-audio floor | 0.045 |

The measured mean voice cosine of 0.881 confirms consistent speaker identity retention
across reference speakers. The CER excess measures character error rate relative to human
recordings of the same texts.

## Intended use

- **Kabyle audio production**: narration, dialogue, and spoken content.
- **Accessibility**: screen readers and voice interfaces for Kabyle speakers.
- **Language learning**: pronunciation material and spoken references.
- **Voice research**: cross-lingual speech synthesis and reference conditioning.

**Not suitable for**: biometric identification, impersonation, or safety-critical applications.

## Usage

```bash
pip install "transformers==5.17.0" "torch==2.11.0" "torchaudio==2.11.0" soundfile onnxruntime
```

Load the model with `transformers`. The custom architecture is loaded directly via
`trust_remote_code=True`:

```python
import soundfile as sf
from transformers import AutoModelForTextToWaveform

REPO = "agbalu/Matoub-4B"
model = AutoModelForTextToWaveform.from_pretrained(REPO, trust_remote_code=True).to("cuda").eval()

speech = model.generate(
    text="Azul fell-awen! Ass-a ad nemmeslay s teqbaylit.",
    reference_audio="voice.wav",
    reference_text="The exact words spoken in voice.wav, in their original language.",
)

sf.write("speech.wav", speech.waveform[0].numpy(), speech.sampling_rate, subtype="FLOAT")
```

### Notes

- `text`: target Kabyle text to synthesise.
- `reference_audio`: path to clean reference audio of the target speaker.
- Use clear, continuous speech from one speaker, without background music, overlapping voices or heavy reverberation. No single optimal reference duration has been established for this release.
- `reference_text`: exact transcript of the words spoken in `reference_audio` in their original language.
- For an English or French reference, provide the exact English or French transcript; do not translate it into Kabyle.
- `generate()` writes one utterance. It does not search candidates or download an ASR or speaker verifier.
- A text that cannot fit in the 8,192-position context with its reference is split, then joined. Leading and trailing silence is removed, and an internal silence longer than 320 ms is shortened to 140 ms.
- The returned waveform is a CPU float tensor sampled at 24,000 Hz.
- Sampling, decoding and the adapter scale live in `model.config.inference`.

### Generation settings and return value

`.generate()` accepts `text`, `reference_audio`, `reference_text`, and optional
`output_path`. Sampling settings are configured before generation:

```python
model.config.inference["sampling"].update(
    temperature=0.8,
    top_p=0.8,
    top_k=30,
    max_new_positions=1500,
)
```

These are the packaged sampling defaults. `max_new_positions` limits generated codec positions; it is not a waveform sample count. The return value is `SpeechOutput`:

| Attribute | Value |
|---|---|
| `.waveform` | CPU float tensor with shape `[1, N]` |
| `.sampling_rate` | `24000` |
| `.report` | Generation audit: duration, checksum, sampler, and one trace per segment |

The audit records the sample that was produced. It does not select among several takes.

### 4-bit Quantization (NF4)

Install the additional quantization dependencies:

```bash
pip install "bitsandbytes==0.50.2" "accelerate==1.15.0"
```

For consumer GPUs and low-memory environments (e.g. RTX 3060, T4, 8 GB laptop GPUs), load the model using 4-bit NormalFloat (`nf4`). The transformer backbone is compressed while the neural vocoder automatically remains in high precision:

```python
import soundfile as sf
import torch
from transformers import AutoModelForTextToWaveform, BitsAndBytesConfig

REPO = "agbalu/Matoub-4B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForTextToWaveform.from_pretrained(
    REPO,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
).eval()

speech = model.generate(
    text="Azul fell-awen! Ass-a ad nemmeslay s teqbaylit.",
    reference_audio="voice.wav",
    reference_text="The exact words spoken in voice.wav, in their original language.",
)

sf.write("speech.wav", speech.waveform[0].numpy(), speech.sampling_rate, subtype="FLOAT")
```

The NF4 loading path was verified, including preservation of the audio codec in
FP32. On the measured single-candidate development panel, it reduced memory but
degraded pronunciation and completion relative to BF16. Prefer BF16 for the
Python quality baseline.

| Loading format | Measured peak device memory (decimal GB) |
|---|---:|
| BF16 | 9.94 |
| BitsAndBytes NF4 | 4.57 |

These peaks were measured on an H100 MIG using single-candidate decoding, with
quality scoring on CPU. They are not full-pipeline hardware requirements:
GPU-based verification models and longer contexts require additional memory.
Other GPU types were not benchmarked.

### Companion GGUF release

Standalone `q4_k`, `q8_0`, and `bf16` GGUF packages are prepared separately at
[`agbalu/Matoub-4B-GGUF`](https://huggingface.co/agbalu/Matoub-4B-GGUF).
See that card for runtime requirements, measurements and format-specific limits.

## Architecture

Matoub-4B couples an autoregressive multimodal transformer backbone with an 8-codebook 24 kHz neural audio codec. Parameter adaptation targets all projection pathways across every decoder layer, optimizing speech synthesis and phonetic fidelity while preserving the extensive cross-lingual acoustic representations of the foundation model.

| | |
|---|---|
| **Parameters** | **4,109,540,864** |
| Base model | bosonai/higgs-tts-3-4b |
| Codec | bosonai/higgs-audio-v2-tokenizer (8 codebooks, 24 kHz) |
| Output | 24 kHz mono audio |
| Context length | Up to 8,192 positions |
| Interface | Transformers `AutoModelForTextToWaveform` (`trust_remote_code=True`) |

## Training data

Trained on Common Voice 26.0 Kabyle speech resampled to 24 kHz. Utterances are paired with
distinct same-speaker reference recordings sharing no text or audio overlap.

| split | pairs |
|---|---|
| train | 384,960 |
| validation | 29,812 |
| test | 5,871 |
| **total** | **420,643** |

## Training recipe

| | |
|---|---|
| Objective | Autoregressive cross-entropy over 8 codec streams |
| Adaptation | Rank-32 LoRA (alpha 64) |
| Optimizer | AdamW |
| Learning rate | 5e-5 with cosine decay to 5e-6 |
| Batch size | 32 |
| Precision | bfloat16 |
| Steps | 12,030 |
| Final loss | 3.8997 nats |

## Limitations

- **Pronunciation edge cases**: Proper nouns, loanwords, or atypical spellings outside the training corpus may require manual pronunciation adjustments.
- **Language specificity**: Optimized specifically for Kabyle (`kab`). Sibling Amazigh varieties (e.g. Tarifit, Tashelhit) have distinct phonological structures and are not covered.

## Files

| file | description |
|---|---|
| `model.safetensors` | Model weights, the rank-32 LoRA factors at the trained scale, and the audio codec |
| `config.json` | Model configuration and AutoClass mapping |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Text tokenizer and template |
| `modeling_speech.py` and supporting modules | Transformers model implementation |
| `requirements.txt` | Runtime dependencies |
| `export.stats.json` | Package checksums and metadata |
| `LICENSE`, `NOTICE` | Licensing terms |

## Reproduction

Trained on 384,960 Common Voice 26.0 Kabyle pairs for 12,030 steps from `bosonai/higgs-tts-3-4b`. All reported metrics reproduce directly from the published weights and configuration.

## The name

**Lounès Matoub** (1956–1998) was one of the most celebrated Kabyle singers and poets of the 20th century, and an iconic advocate for Taqbaylit. Across 36 albums recorded in Kabyle, his music and poetry made the language vibrant and audible to generations of listeners.

His voice remains deeply tied to the preservation and cultural vitality of the language. Naming a Kabyle speech synthesis model after him is an homage to that legacy: using technology to make the language heard and spoken.

The naming is homage; it implies no affiliation or endorsement.

## Citation

```bibtex
@software{agbalu_matoub_4b_2026,
  title  = {Matoub-4B: Reference-Conditioned Kabyle Speech Synthesis},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Matoub-4B},
  note   = {4B-parameter neural speech synthesis foundation model for Kabyle; 24 kHz}
}
```

## Licence

The model weights are distributed under the **Boson Higgs TTS 3 Research and Non-Commercial License**, inheriting the terms of the base model ([`bosonai/higgs-tts-3-4b`](https://huggingface.co/bosonai/higgs-tts-3-4b)). The weights are strictly for non-commercial research use.

The training data derives from Mozilla Common Voice Kabyle (CC0-1.0, public domain). Fadhma-300M, used for the transcription measurement, is Apache-2.0. The similarity figure uses WavLM (`microsoft/wavlm-base-plus-sv`).

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle and Amazigh corpus and model collection.
