---
base_model: agbalu/Matoub-4B
language:
- kab
license: other
license_name: boson-higgs-research-and-non-commercial
license_link: LICENSE
pipeline_tag: text-to-speech
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- text-to-speech
- voice-cloning
- gguf
---

# Matoub-4B (GGUF)

Standalone GGUF weights for [Matoub-4B](https://huggingface.co/agbalu/Matoub-4B),
a reference-conditioned Kabyle speech synthesis model producing 24 kHz mono audio.

## Provided quantizations

| File | Size (decimal GB) | Use |
|---|---:|---|
| `Matoub-4B-q4_k.gguf` | 3.912 | Recommended compact option on the measured development panel |
| `Matoub-4B-q8_0.gguf` | 5.934 | Higher-precision decoder option |
| `Matoub-4B-bf16.gguf` | 9.724 | Native runtime precision baseline |

All formats preserve **all 528 codec tensors in FP32**, with values verified
against the source. Quantization applies to eligible generator tensors. Required
configuration and tokenizers are embedded. No separate codec download is needed.

## Measurements

Measured on one H100 NVL MIG 3g.47gb, four text/reference cases and two seeds per
case. Each request used one decoding attempt, a 600-position limit, temperature
0.8, top-p 0.8 and top-k 30.

| Format | Peak device memory (GB) | RTF on completed outputs | Completed / attempted |
|---|---:|---:|---:|
| BF16 | 9.904 | 0.290 | 5 / 8 |
| Q8_0 | 6.694 | 0.284 | 4 / 8 |
| Q4_K | 4.716 | 0.138 | 5 / 8 |

RTF is synthesis time divided by generated audio duration; lower is faster.
It includes reference processing and waveform decoding, excludes model loading
and quality scoring, and is weighted by successful audio duration. Failed
requests are counted separately. Different completion subsets prevent interpreting
these values as a controlled speedup ratio. Memory is a sampled peak above idle
device usage, not a minimum hardware requirement.

On the four requests completed by both Q4_K and native BF16 (280 normalized
characters), Fadhma-300M CER was 7.14% versus 8.21%; WeSpeaker cosine was 0.542
versus 0.546. Predicted SQUIM quality was 4.180 versus 4.191. This small panel
supports Q4_K as a compact option; it does not prove equivalent quality for all
speakers or texts. Q8_0 is offered for precision preference, not as a measured
quality winner. Q4_0 is excluded after weaker results.

## Usage

Use [audio.cpp](https://github.com/0xShug0/audio.cpp), verified at v0.9.4,
commit `f7ac446fb75e594a6913c846bea6c85b637f6ac0`, with CUDA on SM90.
This speech GGUF requires that runtime's `higgs_audio_tts` family; a generic
text-only GGUF loader is not sufficient. CPU and other accelerators were not
qualified in this release.

```bash
audiocpp_cli --task clon --family higgs_audio_tts \
  --model Matoub-4B-q4_k.gguf --backend cuda \
  --text "Azul fell-awen! Ass-a ad nemmeslay s teqbaylit." \
  --voice-ref voice.wav \
  --reference-text "The exact words spoken in voice.wav." \
  --max-tokens 1500 \
  --session-option higgs_audio_tts.codec_weight_type=f32 \
  --out speech.wav
```

Use a clean single-speaker reference and its exact transcript. The native runtime
has its own decoding path. It does not include the Transformers package's ASR,
speaker verification or multi-candidate selection. The example's larger limit
does not guarantee completion; generation may still repeat or reach the limit.

## Main model and limitations

See [Matoub-4B](https://huggingface.co/agbalu/Matoub-4B) for the Python interface,
intended use, pronunciation limitations and citation. Voice identity and
pronunciation vary across reference recordings. Use voices with permission and
identify synthetic speech appropriately.

## Licence

The weights follow the Boson Higgs TTS 3 Research and Non-Commercial License.
Read the included `LICENSE` and `NOTICE` before use or redistribution.

Boson Higgs TTS 3 is licensed under the Boson Higgs TTS 3 Research and Non-Commercial License, Copyright (c) Boson AI USA, Inc. All Rights Reserved.
