---
language:
- kab
license: apache-2.0
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
- preview
pipeline_tag: text-to-speech
---

# Matoub-82M · Preview

A text-to-speech model for **Kabyle** (Taqbaylit, `kab`), released as a **preview checkpoint**. An 82M-parameter StyleTTS2 fine-tune of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) trained on 21,953 restored Common Voice Kabyle clips from a fifties male speaker. It synthesises 24 kHz speech that reproduces the gemination, spirantisation, emphatics, and pharyngeals of Kabyle phonology in a native speaker voice — twice the sample rate of `mms-tts-kab`, the incumbent Kabyle TTS model, and under a licence that permits commercial use where that one does not.

The work that makes it Kabyle is the front end. Kokoro's token table maps 114 symbols and carries none of `ˤ ʕ ħ` — pharyngealisation, and the letters `ɛ` and `ḥ` — while its G2P is built with `unk=''` and drops a phoneme it cannot represent rather than raising, which would have deleted three consonants from every training target behind a healthy loss curve. The 42-symbol Kabyle inventory was diffed against that table, the three missing symbols assigned to unused embedding rows and trained, the affricate tie bar folded onto `ʧ` and `ʤ`, and the front end made to validate against the vocabulary and raise. That is why the emphatics and pharyngeals survive to the decoder.

Released as a preview: it publishes one voice, and its Cycle-CER is not yet measured.

Named after **Lounès Matoub** (1956-1998), Kabyle singer, poet, and tireless voice of Taqbaylit, who gave his life to its language and culture.

## Results

The baseline for Kabyle TTS is `mms-tts-kab` (Meta's MMS). Cycle-CER measures the acoustic distortion introduced by synthesis: synthesise, transcribe with [`agbalu/Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M), measure character error rate against the original text.

| system | cycle-CER | real-audio control CER | delta |
|---|---|---|---|
| `mms-tts-kab` | 11.89 | 8.33 | +3.56 |
| **Matoub-82M** | *not yet measured* | | |

**That baseline row is itself a result of this project.** `mms-tts-kab` had been published without a Kabyle score by anyone; the +3.56 delta above was measured here on 2026-08-14 over 1,000 held-out non-biblical prompts, against a real-audio control on the same text. Kabyle TTS now has a number to be judged against. Matoub-82M's own Cycle-CER is not yet measured, and this card carries no claim about it until it is.

Three things worth reading carefully.

**The baseline is read scripture at 16 kHz.** `mms-tts-kab` is MMS's per-language VITS checkpoint, trained on New Testament recordings — which typically carry one speaker per language — and it emits 16 kHz under `cc-by-nc-4.0`. Matoub-82M was fine-tuned on Common Voice read speech from a native Kabyle male speaker and emits 24 kHz under Apache-2.0. No listening test has been run between the two, and none is claimed here.

**The training audio has a hard frequency ceiling.** The `kab_male` clips are band-limited at approximately 7.9 kHz -- not 11.5 kHz or 24 kHz -- because the recording conditions for Common Voice Kabyle combined with phone microphones, lossy encoding, and upload artefacts cut the spectral content. The model cannot synthesise what was not in its training data; any evaluation above 7.9 kHz measures silence. This is a property of the Kabyle speech record rather than of this checkpoint — the incumbent synthesises at 16 kHz, so both systems are band-limited, and closing it needs recordings that do not currently exist.

**Diffusion was not trained in this checkpoint.** `lambda_diff: 0.0` in the training config. Passing `beta > 0.0` to the inference function injects Gaussian noise from an untrained sampler directly into the decoder. Use `alpha=0.0, beta=0.0` (pure reference style). This is the correct inference mode for this checkpoint and the one the sample audio was produced with.

## Intended use

Producing spoken Kabyle from text for:

- **Accessibility**: screen readers and audio production for Kabyle-language content.
- **Language learning**: audio for learners studying Taqbaylit.
- **NLP pipeline completion**: the terminal stage of a full Kabyle text pipeline, downstream of [`agbalu/Juba-27M`](https://huggingface.co/agbalu/Juba-27M) (Tifinagh to Latin), [`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) (punctuation and casing), and [`agbalu/Boulifa-48M`](https://huggingface.co/agbalu/Boulifa-48M) (orthography standardisation).

**Not suitable for**: any use requiring speaker consent or biometric match to the source speakers; cloning the voice of any person who has not consented; any decision about a person; any language other than Kabyle. No safety evaluation of any kind has been performed.

## What comes next

This checkpoint establishes the parts that carry forward: a Kabyle phoneme inventory the base model can represent, a G2P front end that raises instead of dropping, a two-stage recipe that trains to a monotone validation curve on 17.9 hours, and an end-to-end path from Kabyle text to 24 kHz audio.

What it does not yet have is a second voice and a perceptual measurement. The `kab_female` Stage 2, the Cycle-CER, and UTMOSv2 and speaker-similarity scoring are the next three, in that order.

This preview stays permanently published. The production voices that follow it are released under their own names as standalone repositories rather than replacing this checkpoint.

## Usage

**Not a `from_pretrained` model.** StyleTTS2 is not a `transformers` architecture and this is
a training checkpoint rather than an export, so the repository ships `inference.py` and that
is the interface. Download the repository and run from inside it:

```bash
pip install torch torchaudio librosa soundfile huggingface_hub
hf download agbalu/Matoub-82M --local-dir Matoub-82M && cd Matoub-82M
```

```python
from inference import MatoubTTS

tts = MatoubTTS.load()
tts.synthesise("Azul fell-awen, amek i telliḍ taṣebḥit-a?", "output.wav")
```

Or from the command line:

```bash
python inference.py --text "Azul fell-awen, amek i telliḍ taṣebḥit-a?" --out output.wav
```

The synthesis pipeline:

1. **G2P** -- converts Kabyle Latin text to IPA, folding affricate tie-bar sequences (`t͡ʃ` -> `ʧ`, `d͡ʒ` -> `ʤ`) to the symbols in Kokoro's token table.
2. **Style extraction** -- a reference clip from the training voice is encoded by `style_encoder` and `predictor_encoder` to produce a 256-dim speaker style vector.
3. **Duration and pitch prediction** -- `bert` (PL-BERT, 12 layers), `bert_encoder`, `predictor`, and `predictor_encoder` predict phoneme durations and F0 contours from the token sequence and style vector.
4. **Waveform decoding** -- the HiFi-GAN `decoder` renders 24 kHz mono audio.

## Architecture

Matoub-82M is a StyleTTS2 model initialised from Kokoro-82M weights and fine-tuned in two stages:

| | |
|---|---|
| Parameters | **82M** (Kokoro base) |
| Base model | hexgrad/Kokoro-82M |
| Vocoder | HiFi-GAN decoder |
| Style encoder | 128-dim acoustic style vector |
| Predictor encoder | 128-dim prosodic style vector |
| Language model | PL-BERT (12 layers) + BERT encoder projection |
| Duration predictor | LSTM + linear projection |
| F0 predictor | JDC pitch extractor |
| Discriminators | MPD + MSD (Stage 1 only) |
| Token table | 178 tokens (Kokoro base), 3 new rows trained for Kabyle phonemes |
| Sample rate | 24 kHz |
| Mel filterbank | 80 bins, f_min 0, f_max 8000, n_fft 2048, hop 300 |

**Stage 1 (multi-speaker)** trains `text_encoder`, `style_encoder`, `decoder`, `mpd`, `msd` on both voices merged under global speaker IDs. It builds the acoustic backbone from the Kokoro base.

**Stage 2 (per-voice)** freezes the Stage 1 acoustic modules and fine-tunes `bert`, `bert_encoder`, `predictor`, `predictor_encoder` -- the language-model and duration stack -- on one voice at a time. It is where Kabyle prosody and phoneme timing are learned.

## Training data

**Corpus:** Common Voice Kabyle, restored arm. Clips were amplitude-normalised, silence-trimmed, and filtered: flat-topped (clipped) samples and zero-energy clips were removed entirely.

| voice | clips | speech hours | mean clip length |
|---|---|---|---|
| `kab_male` | 14,679 | ~12.9 h | 3.73 s |
| `kab_female` | 7,274 | ~5.0 h | 3.11 s |
| **total** | **21,953** | **~17.9 h** | |

**The audio quality of `kab_male` is the binding constraint on this checkpoint.** Inspection of the spectrograms shows the signal cut off at approximately 7.9 kHz, consistent with recording on a smartphone through a codec that discards high frequencies before upload. Of the 3.73 s mean clip, 23% is silence. The clips supervise no pause structure and no multi-sentence prosody: the longest clip is 10.5 s and fewer than 1.2% reach 8 s.

This is not a flaw in the data preparation -- it is a measurement of what Common Voice Kabyle recordings contain, and every claim this model makes about audio quality should be read against it.

## Training recipe

**Stage 1 -- multi-speaker acoustic pretraining:**

| | |
|---|---|
| Voices | `kab_male` + `kab_female` merged, global speaker IDs |
| Train / validation | 20,953 clips / 400 clips |
| Batch size | 4 |
| Max sequence length | 200 frames |
| Hardware | NVIDIA A10G 24 GiB (Modal) |
| Runtime | ~10.47 h |
| Epochs trained | **6** |
| Speed | 1.19 s/step (flat across all epochs) |
| Checkpoint | `epoch_1st_00005.pth` |

Stage 1 validation curve (monotone, decelerating -- the last two epochs bought 0.002 each):

| epoch | validation loss |
|---|---|
| 1 | 0.262 |
| 2 | 0.243 |
| 3 | 0.236 |
| 4 | 0.231 |
| 5 | 0.229 |
| **6** | **0.227** |

**Stage 2 -- per-voice language-model fine-tuning (`kab_male`):**

| | |
|---|---|
| Voice | `kab_male` |
| Train / validation | 14,174 clips / 200 clips |
| Steps per epoch | 3,543 |
| Batch size | 4 |
| Max sequence length | 100 frames |
| Hardware | NVIDIA A10G 24 GiB (Modal) |
| Speed (before joint epoch) | 1.97 s/step |
| Speed (from joint epoch) | 3.66 s/step |
| Epochs trained | **4** (iters 13,944) |
| Checkpoint | `epoch_2nd_00003.pth` |
| Validation loss | **0.3475** |

```bash
make modal-matoub TASK=prepare ARM=restored
make modal-matoub TASK=stage2 ARM=restored VOICE=kab_male EPOCHS=5
```

**Stage 2 (`kab_female`) is not yet published.** The female voice requires a separate Stage 2 run; the checkpoint published here covers only the male voice.

## Limitations

**The `kab_male` recording quality defines the quality ceiling.** Phone microphone recordings at ~7.9 kHz effective bandwidth, with 23% silence per clip and a maximum clip length of 10.5 s, are the training distribution. The model cannot exceed what it was shown. The frequency ceiling is the most consequential limitation: 24 kHz output with nothing above 7.9 kHz is broadband silence from 7.9 kHz upward, and it will be audible on any speaker or headphone that reproduces it.

**Single published voice.** The `kab_female` Stage 2 has not been trained to a publishable checkpoint. The card will be updated when it completes.

**Short-clip corpus.** Mean clip length is 3.73 s (male) and 3.11 s (female), of which 23-43% is silence. The model has not been supervised on multi-sentence prosody, pause structure, or paragraph-level intonation. Long sentences are synthesised phoneme-by-phoneme; paragraph rhythm is not modelled.

**One language.** Trained and evaluated on Kabyle. Tarifit, Tashelhit, Central Atlas Tamazight and Shawiya have related but distinct phonologies; none was tested and none should be assumed.

**No safety evaluation of any kind** has been performed.

## What was not measured

- **Cycle-CER against Matoub-82M.** The evaluation has not been completed; the card will be updated when it is.
- **MOS / UTMOS.** No perceptual evaluation has been performed.
- **Speaker similarity.** No speaker embedding comparison against the source voice has been performed.
- **Long-form synthesis quality.** Degradation on multi-sentence or paragraph-length inputs has not been measured.
- **Female voice.** The `kab_female` Stage 2 checkpoint has not been trained.

## Files

| file | size | description |
|---|---|---|
| `epoch_2nd_00003.pth` | ~1.8 GiB | full training state: 13-module `net` dict + AdamW `opt` + `epoch 3`, `iters 13944`, `val_loss 0.3475` |

The published file includes optimiser state and is resumable. A stripped inference-only export is planned for a future revision.

## Reproduction

```bash
make modal-matoub TASK=pull             # download the checkpoint to artifacts/matoub/
make push REPO=matoub                   # restage and upload to agbalu/Matoub-82M
make infer-matoub TEXT="Azul fell-awen, amek i telliḍ taṣebḥit-a?"
```

Full training reproduction:

```bash
make modal-matoub TASK=prepare ARM=restored
make modal-matoub TASK=stage1 ARM=restored EPOCHS=6
make modal-matoub TASK=stage2 ARM=restored VOICE=kab_male EPOCHS=6
```

## The name

**Lounès Matoub** (1956-1998) was the most celebrated Kabyle singer of the 20th century and one of the fiercest advocates for the survival of Taqbaylit. He recorded 36 albums in Kabyle at a time when the Algerian state was suppressing Berber language and culture, making the language audible to an entire generation. He was assassinated on 25 June 1998, ten days before the Arabisation law he had spent years opposing took effect.

His voice is inseparable from the survival of Kabyle as a spoken language in collective memory. Naming a Kabyle TTS model after him is not metaphor; it is acknowledgment that what this model does -- make the language heard -- is what he spent his life doing.

The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_matoub_2026,
  title  = {Matoub-82M: neural speech synthesis for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Matoub-82M},
  note   = {StyleTTS2 fine-tune of Kokoro-82M on 21,953 restored Common Voice clips; preview}
}
```

## Licence

**Apache-2.0** on the weights and the code. The training data derives from Common Voice Kabyle (CC0); the Kokoro base weights are published under Apache-2.0. A permissive grant on weights makes no claim about the voice recordings they were trained on.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection.
