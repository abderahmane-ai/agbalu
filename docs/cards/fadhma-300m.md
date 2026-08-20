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

A 300M-parameter CTC acoustic model for **Kabyle** (Taqbaylit, `kab`, Latin script), trained
on 144.31 hours of read speech and decoded against a 5-gram language model built from three
million normalised Kabyle sentences.

It transcribes Kabyle at **8.01% character error rate** and **25.65% word error rate** on
15,003 held-out utterances from **888 speakers it has never heard**. It is, as far as we can
establish, the first Kabyle speech recognition system published with an error rate measured
on a speaker-disjoint split — and the first CTC result for the language at any size.

Its output is written in canonical Kabyle orthography, because the transcripts it learned
from were repaired before it saw them.

## Results

Common Voice 22.0 Kabyle test split: 15,003 utterances, 16.00 hours, 888 speakers present in
no other split. Hypothesis and reference are reduced by the same normalisation policy before
either is scored.

| decoding | character error rate | word error rate |
|---|---|---|
| **beam search, 5-gram shallow fusion** | **8.01%** | **25.65%** |
| greedy CTC, no language model | 8.53% | 30.12% |

Three things worth reading carefully.

**The language model buys words, not phonemes.** It takes 4.5 points off the word error rate
and 0.5 off the character error rate, which is the shape you should expect: an n-gram cannot
fix a misheard sound, it can only prefer a spelling that is a word. Both rows are reported
because a single number here would hide which half of the system produced it.

**There is no comparison table, deliberately.** Every number in this card was produced by one
harness on this split. No competing system has been run through it, so none appears. Two
figures circulate for Kabyle and neither is comparable: Meta's Omnilingual ASR reports CER
6.2, but that figure lives in `per_language_results_table_7B_llm_asr.csv` — a 7B model with an
LLM decoder, 23× the parameters and the architecture this model deliberately does not use.
No per-language CTC result was published for any language. The honest comparison is
`facebook/omniASR-CTC-300M`: public, Apache-2.0, exactly this size, and unscored on Kabyle by
anyone. Running it costs no training and has not been done. Until it is, treat this model's
error rate as a first measurement rather than a ranking.

**Character error rate and word error rate diverge here for a structural reason.** Kabyle
attaches object pronouns and directional particles with hyphens — `-d`, `-n`, `-awen`, `-iw`.
A hyphen the model writes as a space is one character wrong and two words wrong. At 92%
character accuracy that accounts for a large share of the 25.65%, and it is a property of the
orthography rather than of the acoustics.

## Intended use

Transcribing Kabyle speech: voice notes, oral archives, interviews, broadcast, and speech
input to downstream NLP. The output is canonical Kabyle Latin, so it can be fed to
[Amrouche-1.3B](https://huggingface.co/agbalu/Amrouche-1.3B) or
[Masinissa-31M](https://huggingface.co/agbalu/Masinissa-31M) without a repair step.

**Not suitable for**: any language other than Kabyle; audio with heavy background music or
noise without voice-activity filtering first; or medical, legal or safety-critical
transcription without human review. No safety evaluation of any kind has been performed.

## Why CTC and not an encoder-decoder

This model exists to convert Kabyle's one abundant resource into its scarcest one: the
project has 571 validated hours of speech and roughly 70 million unique tokens of text, and
transcription is what turns the first into the second.

That purpose forbids a decoder. An encoder-decoder carries an internal language model, which
is why Whisper hallucinates fluent text nobody said — and a hallucination here would be
fabricated Kabyle entering a corpus under a provenance record claiming a human spoke it. CTC
emits characters aligned to frames. It can mishear. It cannot invent a sentence.

Whisper is also refuted on measurement, not preference: 6 of the 10 Kabyle-specific letters
have no token in its vocabulary and decode through UTF-8 byte fallback, its fertility on
Kabyle is 3.3600 tokens per word against this project's tokenizer at 1.8837, and at a 3.463 s
mean clip length against a fixed 30 s window, 11.5% of its compute is signal.

## Architecture

Fine-tuned from Meta's Omnilingual ASR SSL encoder
([`ylacombe/omniASR_W2V_300M_SSL`](https://huggingface.co/ylacombe/omniASR_W2V_300M_SSL)),
which lists `kab_Latn` among the 1,668 languages in its pretraining.

| | |
|---|---|
| Total parameters | **315,482,152** |
| Trainable | 311,287,848 |
| Frozen | 4,194,304 — the 7-layer convolutional feature extractor |
| Layers / hidden / heads | 24 / 1,024 / 16 (head size 64) |
| Feed-forward | 4,096, GELU |
| Front end | 7 convolutions, kernel 10,3,3,3,3,2,2, stride 5,2,2,2,2,2,2 — one frame per 20 ms |
| CTC classes | **40**: `[PAD]`, `[UNK]`, `\|`, 36 Kabyle letters and `-` |
| Loss | CTC, `ctc_zero_infinity=True` |

The convolutional front end is frozen, so the acoustic representation learned across Meta's
multilingual pretraining survives contact with 144 hours of one language.

**The 40 classes are derived from the transcripts' own characters — after normalisation, not
before, and the ordering is the whole point.** Common Voice's Kabyle transcripts are raw
contributor text and carry the same homoglyph substitution as every other Kabyle source:
Greek `ε` accounts for **10.6%** of that letter's occurrences. Take the inventory first and
the homoglyph becomes a 41st class, so the model learns to reproduce the corruption at corpus
scale. Gate on the alphabet without normalising and the character is *deleted* instead, which
splits the word in two and makes every word error rate computed over it wrong. Normalising
first avoids both: 40 classes from a 128-character raw inventory, with 1.73% of rows repaired.

Nobody else fine-tuning Kabyle ASR has a reference orthography to check against. That is what
the first four phases of this project bought.

## Training data

**Common Voice 22.0 Kabyle**, CC-0, normalised at `normaliser` **1.3.0+rules1.0.0**.

| split | clips | hours | speakers |
|---|---|---|---|
| train | 152,478 | 144.31 | 162 |
| dev | 15,002 | 15.21 | 135 |
| test | 15,003 | 16.00 | **888** |

Speaker overlap between splits is **zero**, verified on the built splits rather than trusted
from the upstream partition. Every clip passed a CTC feasibility check first — a target longer
than the available acoustic frames makes the loss infinite.

The weights are Apache-2.0 and every clip and transcript is CC-0, so unlike this project's
text models there is no licence composition to disclose: all 182,483 clips are public domain.

## Training recipe

| | |
|---|---|
| Objective | CTC |
| Optimiser | AdamW, lr 1e-4, β (0.9, 0.98), weight decay 0.01 |
| Batch | dynamic micro-batches of **160 audio-seconds**, gradient accumulation 4 |
| Schedule | 250 warmup steps, cosine decay to 1e-5 |
| Steps | **6,568** — 8 epochs |
| Precision | bfloat16 autocast |
| Masking | `mask_time_prob` 0.05, `layerdrop` 0.0 |
| Hardware | one NVIDIA A10G, 24 GiB |
| Seed | 20260813 |

Validation improved monotonically across every evaluation:

| step | 2,500 | 4,500 | 5,250 | 5,750 | 6,000 | **6,568** |
|---|---|---|---|---|---|---|
| dev loss | 0.4275 | 0.3599 | 0.3572 | 0.3506 | 0.3425 | **0.3340** |
| greedy CER | 10.83% | 9.49% | 9.24% | 8.97% | 8.67% | **8.53%** |
| greedy WER | 36.67% | 33.76% | 32.69% | 31.62% | 30.78% | **30.12%** |

**160 audio-seconds is a measured ceiling, not a round number.** At 320 the run died asking
the allocator for 1.95 GiB in a single tensor: wav2vec2's first convolution turns one 16 kHz
channel into 512 channels at 3.2 kHz, and `layer_norm` sits on autocast's fp32 list, so a
budget stated in seconds of audio reaches the allocator expanded 102×. The first step
survived it and the second did not, because AdamW does not allocate its moment buffers until
the first optimizer step — a training loop that survives one step has not proved it fits.

## Language model

`5gram.klm`, 186.65 MB, built from **AƔBALU-Text v1**, the largest clean Kabyle text
corpus assembled — **3,246,174 lines from its 3,041,989 records**, because 204,185 of
them carry an internal newline and `lmplz` reads a sentence per line:

```
lmplz --order 5 --prune 0 1 2 3 --skip_symbols --discount_fallback
build_binary trie 5gram.arpa 5gram.klm
```

`--skip_symbols` is required rather than defensive: the corpus contains lines carrying `<s>`
and `</s>`, and `lmplz` aborts on them. `trie` rather than `probing` because the model is
memory-mapped once per container and array compression is what makes 186 MB affordable beside
the acoustic model.

Those arguments live in `agbalu.speech.lm` as values the test suite asserts on, so what
produced the published binary is checkable rather than a shell line in a document.

Shallow fusion runs at α 0.5, β 1.5 — `pyctcdecode`'s documented defaults, **unswept**. A
sweep needs the dev split decoded once per setting, which is GPU time this has not had, so
the 4.5-point word error reduction above is a floor on what the fusion is worth rather than a
tuned result.

## Usage

Greedy, with `transformers`, `torch` and `librosa` — the architecture is `Wav2Vec2ForCTC`,
one of the library's own, so no `trust_remote_code` is needed:

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

print(processor.batch_decode(logits.argmax(-1))[0])
```

That is **CER 8.53 / WER 30.12**. The 5-gram takes it to 8.01 / 25.65, and needs
`pyctcdecode` and `kenlm`:

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

# `pyctcdecode` reads label 0 as the CTC blank and a literal space as the word delimiter,
# where this vocabulary writes `[PAD]` and `|`.
labels = [token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
labels = ["" if token == "[PAD]" else " " if token in ("|", "[UNK]") else token
          for token in labels]

decoder = build_ctcdecoder(labels, kenlm_model_path=hf_hub_download(REPO, "5gram.klm"))

waveform, _ = librosa.load("sample.wav", sr=16_000, mono=True)
inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
with torch.inference_mode():
    logits = model(inputs.input_values).logits[0].float().numpy()

print(decoder.decode(logits, beam_width=100, alpha=0.5, beta=1.5))
```

Three things above are not stylistic and will produce wrong output if changed.

**The label substitution.** Handed this vocabulary unchanged, `pyctcdecode` emits `[PAD]`
and `|` verbatim in every transcript, and every error rate computed from one is wrong.

**`.float()` before `.numpy()`.** The model runs in bfloat16, which numpy cannot represent:
without it, `numpy()` raises `TypeError: Got unsupported ScalarType BFloat16`.

**`AutoProcessor`, not a bare `Wav2Vec2FeatureExtractor()`.** The published extractor sets
`do_normalize: true`; a default one feeds the model a distribution it never saw and returns
a plausible, worse transcript with nothing raising.

## Limitations

**162 training speakers against 888 in the test split. This is the binding constraint.**
It is a property of the corpus, not of the recipe: Common Voice Kabyle is 571 validated hours
contributed by a small number of people, and no amount of additional training reaches speakers
who are not in it. Wherever this model fails, speaker generalisation is the first place to
look, and the roughly 395 validated hours outside these three splits are what would move it.

**Hyphenated clitics are where character accuracy and word accuracy part company**, as above.

**Read speech only.** Common Voice is read aloud from written prompts. Spontaneous
conversation, overlapping speakers, telephone bandwidth and regional variation are entirely
unbenchmarked, and the register mismatch cuts both ways: the language model was built from
*written* Kabyle, which is not what a prompt reader sounds like either.

**Not evaluated beyond this split.** One corpus, one domain, 15,003 test utterances. Treat
everything else as unmeasured.

**No safety evaluation of any kind** has been performed.

**The output carries no punctuation and no capitals.** The CTC vocabulary is 40 characters and
none of them is a mark, so that is a property of the architecture rather than a shortfall of
the training. [`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) puts both back
— one utterance at a time, which is the shape this model emits.

## Files

| file | description |
|---|---|
| `model.safetensors` | 315,482,152 acoustic parameters |
| `config.json` | the architecture, `Wav2Vec2ForCTC` with a 40-class head |
| `preprocessor_config.json` | the feature extractor, with `do_normalize: true` |
| `vocab.json`, `tokenizer_config.json`, `added_tokens.json` | the 40 CTC classes as a `Wav2Vec2CTCTokenizer`, so `AutoProcessor` resolves |
| `5gram.klm` | the KenLM binary, 186.65 MB |

The training checkpoint, with AdamW's moments and the resume state, is not published. Ask if
you need it.

## Reproduction

```bash
make modal-asr-repack                  # decode the audio once, CPU only
make modal-asr TASK=lm                 # build the 5-gram
make modal-asr-train EPOCHS=8          # train
make modal-asr TASK=evaluate           # score the test split, both decodings
```

The evaluation writes its results to the run's volume rather than to the repository, so the
figures in this card come from that run's own output. Re-running the last command regenerates
them.

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
  title  = {Fadhma-300M: CTC speech recognition for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Fadhma-300M},
  note   = {Fine-tuned from ylacombe/omniASR_W2V_300M_SSL on Common Voice 22.0 Kabyle;
            normaliser 1.3.0+rules1.0.0}
}
```

## Licence

**Apache-2.0** for the weights and code. The audio and transcripts are CC-0 from Mozilla
Common Voice. The language model derives from AƔBALU-Text v1, a third of which has no
resolvable licence — see [Masinissa-31M](https://huggingface.co/agbalu/Masinissa-31M) for that
corpus's composition before redistributing derivatives of the `.klm` binary.
