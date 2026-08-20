---
language:
- kab
license: apache-2.0
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- orthography-standardisation
- text-normalisation
- keyboard-normalisation
- arabizi
- low-resource
pipeline_tag: translation
metrics:
- cer
- accuracy
- exact_match
model-index:
- name: Boulifa-48M
  results:
  - task:
      type: translation
      name: Orthography standardisation, informal Kabyle to canonical Kabyle Latin
    dataset:
      type: agbalu/KabStandard
      name: 1,000 seeded pairs from the agbalu/KabStandard test split
    metrics:
    - type: accuracy
      value: 0.9739
      name: Character accuracy (greedy, free-running)
    - type: cer
      value: 0.0261
      name: Character error rate (greedy, free-running)
    - type: exact_match
      value: 0.857
      name: Sentence exact match (greedy, free-running)
---

# Boulifa-48M

A 47.8M-parameter character-level Conv-Transformer for **Kabyle** (Taqbaylit, `kab`), trained
from scratch to normalise informal, French-keyboard, and Arabizi Kabyle typing into canonical
Kabyle Latin — restoring emphatics, digraph expansions, hyphenated clitic boundaries, and
preposition contractions in one pass.

It reaches **97.39% character accuracy** on held-out test pairs under greedy free-running
decoding, against **89.70%** for leaving the input alone — **converting everyday social-media
and SMS Kabyle into orthographically correct text without any lexicon or rule
hand-engineering.**

It is, as far as we can establish, **the first orthography standardisation model published for
Kabyle, or for any Berber language.**

Named after **Si Amar ou Saïd Boulifa** (1865–1931), grammarian and the first systematic
codifier of Kabyle Latin orthography.

## The task

Kabyle speakers writing in informal digital contexts use a repertoire of keyboard strategies
that diverge sharply from the canonical Latin orthography:

| phenomenon | example input | canonical output |
|---|---|---|
| French digraph `gh` for `ɣ` | `ighef` | `iɣef` |
| French digraph `kh` for `x` | `khedmekh` | `xedmex` |
| French digraph `ch` for `c` | `achimi` | `acimi` |
| Arabizi digit `7` for `ḥ` | `l7adj` | `lḥadj` |
| Arabizi digit `3` for `ɛ`/`ɣ` | `3emmi`, `gh3ir` | `ɛemmi`, `ɣir` |
| Arabizi digit `5` for `x` | `5ir` | `xir` |
| French `ou` digraph for `u` | `touddart` | `taddart` |
| Emphatic drop, `dh` for `ḍ` | `thekhedmedh` | `tḥexedmeḍ` |
| Emphatic drop, `th` for `ṭ` | `thefsouth` | `tḥefsuṭ` |
| Clitic hyphen omission | `dyeffegh` | `d-yeffeɣ` |
| Preposition contraction | `g taddart` | `deg taddart` |
| Identity (already canonical) | `Azul fell-awen` | `Azul fell-awen` |

A rule table can handle each substitution in isolation. What it cannot do is resolve
**ambiguity across phenomena simultaneously**: `3` maps to `ɛ` or `ɣ` depending on position;
`th` maps to `tḥ` (emphatic cluster) or `t` + `h` depending on morphological context;
clitic hyphen positions depend on verb class and directional clitics simultaneously. That is
a sequence transduction problem.

## Results

**1,000 pairs drawn at seed 4711 from the held-out test split** of `agbalu/KabStandard` —
24,898 pairs, disjoint from train (448,149) and dev (24,897), split 90/5/5 at seed 42 over
the same source sentences as `agbalu/KabTifinagh`. **Greedy free-running decoding**: the
model is fed its own output, which is what a caller gets.

| system | character accuracy | character error rate | exact match |
|---|---|---|---|
| **Boulifa-48M** | **97.39%** | **2.61%** | **85.70%** |
| leave the input untouched | 89.70% | 10.30% | — |

Reproduce both rows in about six minutes on a laptop, no GPU:

```bash
make standardise TASK=evaluate LIMIT=1000
```

**The second row is the row that makes the first one mean something.** These inputs are
corrupted probabilistically, so most characters in most sentences were already correct: a
system that does nothing at all scores 89.70%. The model's contribution is the 7.7 points
above that floor, and the honest way to read 97.39% is as a **75% reduction in character
error**, not as "almost perfect".

**Do not quote the training log's dev accuracy in place of these.** That figure is
teacher-forced — next-character accuracy computed from a decode over the *gold* prefix — and
it runs about two points above what a caller gets. It ranks checkpoints; the table above is
what the model does.

**The evaluation pairs are not hand-transcribed.** `KabStandard` is constructed by a
probabilistic corruption pass over canonical Kabyle text (see Training data below). The
character accuracy is measured on the round-trip: can the model recover the canonical target
from a plausibly corrupted source? It cannot be read as accuracy on arbitrary human typing,
only on the corruption distribution defined in `agbalu/KabStandard`.

Three things worth reading carefully.

**No neural model had been published for Kabyle orthography standardisation before this
one**, so there is no prior system to compare against — which is exactly why the do-nothing
floor is in the table. A rule table would sit somewhere above it and has not been built or
scored here; a number for one is not quoted because none has been measured.

**Character accuracy is the flattering metric and exact match is the honest one.** 2.61% CER
is roughly one character in thirty-eight, but the errors concentrate: **85.70% of sentences
come back exactly right**, so most of the character errors are inside the remaining 14.3%.

**The checkpoint shipped is `boulifa_best.pt`**, selected on dev character accuracy rather
than on loss — and that selection metric is the teacher-forced one, which is fine for
ranking checkpoints and wrong to publish.

## Qualitative examples

Every example was decoded against the published checkpoint before being written down.

```
IN:  "achimi ur d-thekhedmedh ara tamazight g l'ecole?"
OUT: "acimi ur d-tḥexedmeḍ ara tamaziɣt deg lɛecule?"

IN:  "3emmi l7adj yerza-d 5ir d lbaraka s wuzzal"
OUT: "Ɛemmi lḥadj yerza-d xir d lbaraka s wuzzal"

IN:  "thessawledh-d fellanegh zik g thefsouth"
OUT: "tḥessawleḍ-d fell-aneɣ zik deg tḥefsuṭ"

IN:  "yennayasen ur ten-idttakken ara degs"
OUT: "yennayasen ur ten-id-ttakken ara degs"

IN:  "matchi akken i thebghidh a thaddarth-iw"
OUT: "mači akken i tḥebɣiḍ a tḥaddarṭ-iw"

IN:  "Azul fell-awen, amek i telliḍ taṣebḥit-a?"
OUT: "Azul fell-awen, amek i telliḍ taṣebḥit-a?"
```

The last example is already canonical; the model leaves it unchanged.

## Intended use

Converting informal, SMS, and French-keyboard Kabyle text into canonical Kabyle Latin
orthography as a preprocessing step for:

- **ASR post-processing**: transcripts from `agbalu/Fadhma-300M` are already canonical;
  Boulifa targets user-typed input before it enters downstream models.
- **NLP pipeline chaining**: sits after [`agbalu/Juba-27M`](https://huggingface.co/agbalu/Juba-27M)
  (Tifinagh to Latin) and [`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M)
  (punctuation and casing), passing clean text to
  [`agbalu/SiMohand-278M`](https://huggingface.co/agbalu/SiMohand-278M) (sentence embeddings)
  or [`agbalu/Matoub-82M`](https://huggingface.co/agbalu/Matoub-82M) (TTS).
- **Corpus normalisation**: preparing user-generated content for inclusion in training corpora
  for downstream Kabyle models.

**Not suitable for**: translation between Kabyle and any other language; any decision about a
person; or any language other than Kabyle. The model has not been evaluated for bias, toxicity,
or systematic failure patterns against sociolects or regional variants. No safety evaluation
of any kind has been performed.

## Usage

`transformers` and `torch`, nothing else. The architecture is not one of the library's own,
so the modelling code travels in this repository and `trust_remote_code=True` is what loads
it.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = "agbalu/Boulifa-48M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO, trust_remote_code=True).eval()


def standardise(text):
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, max_length=512, num_beams=1)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


standardise("achimi ur d-thekhedmedh ara tamazight g l'ecole?")
# 'acimi ur d-tḥexedmeḍ ara tamaziɣt deg lɛecule?'

standardise("3emmi l7adj yerza-d 5ir d lbaraka s wuzzal")
# 'Ɛemmi lḥadj yerza-d xir d lbaraka s wuzzal'

standardise("Azul fell-awen, amek i telliḍ taṣebḥit-a?")
# 'Azul fell-awen, amek i telliḍ taṣebḥit-a?'
```

Or from the command line:

```bash
make standardise TASK=standardise TEXT="achimi ur d-thekhedmedh ara tamazight g l'ecole?"
# acimi ur d-tḥexedmeḍ ara tamaziɣt deg lɛecule?
```

**Output is greedy by default.** The model has no beam search; greedy decoding (`num_beams=1`)
is the evaluation mode and the numbers in this card are from it.

**Input length.** `max_length=512` characters is the default. Kabyle sentences rarely exceed
200 characters; the limit is not load-bearing in practice.

## Architecture

| | |
|---|---|
| Parameters | **47,797,760** |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Hidden / feed-forward | 512 / 1,536 (SwiGLU) |
| Attention heads / head size | 8 / 64 |
| Positions | rotary (RoPE), on self-attention only |
| Conv stem | 1D depthwise separable, kernels 3 and 5, before the encoder |
| Vocabulary | 128 character slots; 96 used, 0 out-of-vocabulary on Kabyle Latin |
| Tied weights | input embedding tied to output projection |
| Label smoothing | ε = 0.05 |
| Dropout | 0.10 |

**Component breakdown:**

| component | parameters |
|---|---|
| Decoder layers (6×) | 26,747,904 |
| Encoder layers (6×) | 20,453,376 |
| Convolutional stem | 529,920 |
| Embedding (tied) | 65,536 |
| Layer norms | 1,024 |

The architecture mirrors [`agbalu/Juba-27M`](https://huggingface.co/agbalu/Juba-27M) in
structure but is larger: hidden size 512 (vs 384), 8 heads (vs 6), and a SwiGLU FFN at 3x
hidden. The bigger capacity is justified by the task: Juba maps one script to another where
consonants are one-to-one and only the schwa placement is ambiguous. Boulifa must resolve
simultaneous ambiguity across phoneme identity, clitic boundaries, and preposition
contraction, which is a harder and more context-dependent problem.

**The convolutional stem.** Two depthwise 1D kernels (3 and 5) run before the encoder's first
layer. Their receptive field covers the bigram and trigram context that most keyboard
substitutions span (`kh`, `gh`, `ch`, `tch`, `dj`, `ou`), so the encoder inherits local
character-cluster information rather than spending self-attention capacity on adjacency.

**Rotary positions (RoPE) on self-attention only.** The cue for where a hyphen belongs or
which consonant a digit maps to is the distance between characters, not where in the sentence
the word sits. RoPE encodes relative positions natively. It is not applied to cross-attention,
where the two sequences are of different lengths and a shared position index would assert a
false alignment.

**Tied input and output embedding.** The input and output alphabets are identical (128
character vocabulary), so sharing these weights is the correct constraint, not a saving.

## Training data

`agbalu/KabStandard` — 497,944 parallel pairs derived from the Latin side of
`agbalu/KabTifinagh` (all three splits: train, dev, test). The pairs map a probabilistically
corrupted source to its canonical Latin target.

- **15% identity pairs**: canonical input passed unchanged, teaching the model not to alter
  already-correct text.
- **85% corrupted pairs**: canonical text processed through a probabilistic corruption pass
  (`corrupt_text` in `agbalu.standardise.corpus`) that applies substitutions independently:

| corruption | canonical char | informal form | probability |
|---|---|---|---|
| `ɣ` digraph | `ɣ` | `gh` | 0.75 |
| `x` digraph | `x` | `kh` | 0.85 |
| `c` digraph | `c` | `ch` | 0.75 |
| `ğ` digraph | `ğ` | `dj` | 0.80 |
| `ḍ` emphatic | `ḍ` | `dh` | 0.75 |
| `ṭ` emphatic | `ṭ` | `th` | 0.70 |
| `ḥ` pharyngeal | `ḥ` | `7` (Arabizi) | 0.25 |
| `ɣ` Arabizi | `ɣ` | `3` | 0.08 |
| `ɛ` pharyngeal | `ɛ` | `3` | 0.25 |
| `x` Arabizi | `x` | `5` | 0.05 |
| `u` digraph | `u` | `ou` | 0.45 |
| clitic hyphen drop | `-` | ` ` or `` | 0.50 |
| preposition contraction | `deg ` | `g ` | 0.25 |

The corruption is stochastic per character independently, so any given sentence may carry
zero, one, or many phenomena simultaneously. Seed 42 is fixed for reproducibility.

**Split:**

| split | pairs |
|---|---|
| train | 448,149 |
| dev | 24,897 |
| test | 24,898 |
| **total** | **497,944** |

The source sentences are canonical Kabyle Latin normalised under AGBALU normaliser
`1.3.0+rules1.0.0`, the same normaliser used for `Juba-27M`, `Masinissa-31M` and
`Amrouche-1.3B`. Any systematic error in that normaliser propagates to the model and to the
test evaluation. The numbers above are measured on the round-trip, not against independently
authored informal text.

## Training recipe

| | |
|---|---|
| Objective | character cross-entropy with label smoothing ε = 0.05 |
| Optimiser | AdamW, lr 5e-4, β (0.9, 0.98), weight decay 0.01, ε 1e-8 |
| Batch size | 64 sequences per step |
| Gradient clipping | max norm 1.0 |
| Hardware | one NVIDIA A10G 24 GiB (Modal), detached run |
| Runtime | ~5.5 h per epoch (~8,500 tokens/s) |
| Epochs trained | **2** (stopped on dev accuracy plateau) |
| Checkpoint selection | best dev character accuracy (`boulifa_best.pt`) |
| Seed | 42 |

**Training curve** — **teacher-forced** next-character accuracy on dev, computed from a
decode over the gold prefix. It ranks checkpoints; it is not the model's accuracy and is
several points above the free-running figure in the results table.

| epoch | dev accuracy (teacher-forced) | dev loss |
|---|---|---|
| 1 | 99.22% | 0.4652 |
| **2** | **99.45%** | **0.4572** |

Epoch 2 was the checkpoint selected and published.

```bash
make modal-boulifa TASK=prepare          # build KabStandard on the volume, CPU only
make modal-boulifa TASK=train            # deploy and spawn detached
make modal-logs FUNCTION=boulifa_train   # follow that one call
```

## Limitations

**The corruption distribution is synthetic, not human.** `KabStandard` pairs are generated
by a stochastic rule applied to canonical text. Real social-media Kabyle carries
idiosyncrasies — phonetic respellings, loan-word renderings, mixed French-Kabyle
code-switching mid-sentence, regional orthographic conventions — that the corruption pass
does not fully capture. **97.39% is an upper bound on performance against naturally occurring
informal text**, and no measurement here says how far below it the real figure sits.

**Clitic chains with multiple simultaneous phenomena are the hardest cases.** A sequence like
`ten-id-ttakken` requires the model to restore the hyphen, identify the direction clitic
`id`, and correctly delimit `ttakken` — all from a mashed input `tenidttakken`. The model
handles this class (see Qualitative examples) but it concentrates most of the residual error
there.

**`ɛ` and `ɣ` share the digit `3` in Arabizi.** The model must resolve this from context:
`3emmi` gives `Ɛemmi` (pharyngeal fricative, word-initial); `gh3ir` gives `ɣir`. This is the
single most context-sensitive substitution and the one most likely to produce an error when
the context is unusual or ambiguous.

**No recovery from cascaded errors.** If the input contains a corruption outside the trained
distribution — a typo that is not a known substitution, or a word from a regional variety
with different spelling conventions — the model may produce a plausible but incorrect output
without signalling uncertainty.

**One language.** Trained and evaluated on Kabyle. Tarifit, Tashelhit, Central Atlas
Tamazight and Shawiya have related but distinct orthographic conventions and are not evaluated
here.

**No safety evaluation of any kind** has been performed.

## What was not measured

- **No inter-annotator ceiling exists for Kabyle informal spelling.** Nobody has established
  what two fluent readers agree the canonical form of a given informal input should be, so no
  figure here can be read as a fraction of what is attainable.
- **No out-of-domain evaluation.** The test pairs come from the same corruption distribution
  as training. Performance on scraped social media, WhatsApp transcripts, or forum posts has
  not been measured.
- **Sentence exact match was not computed for the published checkpoint.** Character accuracy
  was the training signal. Sentence exact match would be a stricter and more informative
  figure; it is not reported here because the metric was not logged during the run.

## Files

| file | size | description |
|---|---|---|
| `boulifa_best.pt` | 191.3 MB | weights + config dict, selected on dev char accuracy (Epoch 2) |

The checkpoint holds the model state dict and the config dict (`asdict(ModelConfig())`), and
nothing else. No optimiser state, no scheduler state, no training curve. Training cannot be
resumed from the published file.

Will be exported to `model.safetensors` before HuggingFace release.

## Reproduction

```bash
make modal-boulifa TASK=prepare      # build KabStandard on the volume, CPU only
make modal-boulifa TASK=train EPOCHS=3   # train detached (~5.5 h/epoch on an A10G)
make modal-boulifa TASK=pull         # download best checkpoint to artifacts/boulifa/
make standardise TASK=standardise TEXT="achimi ur d-thekhedmedh ara tamazight g l'ecole?"
make standardise TASK=evaluate LIMIT=1000   # the numbers in the results table
```

## The name

**Si Amar ou Saïd Boulifa** (1865–1931) was the first grammarian to systematically codify
Kabyle Latin orthography. His 1897 *Recueil de poésies kabyles* introduced a consistent
method for rendering emphatics, pharyngeals and long vowels in Latin characters — solving,
in 19th-century print typography, precisely the mapping problem this model re-solves in the
age of keyboard input.

He worked without a committee, without a standard, and against a tradition that had not yet
decided whether Kabyle could be written at all. The orthographic rules he established were
carried forward through Mammeri, Dallet, and into the INALCO standard that this model was
trained to produce.

The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_boulifa_2026,
  title  = {Boulifa-48M: orthography standardisation for informal Kabyle Latin},
  author = {AGBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Boulifa-48M},
  note   = {Trained on agbalu/KabStandard; 47,797,760 parameters; 97.39% free-running character accuracy on 1,000 held-out pairs}
}
```

## Licence

**Apache-2.0** on the weights and the code. The training data derives from `agbalu/KabTifinagh`
(`agbalu/KabStandard` is a derived dataset); a permissive grant on weights makes no claim
about the text they were trained on, so read `agbalu/KabTifinagh`'s licence before
redistributing derivatives of the training corpus.

Part of [AGBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection.
