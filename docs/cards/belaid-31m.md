---
language:
- kab
license: apache-2.0
base_model: agbalu/Masinissa-31M
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- punctuation-restoration
- truecasing
- token-classification
- asr-post-processing
- low-resource
pipeline_tag: token-classification
library_name: transformers
metrics:
- f1
model-index:
- name: Belaid-31M
  results:
  - task:
      type: token-classification
      name: Punctuation and casing restoration
    dataset:
      type: fsicoli/common_voice_22_0
      name: Common Voice 22.0 Kabyle, decontaminated test split (5,160 sentences)
    metrics:
    - type: f1
      value: 0.793
      name: Macro-F1 over the four marks and NONE
    - type: f1
      value: 0.933
      name: F1, capitalisation of non-initial proper nouns
    - type: accuracy
      value: 0.863
      name: Exact match, marks and casing, whole sentence
  - task:
      type: token-classification
      name: Punctuation and casing restoration, out of domain
    dataset:
      type: imsidag/kabyle-corpus-hca
      name: Long-form Kabyle prose held out of training (45,028 records)
    metrics:
    - type: f1
      value: 0.635
      name: Macro-F1 over the four marks and NONE
---

# Belaid-31M

**Punctuation and capitalisation for Kabyle** (Taqbaylit, `kab`, Latin script). It reads an
unpunctuated, uncased Kabyle sentence — the form every speech recogniser produces — and
predicts, for each word, which mark follows it and whether it takes a capital.

It restores marks at **0.793 macro-F1** and non-initial proper nouns at **0.933**, and it
returns **86.3% of held-out sentences exactly right**, marks and capitals both, against 59.2%
for the rule that a system without a model uses. It is, as far as we can establish, **the first
punctuation and casing restoration model published for Kabyle, or for any Berber language.**

It is built for [`agbalu/Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M), whose CTC
vocabulary holds 40 characters and no punctuation at all, and it is a fine-tune of
[`agbalu/Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M).

**It never edits spelling.** Words come back exactly as they went in, so it cannot introduce a
transcription error into text the recogniser got right — the failure mode that rules out a
generative post-processor for a language whose corpus this output feeds.

## Results

Scored against a rule baseline that capitalises the first word and appends a period. On read
speech that is not a weak opponent: it gets the sentence-final mark right 83.7% of the time,
because most sentences do end in a full stop.

### Common Voice test — 5,160 sentences, 26,969 words

| | baseline | **Belaid-31M** |
|---|---|---|
| **macro-F1 over marks** | 0.227 | **0.793** |
| sentence-final mark accuracy | 0.837 | **0.970** |
| exact match, whole sentence | 0.592 | **0.863** |
| non-initial proper nouns, F1 | 0.000 | **0.933** |

| label | support | predicted | P | R | F1 |
|---|---|---|---|---|---|
| `NONE` | 20,904 | 20,897 | 0.989 | 0.988 | 0.988 |
| `COMMA` | 804 | 843 | 0.680 | 0.713 | 0.696 |
| `PERIOD` | 4,357 | 4,351 | 0.978 | 0.976 | 0.977 |
| `QUESTION` | 861 | 838 | 0.911 | 0.886 | 0.898 |
| `COLON` | 43 | 40 | 0.625 | 0.581 | 0.602 |

Question marks are the clearest result: **0.898 F1 where the baseline scores 0.000**, because
Kabyle marks interrogatives lexically — `acu`, `anida`, `amek`, `melmi` — and the encoder reads
them. Capitalisation of names inside a sentence goes the same way, 0.933 from a floor of zero.

**`COLON`'s 0.602 rests on 43 examples and carries one fifth of the macro.** On the
out-of-domain split's 13,544 it is 0.721. Quote the macro with that support named, or quote the
four-class figure.

### Out of domain — 45,028 records of long-form prose, 1,000,928 words

A source held out of training entirely, and a different shape of text: Common Voice records are
one sentence each, these average 22 words across several.

| | baseline | **Belaid-31M** |
|---|---|---|
| **macro-F1 over marks** | 0.168 | **0.635** |
| sentence-final mark accuracy | 0.925 | **0.955** |
| non-initial proper nouns, F1 | 0.000 | **0.674** |
| `COMMA` F1 | 0.000 | 0.580 |
| `COLON` F1 | 0.000 | 0.721 |

The absolute figure falls, and **the margin over the baseline widens** — 3.8× here against 3.5×
on test — because the rule degrades faster on multi-sentence text than the model does. Both
numbers are reported because a model evaluated only in its training domain has not been
evaluated.

## Intended use

Putting punctuation and capitals back onto text that has none: the output of
[`Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M) or any other Kabyle speech
recogniser, subtitle and caption tracks, and lowercased corpus text being prepared for
reading rather than for training.

**One utterance at a time is what it was fitted on.** The training split is Common Voice,
where every record is a single sentence, and that is the shape the 0.793 above measures.

**Not suitable for**: segmenting a paragraph or a book page into sentences — it restores
marks, not boundaries, and the out-of-domain numbers above are what that costs; any decision
about a person; or any language other than Kabyle.

## Usage

`transformers` and `torch`, nothing else. The architecture is not one of the library's own,
so the modelling code travels in this repository and `trust_remote_code=True` is what loads
it.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(
    "agbalu/Belaid-31M", trust_remote_code=True
).eval()
tokenizer = AutoTokenizer.from_pretrained("agbalu/Belaid-31M", trust_remote_code=True)

model.restore(["ur zmireɣ ara ad d-aseɣ assa", "anida i tedduḍ a yelli"], tokenizer)
# ['Ur zmireɣ ara ad d-aseɣ assa.', 'Anida i tedduḍ a yelli?']
```

`restore` takes a string or a list and is the interface. Calling the model directly returns
`punctuation_logits` and `case_logits`, both `[batch, subwords, classes]` — but the labels are
per **word** while the model sees subwords, so the alignment inside `restore` is what makes
them mean anything. Reimplement it only deliberately.

**Normalise the input first.** The vocabulary is Mammeri-16k, built over text where `ɛ` is
U+025B and never Greek epsilon; homoglyph-corrupted text fragments into byte pieces and the
labels degrade with it.

## Label scheme

| punctuation | casing |
|---|---|
| `NONE`, `COMMA`, `PERIOD`, `QUESTION`, `COLON` | `LOWER`, `UPPER_INIT` |

Two classes were removed on evidence rather than by preference.

`!` and `;` fold into `PERIOD`. Together they are 0.406% of tokens, and the corpus carries
`Medlet idlisen-nwen!` beside `Medlet idlisen-nwen.` — the same sentence under both marks, so
the distinction is not recoverable from text and a class for it costs macro-F1 without buying
accuracy.

`ALL_CAPS` was a third casing class. It occurs **once** in the development split and three times
in test, and the model carrying it emitted fourteen spurious all-caps words for the three real
ones. Restoring an acronym to full capitals is therefore out of scope, deliberately: a class
that cannot be scored cannot be claimed.

## Architecture

Two token-classification heads over the Masinissa encoder — 12 layers, 384 hidden, 6 attention
heads, 1,280 intermediate, log-bucketed relative positions, and a 16,000-piece Unigram
vocabulary.

The heads are separate because the tasks are. Casing is lexical — whether a word is a name —
and punctuation is syntactic. A single softmax over their product would spend capacity on
combinations that never occur and make the two error rates impossible to read apart.

| | |
|---|---|
| parameters | 31,423,751 |
| of which the two heads | 299,911 |
| context | 128 subwords per call |
| precision | fp32 |

## Training

| | |
|---|---|
| training rows | 1,262,922 |
| epochs | 3 |
| checkpoint | step 11,097 of 14,802, selected on development macro-F1 |
| development score there | 0.8041 marks, 0.9368 non-initial casing |
| hardware | one A10G, 24.3 minutes |

Selection is on development macro-F1 and not on loss, and here that is not a formality: the
minimum-loss checkpoint was step 9,864 and scores lower on both heads, so selecting on loss
would have published a worse model.

Class weighting was tried and removed. Capped inverse-frequency weights at 20× drove comma
recall to 0.919 at precision 0.381 — some 1,200 marks inserted that were not there. The label
distribution is close to IWSLT2011's, where published systems fine-tune unweighted, so the
weighting was dropped rather than tuned.

## Decontamination

**58.2% of Common Voice Kabyle transcripts also appear in AƔBALU-Text v1**, because most are
Tatoeba sentences and Tatoeba is in the corpus. Every clip whose text appears in that corpus was
excluded from the evaluation splits — which decontaminates the encoder at the same time, since
Masinissa was pretrained on the same file. The rows scored above are unseen by both the heads
and the backbone.

A transcript carrying no final mark is dropped rather than labelled `NONE`. Reading them settles
what they are: no punctuation *and* no capitals, a contributor who typed neither. That is typing
habit, and training on it teaches transcriber style rather than Kabyle.

## Limitations

**It restores marks within an utterance, not sentence boundaries across a paragraph.** An
out-of-domain record carries 1.75 periods on average and the model predicts 1.04. It learned
*one period, at the end* from Common Voice, where every record is one sentence by construction,
and question recall follows the same pattern — 0.408 out of domain against 0.886 on test,
because a question inside running prose is mid-record. **This is the input shape the model was
built for**: ASR emits one utterance at a time, and that is where 0.793 applies. Segmenting a
paragraph into sentences is a separate problem and this model does not solve it.

**Comma placement is the weakest mark, and its ceiling is unknown.** Precision is 0.680 on test
and 0.537 out of domain. Kabyle has no codified comma convention, and the references come from
corpus writers who do not agree with each other, so an unmeasured share of these disagreements
are placements a second human would also dispute. The comma figures are a floor of untested
tightness, not a measured distance from what is achievable.

**Text only.** Pauses carry sentence-boundary information and this model never hears them. The
acoustic branch is designed and deliberately unbuilt: on measurement, text answers *which* mark
well and *where* the boundary falls poorly, and the audio is what would close the second gap.

**One language.** Trained and evaluated on Kabyle. Tashelhit, Tarifit, Central Atlas Tamazight
and Tamasheq are related but distinct languages; none was tested and none should be assumed.

**No safety evaluation of any kind** has been performed.

## What was not measured

- **No inter-annotator ceiling exists for Kabyle punctuation.** Nobody has established what two
  fluent readers agree on, so no figure here can be read as a fraction of what is attainable.
- **Segmentation is unscored**, because no long-form Kabyle audio with punctuated transcripts
  exists anywhere to score it on.
- **Latency was not benchmarked** against the recogniser it follows.

## Files

| file | bytes | SHA-256 |
|---|---|---|
| `model.safetensors` | 125,708,716 | `6ac511717b137eb7…` |
| `tokenizer.json` | 1,020,068 | `852cfe5f6aa50ae1…` |
| `agbalu-tok-base-16k.model` | 260,433 | `c8094fccd936d2e8…` |
| `modeling_belaid.py` | 17,876 | `0aa17d3a6f277103…` |
| `configuration_belaid.py` | 2,465 | `672a6ad80d733825…` |
| `config.json` | 852 | `f27945c4243cd4a9…` |
| `tokenizer_config.json` | 221 | `cc21f3d5467039f7…` |
| `__init__.py` | 89 | `76c00392e3348454…` |

115 tensors. The twelve relative-position tables are derived from the config and rebuilt on
load, so they are absent from the weights on purpose — 25.2 MB that would otherwise be shipped
and then reported as unexpected keys.

## Reproduction

```
make punctuation TASK=corpus
make modal-upload TASK=punctuation
make modal-punctuation TASK=train EPOCHS=3 RUN=punctuation-v2
make modal-punctuation TASK=evaluate SPLIT=test RUN=punctuation-v2
make modal-punctuation TASK=evaluate SPLIT=ood RUN=punctuation-v2
make release REPO=belaid
```

The evaluation scores the model and the rule baseline **in the same pass over the same rows**,
because a baseline computed elsewhere on a different sample is not a comparison.

## The name

**Belaïd At Ali** (1909–1950) is the founder of written Kabyle prose. Asked in the 1940s to set
down oral tales, he did more than transcribe them — he composed, trying out the novel and the
short story in a language that had carried neither. *Les Cahiers de Belaïd*, published
posthumously in 1964, is the first body of Kabyle prose in Latin characters.

Writing prose where none existed meant deciding where a sentence ends, where a clause breaks and
what takes a capital — with no precedent to copy. That set of decisions is exactly this model's
label set, and it is why the model carries his name rather than a grammarian's.

He sits beside [`Fadhma-300M`](https://huggingface.co/agbalu/Fadhma-300M) by design: Fadhma Aït
Mansour Amrouche wrote down the songs she had inherited, and Belaïd made Kabyle a written prose
language. One recovers the words; the other renders them as writing.

## Citation

```bibtex
@software{agbalu_belaid_2026,
  title  = {Belaid-31M: punctuation and casing restoration for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Belaid-31M},
  note   = {Two token-classification heads over Masinissa-31M; macro-F1 0.793 over marks}
}
```

## Licence

**Apache-2.0** on the weights and the code, for the patent grant. A permissive grant on
weights does not relicense the text they were trained on: over AƔBALU-Text v1's 3,041,989
sentences the composition is unclear 34.9%, permissive 32.0%, share-alike 31.0%,
non-commercial 2.0%.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle corpus and model collection. The
naming is homage.
