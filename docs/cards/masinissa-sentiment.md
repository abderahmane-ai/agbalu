---
language:
- kab
license: apache-2.0
base_model: agbalu/Masinissa-31M
library_name: transformers
tags:
- berber
- amazigh
- kabyle
- sentiment-analysis
- text-classification
- conformal-prediction
- low-resource
pipeline_tag: text-classification
metrics:
- accuracy
- f1
model-index:
- name: Masinissa-Sentiment
  results:
  - task:
      type: text-classification
      name: Kabyle three-class sentiment
    dataset:
      type: agbalu/KabSentiment
      name: KabSentiment held-out test split (1,500 sentences)
    metrics:
    - type: accuracy
      value: 0.9027
      name: Accuracy
    - type: f1
      value: 0.9026
      name: Macro-F1
---

# Masinissa-Sentiment

**The first sentiment classifier for any Berber language.** Three classes — negative,
neutral, positive — over Kabyle (Taqbaylit, `kab_Latn`), in 31.0M parameters.

On the held-out split of [`KabSentiment`](https://huggingface.co/datasets/agbalu/KabSentiment)
it reaches **90.27% accuracy and 0.9026 macro-F1**. Against the same encoder fine-tuned
once in the ordinary way it is **+1.47 points of accuracy**, and against a linear probe on
the frozen encoder **+12.74** — all three scored on the same 1,500 rows, in this
repository, by the commands at the bottom of this card.

It ships calibrated. A temperature and a split-conformal quantile travel in `config.json`
and are applied by `classify()`, so the model returns a *set* of labels wherever the text
carries no polarity of its own — **94.00% empirical coverage at a mean set size of 1.13
labels**.

**Under total homoglyph corruption it does not move at all**: 90.27%, unchanged, because
the repair for Kabyle's endemic Greek-epsilon defect is inside the inference path rather
than assumed of the caller.

## The task

Searched on 2026-08-25, the Hugging Face Hub holds **no sentiment model for Kabyle,
Tashelhit, Tarifit, Tamasheq, Central Atlas Tamazight or Shawiya** — zero results for
`kabyle sentiment`, `taqbaylit sentiment`, `amazigh sentiment`, `berber sentiment` and
`tamazight sentiment`, against a control query that returns results. Of the seven models
the Hub tags `kab` under `text-classification`, every one is a language identifier or a
prompt-guard. There was nothing to fine-tune from and nothing to beat.

Sentiment in Kabyle is not carried where an English-trained system looks for it.
*Ulac d acu i ixuṣṣen* — "there is nothing lacking" — is a compliment assembled entirely
out of a negation, and the negative particle `ur … ara` wraps the verb rather than
preceding it. Polarity lives in the construction as often as in the vocabulary.

## Results

**1,500 held-out sentences**, 521 negative / 490 neutral / 489 positive. The corpus is
balanced at exactly 5,000 per class and the split is random, so each split is
approximately rather than exactly balanced. **String overlap between train and test is
zero**, measured.

| system | setting | accuracy | macro-F1 |
|---|---|---|---|
| Masinissa-31M | frozen encoder, linear probe | 77.53% | 0.7764 |
| Masinissa-31M | ordinary fine-tune | 88.80% | 0.8880 |
| **Masinissa-Sentiment** | **five seeds, weight-averaged, calibrated** | **90.27%** | **0.9026** |

Three rows, one split, one build of the evaluation set, all produced here. No figure in
this card is a third party's reported number carried into a comparison.

**The gap between the probe and the fine-tune is the task**, and it is 11.27 points: a
frozen encoder and one linear layer already reach 86% of the final accuracy, and what
unfreezing adds is the non-linearity. The 1.47 points on top of that are what the seed
sweep, the weight average and the multi-sample head buy.

### Per class

| label | support | precision | recall | F1 |
|---|---|---|---|---|
| `neutral` | 490 | 0.911 | 0.920 | **0.916** |
| `negative` | 521 | 0.896 | 0.906 | **0.901** |
| `positive` | 489 | 0.902 | 0.881 | **0.891** |

| | → negative | → neutral | → positive |
|---|---|---|---|
| **negative** | 472 | 19 | 30 |
| **neutral** | 22 | 451 | 17 |
| **positive** | 33 | 25 | 431 |

Every class clears 0.89 F1 on three-figure support, and the spread across them is 2.5
points — there is no weak class carrying the average, which is the failure a macro over
three balanced classes is otherwise good at hiding.

**Positive recall, 0.881, is the lowest figure in the table** and the error is one-sided:
33 positive sentences read as negative against 30 the other way. That is the negation
construction above. It is the direction to attack next, and it is a property of the
language rather than of the fit.

### Robustness to the corruption that is actually in Kabyle text

2.60–3.19% of the rows in this language's two largest text exports carry Greek `ε` U+03B5
where Latin `ɛ` U+025B belongs. **826 of the 1,500 test sentences contain a character
exposed to that defect.**

| | accuracy | macro-F1 |
|---|---|---|
| clean | 90.27% | 0.9026 |
| every exposed character corrupted, through the shipped path | **90.27%** | **0.9026** |
| every exposed character corrupted, normaliser bypassed | 85.67% | 0.8563 |

**Nothing moves on the shipped path**, because `agbalu.normalise` runs before the tokenizer
and repairs the substitution. Bypass it and the model still returns **85.67%** on text
where more than half the sentences have been damaged — eight points above the frozen
probe's *clean* number. The encoder absorbs the corruption; the front end removes it.

### The seeds, and the average

Five seeds — 42, 1337, 2026, 7, 999 — each fine-tuned for 6 epochs from the same
initialisation, each selecting its own best epoch on dev.

| | dev macro-F1 |
|---|---|
| best single seed (42) | 0.9119 |
| mean of the five | 0.9105 |
| standard deviation | 0.0011 |
| **uniform weight average** | **0.9126** |

The average — [Wortsman et al., ICML 2022](https://arxiv.org/abs/2203.05482) — is what
ships, because it beat every individual run on dev. It is taken as a *candidate*: where
runs land in different basins the average is worse than all of them, and the trainer falls
back to the best seed and records that it did. Here it won.

A 0.0011 standard deviation is what justifies the sweep rather than undermining it. It is
the same order as most differences anyone reports from a fine-tune this size, so a
single-seed headline would not be separable from the seed — and the reported number is the
average of five runs rather than the best of five draws.

### Where the labels come from, and why they transfer

**`KabSentiment` is the training set** — 12,000 rows — as well as the evaluation set, and
its labels are projected. Each row is a Tatoeba pair: the **English** side is scored by
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
and the label is carried across to the human-written Kabyle translation. Only rows the
classifier is at least 0.80 confident about are kept, which rejects 57% of candidates.

That teacher is a purpose-built instrument, not a general model asked a side question: a
RoBERTa-base trained on ~124M tweets from January 2018 to December 2021 and fine-tuned on
the TweetEval sentiment benchmark ([TimeLMs](https://arxiv.org/abs/2202.03829), Loureiro et
al., 2022). It is among the most heavily used classifiers on the Hub, at 3.1M downloads.
Its label order — 0 negative, 1 neutral, 2 positive — is asserted against
`model.config.id2label` at build time rather than assumed, because a silent reordering
would invert the corpus.

**The transfer is measured, not asserted.** If the projection only survived on rows the
teacher found easy, accuracy would collapse at the bottom of the confidence range. It does
not:

| teacher's confidence in its English label | rows | this model's accuracy |
|---|---|---|
| 0.80 – 0.90 | 889 | **87.85%** |
| 0.90 – 0.98 | 571 | **93.70%** |
| 0.98 – 1.00 | 40 | **95.00%** |

**On the 889 rows the teacher was least sure about — 59% of the split — the model still
agrees 87.85% of the time**, within 2.4 points of its figure over the whole split and
above the frozen probe's by more than ten. Accuracy then rises monotonically with the
teacher's own confidence, 87.85 → 93.70 → 95.00, which is what a well-behaved projection
looks like: the two systems are uncertain about the same sentences, in the same order,
across two languages and two orthographies. A projection that had merely been memorised
would be flat across these bands, or would fall off the bottom of them.

By length, in English source words:

| | rows | accuracy |
|---|---|---|
| 4–7 words | 1,194 | **91.04%** |
| 8–12 words | 277 | 87.36% |
| 13–25 words | 29 | 86.21% |

The corpus is gated at 4–25 English words, so the model is fitted on short sentences and
is strongest there. The 29-row band carries its support and no further claim.

**What this does and does not establish.** It establishes that the Kabyle model reproduces
a strong English sentiment instrument's judgement on human-written Kabyle, including where
that instrument was itself unsure. It does not establish accuracy against a human Kabyle
annotator, because no gold Kabyle sentiment set exists anywhere to measure that with.
Building one is the next thing this task needs.

### Conformal prediction sets

Both corrections are fitted on **dev** and reported on **test**.

| | |
|---|---|
| temperature | 0.9811 |
| conformal quantile `q_hat` | 0.8022, at target coverage 95% |
| **empirical coverage on test** | **94.00%** |
| **mean set size** | **1.13 labels** — 1,312 singletons, 181 pairs, 7 triples |
| sets larger than one label | 12.53%, 188 of 1,500 |
| expected calibration error | 1.52% raw, 1.80% temperature-scaled |

**The temperature is 0.9811, and it moves calibration 0.28 points the wrong way on test.**
The fine-tune came out marginally *under*confident, and the objective minimises negative
log-likelihood on dev rather than ECE on test. It is reported because it is what the model
applies, and it is kept because `q_hat` was fitted against the scaled probabilities and the
two are one calibration; at 0.98 it decides nothing. **A raw ECE of 1.52% is the good
outcome**: this classifier arrived from the fine-tune already calibrated, and there was
nothing for the temperature to correct.

**A widened set is the model declining to guess, and it widens where a fluent reader
would.** Real rows from the test split:

| sentence | set | projected label |
|---|---|---|
| *Aselkim-a yif winna.* | `negative`, `positive` | positive |
| *Mary tessen ad teskerkes, naɣ ?* | `negative`, `positive` | negative |
| *Acu n wugur yellan di tikti-agi-inu?* | `negative`, `neutral` | negative |

A bare comparison, a tag question, an open question. In each the polarity is in the
situation and not in the words, and a forced singleton would be a guess with a confidence
attached to it.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

name = "agbalu/Masinissa-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)

for row in model.classify([
    "Ulac d acu i ixuṣṣen, kulci yelha aṭas.",
    "Ayagi d ayen n diri, ur yelhi ara maḍi.",
    "Yiwen wass, argaz-nni yuli ɣer taddart.",
], tokenizer):
    print(row["label"], round(row["confidence"], 4), row["prediction_set"])
# positive 0.9225 ('positive',)
# negative 0.9825 ('negative',)
# neutral 0.9857 ('neutral',)
```

`classify` applies the temperature and the conformal quantile the numbers above were
measured with; both travel in `config.json`. The plain `forward` returns raw logits and
does neither, so a pipeline reading `logits.argmax()` gets the same label with an
uncalibrated confidence and no prediction set:

```python
logits = model(**tokenizer("Azul fell-awen", return_tensors="pt")).logits
```

**Normalise Kabyle text before the tokenizer sees it.** The vocabulary was built over
normalised text, and the Greek `ε` U+03B5 is a different piece from the Latin `ɛ` U+025B it
stands in for — the 4.6-point gap in the robustness table is exactly that difference.
`agbalu.normalise` in the project repository is the reference implementation, and the
classifier in that package applies it for you:

```python
from agbalu.sentiment import SentimentClassifier

classifier = SentimentClassifier.load(
    "artifacts/masinissa-sentiment", "artifacts/tokenizer/agbalu-tok-base-16k.model"
)
print(classifier.predict("Ulac d acu i ixuṣṣen, kulci yelha aṭas.")[0].as_dict())
```

## Intended use

Sorting and triaging Kabyle text at corpus scale: tagging comment threads, survey
responses, reviews or forum posts by polarity, ranking a backlog, and routing the genuinely
ambiguous cases to a person. Use the prediction set rather than the argmax wherever a wrong
label gets acted on — 12.53% of real sentences are cases where the model has something more
useful to say than one word.

**Not suitable for** any decision about a person, for measuring public opinion, for
Tifinagh or Arabic-script Kabyle, for the sibling Berber languages, or as a moderation
component that acts without review. No safety evaluation of any kind has been performed.

## Architecture

| | |
|---|---|
| Parameters, trained | 31,199,491 |
| Parameters, published | 31,035,651 |
| Trunk | [Masinissa-31M](https://huggingface.co/agbalu/Masinissa-31M) unchanged — 12 layers, 384 hidden, 6 heads, 1,280 feed-forward |
| Positions | 32 log-bucketed relative buckets over a 512-token window |
| Pooling | masked mean over the real positions |
| Head | LayerNorm, 384 → 192 GeLU bottleneck, LayerNorm, five-way multi-sample dropout, 192 → 3 |
| Vocabulary | [Mammeri-Tok](https://huggingface.co/agbalu/Mammeri-Tok), 16,000 SentencePiece unigram pieces |

**The head is 75,651 parameters — 0.24% of the model.** Everything this classifier knows
about Kabyle it knew before it saw a label, which is what the frozen probe's 77.53%
measures directly.

Two things are absent from the download on purpose. The masked-language-model head the
encoder was pretrained with is 163,840 parameters that no forward pass here reads — its
output projection is tied to the embedding, so dropping it costs nothing but its own bias
and hidden layer. The twelve relative-position bucket tables are **3,145,728 int64 entries,
25.2 MB**, derived from the config and rebuilt on load.

**Mean pooling, not `[CLS]`.** This encoder is masked-language-model pretrained with no
next-sentence objective, so nothing ever trained that position to summarise a sentence.

**Multi-sample dropout** reads the bottleneck through five dropout masks at once and
averages the loss over them — [Inoue, 2019](https://arxiv.org/abs/1905.09788). The encoder
runs before the branch, so this costs one extra Linear per mask and delivers the gradient
of a five-way ensemble. Inference takes the undropped path, which is the one every number
above was measured through.

## Training

| | |
|---|---|
| data | **`agbalu/KabSentiment` — 12,000 train, 1,500 dev, 1,500 test** |
| schedule | 6 epochs per seed, batch 64, 10% linear warmup then cosine decay |
| learning rates | encoder 3e-5, head 1e-4 |
| regularisation | weight decay 0.01, off norms and biases; gradient clip 1.0; label smoothing 0.05 |
| hardware | one A10G |

**Two learning rates**, because the encoder arrives pretrained and the head does not: one
rate large enough to move the head undoes the representation.

**Selection is on dev, and test is read once.** Choosing the epoch, the seed or the average
on test would report the maximum of thirty draws as though it were one.

## Decontamination

**99.5% of the test sentences — 1,493 of 1,500 — are in AƔBALU-Text v1**, the encoder's
pretraining corpus, because the Tatoeba monolingual export is one of that corpus's 3,041,989
rows. Unlike Belaid, this model cannot exclude the overlap: the corpus the labels are
projected onto *is* Tatoeba, so removing it would remove the task.

What that does and does not mean. The encoder saw the **text**, unlabelled, under a
masked-token objective. **No sentiment supervision exists anywhere in pretraining** — the
label is a fact about an English sentence that entered this project only when
`make modal-sentiment TASK=build` ran, long after the encoder was trained. This is the
ordinary situation for a fine-tune of a pretrained encoder on in-domain text, and it is
the situation the frozen-probe row measures directly: 77.53% is what the pretrained
representation is worth on rows it has read and never been told about.

**Train and test share no string**, measured over all 13,500 rows of the two splits. The one
known leak in `KabSentiment` is a single sentence shared between train and dev, which is in
Limitations below and touches epoch selection rather than any figure on this card.

The join is positive-controlled by construction: a broken key returns zero matches, and this
one returned 1,493.

## Limitations

**The labels are projected, not annotated.** The measured transfer is in Results above, and
what it bounds is agreement with a strong English instrument on human-written Kabyle. It
does not bound accuracy against a human Kabyle annotator, and no data exists to.

**The labelling classifier was never validated on Kabyle.** Its own three-class accuracy on
this language is unmeasured. The 0.80 gate filters by the teacher's confidence in its
*English* judgement, which is structural rather than semantic.

**Sarcasm, irony and mixed sentiment are outside what a projected label can teach.** Where
the English source carries the tone and the Kabyle translation does not, the label follows
the English.

**Written standard Kabyle only.** No spoken register, no code-switching, no sub-dialectal
orthography, and no Kabyle-French mixing — which is how a great deal of Kabyle is actually
written online.

**Sentences, not documents.** The corpus is gated at 4–25 English words and accuracy falls
from 91.04% to 87.36% across that range. A paragraph is outside the input shape.

**One `KabSentiment` dev sentence is also in train.** `Ulac ǧahennama yugaren ta.` appears
as `kab_sent_train_06003` and `kab_sent_dev_00065`: two English sentences that translate
identically into Kabyle. The split was keyed on the source pair. One row of 1,500, and it
touches dev selection rather than any test figure — train and test share **no** string.

**`q_hat` was fitted on the same dev split the epoch and the average were selected on.**
Coverage is measured on test, which is what makes it a measurement rather than a fit, but
the guarantee rests on dev and test being exchangeable and that is not established here
independently.

**Neutral is over-represented in the source pool** — 31,119 candidates against 8,027
positive — and the release is a subsample capped by the smallest class.

## Files

| file | |
|---|---|
| `model.safetensors` | 124.2 MB, 31,035,651 parameters |
| `config.json` | shapes, label order, temperature and conformal quantile |
| `modeling_masinissa_sentiment.py`, `configuration_masinissa_sentiment.py` | the architecture; `transformers` and `torch` only |
| `tokenizer.json`, `agbalu-tok-base-16k.model` | the 16,000-piece vocabulary |
| `evaluation.json` | the test report, the seed sweep and the calibration, in full |
| `export.stats.json` | what the export dropped, and a checksum per file |

No optimiser state, no scheduler state, no RNG. Training cannot be resumed from these
files.

## Reproduction

```bash
make modal-sentiment TASK=build          # project the labels and write the splits
make modal-sentiment TASK=benchmark      # the frozen probe and the ordinary fine-tune
make modal-sentiment TASK=train          # five seeds, the average, the calibration
make modal-sentiment-pull                # fetch the checkpoint and its report
make release REPO=masinissa-sentiment    # export, stage, and load back as a downloader would
```

`make release` refuses a directory that will not load through `AutoModelForSequenceClassification`
with `trust_remote_code=True`, that disagrees with the training tokenizer on any probe, or
whose head answers three polar sentences identically.

## The name

**Masinissa** (c. 238–148 BCE) was the first king of a united Numidia. The classifier
carries the encoder's name because it *is* that encoder, with 0.24% more weights on top.

The naming is homage and implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_masinissa_sentiment_2026,
  title  = {Masinissa-Sentiment: three-class sentiment for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Masinissa-Sentiment},
  note   = {31.0M parameters; 90.27% accuracy and 0.9026 macro-F1 on 1,500 held-out sentences}
}
```

Please also cite the Tatoeba project for the source sentences, and Cardiff NLP for
`cardiffnlp/twitter-roberta-base-sentiment-latest`, which produced the labels.

## Licence

**Apache-2.0** on the weights and the code, for the patent grant. A permissive grant on
weights does not relicense the text they were trained on: `KabSentiment` is **CC-BY-4.0**
over Tatoeba-derived sentences, and the encoder underneath was pretrained on AƔBALU-Text
v1, whose composition is **unclear 34.9%, permissive 32.0%, share-alike 31.0%,
non-commercial 2.0%**. `unclear` is the absence of a licence, not a permissive one.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle and Amazigh corpus and model
collection.
