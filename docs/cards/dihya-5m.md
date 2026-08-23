---
language:
- kab
- shi
- rif
- taq
- tzm
- shy
license: apache-2.0
library_name: transformers
tags:
- berber
- amazigh
- kabyle
- tashelhit
- tarifit
- tamasheq
- tamazight
- shawiya
- language-identification
- conformal-prediction
- low-resource
pipeline_tag: text-classification
metrics:
- accuracy
- f1
model-index:
- name: Dihya-5M
  results:
  - task:
      type: text-classification
      name: Berber language identification, six Latin-script varieties plus rejection
    dataset:
      type: berber-lid-heldout
      name: AƔBALU Berber LID held-out set (1,050 sentences, 150 per class)
    metrics:
    - type: accuracy
      value: 0.8552
      name: Accuracy over 7 classes
    - type: f1
      value: 0.8562
      name: Macro-F1 over 7 classes
---

# Dihya-5M

A 5.19M-parameter byte-level Conv-Transformer that separates six Latin-script Amazigh varieties
from each other and from an explicit `NOT_AMAZIGH` rejection class: Kabyle (`kab_Latn`),
Tashelhit (`shi_Latn`), Tarifit (`rif_Latn`), Tamasheq (`taq_Latn`), Central Atlas Tamazight
(`tzm_Latn`) and Shawiya (`shy_Latn`).

On a balanced, source-stratified held-out set of 1,050 sentences it reaches **85.52% accuracy and
0.8562 macro-F1** over the seven classes. Scored on the six Berber classes alone — the label set
the public identifiers can be asked about — it reaches **84.44% accuracy**, against **41.78% for
GlotLID and 28.33% for NLLB's `lid218e`** measured in the same pass over the same rows.

It ships with a split-conformal quantile, so it can return a *set* of labels rather than one, for
pipelines where a silent misclassification costs more than a widened answer.

## The task

These are not typologically distant languages. The six varieties share much of a core lexicon —
*aman* (water), *argaz* (man), *awal* (word) — and one Latin orthography. The evidence that
separates them is affixal, phonotactic and clitic, and a short sentence of shared vocabulary
often carries none of it.

That matters beyond the benchmark. NLLB's own identifier mined 90.1% of this project's parallel
corpus, and it labels the siblings Kabyle: **91.3% of Shawiya, 80.7% of Tarifit, 79.3% of
Tashelhit and 73.3% of Central Atlas Tamazight**, measured here. Datasets labelled "Kabyle" or
"Tamazight" inherit that.

## Results

**1,050 sentences, 150 per class**, drawn round-robin across each language's sources so that no
class is one corpus wearing a language's name. Disjoint from train (77,756) and dev (4,093).

### Against the public identifiers

All three scored in one pass over the same 900 Berber rows. `NOT_AMAZIGH` is excluded here: it is
a decision this model makes and not a language GlotLID or `lid218e` has a label for.

| system | accuracy | classes it can name | macro-F1 over those |
|---|---|---|---|
| **Dihya-5M** | **84.44%** | **6 of 6** | 0.8493 |
| GlotLID | 41.78% | 3 of 6 | 0.6837 |
| NLLB `lid218e` | 28.33% | 2 of 6 | 0.6022 |

The last column is three different statistics — each system's macro is taken over the classes it
can name, so GlotLID's 0.6837 averages three classes and Dihya's 0.8493 averages six. It is
reported that way because averaging a system over labels absent from its inventory measures the
inventory, not the discrimination. The accuracy column is the comparable number.

**The quantity that matters downstream is how much sibling text ends up labelled Kabyle**, since
that is what a filter keyed on `kab` silently ingests:

| true language | → Kabyle, Dihya | → Kabyle, GlotLID | → Kabyle, `lid218e` |
|---|---|---|---|
| Tashelhit | **2.0%** | 17.3% | 79.3% |
| Tarifit | **6.7%** | 90.0% | 80.7% |
| Tamasheq | **7.3%** | 0.0% | 6.7% |
| Central Atlas Tamazight | **16.0%** | 66.7% | 73.3% |
| Shawiya | **19.3%** | 96.0% | 91.3% |

GlotLID sends no Tamasheq to Kabyle because it has a good Tamasheq label, and it is the one place
a public system is not the weaker option — its Tamasheq F1 is 0.891 against Dihya's 0.921, close
enough that the ordering should not be leaned on.

### Per class, all seven

The seven-class report, which is what `held-out-report.json` in this repository holds.

| label | support | precision | recall | F1 |
|---|---|---|---|---|
| `NOT_AMAZIGH` | 150 | 0.949 | 0.987 | 0.967 |
| `taq_Latn` | 150 | 1.000 | 0.847 | 0.917 |
| `rif_Latn` | 150 | 0.916 | 0.873 | 0.894 |
| `shi_Latn` | 150 | 0.811 | 0.973 | 0.885 |
| `shy_Latn` | 150 | 0.936 | 0.687 | 0.792 |
| `tzm_Latn` | 150 | 0.952 | 0.660 | 0.780 |
| `kab_Latn` | 150 | 0.626 | 0.960 | 0.758 |

**Kabyle's precision is the lowest number here, and it is the one to read first.** At 0.626 the
model still over-assigns Kabyle — it recovers 96.0% of the Kabyle and pays for it by pulling in
sibling text. That is the same failure direction as the systems above, an order of magnitude
smaller. Anyone building a Kabyle-only corpus should treat a `kab_Latn` label as a filter, not as
a verdict.

**Recall on Tamazight and Shawiya is the weak side.** 0.660 and 0.687 mean a third of each is
missed, mostly to Kabyle and Tashelhit.

### Where it is worst

Selection and reporting are on `(language, domain)` cells, not on classes, because a class average
hides a domain. Two cells are worth naming:

| cell | support | recall |
|---|---|---|
| `shy_Latn/interface` | 3 | 0.000 |
| `tzm_Latn/interface` | 61 | 0.410 |

**`tzm_Latn/interface` is a real defect**: software-localisation strings in Central Atlas
Tamazight are short, formulaic, and share their register with the Tashelhit and Kabyle
localisations they were translated alongside. Two in five are missed.

**`shy_Latn/interface` at 0.000 is three examples**, and three examples do not measure anything.
It is listed because it is the worst cell in the file and omitting it would make the file
disagree with the card, not because it supports a conclusion.

### Conformal prediction sets

Calibrated by split conformal at target coverage 99.0%. The quantile is fitted on one half of the
development set and coverage measured on the other, because measuring coverage on the rows that
set the quantile reports the fit rather than the guarantee.

| | |
|---|---|
| fitted on | 2,047 dev sentences |
| coverage measured on | 2,046 held-out dev sentences |
| target coverage | 99.0% |
| empirical coverage | **98.78%** |
| quantile `q̂` | 0.9979 |
| mean set size | **1.098 labels** |

**This is measured on development data, not on the 1,050-sentence test set.** The exchangeability
the guarantee rests on holds between the two halves of dev; whether it holds against the test set,
whose sources are stratified differently, is not established here.

## Usage

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "agbalu/Dihya-5M", trust_remote_code=True
)

for row in model.identify([
    "Azul fell-awen, amek tettilim ass-a?",
    "Mayemmi ur teccid ticemrarin nnecm a gma?",
    "Bonjour tout le monde, comment allez-vous?",
]):
    print(row["language"], round(row["confidence"], 4), row["prediction_set"])
# kab_Latn 0.9959 ('kab_Latn', 'shy_Latn')
# rif_Latn 0.9998 ('rif_Latn',)
# NOT_AMAZIGH 0.9997 ('NOT_AMAZIGH',)
```

`identify` applies the logit adjustment and the conformal quantile that the numbers above were
measured with; both travel in `config.json`. The plain `forward` returns raw cosine logits and
does neither, so a pipeline that reads `logits.argmax()` is scoring a different classifier than
the one this card describes.

The tokenizer is the 256 UTF-8 byte values and is only needed for batching:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("agbalu/Dihya-5M")
model = AutoModelForSequenceClassification.from_pretrained(
    "agbalu/Dihya-5M", trust_remote_code=True
)
batch = tokenizer(["Azul fell-awen"], return_tensors="pt", padding=True, truncation=True)
logits = model(**batch).logits
```

**Prediction sets widen where the text is genuinely ambiguous.** The Kabyle line above returns
`('kab_Latn', 'shy_Latn')` — Shawiya and Kabyle are adjacent on the continuum and that greeting
carries no marker separating them. A forced singleton there would be a guess with a confidence
attached.

## Intended use

Screening and routing text in corpus pipelines: tagging web scrapes and OCR output by variety,
discarding non-Berber lines, and routing input to variety-specific models. Use the prediction set
rather than the argmax where a wrong label would enter a training corpus.

**Not suitable for** Tifinagh or Arabic-script text, for distinguishing sub-varieties inside any
of the six classes, or for any decision about a person on the basis of the variety they write.

## Architecture

| | |
|---|---|
| Parameters, trained | 5,187,712 |
| Parameters, published | 5,089,408 |
| Encoder layers | 6 |
| Hidden / feed-forward | 256 / 704 (SwiGLU) |
| Attention heads | 8 × 32, rotary positions |
| Conv stem | three parallel depthwise-separable 1D convolutions, kernels 3, 5, 7 |
| Pooling | attentive (learned score per position) |
| Head | cosine, scale 24.0 |
| Vocabulary | 258 (256 UTF-8 bytes, `[PAD]`, `[UNK]`) |
| Maximum input | 256 bytes |

The published model is smaller than the trained one because the contrastive projection head
(98,304 parameters) shapes the trunk during training and no forward pass reads it at inference.

**Bytes, not characters or subwords.** The orthographies differ in exactly the characters where
they diverge — `ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ ṯ` are two or three UTF-8 bytes each. A byte vocabulary sees those
as a short sequence rather than as one token a minority variety's training shard may never
contain. Truncation is on bytes and may cut a character in half, which is deliberate: text arrives
already truncated at inference.

**Three kernel widths** because the evidence sits at three scales — a grapheme cluster (`ţ` against
`t`), an affix (the `u-`/`w-` annexation), and a clitic chain.

## Training

| | |
|---|---|
| training rows | 77,756 across 20 `(language, domain)` cells |
| dev rows | 4,093 |
| schedule | 8 epochs, 5,176 steps; batch 12 cells × 10 rows |
| published checkpoint | step 2,898, selected on dev |
| dev score there | 0.9363 macro-F1, 0.8125 worst-group recall |
| hardware | one A10G |

Three objectives address three separate failures of an earlier build that read 98.26% on a dev
split drawn from the training sources and 76.27% on held-out text:

**LDAM margins** for the long tail — `m_k = C / n_k^(1/4)`, from
[Cao et al., NeurIPS 2019](https://arxiv.org/abs/1906.07413). Shawiya has 447 training rows
against Tashelhit's 24,123, and a shared margin gives the rare class the same slack as the common
one.

**Group-DRO** for the domain confound — exponentiated-gradient weights over the twenty
`(language, domain)` cells, from [Sagawa et al., ICLR 2020](https://arxiv.org/abs/1911.08731). It
optimises the worst cell rather than the batch mean.

**Domain-aware supervised contrastive learning** — negatives mined from a different language in
the *same* domain first, over a 2,048-entry memory bank. The hard-negative scheme is
[ConLID's](https://arxiv.org/abs/2506.15304). A same-domain negative can only be separated by the
language.

**Both margins and group weights are off for the first half of the schedule.** Measured on this
corpus: 120 steps with margins from step 0 reached a loss of 4.01 where the same run without them
reached 0.76, at the same accuracy. This is LDAM's own deferred schedule.

**Selection is on the mean of dev macro-F1 and dev worst-group recall, never on the test set and
never on loss.** Macro-F1 alone picks the checkpoint that wins the large domains; worst-group
alone picks one that has learned a single hard cell and forgotten the rest.

## Limitations

**Ambiguity on shared vocabulary is real and is not an error.** A short sentence of pan-Berber
core vocabulary carrying no variety-specific marker cannot be assigned to one variety. The
conformal set reports that with size > 1.

**Kabyle precision is 0.626.** Sibling text still arrives labelled Kabyle at that rate on this set.

**Tifinagh and Arabic script are out of scope.** Byte input from either produces a label. That
label has not been measured.

**Single words are outside the input shape.** The model was fitted on sentences.

**Code-switching is not handled.** One label per input.

**Shawiya cannot be measured well by anyone.** 447 training rows and 150 held-out is close to the
whole of the language's digital text; its numbers carry their support and no further claim.

**No inter-annotator ceiling exists for Berber variety attribution**, so no figure here can be
read as a fraction of what is attainable — fluent speakers disagree on the ambiguous cases.

**Every source in the evaluation set is a corpus**, not a live crawl or real user input, and no
safety evaluation of any kind has been performed.

## Files

| file | description |
|---|---|
| `model.safetensors` | 20.4 MB, 5,089,408 parameters |
| `config.json` | architecture, class order, prior shift and conformal quantile |
| `modeling_dihya.py`, `configuration_dihya.py` | the architecture, `transformers` and `torch` only |
| `tokenizer.json` | the 258-entry byte vocabulary |
| `calibration.json` | the calibration, in full |
| `held-out-report.json` | per class and per `(language, domain)` cell |

No optimiser state, no scheduler state, no RNG. Training cannot be resumed from these files.

## Reproduction

```bash
make lid-dataset                              # build the splits from the source table
make modal-lid EPOCHS=8                       # train on Modal A10G
make modal-lid-pull                           # fetch the checkpoint
make lid TASK=evaluate                        # the 7-class held-out report
make bench TASK=lid                           # Dihya, GlotLID and lid218e in one pass
make release REPO=dihya                       # export and stage for the Hub
```

`make bench TASK=lid` is what produced the comparison table: all three systems, one build of the
evaluation set, one pass. A baseline computed elsewhere on a different sample is not a comparison.

## The name

**Dihya** (also called *al-Kāhina*, died c. 703 CE) was an Amazigh military leader from the Aurès
Mountains of what is now eastern Algeria — the Shawiya homeland — who led the Berber resistance to
the Umayyad conquest of Ifriqiya. Much of what is written about her comes from sources composed
well after her lifetime and is contested; the Aurès and the resistance are the parts that are not.

The naming is homage and implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_dihya_2026,
  title  = {Dihya-5M: language identification across the Berber dialect continuum},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Dihya-5M},
  note   = {5.19M parameters; 85.52% accuracy over seven classes on 1,050 held-out sentences}
}
```

## Licence

**Apache-2.0** on the weights and the code. The training corpus is assembled from sources under
mixed licences, which a permissive grant on weights does not relicense; see
[the datasheet](https://huggingface.co/datasets/agbalu/KabBench) and
`resources/sibling_registry.yaml` in the project repository for per-source terms.

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle and Amazigh corpus and model collection.
