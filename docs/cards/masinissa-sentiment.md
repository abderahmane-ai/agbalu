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

A 31.0M-parameter (31,035,651 published) three-class sentiment classifier for **Kabyle** (Taqbaylit, `kab_Latn`), categorizing text into negative, neutral, and positive sentiment.

On the held-out test split of [`KabSentiment`](https://huggingface.co/datasets/agbalu/KabSentiment) (1,500 sentences), it reaches **90.27% accuracy and 0.9026 macro-F1**. This improves upon an ordinary single-seed fine-tune of Masinissa-31M by **+1.47 points of accuracy** and a linear probe on the frozen encoder by **+12.74 points**, evaluated on the identical test split.

The model includes built-in temperature scaling and split-conformal calibration parameters ($\hat{q} = 0.8022$ at 95% nominal coverage), yielding an empirical test coverage of **94.00% at an average set size of 1.13 labels**.

Under full synthetic homoglyph corruption (Greek `ε` U+03B5 replacing Latin `ɛ` U+025B), the model maintains **90.27% accuracy** via the default preprocessing pipeline, which normalises characters prior to tokenization.

## The task

Prior to this model, no dedicated sentiment classification model was available on the Hugging Face Hub for Kabyle or other Amazigh varieties (`kab`, `shi`, `rif`, `taq`, `tzm`, `shy`). Existing models tagged with `kab` under `text-classification` were limited to language identification and prompt-safety classifiers.

Kabyle sentiment expression frequently relies on syntactic constructions rather than isolated sentiment lexicons. For example, bipartite negation (`ur … ara`) wraps verbs, and litotes such as *Ulac d acu i ixuṣṣen* ("there is nothing lacking") convey strong positive sentiment through negative grammatical forms. Effective classification requires modeling sentence structure and clitic dependencies rather than relying on surface keyword matching.

## Results

The model was evaluated on a held-out test split of **1,500 sentences** (521 negative, 490 neutral, 489 positive) from `KabSentiment`. The dataset is balanced across three classes (5,000 sentences each, 15,000 total). There is zero string overlap between training and test sets.

| System | Setting | Accuracy | Macro-F1 |
|---|---|---|---|
| Masinissa-31M | Frozen encoder, linear probe | 77.53% | 0.7764 |
| Masinissa-31M | Single-seed fine-tune | 88.80% | 0.8880 |
| **Masinissa-Sentiment** | **Five seeds, weight-averaged, calibrated** | **90.27%** | **0.9026** |

All models were evaluated on the same 1,500-sentence test split using identical preprocessing. The frozen encoder linear probe reaches 77.53% accuracy, indicating that the pretrained encoder captures substantial semantic structure prior to task-specific adaptation. Fine-tuning the full encoder improves accuracy to 88.80%, while multi-seed weight averaging and multi-sample dropout yield the final 90.27% accuracy.

### Per-class performance

| Label | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `neutral` | 490 | 0.911 | 0.920 | **0.916** |
| `negative` | 521 | 0.896 | 0.906 | **0.901** |
| `positive` | 489 | 0.902 | 0.881 | **0.891** |

Confusion matrix:

| | → negative | → neutral | → positive |
|---|---|---|---|
| **negative** | 472 | 19 | 30 |
| **neutral** | 22 | 451 | 17 |
| **positive** | 33 | 25 | 431 |

Performance is balanced across all three classes, each exceeding 0.89 F1 with a spread of 2.5 percentage points. Positive recall (0.881) represents the lowest figure in the table, with the primary confusion mode consisting of positive sentences classified as negative (33 cases), predominantly involving negation-based complimentary idioms.

### Robustness to character corruption

Public Kabyle text exports frequently exhibit character corruption where Greek `ε` (U+03B5) is substituted for Latin `ɛ` (U+025B) in 2.60% to 3.19% of records. In the 1,500-sentence test set, 826 sentences contain characters exposed to this substitution.

| Setting | Accuracy | Macro-F1 |
|---|---|---|
| Clean input | 90.27% | 0.9026 |
| All exposed characters corrupted, shipped pipeline | **90.27%** | **0.9026** |
| All exposed characters corrupted, normaliser bypassed | 85.67% | 0.8563 |

When input passes through the default normalisation pipeline before tokenization, performance is completely invariant (90.27% accuracy). If normalisation is bypassed entirely, the model retains 85.67% accuracy under 100% corruption of exposed characters, reflecting the underlying encoder's robustness to subword fragmentation.

### Multi-seed training and weight averaging

Five random seeds (42, 1337, 2026, 7, 999) were fine-tuned for 6 epochs from the base checkpoint, each selecting its best epoch on the validation split.

| Setting | Dev Macro-F1 |
|---|---|
| Best single seed (42) | 0.9119 |
| Mean of five seeds | 0.9105 |
| Standard deviation | 0.0011 |
| **Uniform weight average** | **0.9126** |

A uniform weight average across the fine-tuned checkpoints ([Wortsman et al., ICML 2022](https://arxiv.org/abs/2203.05482)) yielded the highest validation macro-F1 (0.9126) and was selected for the final release.

### Dataset curation and label projection

Training and evaluation were conducted on [`KabSentiment`](https://huggingface.co/datasets/agbalu/KabSentiment) (12,000 train, 1,500 dev, 1,500 test). Labels were projected from English-Kabyle sentence pairs in Tatoeba: the English source text was classified by [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) ([Loureiro et al., TimeLMs, 2022](https://arxiv.org/abs/2202.03829)), and high-confidence labels ($\ge 0.80$ teacher confidence) were assigned to the corresponding Kabyle translations. This confidence threshold filters out 57% of candidate pairs.

Agreement between the Kabyle model and the teacher model was measured across teacher confidence bands on the test split:

| Teacher confidence (English) | Test rows | Kabyle model accuracy |
|---|---|---|
| 0.80 – 0.90 | 889 | **87.85%** |
| 0.90 – 0.98 | 571 | **93.70%** |
| 0.98 – 1.00 | 40 | **95.00%** |

Across the 889 test sentences where the teacher was least confident (0.80–0.90), the model achieves 87.85% agreement. Agreement increases monotonically with teacher confidence up to 95.00% on the highest-confidence bracket.

Accuracy stratified by English sentence length:

| English source length | Test rows | Accuracy |
|---|---|---|
| 4–7 words | 1,194 | **91.04%** |
| 8–12 words | 277 | 87.36% |
| 13–25 words | 29 | 86.21% |

The dataset is filtered to sentences of 4–25 English words. Performance is highest on short sentences (91.04% accuracy on 4–7 words).

### Conformal prediction and calibration

Temperature scaling and split-conformal calibration parameters were fitted on the validation set and evaluated on the test set:

| Parameter / Metric | Value |
|---|---|
| Temperature | 0.9811 |
| Conformal quantile $\hat{q}$ | 0.8022 (target coverage 95%) |
| **Empirical coverage on test** | **94.00%** |
| **Mean set size** | **1.13 labels** (1,312 singletons, 181 pairs, 7 triples) |
| Prediction sets $> 1$ label | 12.53% (188 of 1,500) |
| Expected calibration error (ECE) | 1.52% raw, 1.80% temperature-scaled |

The model achieves 94.00% empirical test coverage with a mean prediction set size of 1.13 labels. Raw expected calibration error is 1.52%.

Prediction sets expand to multiple labels for syntactically ambiguous or context-dependent sentences where polarity is implicit:

| Sentence | Prediction set | Projected label |
|---|---|---|
| *Aselkim-a yif winna.* | `negative`, `positive` | positive |
| *Mary tessen ad teskerkes, naɣ ?* | `negative`, `positive` | negative |
| *Acu n wugur yellan di tikti-agi-inu?* | `negative`, `neutral` | negative |

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

`classify` applies temperature scaling and conformal quantile thresholding using parameters stored in `config.json`. The standard forward pass returns uncalibrated logits:

```python
logits = model(**tokenizer("Azul fell-awen", return_tensors="pt")).logits
```

Using the standard Hugging Face pipeline:

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="agbalu/Masinissa-Sentiment", trust_remote_code=True)
print(pipe("Ulac d acu i ixuṣṣen, kulci yelha aṭas."))
# [{'label': 'positive', 'score': 0.984}]
```

## Intended use

- **Corpus-scale triage**: Classifying comment threads, survey feedback, forum posts, and user reviews by sentiment polarity.
- **Routing ambiguous instances**: Using multi-label prediction sets to identify borderline or underspecified inputs for human review.

**Not suitable for**:
- High-stakes automated decision making affecting individuals.
- Measuring public opinion without demographic controls.
- Text written in Tifinagh or Arabic scripts, or non-Kabyle Amazigh varieties.
- Unmonitored content moderation pipelines.

## Architecture

| Parameter | Value |
|---|---|
| Parameters, trained | 31,199,491 |
| Parameters, published | 31,035,651 |
| Trunk | [Masinissa-31M](https://huggingface.co/agbalu/Masinissa-31M) — 12 layers, 384 hidden, 6 heads, 1,280 intermediate |
| Position embeddings | 32 log-bucketed relative position buckets (512-token maximum) |
| Pooling | Masked mean pooling over non-padding tokens |
| Classification head | LayerNorm, 384 → 192 GeLU bottleneck, LayerNorm, 5-way multi-sample dropout, 192 → 3 |
| Vocabulary | [Mammeri-Tok](https://huggingface.co/agbalu/Mammeri-Tok), 16,000 SentencePiece Unigram tokens |

The classification head contains 75,651 parameters (0.24% of total model capacity). The 163,840-parameter masked language modeling head used during pretraining is excluded from the published checkpoint. Relative position bucket embeddings are dynamically computed from configuration at load time.

Mean pooling is applied across non-padding sequence tokens. Multi-sample dropout ([Inoue, 2019](https://arxiv.org/abs/1905.09788)) evaluates the 192-dimensional bottleneck through five independent dropout masks during training, averaging their classification losses to regularize the head. Standard deterministic inference evaluates without dropout.

## Training recipe

| Parameter | Value |
|---|---|
| Training data | `agbalu/KabSentiment` (12,000 train, 1,500 dev, 1,500 test) |
| Schedule | 6 epochs per seed, effective batch size 64 |
| Learning rates | Encoder: 3e-5, Classification head: 1e-4 |
| Optimizer & schedule | AdamW, 10% linear warmup followed by cosine decay |
| Regularization | Weight decay 0.01 (excluding norms and biases), gradient clipping 1.0, label smoothing 0.05 |
| Hardware | Single NVIDIA A10G GPU |

Differential learning rates are used to preserve pretrained encoder representations while training the classification head from random initialization. Checkpoints are selected based on validation macro-F1.

## Decontamination

1,493 of the 1,500 test sentences (99.5%) appear in the AƔBALU-Text v1 pretraining corpus via Tatoeba. However, pretraining was strictly self-supervised (masked language modeling) without sentiment labels. Sentiment supervision was introduced exclusively during downstream fine-tuning.

String overlap between `KabSentiment` train and test splits is 0.0%. One sentence (`Ulac ǧahennama yugaren ta.`) appears in both train and dev (`kab_sent_train_06003` and `kab_sent_dev_00065`) due to distinct English source sentences translating to the identical Kabyle string.

## Limitations

- **Projected labels**: Sentiment labels originate from an English teacher model and reflect cross-lingual projection rather than direct human annotation in Kabyle.
- **Sarcasm and pragmatic nuance**: Subtle tonal nuances, irony, and culturally specific idioms may not transfer cleanly through cross-lingual projection.
- **Register**: The model is trained on written standard Kabyle in Latin script and does not cover dialectal spoken transcripts, heavy code-switching with French, Arabizi, or Tifinagh.
- **Sentence length**: Training data is bounded between 4 and 25 English words; document-level classification requires chunking or document-level models.
- **Distribution**: The source pool exhibits a higher natural frequency of neutral examples (31,119 candidates) relative to positive examples (8,027); training splits are balanced by subsampling.

## Files

| File | Description |
|---|---|
| `model.safetensors` | 124.2 MB, 31,035,651 parameters (float32) |
| `config.json` | Model configuration, class labels, temperature (0.9811), and conformal quantile (0.8022) |
| `modeling_masinissa_sentiment.py`, `configuration_masinissa_sentiment.py` | PyTorch model implementation with `classify()` method |
| `tokenizer.json`, `agbalu-tok-base-16k.model` | 16,000-piece Mammeri Unigram tokenizer |
| `evaluation.json` | Detailed evaluation report, seed metrics, and calibration data |
| `export.stats.json` | Export verification metadata and SHA-256 checksums |

Training artifacts such as optimizer states and random seeds are excluded from the published release.

## Reproduction

Fine-tuned from `agbalu/Masinissa-31M` across five random seeds (42, 1337, 2026, 7, 999) for 6 epochs with AdamW, learning rate 3e-5 (encoder) and 1e-4 (head), with linear warmup over `agbalu/KabSentiment`. Each run selects its best checkpoint on validation macro-F1. Evaluated over the 1,500-sentence unseen test split, reporting mean macro-F1 and test accuracy. All metrics reproduce deterministically from the published checkpoint.

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

Please also cite the Tatoeba project for the source sentences, and Cardiff NLP for `cardiffnlp/twitter-roberta-base-sentiment-latest`, which produced the labels.

## Licence

**Apache-2.0** on weights and code. `KabSentiment` is distributed under **CC-BY-4.0** over Tatoeba-derived sentences. The underlying encoder was pretrained on AƔBALU-Text v1 (**unclear 34.9%, permissive 32.0%, share-alike 31.0%, non-commercial 2.0%**).

Part of [AƔBALU](https://huggingface.co/agbalu), a Kabyle and Amazigh corpus and model collection.
