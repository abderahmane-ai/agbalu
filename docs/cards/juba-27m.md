---
language:
- kab
license: apache-2.0
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- transliteration
- tifinagh
- schwa-restoration
- low-resource
pipeline_tag: translation
metrics:
- cer
- exact_match
model-index:
- name: Juba-27M
  results:
  - task:
      type: translation
      name: Script conversion, Tifinagh to Kabyle Latin
    dataset:
      type: agbalu/KabTifinagh
      name: agbalu/KabTifinagh test split
    metrics:
    - type: exact_match
      value: 0.9422
      name: Sentence exact match (free-running, 5,000 sentences)
    - type: cer
      value: 0.0033
      name: Character error rate (free-running, 5,000 sentences)
    - type: f1
      value: 0.9351
      name: Schwa placement F1
---

# Juba-27M

A 26.9M-parameter character-level Conv-Transformer sequence-to-sequence model for **Kabyle** (Taqbaylit, `kab`), trained from scratch to convert Neo-Tifinagh (`ⵜⴰⵇⴱⴰⵢⵍⵉⵜ`) into standard Kabyle Latin (`Taqbaylit`) with contextual schwa (`e`) restoration.

While consonant mappings between Neo-Tifinagh and Latin are largely one-to-one, standard Kabyle Neo-Tifinagh omits the neutral vowel `e`. Where schwas appear is governed by local phonotactic constraints in consonant clusters. Juba-27M models this context using a convolutional stem and Transformer encoder-decoder, achieving **94.22% sentence exact match**, **0.33% character error rate (CER)**, and **93.51% schwa placement F1** under free-running greedy decoding.

## Results

Evaluated on the held-out test split of `agbalu/KabTifinagh` (5,000 sentences, free-running autoregressive decoding):

| System | Sentence Exact Match | Character Error Rate (CER) | Schwa Placement F1 |
|---|---|---|---|
| **Juba-27M** | **94.22%** | **0.33%** | **93.51%** |
| Deterministic Character Lookup | 1.16% | 13.55% | 1.61% |

- **Placement Metric:** Schwa placement F1 strictly measures the placement of restored `e` vowels relative to the consonant skeleton rather than count alone.
- **Autoregressive Decoding:** Scored under free-running generation without teacher forcing.
- **Error Distribution:** The majority of residual sentence-level errors (5.78%) occur in complex hyphenated clitic chains (e.g. `yefka-yas-d-t.`), where the model predicts schwas and boundary hyphens simultaneously from context.

## Intended Use

- **Script Conversion:** Automated conversion of Neo-Tifinagh Kabyle (`kab_Tfng`) text into standard Kabyle Latin (`kab_Latn`) for downstream NLP tasks.
- **Vowel Restoration:** Contextual restoration of omitted schwas in unvocalized Berber text.
- **Out of Scope:** Translation to non-Berber languages; historical Libyco-Berber or non-Kabyle Tifinagh scripts (e.g. Tuareg).

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = "agbalu/Juba-27M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO, trust_remote_code=True).eval()

def transliterate(text, num_beams=1):
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, num_beams=num_beams, max_length=256)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

print(transliterate("ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?"))
# Output: 'tecfiḍ fell-i ?'

print(transliterate("ⴰⵣⵓⵍ ⴼⵍⵍⴰⵡⵏ, ⴰⵎⴽ ⵜⵜⵉⵍⵉⴹ ?"))
# Output: 'azul fell-awen, amek ttiliḍ ?'
```

**Implementation notes:**
- Batching: Process sequences individually or pad uniformly; the self-attention mechanism attends over padding tokens during inference consistent with evaluation.
- Output casing: The vocabulary is defined over lowercase characters. Sentence-initial and proper noun capitalization can be restored using [`Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M).

## Architecture

| Parameter | Value |
|---|---|
| **Architecture** | Character-level Conv-Transformer (`JubaForSeq2SeqLM`) |
| **Total Parameters** | 26,901,888 |
| **Encoder / Decoder Layers** | 6 / 6 |
| **Hidden / Feed-Forward Dim** | 384 / 1,152 (SwiGLU) |
| **Attention Heads** | 6 heads (head dimension 64) |
| **Positional Embeddings** | Rotary Position Embeddings (RoPE) on self-attention |
| **Convolutional Stem** | 1D depthwise-separable convolution (kernels 3 and 5) |
| **Vocabulary** | 128 character slots (96 active characters; tied input/output embeddings) |
| **Label Smoothing** | $\epsilon = 0.05$ |

The convolutional stem extracts local n-gram context from the consonant skeleton prior to attention, providing the local inductive bias necessary for vowel prediction.

## Training Recipe

| Parameter | Value |
|---|---|
| **Dataset** | `agbalu/KabTifinagh` (497,944 sentence pairs: 398,355 train, 49,794 dev, 49,795 test) |
| **Objective** | Character-level cross-entropy with label smoothing ($\epsilon = 0.05$) |
| **Optimizer** | AdamW (peak lr 5e-4, $\beta=(0.9, 0.95)$, weight decay 0.1) |
| **Batch Size** | Micro-batch 64 $\times$ gradient accumulation 2 = effective batch size 128 |
| **Schedule** | Cosine annealing to $\eta_{\min} = 10^{-5}$ over 15,500 steps (~5 epochs) |
| **Gradient Clipping** | Maximum norm 1.0 |
| **Seed** | 42 |

The Latin targets were normalized under reference normalizer v1.3.0, ensuring consistent orthography across vowels and spirantized consonants.

## Limitations

- **Hyphenated Clitics:** Clitic chains requiring simultaneous hyphen and vowel prediction remain the primary source of error.
- **Unicameral Source:** Neo-Tifinagh text is unicameral and lacks capitalization; output text is emitted in lowercase.
- **Synthetic Alignment:** Training sentence pairs were constructed by applying deterministic Latin-to-Tifinagh projection to normalized text, meaning error rates are measured on round-trip restoration rather than independently transcribed archival Tifinagh.
- **Dialect Scope:** Calibrated specifically for Kabyle phonology; not intended for Tashelhit or Central Atlas Tamazight.

## Files

| File | Description | Size |
|---|---|---|
| `model.safetensors` | Model weights (145 tensors, 26,901,888 parameters) | ~108 MB |
| `config.json` | Architecture configuration and `auto_map` | < 5 KB |
| `configuration_juba.py`, `modeling_juba.py` | Standalone Transformers modeling code | < 30 KB |
| `tokenizer.json`, `tokenizer_config.json` | 96-character alphabet tokenizer | < 10 KB |
| `export.stats.json` | Export manifest and file verification | < 5 KB |

## Reproduction

Trained from scratch with AdamW (lr 5e-4, 15,500 steps) on the 398,355-pair training split of `agbalu/KabTifinagh`. Evaluated with free-running greedy decoding over 5,000 sentences of the held-out test split, comparing character error rate, sentence exact match, and schwa placement F1 against the deterministic baseline. All reported metrics reproduce deterministically from the published checkpoint.

## The name

**Juba II** (r. 25 BCE – 23 CE) was a Numidian king who wrote in Greek on geography and
Berber history, and whose court at Caesarea worked across scripts and languages at once.
The model that carries his name does the inverse of his career: it recovers Kabyle Latin
orthography from a script that encodes it without vowels.

This project had previously ruled Juba out for the translation model, on the grounds that a
Roman client king writing in the empire's language is the wrong figure for a
language-sovereignty project. That reasoning was about translation. Moving *between
scripts* is what his life actually was, and it is what this model does.

The naming is homage; it implies no endorsement by anyone.

## Citation

```bibtex
@software{agbalu_juba_2026,
  title  = {Juba-27M: Character-Level Script Conversion and Schwa Restoration for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Juba-27M},
  note   = {Trained on agbalu/KabTifinagh; normaliser 1.3.0+rules1.0.0}
}
```

## Licence

- **Weights and Code:** Apache-2.0.
- **Training Data:** Derived from `agbalu/KabTifinagh`.
