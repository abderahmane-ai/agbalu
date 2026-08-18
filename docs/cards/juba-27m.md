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

A 26.9M-parameter character-level Conv-Transformer for **Kabyle** (Taqbaylit, `kab`),
trained from scratch to convert Neo-Tifinagh (`ⵜⴰⵇⴱⴰⵢⵍⵉⵜ`) into standard Kabyle Latin
(`Taqbaylit`) — and, in doing so, to restore the vowel `e` that Tifinagh does not write.

**The vowel is the whole task.** Every consonant maps one-to-one between the two scripts,
so a lookup table converts them in an afternoon. Kabyle Neo-Tifinagh omits the schwa, and
where it belongs is a function of the surrounding consonants. That is a sequence problem,
and it is why a character table reaches **1.2% sentence exact match** on the test split
where this model reaches **94.2%**.

## Results

Test split of `agbalu/KabTifinagh`: 49,795 sentences, disjoint from train and dev, never
seen during training or model selection. **Free-running greedy decoding** — the model is
fed its own output, which is what a caller gets.

| system | sentence exact match | character error rate | schwa placement F1 |
|---|---|---|---|
| **Juba-27M** | **94.22%** | **0.33%** | **93.51%** |
| deterministic character table | 1.16% | 13.55% | 1.61% |

Both rows are measured on the same 5,000 sentences in the same pass, by
`agbalu.bench.tifinagh`. The character table's figures over the full 49,795 are 1.02% /
13.49% / 1.72%, so the sample is not flattering it.

Reproduce either without a GPU:

```bash
make tifinagh TASK=evaluate LIMIT=5000   # the model
make bench TASK=tifinagh                 # the table, over the whole split
```

Three things worth reading carefully.

**The baseline is a table, not another model.** No neural model had been trained for
Kabyle script conversion or schwa restoration before this one. The comparison is therefore
against the only tool that existed, and the gap is not incremental: the table gets 1.2% of
sentences exactly right because it gets almost every schwa wrong, and one wrong character
fails a sentence.

**Schwa F1 measures placement, not count.** A hypothesis with the right *number* of `e`
in the wrong positions scores 100% under a count and 0% here. The metric removes every `e`
from both strings, compares the consonant skeletons, and matches the vowels by their index
into that skeleton; a hypothesis whose skeleton differs has no position to be judged at, so
its vowels are charged to both error columns rather than dropped. 251 of 5,000 sentences
fall in that class.

**Character accuracy overstates this task and sentence exact match is the honest number.**
0.33% CER means roughly one character in three hundred; 5.78% of sentences still contain
at least one error. Most are hyphenated clitic chains — `yefka-yas-d-t.` — where the model
must predict schwa placement *and* hyphen boundaries at once, and the hyphens are not in
the Tifinagh source at all.

## Intended use

Converting Neo-Tifinagh Kabyle (`kab_Tfng`) into Kabyle Latin (`kab_Latn`): archival and
educational corpora into a form the rest of the NLP stack can read, and schwa restoration
for any Tifinagh source where the vowel was omitted by convention.

The reverse direction needs no model. Latin to Tifinagh is a character table, and it is
lossless in that direction because Tifinagh simply does not write the schwa this model
restores.

**Not suitable for**: translation between Kabyle and any other language (use
[Amrouche-1.3B](https://huggingface.co/agbalu/Amrouche-1.3B)); any decision about a person;
or any language other than Kabyle. Tarifit, Tashelhit and Central Atlas Tamazight share
part of the orthography and are *not* evaluated here — this project does not treat them as
Kabyle, and neither should a caller.

## Usage

`transformers` and `torch`, nothing else. The architecture is not one of the library's own,
so the modelling code travels in this repository and `trust_remote_code=True` is what loads
it.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = "agbalu/Juba-27M"
tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO, trust_remote_code=True).eval()


def transliterate(text, num_beams=4):
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, num_beams=num_beams, max_length=256)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


transliterate("ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?")
# 'tecfiḍ fell-i ?'
transliterate("ⴰⵣⵓⵍ ⴼⵍⵍⴰⵡⵏ, ⴰⵎⴽ ⵜⵜⵉⵍⵉⴹ ?")
# 'azul fell-awen, amek ttiliḍ ?'
```

Every example above was executed against the published directory before being written down,
and both decodings are **free-running** — the model is fed its own output. `num_beams=1` is
the greedy decode; on these sentences it agrees with beam search.

**Batch rows of equal length, or decode one at a time.** The model attends over padding and
`attention_mask` is accepted and ignored, because that is how the weights were evaluated;
honouring a mask here would return different logits from the numbers above.

There is **no key-value cache**. Every step re-reads the whole prefix, which is what the
published evaluation did and what makes the two agree. Inference is a few milliseconds per
sentence on CPU; nothing here needs a GPU.

**Output is lowercase.** The model is defined over a case-folded alphabet and does not
restore capitalisation; `.capitalize()` covers the sentence-initial case and proper nouns
do not have a solution here. A snippet showing capitalised output would not reproduce.

## Architecture

| | |
|---|---|
| Parameters | **26,901,888** |
| Encoder / decoder layers | 6 / 6 |
| Hidden / feed-forward | 384 / 1,152 (SwiGLU) |
| Attention heads / head size | 6 / 64 |
| Positions | rotary (RoPE), on self-attention only |
| Conv stem | 1D depthwise separable, kernels 3 and 5, before the encoder |
| Vocabulary | 128 character slots; 96 used, 0 out-of-vocabulary on Kabyle |
| Tied weights | input embedding is the output projection |
| Label smoothing | ε = 0.05 |

**The convolutional stem is not decoration.** Tifinagh writes consonant clusters with the
vowel deleted, so the evidence for where a schwa belongs is the adjacent two to five
characters. Depthwise kernels of 3 and 5 hand the first encoder layer that window already
computed, instead of spending attention capacity on adjacency.

**Rotary rather than absolute positions**, because the cue is the distance between
consonants, not where in the sentence the cluster sits. RoPE is applied to self-attention
only: cross-attention relates two sequences of different lengths, and a shared index there
would assert an alignment that schwa restoration falsifies.

**Tied embedding and output projection**, because the input and output alphabets are the
same alphabet. That is the correct constraint, not a saving.

## Training data

`agbalu/KabTifinagh`, `script_conversion` config — 497,944 sentence pairs, split
398,355 / 49,794 / 49,795 at seed 42 with no sentence in two splits.

The Latin side is normalised under **1.3.0+rules1.0.0**, the same normaliser as
Masinissa-31M and Amrouche-1.3B. That is load-bearing rather than tidy: a corpus where `ɛ`
is sometimes Greek epsilon teaches two consonant contexts where the language has one.

The 123,852 English and 205,637 French trilingual alignments in the same dataset were not
used to train this model.

## Training recipe

| | |
|---|---|
| Objective | character cross-entropy, label smoothing ε = 0.05 |
| Optimiser | AdamW, lr 5e-4, β (0.9, 0.95), weight decay 0.1 |
| Batch | micro-batch 64 × gradient accumulation 2 = 128 |
| Schedule | cosine annealing to η_min 1e-5 over 15,500 steps |
| Steps | 15,500 optimizer steps, ~5 epochs |
| Gradient clipping | max norm 1.0 |
| Hardware | one NVIDIA A10G (Modal), detached |
| Seed | 42 |

```bash
make modal-tifinagh TASK=train
```

## Limitations

**Hyphenated clitic chains are the hardest cases** and account for most of the 5.78%
sentence-level error. The hyphens are absent from the Tifinagh source and are predicted
entirely from morphosyntactic context.

**Capitalisation is not restored**, as above — Neo-Tifinagh is unicameral, so the information
is not in the input. [`agbalu/Belaid-31M`](https://huggingface.co/agbalu/Belaid-31M) supplies
it downstream, at 0.933 F1 on non-initial proper nouns.

**The training pairs are rule-derived, not human-transcribed.** The Tifinagh side was
produced by applying a deterministic Latin→Tifinagh mapping to Kabyle Latin text, and the
model learns to invert it. Any systematic error in that mapping is inherited, and the error
rates above are measured on the round trip — not against independently authored Tifinagh.
This is the single largest caveat on the numbers.

**Kabyle Neo-Tifinagh only.** Tuareg Tifinagh, Moroccan variants and Libyco-Berber
inscriptions are out of scope and untested.

**One task, one corpus.** OCR post-correction, keyboard input and archival digitisation
are plausible uses and are not evaluated here.

**No safety evaluation of any kind** has been performed.

## Files

| file | contents |
|---|---|
| `model.safetensors` | 145 tensors, 26,901,888 parameters |
| `config.json` | the architecture, and the `auto_map` that points at the code below |
| `configuration_juba.py`, `modeling_juba.py` | the architecture in code, importing only `torch` and `transformers` |
| `tokenizer.json`, `tokenizer_config.json` | the 96-character alphabet, id for id with the one the model was trained on |
| `export.stats.json` | what the source checkpoint held, and what the export dropped |

The training checkpoint this was exported from holds **the weights, the config and the
final dev accuracy — and nothing else**. It carries no optimizer state, no scheduler state
and no step-by-step validation curve, so training cannot be resumed from the published
files. `export.stats.json` records the source's contents, which is what makes that
checkable rather than a claim.

`lm_head.weight` is absent from `model.safetensors` on purpose: it is `embed.weight` under
a second name, safetensors refuses to write shared storage twice, and the module re-ties it
on load.

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
  title  = {Juba-27M: character-level script conversion and schwa restoration for Kabyle},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Juba-27M},
  note   = {Trained on agbalu/KabTifinagh; normaliser 1.3.0+rules1.0.0}
}
```

## Licence

**Apache-2.0** on the weights and the code. The training data derives from AƔBALU-Text v1;
a permissive grant on weights makes no claim about the text they were trained on, so read
`agbalu/KabTifinagh`'s licence before redistributing derivatives.
