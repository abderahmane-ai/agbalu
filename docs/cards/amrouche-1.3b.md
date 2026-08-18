---
language:
- kab
- en
- fr
license: apache-2.0
base_model: facebook/nllb-200-distilled-1.3B
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- translation
- nllb
- low-resource
pipeline_tag: translation
metrics:
- chrf
- bleu
model-index:
- name: Amrouche-1.3B
  results:
  - task:
      type: translation
      name: Translation (English to Kabyle)
    dataset:
      type: flores_plus
      name: FLORES+ devtest, orthographically corrected kab_Latn
    metrics:
    - type: chrf
      value: 36.34
      name: chrF++
    - type: bleu
      value: 10.86
      name: BLEU
---

# Amrouche-1.3B

A machine translation model for **Kabyle** (Taqbaylit, `kab`), fine-tuned from
[NLLB-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) on
544,729 human-authored pairs — everything in the AƔBALU parallel corpus that NLLB did *not*
mine itself.

It beats its own base model in **all four directions**, and by the widest margin where it
matters most: **into** Kabyle.

## Results

chrF++ and BLEU on FLORES+ devtest, against the orthographically corrected `kab_Latn`
reference. Same harness for every row.

| direction | chrF++ base → ours | BLEU base → ours |
|---|---|---|
| kab→eng | 44.60 → **46.25** (+1.65) | 23.06 → **25.29** (+2.23) |
| **eng→kab** | 31.53 → **36.34** (**+4.81**) | 7.74 → **10.86** (**+40.3%**) |
| kab→fra | 42.00 → **45.10** (+3.10) | 19.46 → **22.10** (+2.64) |
| **fra→kab** | 30.21 → **34.43** (**+4.22**) | 6.33 → **8.39** (**+32.5%**) |

**10.86 BLEU on eng→kab against NLLB's own published 6.2 for Kabyle**, which this same
harness reproduces at 6.02 on the 600M model — so the comparison is calibrated, not asserted.

**The gains are asymmetric on purpose.** Generating Kabyle is the unsolved half: kab→eng was
already at 44.60 on the base model because English generation is solved, and it gains 1.65.
eng→kab gains 4.81. The corpus is doing its work on the side where there was work to do.

### The orthography gap closes to zero

Every result is scored twice — as-published, and with both sides passed through the
reference normaliser — and the difference reported. FLORES+ `kab_Latn` is 16.2%
homoglyph-corrupted and was never revised upstream, so a system that spells Kabyle
correctly is penalised by the reference itself.

| | base NLLB-1.3B | Amrouche-1.3B |
|---|---|---|
| eng→kab gap | +0.17 | **0.00** |
| fra→kab gap | +0.19 | **0.00** |

Normalising both sides no longer changes the score in any direction. The model already
spells Kabyle canonically — the corpus-level orthographic repair showing up as model output.

## Intended use

Translation between Kabyle and English or French, in either direction. For a source language
this model was not trained on, pivot through English: `X→eng` with stock NLLB, then
`eng→kab` here, which is the direction with the largest gain.

**Not suitable for**: kab→X where X is neither English nor French — the base model's ability
survives but gains nothing here, and pivoting through `kab→eng` (46.25 chrF++, the strongest
direction) will do better. Also not suitable for safety-critical, legal or medical
translation. No adequacy evaluation by human annotators has been performed on the training
corpus or the output.

## Usage

**This model has a trimmed vocabulary, so `from_pretrained` alone is not enough.** The
embedding was cut to the 52,209 tokens the fine-tuning corpus uses, while the tokenizer
shipped beside it still speaks NLLB's full 256,206. Ids must be translated in both
directions or the output is fluent nonsense. `keep.json` is the translation table, and it
is in this repository.

```python
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = "agbalu/Amrouche-1.3B"

tokenizer = AutoTokenizer.from_pretrained(REPO)              # full 256k vocabulary
model = AutoModelForSeq2SeqLM.from_pretrained(REPO).eval()   # trimmed 52,209 rows

keep = torch.tensor(json.load(open(hf_hub_download(REPO, "keep.json")))["keep"])
to_new = torch.full((max(len(tokenizer), int(keep.max()) + 1),), -1, dtype=torch.long)
to_new[keep] = torch.arange(len(keep))
unk = to_new[tokenizer.unk_token_id]


def translate(text, source="eng_Latn", target="kab_Latn"):
    tokenizer.src_lang = source
    batch = tokenizer(text, return_tensors="pt", padding=True)
    ids = to_new[batch["input_ids"]]
    batch["input_ids"] = ids.masked_fill(ids < 0, unk)       # unscanned token -> unk
    with torch.inference_mode():
        out = model.generate(
            **batch,
            forced_bos_token_id=int(to_new[tokenizer.convert_tokens_to_ids(target)]),
            num_beams=4,
            max_length=256,
        )
    return tokenizer.batch_decode(keep[out], skip_special_tokens=True)


translate(["The house is big.", "I speak Kabyle.", "Water is life."])
# ['Meqqer uxxam-nni.', 'Heddreɣ taqbaylit.', 'Aman d tudert.']
```

**Ids cross the boundary in four places**, and a version that handles three of them returns
plausible text in the wrong language: the source `input_ids`, the forced target-language
token, the generated ids on the way back, and `unk` itself. Language codes are NLLB's —
`kab_Latn`, `eng_Latn`, `fra_Latn` — and all of them survive the trim.

Special tokens are unaffected: ids 0–49 map to themselves, so `<s>`, `<pad>`, `</s>` and
`<unk>` keep their usual values.

## Training

| | |
|---|---|
| Base | `facebook/nllb-200-distilled-1.3B` |
| Corpus | **544,729 pairs → 1,089,458 examples** (train 1,085,458 / dev 4,000), all four directions |
| Selection | everything NLLB did not mine, minus 7,805 hard-defect pairs |
| Recipe | published, arXiv 2602.04442 — effective batch 2,048, lr 2e-4, 2 epochs |
| Optimiser | Adafactor with gradient checkpointing (24 GiB A10) |
| Steps | **1,050**, 6.9 hours on one A10, final train loss 2.372 |
| Best eval loss | **2.302** |
| Vocabulary | trimmed to **52,209 of 256,206 tokens (20.4%)** → 1,161,745,408 parameters |

**The corpus deliberately excludes NLLB's own mined bitext.** 90.1% of the available Kabyle
parallel data is NLLB-mined with unmeasured precision; training a NLLB derivative on it would
be distilling the base model's own output. What remains — 552,534 non-mined pairs, 512,049 of
them defect-free — is still **7.7× more public Kabyle bitext than NLLB's paper reports having
seen** (72,000 sentences, Table 12).

**The fine-tuning corpus was checked for sibling contamination before any GPU time was
spent**: 98.31% of its judged Kabyle side is `kab_Latn`, and 14 lines of 38,486 carry a Berber
sibling label. The 1.69% residue is untranslated English and French strings from localisation
exports, not sibling text.

**LoRA was ruled out** by measurement in the literature, not by preference — arXiv 2404.04212
reports 18.63 BLEU against full fine-tuning's 30.25 on this class of task.

## Limitations

**The vocabulary is trimmed, and this understates the scores above.** The embedding table was
scanned from the fine-tuning corpus and cut to 20.4% of NLLB's. FLORES+ then contains tokens
the scan never saw — 0.07–0.22% of tokens, but **28–55 sentences per thousand** — which are
mapped to `unk` at generation. The baselines carry no such handicap, so the reported gains are
a floor.

**Adequacy is unmeasured, and will stay unmeasured.** 90.1% of the *available* Kabyle parallel
data is NLLB bitext-mined with unmeasured precision. This model's corpus excludes it, but the
excluded portion's quality was never human-annotated either. The measured mechanical defect
rate is a lower bound on errors, **not** a precision figure — do not quote it as one.

**Sibling-language contamination is bounded, not cleared.** Neither GlotLID nor NLLB's own
`lid218e` can *name* Tarifit, Central Atlas Tamazight or Shawiya, so a `kab_Latn` label cannot
exclude them. Measured on a balanced set, `lid218e` labels 87–95% of Tashelhit, Tarifit and
Central Atlas Tamazight as Kabyle — and that identifier is what mined most public Kabyle
bitext.

**One benchmark, one domain.** FLORES+ is Wikipedia-derived prose. Nothing here says how the
model behaves on speech transcripts, dialogue, or the localisation strings that make up a
sixth of the training corpus.

## Licence composition of the training text

The weights are Apache-2.0. That grant does not relicense the text they were trained on. By
licence, the 5.56M-pair parallel corpus this fine-tuning set was drawn from is: `unclear`
4,871,469 pairs, permissive 376,389, non-commercial 297,058, share-alike 16,710. The
`unclear` bulk — not the non-commercial slice — is the real redistribution risk, and it is
published here rather than left for someone to discover.

## The name

**Taos Amrouche** (1913–1976) sang exclusively in Kabyle and wrote in French. She published
*Jacinthe noire* in 1947, the first novel by a Kabyle woman; from 1936 she collected and
performed the Kabyle songs her mother had preserved; and she co-founded the Académie berbère
in 1966. One of her albums is *Chants sauvés de l'oubli* — songs saved from oblivion.

A life spent carrying Kabyle into another language without ever surrendering it is what a
translation model is for. Her mother, **Fadhma Aït Mansour Amrouche**, who from 1930 began
writing down the songs and tales inherited from her ancestors, gives her name to the speech
recognition model — the two halves of one family's work, split the way the two models are.

The naming is homage. It implies no endorsement, and neither Taos Amrouche nor her family is
affiliated with this work.

## Citation

```bibtex
@software{agbalu_amrouche_2026,
  title  = {Amrouche-1.3B: Kabyle machine translation},
  author = {AƔBALU},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Amrouche-1.3B},
  note   = {Fine-tuned from facebook/nllb-200-distilled-1.3B on 544,729 non-mined pairs}
}
```

## Licence

**Apache-2.0** for the weights. NLLB-200 itself is CC-BY-NC-4.0; check the base model's terms
for your use case, and read the training-text composition above before redistributing
derivatives.
