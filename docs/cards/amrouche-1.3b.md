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

A machine translation model for **Kabyle** (Taqbaylit, `kab_Latn`), fine-tuned from
[`facebook/nllb-200-distilled-1.3B`](https://huggingface.co/facebook/nllb-200-distilled-1.3B) on
544,729 human-authored parallel sentence pairs from the AƔBALU parallel corpus (excluding NLLB web-mined data).

Amrouche-1.3B outperforms the base NLLB-200 model in all four translation directions between Kabyle, English, and French, with the largest gains observed when translating into Kabyle (**36.34 chrF++ / 10.86 BLEU on English → Kabyle** and **34.43 chrF++ / 8.39 BLEU on French → Kabyle**). The model features an embedding matrix trimmed to the 52,209 tokens active in the training domain, reducing model size and memory overhead.

## Results

All models are evaluated on FLORES+ devtest using sacreBLEU and chrF++ against the orthographically corrected `kab_Latn` reference:

| Direction | chrF++ (Base → Amrouche-1.3B) | BLEU (Base → Amrouche-1.3B) | Relative BLEU Gain |
|---|---|---|---|
| **eng → kab** | 31.53 → **36.34** (+4.81) | 7.74 → **10.86** | +40.3% |
| **fra → kab** | 30.21 → **34.43** (+4.22) | 6.33 → **8.39** | +32.5% |
| **kab → eng** | 44.60 → **46.25** (+1.65) | 23.06 → **25.29** | +9.7% |
| **kab → fra** | 42.00 → **45.10** (+3.10) | 19.46 → **22.10** | +13.6% |

- **Baseline Calibration:** The 10.86 BLEU score on `eng → kab` compares against NLLB's published baseline of 6.2 BLEU for Kabyle, reproduced at 6.02 BLEU on `nllb-200-distilled-600M` under the identical evaluation harness.
- **Target-Side Generation:** Gains are concentrated when generating Kabyle (+4.81 chrF++ on `eng → kab` and +4.22 on `fra → kab`), addressing the base model's weakness in target-language syntax and morphology, while also improving Kabyle-to-English (+1.65 chrF++) and Kabyle-to-French (+3.10 chrF++).

### Orthographic Consistency

Because the original FLORES+ `kab_Latn` dataset contains 16.2% homoglyph corruption (e.g. Greek `ε` U+03B5 instead of Latin `ɛ` U+025B), evaluation is conducted both on raw text and with both hypothesis and reference passed through the versioned AƔBALU normalizer:

| Evaluation Condition | Base NLLB-1.3B | Amrouche-1.3B |
|---|---|---|
| **eng → kab Orthography Gap (chrF++)** | +0.17 | **0.00** |
| **fra → kab Orthography Gap (chrF++)** | +0.19 | **0.00** |

Normalizing hypotheses and references yields identical scores (an orthography gap of 0.00), demonstrating that Amrouche-1.3B natively outputs canonical Kabyle Latin characters (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`) without phonetic or character drift.

## Intended Use

- **Direct Translation:** Bidirectional translation between Kabyle and English or French.
- **Pivoting:** For source languages outside English and French, pivot through English (`X → eng` using standard NLLB-200, followed by `eng → kab` using Amrouche-1.3B).
- **Out of Scope:** Direct translation from Kabyle to languages other than English or French (pivoting through `kab → eng` produces significantly higher fidelity). Not certified for safety-critical, clinical, or legally binding translations without qualified human review.

## Usage

Amrouche-1.3B uses standard `transformers` and `torch` with `M2M100ForConditionalGeneration` (no `trust_remote_code` required).

Because the embedding table was trimmed from NLLB's full 256,206 vocabulary to the 52,209 tokens active in the training data, token IDs must be remapped using `keep.json`:

```python
import json
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO = "agbalu/Amrouche-1.3B"

tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO).eval()

keep_data = json.load(open(hf_hub_download(REPO, "keep.json")))["keep"]
keep = torch.tensor(keep_data)
to_new = torch.full((max(len(tokenizer), int(keep.max()) + 1),), -1, dtype=torch.long)
to_new[keep] = torch.arange(len(keep))
unk = to_new[tokenizer.unk_token_id]

def translate(text, source="eng_Latn", target="kab_Latn"):
    tokenizer.src_lang = source
    batch = tokenizer(text, return_tensors="pt", padding=True)
    ids = to_new[batch["input_ids"]]
    batch["input_ids"] = ids.masked_fill(ids < 0, unk)
    with torch.inference_mode():
        out = model.generate(
            **batch,
            forced_bos_token_id=int(to_new[tokenizer.convert_tokens_to_ids(target)]),
            num_beams=4,
            max_length=256,
        )
    return tokenizer.batch_decode(keep[out], skip_special_tokens=True)

print(translate(["The house is big.", "I speak Kabyle.", "Water is life."]))
# Output: ['Meqqer uxxam-nni.', 'Heddreɣ taqbaylit.', 'Aman d tudert.']
```

**Vocabulary Mapping Details:**
Token IDs cross between the tokenizer and model representations across four interfaces:
1. Source `input_ids` mapped to the trimmed indices (unseen tokens map to `unk`).
2. Target-language forced BOS token ID (`kab_Latn`, `eng_Latn`, `fra_Latn` are all preserved).
3. Generated output token IDs mapped back to full tokenizer IDs via `keep`.
4. Special tokens (IDs 0–49, including `<s>`, `<pad>`, `</s>`, `<unk>`) map identically (identity mapping).

## Architecture

The model retains the standard NLLB `M2M100ForConditionalGeneration` architecture, modifying only the embedding matrix dimensions:

| Parameter | Specification |
|---|---|
| **Parameters** | **1,161,745,408** |
| **Base Model** | `facebook/nllb-200-distilled-1.3B` |
| **Encoder / Decoder Layers** | 24 / 24 |
| **Hidden / Feed-Forward Dim** | 1,024 / 8,192 (ReLU) |
| **Attention Heads** | 16 encoder, 16 decoder |
| **Maximum Context Window** | 1,024 tokens (sinusoidal positional embeddings) |
| **Vocabulary Size** | **52,209** of base 256,206 tokens (tied input/output embeddings) |

Trimming 203,997 unused vocabulary rows reduces model size by 0.84 GB without removing active Kabyle, English, or French tokens.

## Training Recipe

| Parameter | Specification |
|---|---|
| **Corpus Size** | **544,729 parallel pairs → 1,089,458 examples** (1,085,458 train / 4,000 dev across all 4 directions) |
| **Data Selection** | Non-mined, human-authored pairs; excluded 7,805 pairs with severe character defects |
| **Recipe** | Full parameter fine-tuning (following arXiv 2602.04442) |
| **Optimizer** | Adafactor with gradient checkpointing |
| **Effective Batch Size** | 2,048 sequences (learning rate 2e-4) |
| **Training Steps** | **1,050 steps** (2 epochs; final train loss 2.372) |
| **Validation Loss** | **2.302** |
| **Vocabulary Trimming** | **52,209 tokens** (20.4% of base 256,206 vocabulary) |

- **Exclusion of Mined Bitext:** Approximately 90.1% of publicly available Kabyle parallel data is bitext mined by NLLB with unverified precision. To prevent distilling base-model errors, all mined pairs were excluded. The remaining 544,729 human-authored pairs provide 7.7× more verified Kabyle bitext than reported in the original NLLB paper (72,000 sentences).
- **Language Verification:** Audited against language identification models to prevent sibling Berber contamination: 98.31% of the Kabyle text is confirmed `kab_Latn`, with 14 lines out of 38,486 flagged as sibling Berber. The remaining 1.69% consists of untranslated English and French strings from software localization exports.
- **Full Fine-Tuning vs. LoRA:** Full parameter fine-tuning was selected over parameter-efficient methods based on benchmark evidence for low-resource translation (e.g. arXiv 2404.04212, which reported 30.25 BLEU for full fine-tuning compared to 18.63 for LoRA).

## Limitations

- **Vocabulary Trimming Floor:** Because the embedding table was trimmed to the active vocabulary of the training split, words in evaluation sets that were absent during training (0.07% to 0.22% of tokens, affecting 28–55 sentences per thousand in FLORES+) are mapped to `unk`. The baseline model does not have this constraint, making the reported benchmark improvements a conservative lower bound.
- **Human Adequacy Evaluation:** Automated metrics (BLEU, chrF++) are reported. Formal human adequacy and fluency evaluations have not yet been conducted on large-scale Kabyle machine translation outputs.
- **Sibling Language Identifiers:** Standard language identification tools (such as GlotLID and NLLB `lid218e`) have limited support for distinguishing Northern Berber varieties (Tarifit, Central Atlas Tamazight, Shawiya), occasionally conflating them with `kab_Latn`. While the training corpus was curated to exclude non-Kabyle content, automated filtering at the web scale retains residual ambiguity.
- **Domain Specificity:** Benchmark scores are derived from FLORES+ (formal Wikipedia prose). Generalization to colloquial speech, conversational register, or heavily code-switched text has not been systematically evaluated.

## Files

| File | Description | Size |
|---|---|---|
| `model.safetensors` | Model weights (1,161,745,408 parameters) | ~4.65 GB |
| `config.json` | Architecture configuration (`vocab_size: 52209`) | < 5 KB |
| `generation_config.json` | Decoder generation parameters and special token IDs | < 1 KB |
| `keep.json` | Token mapping index containing the 52,209 active base vocabulary IDs | ~300 KB |
| `tokenizer.json`, `tokenizer_config.json` | Full unmodified NLLB tokenizer (256,206 tokens) to preserve language routing | ~17 MB |

## Reproduction

Fine-tuned from `facebook/nllb-200-distilled-1.3B` on 544,729 verified parallel pairs using the trimmed embedding matrix (52,209 tokens) with Adafactor (effective batch size 2,048, 1,050 steps). Evaluated across all four translation directions on the decontaminated FLORES+ `kab_Latn` test split using sacreBLEU and chrF++. All evaluation metrics reproduce deterministically from the published checkpoint.

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

- **Model Weights & Code:** Apache-2.0.
- **Base Model Terms:** Subject to Meta's NLLB-200 licence terms (CC-BY-NC-4.0).
- **Training Data Composition:** The 5.56M-pair parallel corpus from which this fine-tuning set was curated consists of:
  - `unclear`: 4,871,469 pairs (87.6%)
  - `permissive`: 376,389 pairs (6.8%)
  - `non-commercial`: 297,058 pairs (5.3%)
  - `share-alike`: 16,710 pairs (0.3%)

`unclear` indicates sources lacking an explicit machine-readable licence. Full provenance records are maintained in the AƔBALU corpus registry.
