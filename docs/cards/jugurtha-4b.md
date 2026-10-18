---
language:
- kab
- en
- fr
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- text-generation
- causal-lm
- instruction-tuned
- reasoning
- thinking
- sovereign-ai
- low-resource
metrics:
- accuracy
- chrf
- bleu
model-index:
- name: Jugurtha-4B
  results:
  - task:
      type: text-generation
      name: Long-Context Retrieval (Needle-in-a-Haystack, 32k)
    dataset:
      type: ruler_niah
      name: RULER NIAH Single (32,768 tokens)
    metrics:
    - type: accuracy
      value: 100.0
      name: needle-in-a-haystack (32k context)
  - task:
      type: text-generation
      name: Mathematical Reasoning (GSM8K)
    dataset:
      type: gsm8k
      name: GSM8K (chat)
    metrics:
    - type: exact_match
      value: 80.89
      name: gsm8k
  - task:
      type: text-generation
      name: Multidisciplinary Knowledge & Reasoning (MMLU-Pro)
    dataset:
      type: TIGER-Lab/MMLU-Pro
      name: MMLU-Pro (chat)
    metrics:
    - type: exact_match
      value: 62.86
      name: mmlu_pro
  - task:
      type: text-generation
      name: Instruction Following (IFEval)
    dataset:
      type: google/IFEval
      name: IFEval
    metrics:
    - type: accuracy
      value: 54.16
      name: ifeval (prompt-level strict)
  - task:
      type: text-classification
      name: Kabyle Topic Classification (SIB-200)
    dataset:
      type: Davlan/sib200
      name: SIB-200 (kab_Latn, 7-shot)
    metrics:
    - type: accuracy
      value: 73.74
      name: sib-200 (kab_Latn)
  - task:
      type: translation
      name: Translation (Kabyle to English)
    dataset:
      type: openlanguagedata/flores_plus
      name: FLORES+ devtest (kab_Latn -> eng_Latn)
    metrics:
    - type: chrf
      value: 46.17
      name: chrF++
    - type: bleu
      value: 23.79
      name: BLEU
  - task:
      type: translation
      name: Translation (English to Kabyle)
    dataset:
      type: openlanguagedata/flores_plus
      name: FLORES+ devtest (eng_Latn -> kab_Latn)
    metrics:
    - type: chrf
      value: 28.77
      name: chrF++
    - type: bleu
      value: 6.73
      name: BLEU
---

# Jugurtha-4B

A 4.2B-parameter generative model for **Kabyle** (Taqbaylit, `kab`, Latin script) with native
chain-of-thought reasoning in Taqbaylit. It is, as far as we can establish, **the first
general-purpose generative model to produce verified, orthographically correct Kabyle** across
conversation, translation, and multi-step reasoning.

It achieves **100.0% NIAH at 32k context**, **80.89% on GSM8K**, **62.86% on MMLU-Pro**, and lifts
Kabyle topic classification from 20% to **73.74%** on SIB-200 — while retaining **73.96 points**
across seven general-purpose English and French benchmarks.

## Highlights

- **Dual-mode reasoning**: standard conversational mode and transparent step-by-step reasoning in
  Kabyle (`<think>...</think>`) — switchable per request with `enable_thinking`
- **100% NIAH at 32k**: perfect single-needle retrieval across all three RULER variants at full context
- **Native Kabyle generation**: lifts `kab→eng` from 14 chrF++ to **46.17**, `eng→kab` from 10 to
  **28.77**, with the orthography gap closing to **0.00**
- **Multilingual retention**: 73.96-point mean across MMLU, Belebele, GSM8K, ARC-Challenge and three
  others — within 0.20 of the general-purpose baseline
- **No `trust_remote_code`**: standard `transformers` loading, BF16, `device_map="auto"`

## Results

Evaluated on an NVIDIA H100 NVL MIG slice (`lm-eval` 0.4.13). Translation scored 5-shot against the
orthographically corrected FLORES+ `kab_Latn` reference ([`agbalu/KabBench`](https://huggingface.co/datasets/agbalu/KabBench)).

### Kabyle & multilingual

| Benchmark / Task | Metric | Score |
|---|---|---:|
| **Needle-in-a-Haystack (32k)** | Accuracy | **100.00%** |
| **SIB-200 (`kab_Latn`, 7-shot)** | Accuracy (chance 14.29%) | **73.74%** |
| **FLORES+ `kab → eng`** | 5-shot chrF++ / BLEU | **46.17 / 23.79** |
| **FLORES+ `eng → kab`** | 5-shot chrF++ / BLEU | **28.77 / 6.73** |
| **FLORES+ Orthography Gap** | chrF++ (normalised − raw) | **0.00** |
| **Held-out Kabyle** | Bits per character (2,000 docs) | **1.578** |

### Reasoning & knowledge

| Benchmark | Metric | Score |
|---|---|---:|
| **GSM8K** | Exact match | **80.89%** |
| **MMLU-Pro** | Exact match | **62.86%** |
| **IFEval** | Prompt-level strict | **54.16%** |
| **Multilingual Retention Mean** | 7-task mean | **73.96** |

Three things worth reading carefully.

**General-purpose models cannot generate Kabyle.** On 5-shot `eng→kab`, models of this class
score 10–14 chrF++ because they copy English source words or loop into repetitive tokens, and
label 13 of 16 SIB-200 samples as `"politics"`. Jugurtha-4B lifts translation to **28.77 chrF++**
and topic classification to **73.74%**. The gap is not a matter of scale — it is the absence of
the language from the training distribution.

**The orthography gap closes to zero.** Scored raw and then passed through the versioned AƔBALU
normaliser, Jugurtha-4B's results are identical (`28.7714` vs `28.7714`). The model natively
writes canonical Kabyle Latin — `ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ` — without phonemic degradation. The
normaliser was run on both sides to confirm, not to correct.

**General knowledge is retained.** On the 7-task English and French suite, Jugurtha-4B averages
**73.96 points** — scoring **74.28% on MMLU** and **57.85% on ARC-Challenge**, both at or above
the general-purpose baseline. The held-out Kabyle BPC drops from 4.26 to **1.578**, a compression
gain that reflects the model having genuinely learned the language rather than pattern-matched
superficially.

## Intended use

- **Primary:** Kabyle conversational assistance, reading comprehension, question answering,
  summarisation, cultural explanations, and bidirectional Kabyle ↔ English / French translation.
- **Thinking mode:** Set `enable_thinking=True` for transparent step-by-step reasoning in Kabyle
  inside `<think>...</think>` blocks. Set `enable_thinking=False` (default) for direct, fast
  conversational generation.
- **Not suitable for:** Safety-critical medical diagnosis, pharmacological guidance, or binding
  legal counsel.

## Usage

`transformers` and `torch` in BF16. The architecture is one of the library's own, so no
`trust_remote_code` is required.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "agbalu/Jugurtha-4B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [
    {"role": "user", "content": "Segzi-yi-d s wegzul anta i d tamurt n Yiqbayliyen d wamek i d-tezga."},
]

# Standard mode (thinking OFF — default)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)

reply = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(reply)
```

For step-by-step Kabyle chain-of-thought, set `enable_thinking=True` and sample with
`temperature=0.6`, `top_p=0.95`. The reasoning trace appears between `<think>` and `</think>`.

The sovereign Kabyle identity, humble demeanor, and Latin orthography are **baked directly into the model's default chat template and GGUF metadata**; no manual system prompt configuration is needed across any runtime.

### vLLM

```bash
vllm serve agbalu/Jugurtha-4B --max-model-len 32768 --dtype bfloat16
```

## Architecture

32-layer hybrid causal language model with tied word embeddings and 32,768-token native context.

| | |
|---|---|
| **Parameters** | **4,205,751,296** (text tower) / **4,539,265,536** (total) |
| **Layers** | 32 (24 DeltaNet linear-attention + 8 full-attention) |
| **Context Window** | 32,768 tokens (100% NIAH) |
| **Vocabulary** | 248,320 tokens (tied input/output embeddings) |
| **Format** | BF16, 738 tensors, 2 safetensor shards |

## Training

### Data

Jugurtha-4B was trained on three streams:

1. **Kabyle pretraining corpus (800.8M tokens):** Cleaned Kabyle web and document prose,
   filtered with [`Dihya-5M`](https://huggingface.co/agbalu/Dihya-5M) and deduplicated, human-authored
   parallel bitext from the AƔBALU corpus, and curated multilingual replay to preserve
   general-purpose capability.
2. **Curated instruction set:** High-density Kabyle dialogues covering native conversation,
   translation, question answering, and reasoning — including verified Kabyle chain-of-thought
   traces for think-mode instances.
3. **Decontamination:** Filtered against FLORES+ `kab_Latn` and SIB-200 with a 13-word n-gram
   and 8-word sentence boundary. Measured overlap: **0 sentences**, verified with positive controls.

### Recipe

| | |
|---|---|
| **Hardware** | Single NVIDIA H100 NVL MIG (`3g.47gb`) |
| **Stage 1 — CPT** | Full-parameter continued pretraining on 800.8M tokens |
| **Stage 2 — Merge** | Task-vector arithmetic into a 4B instruction-tuned backbone |
| **Stage 3 — SFT** | Supervised fine-tuning on curated Kabyle dialogues, 1 epoch |

## Limitations

- **Informal keyboards & Arabizi:** Trained on canonical Kabyle Latin (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`).
  Noisy Arabizi (digits `3`, `7`, `9`) or unstandardised spelling inflates token fertility.
  Preprocess informal input with [`Boulifa-48M`](https://huggingface.co/agbalu/Boulifa-48M) before
  prompting.
- **Reasoning vs. standard mode:** Think mode aligns the reasoning process to Kabyle, which trades
  some raw English benchmark score. Use standard mode when latency matters or English-only tasks are
  primary.
- **Corpus factual bounds:** Knowledge in low-resource cultural domains reflects the verified corpus;
  unverified claims from public web text are filtered but may surface in niche topics.
- **Medical & legal:** Not certified for clinical diagnosis, drug prescribing, or legal advice.

## Files

| File | Contents |
|---|---|
| `model.safetensors-00001-of-00002.safetensors` | Shard 1 of 2 (BF16 weights) |
| `model.safetensors-00002-of-00002.safetensors` | Shard 2 of 2 (BF16 weights, 738 tensors total) |
| `config.json` | Architecture configuration |
| `generation_config.json` | Default sampling and EOS generation parameters |
| `chat_template.jinja` | Native dual-mode chat template |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer (248,320 vocabulary) |

## The name

**Jugurtha** (c. 160–104 BCE) was the grandson of King Masinissa of Numidia. When Rome sought to
partition and subjugate Numidia after Masinissa's death, Jugurtha waged a seven-year war to
preserve Numidian sovereignty — and was never taken in battle.

Within the AƔBALU model family, [`Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M)
and **Jugurtha-4B** stand at the two ends of the stack: the grandfather who unified disparate tribes
into one nation, and the grandson who refused to surrender its independence.

The naming is homage. It implies no endorsement, and neither Jugurtha nor his descendants are
affiliated with this work.

## Citation

```bibtex
@software{agbalu_jugurtha_2026,
  title  = {Jugurtha-4B: A Sovereign Kabyle Generative Foundation Model},
  author = {Ainouche, Abderahmane and {AƔBALU Project}},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Jugurtha-4B},
}
```

## Licence

**Apache-2.0** for the weights and configurations. **That grant does not relicense the text the
weights were trained on**, and a third of that text has no licence anyone could resolve. The
composition is published so you can make your own judgement:

| redistribution | share |
|---|---|
| **unclear** | **34.9%** |
| permissive | 32.0% |
| share-alike | 31.0% |
| non-commercial | 2.0% |

`unclear` is not a permissive category — it is the absence of a resolvable licence. A
permissive-only rebuild is possible from the 32.0% permissive sentences. If you need one, open
an issue.
