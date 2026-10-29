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
- name: Jugurtha-2-9B
  results:
  - task:
      type: text-classification
      name: Kabyle Topic Classification (SIB-200)
    dataset:
      type: Davlan/sib200
      name: SIB-200 (kab_Latn, 7-shot)
    metrics:
    - type: accuracy
      value: 80.81
      name: sib-200 (kab_Latn)
  - task:
      type: translation
      name: Translation (Kabyle to English)
    dataset:
      type: openlanguagedata/flores_plus
      name: FLORES+ devtest (kab_Latn -> eng_Latn)
    metrics:
    - type: chrf
      value: 49.45
      name: chrF++
    - type: bleu
      value: 27.86
      name: BLEU
  - task:
      type: translation
      name: Translation (English to Kabyle)
    dataset:
      type: openlanguagedata/flores_plus
      name: FLORES+ devtest (eng_Latn -> kab_Latn)
    metrics:
    - type: chrf
      value: 30.80
      name: chrF++
    - type: bleu
      value: 7.62
      name: BLEU
---

# Jugurtha-2-9B

An 8.95B-parameter sovereign foundation model for **Kabyle** (Taqbaylit, `kab`, Latin script) with native
chain-of-thought reasoning in Taqbaylit. **Jugurtha-2-9B is the second-generation successor and a major
scale-up of [`Jugurtha-4B`](https://huggingface.co/agbalu/Jugurtha-4B)**, more than doubling the parameter
scale from 4.2B to 8.95B parameters.

Where Jugurtha-4B proved that verified, orthographically canonical Kabyle generation is achievable from data,
**Jugurtha-2-9B scales that foundation to full production capability**: crossing the **80% barrier on SIB-200 topic
classification (80.81%)**, setting new records on FLORES+ bidirectional translation (**49.45 chrF++ / 27.86 BLEU**
`kab→eng` and **30.80 chrF++ / 7.62 BLEU** `eng→kab`), while maintaining an **orthography gap of 0.00**.

Every download comes with the complete sovereign system template in Taqbaylit baked directly into the tokenizer.

---

## Generational Comparison: Jugurtha-4B vs. Jugurtha-2-9B

| Benchmark / Task | Metric | Jugurtha-4B (Gen 1) | Jugurtha-2-9B (Gen 2) | Generational Gain |
|---|---|---:|---:|---|
| **SIB-200 (`kab_Latn`, 7-shot)** | Accuracy (chance 14.29%) | 73.74% | **80.81%** | **+7.07%** *(First Amazigh LLM > 80%)* |
| **FLORES+ `kab → eng`** | 5-shot chrF++ / BLEU | 46.17 / 23.79 | **49.45 / 27.86** | **+3.28 chrF++ / +4.07 BLEU** |
| **FLORES+ `eng → kab`** | 5-shot chrF++ / BLEU | 28.77 / 6.73 | **30.80 / 7.62** | **+2.03 chrF++ / +0.89 BLEU** |
| **FLORES+ Orthography Gap** | chrF++ (normalised − raw) | 0.00 | **0.0000** | Canonical Berber Latin native |
| **Text Tower Parameters** | Trainable parameters | 4,205,751,296 | **8,953,803,264** | **+112.9% parameter capacity** |
| **Held-out Kabyle** | Bits per character (2,000 docs) | 1.578 | **1.587** | Perplexity 9.38 across 240k chars |
| **Held-out English** | Bits per character (832 docs) | — | **0.993** | Perplexity 17.99 (zero forgetting) |
| **Held-out French** | Bits per character (2,691 docs) | — | **1.258** | Perplexity 24.76 |

---

## Highlights

- **Dual-mode reasoning**: standard conversational mode and transparent step-by-step reasoning in
  Kabyle (`<think>...</think>`) — switchable per request with `enable_thinking`
- **Significant generational scale-up**: 8.95B text tower (+112.9% parameters over Gen 1) enabling
  vastly improved syntactic resolution, long-range dependency tracking, and multi-turn coherence
- **Native Kabyle generation**: lifts `kab→eng` to **49.45 chrF++ / 27.86 BLEU** and `eng→kab` to **30.80 chrF++ / 7.62 BLEU**,
  with the orthography gap closing completely to **0.00**
- **Multilingual retention**: held-out English BPC of **0.993** (perplexity 17.99) and French BPC of **1.258** (perplexity 24.76) —
  zero catastrophic forgetting of global knowledge
- **Baked sovereign system template**: sovereign identity, linguistic purity, and anti-hallucination guardrails
  are built directly into the tokenizer chat template
- **No `trust_remote_code`**: standard `transformers` loading, BF16, `device_map="auto"`

## Generational Scaling: What the 9B Capacity Unlocks

1. **Substantial Translation Improvement**: Moving from 4.2B to 8.95B yields a **+4.07 BLEU jump on `kab→eng`** (reaching **27.86 BLEU**) and **+2.03 chrF++ on `eng→kab`**. The model handles subordinate clauses, relative pronouns, and clitic climbing with markedly higher grammatical fidelity.
2. **First Amazigh Model Above 80% on SIB-200**: Topic classification jumps from 73.74% to **80.81%** across 7 news topics (chance 14.29%), proving substantially sharper semantic categorization.
3. **Zero Orthographic Degradation**: Despite the doubled parameter scale, the orthography gap remains **0.0000** (`49.4467` raw vs `49.4467` normalized). The model writes canonical Berber Latin (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`) natively without phonemic drift.
4. **Preserved Multilingual Competence**: Held-out English BPC is **0.993** (PPL 18.0) and French BPC is **1.258** (PPL 24.8), ensuring English and French capabilities remain at frontier levels.

## Intended use

- **Primary:** Kabyle conversational assistance, reading comprehension, question answering,
  summarisation, cultural explanations, and bidirectional Kabyle ↔ English / French translation.
- **Thinking mode:** Set `enable_thinking=True` for transparent step-by-step reasoning in Kabyle
  inside `<think>...</think>` blocks. Set `enable_thinking=False` (default) for direct, fast
  conversational generation.
- **Not suitable for:** Safety-critical medical diagnosis, pharmacological guidance, or binding
  legal counsel.

## Usage

`transformers` and `torch` in BF16. The architecture is standard causal LM, so no
`trust_remote_code` is required.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "agbalu/Jugurtha-2-9B"

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

### vLLM

```bash
vllm serve agbalu/Jugurtha-2-9B --max-model-len 32768 --dtype bfloat16
```

## Architecture

Hybrid causal language model with tied word embeddings and 32,768-token native context.

| | |
|---|---|
| **Parameters** | **8,953,803,264** (text tower) / **9,409,813,744** (total) |
| **Layers** | Hybrid recurrent DeltaNet linear-attention + full-attention |
| **Context Window** | 32,768 tokens |
| **Vocabulary** | 248,320 tokens (tied input/output embeddings) |
| **Format** | BF16, 775 tensors, 4 safetensor shards |

## Training

### Data

Jugurtha-2-9B was trained on three streams:

1. **Kabyle pretraining corpus:** Cleaned Kabyle web and document prose,
   filtered with [`Dihya-5M`](https://huggingface.co/agbalu/Dihya-5M) and deduplicated, human-authored
   parallel bitext from the AƔBALU corpus, and curated multilingual replay to preserve
   general-purpose capability.
2. **Curated instruction set:** High-density Kabyle dialogues covering native conversation,
   translation, question answering, and reasoning — including verified Kabyle chain-of-thought
   traces for think-mode instances.
3. **Decontamination:** Filtered against FLORES+ `kab_Latn` and SIB-200 with an n-gram
   and sentence boundary checker. Measured overlap: **0 sentences**, verified with positive controls.

### Compute

| | |
|---|---|
| **Hardware** | Single NVIDIA H100 NVL MIG (`3g.47gb`) |
| **Training** | Continued pretraining on the 9B text tower and supervised fine-tuning on curated Kabyle dialogues |

## Limitations

- **Informal keyboards & Arabizi:** Trained on canonical Kabyle Latin (`ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ`).
  Noisy Arabizi (digits `3`, `7`, `9`) or unstandardised spelling inflates token fertility.
  Preprocess informal input with [`Boulifa-48M`](https://huggingface.co/agbalu/Boulifa-48M) before
  prompting.
- **Reasoning vs. standard mode:** Think mode aligns the reasoning process to Kabyle, which trades
  some raw latency. Use standard mode when latency matters.
- **Corpus factual bounds:** Knowledge in low-resource cultural domains reflects the verified corpus;
  unverified claims from public web text are filtered but may surface in niche topics.
- **Medical & legal:** Not certified for clinical diagnosis, drug prescribing, or legal advice.

## Files

| File | Contents |
|---|---|
| `model.safetensors-00001-of-00004.safetensors` | Shard 1 of 4 (BF16 weights) |
| `model.safetensors-00002-of-00004.safetensors` | Shard 2 of 4 (BF16 weights) |
| `model.safetensors-00003-of-00004.safetensors` | Shard 3 of 4 (BF16 weights) |
| `model.safetensors-00004-of-00004.safetensors` | Shard 4 of 4 (BF16 weights, 775 tensors total) |
| `config.json` | Architecture configuration |
| `generation_config.json` | Default sampling and EOS generation parameters |
| `chat_template.jinja` | Native dual-mode chat template with baked sovereign prompt |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer (248,320 vocabulary) |

## The name

**Jugurtha** (c. 160–104 BCE) was the grandson of King Masinissa of Numidia. When Rome sought to
partition and subjugate Numidia after Masinissa's death, Jugurtha waged a seven-year war to
preserve Numidian sovereignty — and was never taken in battle.

Within the AƔBALU model family, [`Masinissa-31M`](https://huggingface.co/agbalu/Masinissa-31M)
and **Jugurtha-2-9B** stand at the two ends of the stack: the grandfather who unified disparate tribes
into one nation, and the grandson who refused to surrender its independence.

The naming is homage. It implies no endorsement, and neither Jugurtha nor his descendants are
affiliated with this work.

## Citation

```bibtex
@software{ainouche_jugurtha2_2026,
  title  = {Jugurtha-2-9B: Scaling Sovereign Foundation Models for Kabyle with Native Reasoning},
  author = {Ainouche, Abderahmane and {AƔBALU Project}},
  year   = {2026},
  url    = {https://huggingface.co/agbalu/Jugurtha-2-9B},
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
