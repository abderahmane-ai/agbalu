---
base_model: agbalu/Jugurtha-2-9B
language:
- kab
- en
- fr
license: apache-2.0
pipeline_tag: text-generation
tags:
- kabyle
- taqbaylit
- berber
- amazigh
- text-generation
- causal-lm
- reasoning
- thinking
- sovereign-ai
- gguf
- ollama
- lm-studio
---

# Jugurtha-2-9B (GGUF)

Quantized GGUF releases for [**Jugurtha-2-9B**](https://huggingface.co/agbalu/Jugurtha-2-9B), the second-generation 8.95B-parameter sovereign foundation model for **Kabyle** (Taqbaylit, `kab`, Latin script) with native chain-of-thought reasoning.

Every GGUF carries the complete sovereign system prompt baked directly into its metadata header.

Optimized for consumer GPUs, Apple Silicon, **LM Studio**, and **Ollama**.

---

## Provided Quantizations

| File | Size (approx) | Format | Description |
|---|---|---|---|
| `Jugurtha-2-9B-Q4_K_M.gguf` | **~5.5 GB** | Q4_K_M | **Recommended.** Balanced 4-bit quant; optimal sweet spot of speed, RAM footprint, and reasoning. Default for Ollama. |
| `Jugurtha-2-9B-Q5_K_M.gguf` | **~6.6 GB** | Q5_K_M | **High precision.** 5-bit quant; recommended for complex analytical tasks and high linguistic nuance. |
| `Jugurtha-2-9B-Q6_K.gguf` | **~7.6 GB** | Q6_K | **Near-lossless.** 6-bit quant; virtually indistinguishable from BF16 across all linguistic benchmarks. |
| `Jugurtha-2-9B-Q8_0.gguf` | **~9.7 GB** | Q8_0 | **Full reference.** 8-bit quant; bit-for-bit benchmark match to full BF16 precision. |
| `Jugurtha-2-9B-Q3_K_M.gguf` | **~4.5 GB** | Q3_K_M | **Lightweight.** 3-bit quant; designed for resource-constrained laptops, edge devices, and mobile environments. |

---

## Direct Usage with Ollama

Run directly from the Hugging Face Hub without manual file management:

```bash
# Direct fast responses (Thinking OFF)
ollama run hf.co/agbalu/Jugurtha-2-9B-GGUF --think=false

# Full Chain-of-Thought reasoning in Taqbaylit (Thinking ON)
ollama run hf.co/agbalu/Jugurtha-2-9B-GGUF --think=true
```

Inside an interactive Ollama session, you can toggle thinking at any time:
- `/set nothink` — immediate, concise answers
- `/set think` — detailed internal deliberation in Taqbaylit inside `<think>...</think>`

---

## Direct Usage with LM Studio

1. Open **LM Studio**.
2. Search for `agbalu/Jugurtha-2-9B-GGUF`.
3. Choose your quantization (`Q4_K_M` recommended) and click **Download**.
4. Click **Load Model**. Ensure GPU Offload is set to **Max** for peak token throughput.
5. The baked sovereign system prompt in Taqbaylit is recognized and applied automatically.
6. Toggle the **Thinking / Reasoning** switch in the sidebar to control reasoning traces.

---

## Direct Usage with llama.cpp

Fast GPU/CPU inference with `llama-cli`:

```bash
# Thinking OFF (direct conversational response)
llama-cli -m Jugurtha-2-9B-Q4_K_M.gguf -ngl 99 -t 8 \
    -p "<|im_start|>user\nSegzi-yi-d s wegzul d acu i d AƔBALU.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" \
    -n 512

# Thinking ON (chain-of-thought in Taqbaylit)
llama-cli -m Jugurtha-2-9B-Q4_K_M.gguf -ngl 99 -t 8 \
    -p "<|im_start|>user\nAcḥal i d 25 x 4?<|im_end|>\n<|im_start|>assistant\n<think>\n" \
    -n 512
```

---

## Main Repository & Full Evaluations

For benchmark scores (FLORES+, SIB-200, BPC), evaluations, and BF16 weights, see the primary repository: [**agbalu/Jugurtha-2-9B**](https://huggingface.co/agbalu/Jugurtha-2-9B).

## Licence

**Apache-2.0**
