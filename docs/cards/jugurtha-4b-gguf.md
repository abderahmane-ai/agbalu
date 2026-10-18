---
base_model: agbalu/Jugurtha-4B
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

# Jugurtha-4B (GGUF)

Quantized GGUF weights for [**Jugurtha-4B**](https://huggingface.co/agbalu/Jugurtha-4B), the first instruction-tuned foundation model for **Kabyle** (Taqbaylit, `kab`, Latin script) with native chain-of-thought reasoning.

Optimized for consumer hardware, edge devices, **LM Studio**, and **Ollama**.

## Provided Quantizations

| File | Size | Format | Description |
|---|---|---|---|
| `Jugurtha-4B-Q4_K_M.gguf` | **2.6 GB** | Q4_K_M | **Recommended.** Balanced 4-bit quant; fits on 8 GB+ laptops with peak speed. Default for Ollama. |
| `Jugurtha-4B-Q5_K_M.gguf` | **3.0 GB** | Q5_K_M | **High precision.** 5-bit quant; recommended for complex reasoning on 16 GB+ systems. |
| `Jugurtha-4B-Q8_0.gguf` | **4.3 GB** | Q8_0 | **Near-lossless.** 8-bit quant; identical benchmark accuracy to BF16. |

---

## Direct Usage with Ollama

Run directly from Hugging Face without manual downloads:

```bash
# Direct fast responses (Thinking OFF)
ollama run hf.co/agbalu/Jugurtha-4B-GGUF --think=false

# Full Chain-of-Thought reasoning in Taqbaylit (Thinking ON)
ollama run hf.co/agbalu/Jugurtha-4B-GGUF --think=true
```

Inside an interactive Ollama session, you can toggle thinking at any time:
- `/set nothink` — immediate, concise answers
- `/set think` — detailed internal reasoning in Taqbaylit

---

## Direct Usage with LM Studio

1. Open **LM Studio**.
2. Search for `agbalu/Jugurtha-4B-GGUF`.
3. Choose your quantization (`Q4_K_M` recommended) and click **Download**.
4. Click **Load Model**. Ensure GPU Offload is set to **Max** for peak speed (40–70 tok/s on Apple Silicon / CUDA).
5. Toggle the **Thinking / Reasoning** switch in the sidebar to turn reasoning on or off.

---

## Direct Usage with llama.cpp

For fast multi-threaded GPU inference:

```bash
# Thinking OFF (direct answer)
llama-cli -m Jugurtha-4B-Q4_K_M.gguf -ngl 99 -t 8 \
    -p "<|im_start|>user\nSegzi-yi-d s wegzul d acu i d Agbalu.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" \
    -n 512

# Thinking ON (chain-of-thought in Taqbaylit)
llama-cli -m Jugurtha-4B-Q4_K_M.gguf -ngl 99 -t 8 \
    -p "<|im_start|>user\nAcḥal i d 25 x 4?<|im_end|>\n<|im_start|>assistant\n<think>\n" \
    -n 512
```

---

## Main Model & Details

For full model benchmarks, training recipe, and architecture details, refer to the main repository: [**agbalu/Jugurtha-4B**](https://huggingface.co/agbalu/Jugurtha-4B).

## Licence

**Apache-2.0**
