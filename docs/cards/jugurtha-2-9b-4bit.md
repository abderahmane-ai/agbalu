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
- mlx
- mlx-my-repo
- 4bit
- apple-silicon
---

# Jugurtha-2-9B-MLX-4bit (MLX)

4-bit quantized MLX format for [**Jugurtha-2-9B**](https://huggingface.co/agbalu/Jugurtha-2-9B), the second-generation 8.95B-parameter sovereign foundation model for **Kabyle** (Taqbaylit, `kab`, Latin script) with native chain-of-thought reasoning.

Natively optimized for Apple Silicon (M1, M2, M3, M4) using Apple's [MLX](https://github.com/ml-explore/mlx) framework. Runs with high token generation throughput and minimal memory footprint (~5.5 GB unified RAM).

The complete sovereign system prompt in Taqbaylit is baked directly into the tokenizer configuration.

---

## Quick Start

### Installation

```bash
pip install mlx-lm
```

### Python API

```python
from mlx_lm import generate, load

model, tokenizer = load("agbalu/Jugurtha-2-9B-MLX-4bit")

messages = [{"role": "user", "content": "Segzi-yi-d s wegzul d acu i d AƔBALU."}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    verbose=True,
)
print(response)
```

### Command Line Interface

```bash
# Direct generation (Thinking OFF)
mlx_lm.generate \
  --model agbalu/Jugurtha-2-9B-MLX-4bit \
  --prompt "<|im_start|>user\nAzul, amek telliḍ?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" \
  --max-tokens 512

# Transparent reasoning in Taqbaylit (Thinking ON)
mlx_lm.generate \
  --model agbalu/Jugurtha-2-9B-MLX-4bit \
  --prompt "<|im_start|>user\nAcḥal i d 25 x 4?<|im_end|>\n<|im_start|>assistant\n<think>\n" \
  --max-tokens 512

# Interactive local server / web chat
mlx_lm.server --model agbalu/Jugurtha-2-9B-MLX-4bit
```

---

## Dual-Mode Reasoning

Jugurtha-2-9B supports native dual-mode reasoning on Apple Silicon:
- **Thinking Mode ON**: Start assistant generation with `<think>\n` to let the model deliberate internally in Taqbaylit before delivering the final answer.
- **Thinking Mode OFF**: Start assistant generation with `<think>\n\n</think>\n\n` for immediate, direct conversational generation.

---

## Full Model & Benchmark Details

For full evaluations (FLORES+, SIB-200, BPC), training methodology, and safetensors BF16 weights, visit the primary repository: [**agbalu/Jugurtha-2-9B**](https://huggingface.co/agbalu/Jugurtha-2-9B).

## Licence

**Apache-2.0**
