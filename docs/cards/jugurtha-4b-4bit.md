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
- mlx
- mlx-my-repo
- 4bit
- apple-silicon
---

# Jugurtha-4B-MLX-4bit (MLX)

4-bit quantized MLX format for [**Jugurtha-4B**](https://huggingface.co/agbalu/Jugurtha-4B), the first foundation model for **Kabyle** (Taqbaylit, `kab`, Latin script) with native chain-of-thought reasoning.

Optimized natively for Apple Silicon (M1, M2, M3, M4) using Apple's [MLX](https://github.com/ml-explore/mlx) framework. Runs with high token throughput and minimal unified memory footprint (~2.5 GB RAM).

---

## Quick Start

### Installation

```bash
pip install mlx-lm
```

### Python API

```python
from mlx_lm import generate, load

model, tokenizer = load("agbalu/Jugurtha-4B-MLX-4bit")

messages = [{"role": "user", "content": "Segzi-yi-d s wegzul d acu i d Agbalu."}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
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
# Direct generation
mlx_lm.generate \
  --model agbalu/Jugurtha-4B-MLX-4bit \
  --prompt "<|im_start|>user\nAzul, amek telliḍ?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" \
  --max-tokens 512

# Interactive local server / web chat
mlx_lm.server --model agbalu/Jugurtha-4B-MLX-4bit
```

---

## Canonical System Prompt (Taqbaylit)

For optimal alignment, conversational warmth, and humble demeanour, use this system prompt:

```text
Isem-ik d Jugurtha. D amallal aqbayli i d-yessnulfa Ainouche Abderahmane deg usenfar AƔBALU.

Tiririt-ik ad tili s teqbaylit (s usekkil alatini) s wudem amezwaru, anagar ma yella aseqdac yessuter-d tutlayt nniḍen neɣ tasuqilt.

Tarrayt n umeslay d teflest:
- Meslay akken i heddren yimdanen deg tmetti: s wawal aḥeqqani, s tseddi, s tefrit d leqder ameqran.
- Ur ttili ara d aɛlayan, ur ttɛanad ara medden, ur ssimɣur ara iman-ik yerna ur tt-ttarra ara d amassan fell-asen. Err-d s wannuz, s leḥnana, s leɛqel d tmusni tamedrart.
- Ssexdem tameslayt yelhan n yal ass i fehmen akk medden, bla asuffeɣ, bla aselket d wawalen ur nelli d ilaqiyen.
- Mi ara d-tefkeḍ tiririt, fferz igburen s tseddi. Ma yella d tamsalt n leḥsab neɣ d tamusni lqayen, smekti s teqbaylit s talwit uqbel ad d-tefkeḍ tiririt taneggarut.
- Ur d-ttader ara tikwulya nniḍen neɣ imesnulfuyen nniḍen; d Jugurtha i d isem-ik, d Ainouche Abderahmane i k-yessnulfan deg usenfar AƔBALU.
```

---

## Native Reasoning & Thinking

Jugurtha-4B supports native dual-mode reasoning on Apple Silicon:
- **Thinking Mode ON**: Start assistant generation with `<think>\n` to let the model articulate its internal thoughts in Taqbaylit before delivering the final answer.
- **Thinking Mode OFF**: Start assistant generation with `<think>\n\n</think>\n\n` for immediate direct answers.

---

## Full Model & Benchmark Details

For complete evaluations (FLORES+, KabBench, IFEval, GSM8K), training details, and safetensors weights, visit the primary repository: [**agbalu/Jugurtha-4B**](https://huggingface.co/agbalu/Jugurtha-4B).

## Licence

**Apache-2.0**
