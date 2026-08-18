"""The base Jugurtha adapts, and the A10G arithmetic that chose its size (task 11.3).

`Qwen/Qwen3.5-2B`, Apache-2.0 and ungated: 2,274,069,824 parameters of which the text tower
is 1,881,825,088. `AutoModelForCausalLM` loads that tower alone and leaves the 331.4M vision
tower and the 60.8M multi-token-prediction head on disk.

Sizes are ruled out by the clock, not by memory. Full continued pretraining costs about
`6 * N` FLOPs per token; against one epoch of the CPT corpus and the 21.1 TFLOP/s measured
on an A10, a two-GPU container gives:

| model | text tower | h/epoch on A10:2 |
|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 752,393,024 | 6.8 |
| **`Qwen/Qwen3.5-2B`** | **1,881,825,088** | **17.1** |
| `Qwen/Qwen3.5-4B` | 4,205,751,296 | 38.3 |
| `Qwen/Qwen3.5-9B` | 8,953,803,264 | 81.5 |

Qwen 3.6 and 3.8 ship nothing below 27B and there is no 3.7, so 3.5 is the whole frontier at
this size. The post-trained checkpoint rather than `-Base`: no Kabyle instruction data exists
anywhere, so inherited instruction-following is worth more than the perplexity a base
checkpoint starts with.
"""

from __future__ import annotations

from typing import Final

BASE: Final = "Qwen/Qwen3.5-2B"
"""The one model. Read by the corpus counter, the fertility table and the baseline scorer,
so the tokens are counted in the vocabulary the run is scored in."""

TEXT_TOWER_PARAMETERS: Final = 1_881_825_088
"""Read from the checkpoint's safetensors header, excluding the vision tower and the MTP
head. This is `N` in the `6 * N * D` that sizes a run."""
