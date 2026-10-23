"""The checkpoints Jugurtha is built from.

`Qwen/Qwen3.5-4B-Base`, Apache-2.0 and ungated. Continued pretraining starts from the base
rather than the post-trained model, and instruction following is recovered afterwards by
adding the CPT delta to `INSTRUCT`, which shares the architecture tensor for tensor.
AfriqueLLM (ACL 2026, arXiv 2601.06395) adapted this exact base and reports it the strongest
4B result after CPT.

Parameter counts are read from the published safetensors headers on 2026-10-14. The text
tower excludes the vision tower and the multi-token-prediction head, which
`AutoModelForCausalLM` leaves on disk.
"""

from __future__ import annotations

from typing import Final

BASE: Final = "Qwen/Qwen3.5-4B-Base"
"""Read by the corpus counter, the packer, the trainer and the evaluator, so tokens are
counted in the vocabulary the run is trained and scored in."""

INSTRUCT: Final = "Qwen/Qwen3.5-4B"
"""The post-trained twin of `BASE`, the target the CPT delta is merged into."""

BASE_9B: Final = "Qwen/Qwen3.5-9B-Base"
"""The 9B base checkpoint for Jugurtha-9B."""

INSTRUCT_9B: Final = "Qwen/Qwen3.5-9B"
"""The post-trained twin of `BASE_9B`."""

REVISIONS: Final[dict[str, str]] = {
    BASE: "1001bb4d826a52d1f399e183466143f4da7b741b",
    INSTRUCT: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    BASE_9B: "68c46c4b3498877f3ef123c856ecfde50c39f404",
    INSTRUCT_9B: "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
"""Pinned commits. `BASE`'s is the snapshot `jugurtha-4b-v1` started from (the only one in the
pod's cache, 2026-10-16), so the CPT delta is taken against the weights it was trained from."""

PILOT: Final = "Qwen/Qwen3.5-0.8B-Base"
"""Same architecture and tokenizer at a fifth of the size. The optimizer parity control
runs on it, where an fp32 AdamW reference fits beside the 8-bit arm."""

TEXT_TOWER_PARAMETERS: Final[dict[str, int]] = {
    BASE: 4_205_751_296,
    INSTRUCT: 4_205_751_296,
    BASE_9B: 8_953_803_264,
    INSTRUCT_9B: 8_953_803_264,
    PILOT: 752_393_024,
}
"""`N` in the `6 * N * D` that sizes a run and in the model-FLOP utilisation it reports."""
