"""The Kabyle encoder: architecture, masking, training loop and checkpointing.

An LTG-BERT/GPT-BERT reimplementation, trained on AƔBALU-Text v1.

This module imports nothing, deliberately. `torch` is an optional extra, so re-exporting
here would make `config` and `lock` — the two modules that do not need it — unreachable on
an install without a training stack.
"""
