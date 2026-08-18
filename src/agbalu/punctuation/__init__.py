"""Restoring punctuation and casing on what the ASR model emits.

Fadhma's vocabulary is 36 letters, `-` and a word delimiter, so its output carries no marks
and no capitals. This package puts them back as two token-classification heads over the
Kabyle encoder, scored on the split of Common Voice that is absent from the encoder's own
pretraining corpus.

`docs/punctuation_design.md` holds the measurements. This module imports nothing: `labels`
and `corpus` are pure and must stay reachable on an install without a training stack.
"""
