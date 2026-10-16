"""What the language identifiers, Boulifa and the CPT model say about a candidate (task 11.22).

Each signal reads the candidate's Kabyle: every assistant answer and reasoning, joined.

- `glotlid_kabyle`: GlotLID's label for the text is `kab_Latn`. It names French, English
  and Arabic, which Dihya cannot.
- `dihya_sibling`: the label Dihya gives most windows, when that label is another Berber
  language; None when it is Kabyle or not Amazigh at all.
- `boulifa_edit_rate`: characters Boulifa rewrites, over the characters read. Boulifa's own
  97.39% is measured on its task and proves nothing about grammar, so a high rate only sends
  a candidate to review.
- `fluency_bpc`: bits per character under the CPT model, which needs the GPU and is computed
  on the pod in its own pass.

`agbalu.llm.sft_data` decides on these; nothing here accepts or rejects.
"""

from __future__ import annotations

import math
from collections import Counter as Tally
from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from agbalu.llm.documents import windows
from agbalu.llm.evaluate import Scorer, Tokenizer, summed_nll
from agbalu.llm.packing import concatenate
from agbalu.llm.sft_data import assistant_texts
from agbalu.llm.teacher import Candidate
from agbalu.speech.metrics import edit_distance

GLOTLID_KABYLE: Final = "kab_Latn"
DIHYA_NOT_SIBLING: Final = frozenset({"kab_Latn", "NOT_AMAZIGH"})
LINE_CHARACTERS: Final = 400
"""Boulifa reads at most 512 tokens; lines are cut well inside that."""


class SignalError(Exception):
    """A signal cannot be computed from what it was given."""


class Labeller(Protocol):
    def identify(self, texts: Sequence[str]) -> list[str]: ...


class Rewriter(Protocol):
    def standardise(self, text: str) -> str: ...


def kabyle_text(candidate: Candidate) -> str:
    return "\n".join(assistant_texts(candidate))


def sibling(labels: Sequence[str]) -> str | None:
    """The label most windows got, when it is a Berber language other than Kabyle."""
    if not labels:
        return None
    winner, _ = Tally(labels).most_common(1)[0]
    return None if winner in DIHYA_NOT_SIBLING else winner


def pieces(text: str, size: int = LINE_CHARACTERS) -> list[str]:
    """`text` as lines no longer than `size`, cut at the last space before the limit."""
    out: list[str] = []
    for line in text.splitlines():
        rest = line.strip()
        while len(rest) > size:
            cut = rest.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            out.append(rest[:cut])
            rest = rest[cut:].strip()
        if rest:
            out.append(rest)
    return out


def edit_rate(text: str, rewriter: Rewriter) -> float:
    """Characters Boulifa changes, over the characters it reads; 0 for empty text."""
    lines = pieces(text)
    read = sum(len(line) for line in lines)
    if read == 0:
        return 0.0
    changed = sum(edit_distance(list(line), list(rewriter.standardise(line))) for line in lines)
    return changed / read


def language_signals(
    candidates: Sequence[Candidate], glotlid: Labeller, dihya: Labeller, rewriter: Rewriter
) -> dict[str, dict[str, object]]:
    """The three signals that run on a laptop, per candidate id."""
    texts = [kabyle_text(candidate) for candidate in candidates]
    flat = [" ".join(text.split()) for text in texts]
    glot = glotlid.identify(flat)
    out: dict[str, dict[str, object]] = {}
    for candidate, text, label in zip(candidates, texts, glot, strict=True):
        parts = windows(" ".join(text.split()))
        out[candidate.id] = {
            "glotlid_kabyle": label == GLOTLID_KABYLE,
            "glotlid_label": label,
            "dihya_sibling": sibling(dihya.identify(parts)),
            "boulifa_edit_rate": round(edit_rate(text, rewriter), 4),
        }
    return out


def percentile(values: Sequence[float], share: float) -> float:
    """The nearest-rank percentile; `share` in (0, 1]."""
    if not values or not 0 < share <= 1:
        message = f"a percentile needs values and a share in (0, 1], got {len(values)}, {share}"
        raise SignalError(message)
    ordered = sorted(values)
    return ordered[max(0, math.ceil(share * len(ordered)) - 1)]


FLUENCY_FLOOR: Final = 0.95
"""The threshold is held-out Kabyle's 95th percentile: a candidate less fluent than 95% of
real Kabyle sentences is sent to review."""


def bits_per_character(scorer: Scorer, tokenizer: Tokenizer, texts: Sequence[str]) -> list[float]:
    """Each text scored alone under the model, `eos` first as in training."""
    out: list[float] = []
    for text, ids in zip(texts, tokenizer.encode_batch(texts), strict=True):
        if not ids or not text:
            message = "an empty text has no bits per character"
            raise SignalError(message)
        nats, _ = summed_nll(scorer, [concatenate([[tokenizer.eos, *ids]])])
        out.append(nats / math.log(2) / len(text))
    return out


def merged(*parts: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Signals from separate passes, per candidate; a candidate in one pass and not another is
    left incomplete, which `sft_data.soft_flags` refuses by name."""
    out: dict[str, dict[str, object]] = {}
    for part in parts:
        for name, values in part.items():
            out.setdefault(name, {}).update(values)
    return out
