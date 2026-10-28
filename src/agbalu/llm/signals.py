"""What the language identifiers and the CPT model say about a candidate (task 11.22).

Each signal reads the candidate's Kabyle: every assistant answer and reasoning, joined.

- `glotlid_kabyle`: GlotLID's label for the text is `kab_Latn`. It names French, English
  and Arabic, which Dihya cannot.
- `dihya_sibling`: the label Dihya gives most windows, when that label is another Berber
  language; None when it is Kabyle or not Amazigh at all.
- `fluency_bpc`: bits per character under the CPT model, which needs the GPU and is computed
  on the pod in its own pass.

Boulifa's edit rate was dropped on 2026-10-17: it decodes one character at a time and ran
over 20 minutes on 839 candidates without finishing, for a flag that only sent a candidate
to review. The teacher's Kabyle passes through the versioned normaliser either way.

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
from agbalu.llm.progress import stage
from agbalu.llm.sft_data import assistant_texts
from agbalu.llm.teacher import Candidate

GLOTLID_KABYLE: Final = "kab_Latn"
DIHYA_NOT_SIBLING: Final = frozenset({"kab_Latn", "NOT_AMAZIGH"})


class SignalError(Exception):
    """A signal cannot be computed from what it was given."""


class Labeller(Protocol):
    def identify(self, texts: Sequence[str]) -> list[str]: ...


def kabyle_text(candidate: Candidate) -> str:
    return "\n".join(assistant_texts(candidate))


def sibling(labels: Sequence[str]) -> str | None:
    """The label most windows got, when it is a Berber language other than Kabyle."""
    if not labels:
        return None
    winner, _ = Tally(labels).most_common(1)[0]
    return None if winner in DIHYA_NOT_SIBLING else winner


def language_signals(
    candidates: Sequence[Candidate], glotlid: Labeller | None, dihya: Labeller
) -> dict[str, dict[str, object]]:
    """The two language signals, per candidate id."""
    flat = [" ".join(kabyle_text(candidate).split()) for candidate in candidates]
    if glotlid is not None:
        with stage("glotlid", texts=len(flat)):
            glot = glotlid.identify(flat)
    else:
        glot = [GLOTLID_KABYLE for _ in flat]
    with stage("dihya", texts=len(flat)):
        siblings = [sibling(dihya.identify(windows(text))) for text in flat]
    return {
        candidate.id: {
            "glotlid_kabyle": label == GLOTLID_KABYLE,
            "glotlid_label": label,
            "dihya_sibling": other,
        }
        for candidate, label, other in zip(candidates, glot, siblings, strict=True)
    }


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
