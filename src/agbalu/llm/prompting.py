"""AƔBALU-Bench translation as a prompt (task 11.2).

An instruction-tuned base has no forced target-language token, so the direction is carried
by the instruction and by the shots. Both matter: the shots are what stop the model
answering *about* the sentence instead of translating it, and they set the output shape the
extraction below depends on.

Shots come from FLORES+ `dev`. Scoring is on `devtest`, and dev exists precisely so that
prompt construction never touches the split a number is reported on.

Scoring is `agbalu.bench.mt.score`, unchanged, so the result sits in the same table as
NLLB-1.3B and Amrouche rather than in a table of its own.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    import torch

LANGUAGE_NAME: Final[dict[str, str]] = {
    "kab": "Kabyle",
    "eng": "English",
    "fra": "French",
    "arb": "Arabic",
    "spa": "Spanish",
    "deu": "German",
}
"""Endonyms would be the better prompt for a model that knows them; these are the names the
base model's own pretraining data uses, and the direction has to be unambiguous before it
can be idiomatic."""

SHOTS: Final = 5
SHOT_SEED: Final = 20260810
MAX_NEW_TOKENS: Final = 256
TOKEN_SLACK: Final = 32

LABEL: Final = re.compile(r"^\s*(translation|traduction|kabyle|english|french)\s*[:\-]\s*", re.I)
QUOTES: Final = "\"'“”«»‹›„"


class PromptError(Exception):
    """A direction cannot be turned into a prompt."""


Message = dict[str, str]
"""One chat turn — `role` and `content` — in the shape `apply_chat_template` takes.

A plain `dict` rather than a `TypedDict` because the library's parameter is
`list[dict[str, str]]`, and a list is invariant: a `list[TypedDict]` cannot be passed to it
without either a copy on every call or an ignore comment covering the whole argument.
`messages` is the only constructor, and the suite asserts the roles it emits."""


@dataclass(frozen=True, slots=True)
class Shot:
    """One demonstration pair."""

    source: str
    target: str


class Encoded(Protocol):
    """A tokenizer's batch output: indexable, and movable to a device."""

    def __getitem__(self, key: str, /) -> object: ...

    def to(self, device: str) -> Encoded: ...


class ChatTokenizer(Protocol):
    """The slice of a `transformers` tokenizer the generation path drives.

    Three of these return a union decided by the call's own flags — `apply_chat_template`
    by `tokenize`, `decode` by whether the ids are one row or many, `generate` on
    `TextGenerator` by `return_dict_in_generate`. The unions are declared rather than
    asserted away, so `require_str` and `require_tensor` narrow them where the flag is
    actually known, and changing a flag becomes a caught error rather than a `TypeError`
    raised somewhere further down.
    """

    def apply_chat_template(
        self, conversation: list[Message], *, tokenize: bool, add_generation_prompt: bool
    ) -> object: ...

    def __call__(
        self,
        text: list[str],
        *,
        return_tensors: str,
        padding: bool,
        padding_side: str,
        add_special_tokens: bool,
    ) -> Encoded: ...

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str | list[str]: ...


class TextGenerator(Protocol):
    """The slice of a causal LM that produces continuations."""

    def generate(
        self,
        *,
        input_ids: object,
        attention_mask: object,
        max_new_tokens: int,
        do_sample: bool,
    ) -> object: ...


def require_str(value: object, what: str) -> str:
    if not isinstance(value, str):
        message = f"{what} returned {type(value).__name__}, not a string"
        raise PromptError(message)
    return value


def require_tensor(value: object, what: str) -> torch.Tensor:
    import torch

    if not isinstance(value, torch.Tensor):
        message = f"{what} returned {type(value).__name__}, not a tensor"
        raise PromptError(message)
    return value


def language_name(code: str) -> str:
    if code not in LANGUAGE_NAME:
        message = f"no language name for {code!r}; have {sorted(LANGUAGE_NAME)}"
        raise PromptError(message)
    return LANGUAGE_NAME[code]


def split_direction(direction: str) -> tuple[str, str]:
    if direction.count("-") != 1:
        message = f"{direction!r} is not `src-tgt`"
        raise PromptError(message)
    source, target = direction.split("-")
    return language_name(source), language_name(target)


def instruction(direction: str) -> str:
    source, target = split_direction(direction)
    return (
        f"Translate the {source} sentence the user sends into {target}. "
        f"Reply with the {target} translation only, on one line, with no explanation."
    )


def shots(
    sources: Sequence[str], targets: Sequence[str], count: int = SHOTS, *, seed: int = SHOT_SEED
) -> tuple[Shot, ...]:
    """`count` demonstrations drawn from a parallel split, reproducibly.

    Sampled rather than taken from the head: FLORES+ is ordered by document, so the first
    rows are consecutive sentences of one article and would demonstrate one domain.
    """
    if len(sources) != len(targets):
        message = f"{len(sources)} sources against {len(targets)} targets"
        raise PromptError(message)
    if not 0 <= count <= len(sources):
        message = f"cannot draw {count} shots from {len(sources)} pairs"
        raise PromptError(message)
    rng = random.Random(seed)  # noqa: S311 — a fixed-seed draw, not a secret
    chosen = sorted(rng.sample(range(len(sources)), count))
    return tuple(Shot(source=sources[i], target=targets[i]) for i in chosen)


def messages(direction: str, text: str, examples: Sequence[Shot] = ()) -> list[Message]:
    """The chat turns for one sentence.

    `assistant` for the demonstration answers: the base's template accepts `system`, `user`
    and `assistant` and raises `TemplateError` on anything else, so a shot written under a
    different name kills the run rather than degrading it.
    """
    turns: list[Message] = [{"role": "system", "content": instruction(direction)}]
    for shot in examples:
        turns.append({"role": "user", "content": shot.source})
        turns.append({"role": "assistant", "content": shot.target})
    turns.append({"role": "user", "content": text})
    return turns


def length_ordered_batches(lengths: Sequence[int], size: int) -> list[list[int]]:
    """Indices grouped into batches of similar length, longest first.

    Padding is the cost of a ragged batch, and a corpus-order batch pads every sentence up
    to its longest neighbour — worth 5.0× in padded tokens on the pivot corpus. Indices are
    yielded rather than texts so the caller restores input order, which scoring depends on.
    Longest first so the largest allocation happens on the first batch rather than after
    memory is fragmented.
    """
    if size < 1:
        message = f"batch size must be positive, got {size}"
        raise PromptError(message)
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    return [order[start : start + size] for start in range(0, len(order), size)]


def new_tokens(lengths: Sequence[int]) -> int:
    """The generation budget for one batch, from its source lengths in tokens."""
    if not lengths:
        message = "an empty batch has no generation budget"
        raise PromptError(message)
    return min(MAX_NEW_TOKENS, TOKEN_SLACK + 2 * max(lengths))


def hypothesis(text: str) -> str:
    """The translation out of a chat reply.

    An instruction-tuned model answers in prose: a label, a quoted sentence, sometimes a
    note underneath. Everything after the first non-empty line is dropped, because a
    reference is one sentence and scoring a paragraph against it measures obedience rather
    than translation.
    """
    for line in text.strip().splitlines():
        cleaned = LABEL.sub("", line).strip().strip(QUOTES).strip()
        if cleaned:
            return cleaned
    return ""
