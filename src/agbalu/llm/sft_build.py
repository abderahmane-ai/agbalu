"""Accepted conversations and raw-text replay as the SFT rows on disk (task 11.23).

Input: `sft-v1.jsonl`, one accepted conversation a line with its split (`train` or
`validation`) and turns. Every assistant turn becomes one segment through `agbalu.llm.chat`.
Replay is sized to `SFT_REPLAY_SHARE` of the supervised tokens and split across the CPT
inputs by the stream's own shares, so SFT keeps reading what CPT read in the proportion it
read it. Each replay file is sampled by a seeded Bernoulli draw over the whole file, at a
rate set from its stats file's token count, and then cut to the budget by a seeded order:
a prefix of a source-ordered file is not a sample.

Every conversation and replay document is accounted for: written, or rejected with its
reason.
"""

from __future__ import annotations

import json
import random
from collections import Counter as Tally
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

from agbalu.llm.chat import ChatError, TemplateTokenizer, Turn, samples
from agbalu.llm.recipe import SFT_REPLAY_SHARE, SFT_ROW_TOKENS, SFT_SEED
from agbalu.llm.sft_rows import Segment, from_document, from_sample, pack, summary, write

REPLAY_SHARES: Final[dict[str, float]] = {"kabyle": 0.574, "aligned": 0.096, "replay": 0.330}
"""The CPT stream's token shares (`cpt-v2.stats.json`), the mix SFT replay keeps."""

OVERDRAW: Final = 1.25
"""The Bernoulli rate is set this far above the budget, so a draw almost never falls short;
the surplus is cut by a seeded order."""

SPLITS: Final = ("train", "validation")
ROLES: Final = ("system", "user", "assistant")


class BuildError(Exception):
    """The conversations or the replay cannot make the SFT rows."""


class TextTokenizer(Protocol):
    """The base tokenizer that encodes replay text exactly as continued pretraining did."""

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...

    @property
    def eos(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    split: str
    turns: tuple[Turn, ...]


def parse_turn(raw: object, where: str) -> Turn:
    if not isinstance(raw, Mapping):
        message = f"{where}: a turn is not an object"
        raise BuildError(message)
    role, content, reasoning = raw.get("role"), raw.get("content"), raw.get("reasoning")
    if role not in ROLES or not isinstance(content, str):
        message = f"{where}: turn role {role!r} or content of type {type(content).__name__}"
        raise BuildError(message)
    if reasoning is not None and not isinstance(reasoning, str):
        message = f"{where}: reasoning of type {type(reasoning).__name__}"
        raise BuildError(message)
    return Turn(role, content, reasoning)


def read_conversations(path: Path) -> list[Conversation]:
    if not path.is_file():
        message = f"{path} not found; it is written by `make llm TASK=sft-build`"
        raise BuildError(message)
    found: list[Conversation] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            where = f"{path}:{number}"
            row = json.loads(line)
            identifier, split, turns = row.get("id"), row.get("split"), row.get("turns")
            if not isinstance(identifier, str) or split not in SPLITS:
                message = f"{where}: id {identifier!r} or split {split!r} is not usable"
                raise BuildError(message)
            if identifier in seen:
                message = f"{where}: id {identifier} repeats"
                raise BuildError(message)
            if not isinstance(turns, list):
                message = f"{where}: turns is not a list"
                raise BuildError(message)
            seen.add(identifier)
            parsed = tuple(parse_turn(turn, where) for turn in turns)
            found.append(Conversation(identifier, split, parsed))
    return found


@dataclass(slots=True)
class Tallies:
    conversations: Tally[str] = field(default_factory=Tally)
    rejected: Tally[str] = field(default_factory=Tally)
    replay_tokens: Tally[str] = field(default_factory=Tally)
    replay_documents: Tally[str] = field(default_factory=Tally)
    truncated: int = 0


def conversation_segments(
    conversations: Sequence[Conversation],
    tokenizer: TemplateTokenizer,
    tallies: Tallies,
    capacity: int = SFT_ROW_TOKENS,
) -> dict[str, list[Segment]]:
    """Segments per split; a turn that will not render or will not fit is rejected by name."""
    out: dict[str, list[Segment]] = {split: [] for split in SPLITS}
    for conversation in conversations:
        try:
            rendered = samples(conversation.id, conversation.turns, tokenizer)
        except ChatError:
            tallies.rejected["unrenderable"] += 1
            continue
        if any(len(sample.input_ids) > capacity for sample in rendered):
            tallies.rejected["longer_than_a_row"] += 1
            continue
        tallies.conversations[conversation.split] += 1
        out[conversation.split].extend(from_sample(sample) for sample in rendered)
    return out


def supervised_tokens(segments: Sequence[Segment]) -> int:
    """What the trainer scores: supervised tokens, never a segment's first token."""
    return sum(sum(segment.supervised[1:]) for segment in segments)


def replay_budget(conversation_tokens: int, share: float = SFT_REPLAY_SHARE) -> int:
    if conversation_tokens < 1:
        message = "no supervised conversation token to size the replay against"
        raise BuildError(message)
    return round(conversation_tokens * share / (1 - share))


def bernoulli_lines(path: Path, rate: float, seed: int) -> Iterator[str]:
    """Each line's text kept with probability `rate`, over the whole file."""
    draw = random.Random(seed)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if draw.random() < rate:
                text = json.loads(line).get("text")
                if not isinstance(text, str):
                    message = f"{path}:{number} has no text"
                    raise BuildError(message)
                yield text


def replay_segments(
    sources: Mapping[str, tuple[Path, int]],
    budget: int,
    tokenizer: TextTokenizer,
    tallies: Tallies,
) -> list[Segment]:
    """`budget` tokens across the CPT inputs by `REPLAY_SHARES`; `sources` maps each kind to
    its file and the token count its stats file records. A draw that falls short is redrawn
    at twice the rate under the next seed, up to the whole file."""
    if set(sources) != set(REPLAY_SHARES):
        message = f"replay sources {sorted(sources)} are not {sorted(REPLAY_SHARES)}"
        raise BuildError(message)
    segments: list[Segment] = []
    for offset, (kind, share) in enumerate(sorted(REPLAY_SHARES.items())):
        path, file_tokens = sources[kind]
        target = round(budget * share)
        rate = min(1.0, OVERDRAW * target / max(1, file_tokens))
        attempt = 0
        while True:
            chosen = draw_documents(
                path, rate, SFT_SEED + 100 * offset + attempt, target, tokenizer
            )
            if chosen.tokens >= target or rate >= 1.0:
                break
            rate, attempt = min(1.0, 2 * rate), attempt + 1
        if chosen.tokens < target:
            message = f"replay {kind}: {chosen.tokens:,} tokens in all of {path}, budget {target:,}"
            raise BuildError(message)
        segments.extend(chosen.segments(tokenizer.eos, kind))
        tallies.rejected.update({f"replay_{kind}_empty": chosen.empty} if chosen.empty else {})
        tallies.truncated += chosen.truncated
        tallies.replay_documents[kind] = len(chosen.documents)
        tallies.replay_tokens[kind] = chosen.tokens
    return segments


@dataclass(slots=True)
class Draw:
    documents: list[list[int]] = field(default_factory=list)
    tokens: int = 0
    empty: int = 0
    truncated: int = 0

    def segments(self, eos: int, kind: str) -> list[Segment]:
        return [from_document(ids, eos, f"replay-{kind}") for ids in self.documents]


def draw_documents(
    path: Path, rate: float, seed: int, target: int, tokenizer: TextTokenizer
) -> Draw:
    """A Bernoulli draw over `path`, shuffled by `seed`, kept until exactly `target` tokens; a
    document longer than a row, its leading `eos` included, is cut to the row, and the last
    document is cut to what the budget has left. Without that cut, one long document put
    replay at 11.4% of an 812-conversation set (2026-10-17)."""
    texts = list(bernoulli_lines(path, rate, seed))
    random.Random(seed).shuffle(texts)
    chosen = Draw()
    for text, ids in zip(texts, tokenizer.encode_batch(texts), strict=True):
        if chosen.tokens >= target:
            break
        if not ids or not text.strip():
            chosen.empty += 1
            continue
        kept = ids[: min(SFT_ROW_TOKENS - 1, target - chosen.tokens)]
        chosen.truncated += len(kept) < len(ids)
        chosen.documents.append(kept)
        chosen.tokens += len(kept)
    return chosen


SHARE_TOLERANCE: Final = 0.01
"""How far the built replay share may sit from `SFT_REPLAY_SHARE`: one document overshoots
each kind's budget by at most its own length."""


@dataclass(frozen=True, slots=True)
class Outputs:
    train: Path
    validation: Path


def build(
    conversations: Sequence[Conversation],
    chat_tokenizer: TemplateTokenizer,
    text_tokenizer: TextTokenizer,
    replay: Mapping[str, tuple[Path, int]],
    outputs: Outputs,
) -> dict[str, object]:
    """Write the training and validation rows; return their stats after every check."""
    tallies = Tallies()
    segments = conversation_segments(conversations, chat_tokenizer, tallies)
    kept = sum(tallies.conversations.values())
    rejected = sum(tallies.rejected.values())
    if kept + rejected != len(conversations):
        message = f"{len(conversations)} conversations read, {kept} kept, {rejected} rejected"
        raise BuildError(message)
    if not segments["train"] or not segments["validation"]:
        counts = {split: len(found) for split, found in segments.items()}
        message = f"a split has no turn to train or validate on: {counts}"
        raise BuildError(message)
    conversation_tokens = supervised_tokens(segments["train"])
    budget = replay_budget(conversation_tokens)
    drawn = replay_segments(replay, budget, text_tokenizer, tallies)
    replay_tokens = supervised_tokens(drawn)
    share = replay_tokens / (replay_tokens + conversation_tokens)
    if abs(share - SFT_REPLAY_SHARE) > SHARE_TOLERANCE:
        message = f"replay is {share:.4f} of supervised tokens, not {SFT_REPLAY_SHARE}"
        raise BuildError(message)
    train_rows = pack([*segments["train"], *drawn])
    validation_rows = pack(segments["validation"])
    write(train_rows, outputs.train)
    write(validation_rows, outputs.validation)
    return {
        "conversations": dict(tallies.conversations),
        "rejected": dict(tallies.rejected),
        "conversation_supervised_tokens": conversation_tokens,
        "replay_budget": budget,
        "replay_supervised_tokens": replay_tokens,
        "replay_share": round(share, 4),
        "replay_tokens_by_kind": dict(tallies.replay_tokens),
        "replay_documents_by_kind": dict(tallies.replay_documents),
        "replay_truncated": tallies.truncated,
        "train": summary(train_rows),
        "validation": summary(validation_rows),
        "seed": SFT_SEED,
    }
