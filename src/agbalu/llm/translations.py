"""Translation conversations built from human sentence pairs (task 11.22).

The answer to a translation request is a human translation, never the teacher's: the
teacher writes only the requests, as templates in the five prompt languages, and each
conversation puts one human pair into one template. The answer is the pair's other side
alone, which also teaches the model to translate without commentary.

The pairs are Tatoeba's (CC-BY 2.0 FR), the one human source in `mt/train.jsonl` whose
sentences are natural and aligned. Read 2026-10-17: `hf.tutlaytai.multilang-to-kabyle`
carries Spanish on its English and French side, `hf.finetranslations-kab` is web-page
debris, `opus.wikimedia-kab` is misaligned, and the localisation sources are interface
strings with placeholders. Sentences naming Tatoeba's stock characters Tom and Mary are
left out, or they would set the register of every request.

Every pair is accounted for: kept, or counted under the filter that dropped it.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter as Tally
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from agbalu.llm.chat import SYSTEM_POLICY
from agbalu.llm.teacher import LANGUAGE_NAMES, Prompts, fill, orthography

SOURCES: Final = frozenset({"opus.tatoeba-kab", "local.tatoeba-kab-eng"})

DIRECTIONS: Final[dict[str, int]] = {"eng-kab": 30, "fra-kab": 30, "kab-eng": 20, "kab-fra": 20}
"""Conversations per direction: `sft_data.TASK_TARGETS["translation"]`, 100, weighted to
translating into Kabyle, the direction the model is for."""

ASKED_IN: Final[dict[str, tuple[str, ...]]] = {
    "kab": ("kab", "eng", "fra", "ara", "zho"),
    "eng": ("kab", "eng", "fra"),
    "fra": ("kab", "eng", "fra"),
}
"""For each language translated into, the languages the request may be written in."""

PER_PAIR: Final = 8
"""Templates asked of the teacher per (request language, target language)."""

MIN_TEMPLATES: Final = 4
"""Fewer usable templates than this for a pair of languages stops the build."""

MIN_WORDS: Final = 6
MAX_WORDS: Final = 40
POLICY_SHARE: Final = 0.7
"""The rest carry no system prompt; an operator prompt needs a writer, and these have none."""

SEED: Final = 20261017
PLACEHOLDER: Final = "{text}"
STOCK_NAMES: Final = re.compile(r"\b(?:Tom|Mary|Tum|Meri)\b")


class TranslationError(Exception):
    """The requests or the pairs cannot make the translation conversations."""


@dataclass(frozen=True, slots=True)
class Template:
    ask: str
    into: str
    text: str


def language_pairs() -> list[tuple[str, str]]:
    """Every (request language, target language) the teacher writes templates for."""
    return [(ask, into) for into, asks in ASKED_IN.items() for ask in asks]


def requests_prompt(prompts: Prompts) -> str:
    """The one prompt that asks the teacher for every request template."""
    pairs = "\n".join(
        f"- ask: {ask} ({LANGUAGE_NAMES[ask]}), into: {into} ({LANGUAGE_NAMES[into]})"
        for ask, into in language_pairs()
    )
    values = {"orthography": orthography(), "per_pair": str(PER_PAIR), "pairs": pairs}
    return fill(prompts.requests, values)


def parse_requests(lines: Sequence[str]) -> tuple[list[Template], Tally[str]]:
    """Templates from a reply; a line that is not one usable template is counted by reason."""
    found: list[Template] = []
    rejected: Tally[str] = Tally()
    seen: set[tuple[str, str, str]] = set()
    allowed = set(language_pairs())
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rejected["not_json"] += 1
            continue
        ask = row.get("ask") if isinstance(row, dict) else None
        into = row.get("into") if isinstance(row, dict) else None
        text = row.get("template") if isinstance(row, dict) else None
        if (ask, into) not in allowed or not isinstance(ask, str) or not isinstance(into, str):
            rejected["unknown_languages"] += 1
        elif not isinstance(text, str) or text.count(PLACEHOLDER) != 1:
            rejected["placeholder"] += 1
        elif text.replace(PLACEHOLDER, "").count("{") or text.replace(PLACEHOLDER, "").count("}"):
            rejected["stray_brace"] += 1
        elif (ask, into, text.strip()) in seen:
            rejected["duplicate"] += 1
        else:
            seen.add((ask, into, text.strip()))
            found.append(Template(ask, into, text.strip()))
    return found, rejected


def check_templates(templates: Sequence[Template]) -> dict[str, int]:
    """Templates per language pair, refused if any pair has fewer than `MIN_TEMPLATES`."""
    counts = Tally((t.ask, t.into) for t in templates)
    short = [
        f"{ask}->{into}" for ask, into in language_pairs() if counts[ask, into] < MIN_TEMPLATES
    ]
    if short:
        message = f"fewer than {MIN_TEMPLATES} templates for {short}; ask the teacher again"
        raise TranslationError(message)
    return {f"{ask}->{into}": counts[ask, into] for ask, into in language_pairs()}


@dataclass(frozen=True, slots=True)
class Pair:
    direction: str
    text: str
    translation: str
    source: str
    record: int

    @property
    def kabyle(self) -> str:
        return self.text if self.direction.startswith("kab") else self.translation


@dataclass(slots=True)
class PairTally:
    read: int = 0
    dropped: Tally[str] = field(default_factory=Tally)
    eligible: Tally[str] = field(default_factory=Tally)


def eligible_pairs(
    rows: Iterable[Mapping[str, object]],
    usable: Callable[[str], bool],
    used: set[str],
    tally: PairTally,
) -> dict[str, list[Pair]]:
    """Every pair that may seed a conversation, per direction. `usable` is false for text in
    an evaluation set or the held-out split; `used` holds Kabyle sentences a seed card has."""
    found: dict[str, list[Pair]] = {direction: [] for direction in DIRECTIONS}
    for record, row in enumerate(rows):
        tally.read += 1
        direction, text = row.get("direction"), row.get("source")
        translation, source = row.get("target"), row.get("source_id")
        reason = drop_reason(direction, text, translation, source)
        if reason is None and isinstance(text, str) and isinstance(translation, str):
            kabyle = text if str(direction).startswith("kab") else translation
            if kabyle in used:
                reason = "used_by_a_card"
            elif not (usable(text) and usable(translation)):
                reason = "evaluation_or_held_out"
            else:
                found[str(direction)].append(
                    Pair(str(direction), text, translation, str(source), record)
                )
                tally.eligible[str(direction)] += 1
                continue
        tally.dropped[reason or "malformed"] += 1
    return found


def drop_reason(direction: object, text: object, translation: object, source: object) -> str | None:
    if direction not in DIRECTIONS:
        return "direction"
    if source not in SOURCES:
        return "source"
    if not isinstance(text, str) or not isinstance(translation, str):
        return "malformed"
    words = (len(text.split()), len(translation.split()))
    if min(words) < MIN_WORDS or max(words) > MAX_WORDS:
        return "length"
    if STOCK_NAMES.search(text) or STOCK_NAMES.search(translation):
        return "stock_names"
    return None


def draw_pairs(eligible: Mapping[str, Sequence[Pair]], seed: int = SEED) -> dict[str, list[Pair]]:
    """`DIRECTIONS[d]` pairs per direction, drawn over every eligible pair; no Kabyle sentence
    is drawn twice, since the file holds each pair in both directions."""
    draw = random.Random(seed)
    taken: set[str] = set()
    chosen: dict[str, list[Pair]] = {}
    for direction, count in DIRECTIONS.items():
        pool = [pair for pair in eligible[direction] if pair.kabyle not in taken]
        if len(pool) < count:
            message = f"{direction}: {len(pool)} eligible pairs for {count} conversations"
            raise TranslationError(message)
        chosen[direction] = draw.sample(pool, count)
        taken.update(pair.kabyle for pair in chosen[direction])
    return chosen


@dataclass(frozen=True, slots=True)
class Provenance:
    """What each row records about where its text came from and how it was normalised."""

    licence_of: Callable[[str], str]
    normalise: Callable[[str], str]
    normaliser_version: str


def conversations(
    chosen: Mapping[str, Sequence[Pair]],
    templates: Sequence[Template],
    provenance: Provenance,
    seed: int = SEED,
) -> list[dict[str, object]]:
    """One accepted-set row per drawn pair, in the format `sft_data.accepted_row` writes."""
    draw = random.Random(seed + 1)
    by_pair: dict[tuple[str, str], list[Template]] = {}
    for template in templates:
        by_pair.setdefault((template.ask, template.into), []).append(template)
    rows: list[dict[str, object]] = []
    for direction, pairs in chosen.items():
        source_language, into = direction.split("-")
        for number, pair in enumerate(pairs, start=1):
            template = draw.choice(by_pair[draw.choice(ASKED_IN[into]), into])
            text = provenance.normalise(pair.text) if source_language == "kab" else pair.text
            answer = provenance.normalise(pair.translation) if into == "kab" else pair.translation
            system = "policy" if draw.random() < POLICY_SHARE else "none"
            turns = [{"role": "user", "content": template.text.replace(PLACEHOLDER, text)}]
            turns.append({"role": "assistant", "content": answer})
            if system == "policy":
                turns.insert(0, {"role": "system", "content": SYSTEM_POLICY})
            rows.append(
                {
                    "id": f"translation-{direction}-{number:03d}",
                    "split": "train",
                    "card": None,
                    "card_split": "train",
                    "task": "translation",
                    "domain": "translation",
                    "think": False,
                    "system": system,
                    "turns": turns,
                    "teacher": {"model": "human translation", "date": None, "batch": None},
                    "request": {"ask": template.ask, "into": template.into},
                    "prompt_version": None,
                    "normaliser": provenance.normaliser_version,
                    "source": pair.source,
                    "licence": provenance.licence_of(pair.source),
                    "record": pair.record,
                }
            )
    return rows


def build(
    rows: Iterable[Mapping[str, object]],
    templates: Sequence[Template],
    usable: Callable[[str], bool],
    used: set[str],
    provenance: Provenance,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """The translation conversations and their stats, every pair read accounted for."""
    per_pair = check_templates(templates)
    tally = PairTally()
    eligible = eligible_pairs(rows, usable, used, tally)
    kept, dropped = sum(tally.eligible.values()), sum(tally.dropped.values())
    if kept + dropped != tally.read:
        message = f"{tally.read} pairs read, {kept} eligible, {dropped} dropped"
        raise TranslationError(message)
    conversations_out = conversations(draw_pairs(eligible), templates, provenance)
    stats: dict[str, object] = {
        "pairs_read": tally.read,
        "eligible": dict(tally.eligible),
        "dropped": dict(sorted(tally.dropped.items())),
        "conversations": dict(Tally(str(row["id"]).rsplit("-", 1)[0] for row in conversations_out)),
        "templates": per_pair,
        "sources": sorted(SOURCES),
        "seed": SEED,
    }
    return conversations_out, stats
