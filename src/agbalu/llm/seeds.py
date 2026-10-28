"""The seed bank: what each generated conversation is about (task 11.22).

1,200 cards, each asking the teacher for `CONVERSATIONS_PER_CARD` conversations, so 12,000
candidates from which the filters keep about 5,000. Every quota in
`docs/llm_post_training_design.md` §4.4–4.5 and §6.4 is a card count here, scaled from the
5,000 conversations to the 1,200 cards:

- the primary task, by card count;
- the subject domain, stratified independently of the task by a seeded shuffle;
- thinking: 240 cards (20%) spread over the domains by the §6.4 quotas;
- the system-prompt condition (60/25/15) and multi-turn (25%), seeded per card.

Where a card's seed comes from follows from its task: grounded tasks and native Kabyle cards
on encyclopaedic domains carry a Kabyle Wikipedia passage; translation and non-Kabyle-input
cards carry a human translation pair; the rest are seeded by task and domain alone. The split
is made per card, before generation, so no rewording can carry an evaluation card's content
into training.
"""

from __future__ import annotations

import json
import random
from collections import Counter as Tally
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, Literal

Split = Literal["train", "validation", "release"]
System = Literal["policy", "none", "operator"]

SEED: Final = 20261016
CONVERSATIONS_PER_CARD: Final = 10

TASK_CARDS: Final[dict[str, int]] = {
    "native_kabyle": 540,
    "non_kabyle_input": 180,
    "translation": 120,
    "reasoning": 120,
    "code": 72,
    "hierarchy": 60,
    "identity": 24,
    "grounded": 84,
}
DOMAIN_CARDS: Final[dict[str, int]] = {
    "everyday": 240,
    "knowledge": 216,
    "culture": 180,
    "geography": 120,
    "civic": 132,
    "maths": 120,
    "code": 84,
    "writing": 60,
    "translation": 48,
}
THINK_CARDS: Final[dict[str, int]] = {
    "maths": 60,
    "knowledge": 43,
    "everyday": 36,
    "code": 29,
    "culture": 24,
    "geography": 14,
    "civic": 14,
    "writing": 10,
    "translation": 10,
}
"""§6.4's 1,000 think-mode conversations as 240 cards, per domain."""

SYSTEM_SHARES: Final[dict[System, float]] = {"policy": 0.60, "none": 0.25, "operator": 0.15}
MULTI_TURN_SHARE: Final = 0.25
MULTI_TURN_USERS: Final = 3
SPLIT_SHARES: Final[dict[Split, float]] = {"validation": 0.05, "release": 0.05}

OTHER_LANGUAGES: Final = ("eng", "fra", "ara", "zho")
KABYLE_INPUT_SHARE: Final = 0.70
"""Native tasks are asked in Kabyle mostly; §4.6's other languages ask the rest."""

GROUNDED_DOMAINS: Final = frozenset({"culture", "geography", "knowledge", "civic"})
PASSAGE_CHARACTERS: Final = 1500
MIN_PASSAGE_CHARACTERS: Final = 400


class SeedError(Exception):
    """The quotas do not add up, or a seed pool is too small for the cards that need it."""


@dataclass(frozen=True, slots=True)
class Passage:
    text: str
    source: str
    record: str
    licence: str
    other: str | None = None
    """The non-Kabyle side of a translation pair; None for a Wikipedia passage."""

    other_language: str | None = None


@dataclass(frozen=True, slots=True)
class Card:
    id: str
    task: str
    domain: str
    think: bool
    input_language: str
    system: System
    user_turns: int
    split: Split
    passage: Passage | None = None

    @property
    def seed_source(self) -> str:
        if self.passage is None:
            return "identity" if self.task in {"identity", "hierarchy"} else "task"
        return "parallel" if self.passage.other is not None else "grounded"


def check_quotas() -> int:
    """Every quota table counts the same cards, and thinking fits inside its domains."""
    total = sum(TASK_CARDS.values())
    if sum(DOMAIN_CARDS.values()) != total:
        message = f"domains count {sum(DOMAIN_CARDS.values())} cards, tasks {total}"
        raise SeedError(message)
    over = {d: n for d, n in THINK_CARDS.items() if n > DOMAIN_CARDS.get(d, 0)}
    if over:
        message = f"more think cards than domain cards in {over}"
        raise SeedError(message)
    return total


def expanded(counts: Mapping[str, int]) -> list[str]:
    return [name for name, count in counts.items() for _ in range(count)]


def system_condition(draw: random.Random) -> System:
    """One condition drawn by `SYSTEM_SHARES`."""
    point = draw.random()
    chosen: System = "policy"
    for name, share in SYSTEM_SHARES.items():
        chosen = name
        if point < share:
            break
        point -= share
    return chosen


def input_language(draw: random.Random, task: str, think: bool) -> str:
    if task == "non_kabyle_input":
        return draw.choice(OTHER_LANGUAGES)
    kabyle_share = 0.5 if think or task == "translation" else KABYLE_INPUT_SHARE
    return "kab" if draw.random() < kabyle_share else draw.choice(OTHER_LANGUAGES)


def allocate(seed: int = SEED) -> list[Card]:
    """Every card's task, domain, mode, language, system condition, turns and split."""
    total = check_quotas()
    draw = random.Random(seed)
    tasks = expanded(TASK_CARDS)
    domains = expanded(DOMAIN_CARDS)
    draw.shuffle(domains)
    think_slots: set[int] = set()
    for domain, count in THINK_CARDS.items():
        members = [i for i, name in enumerate(domains) if name == domain]
        think_slots.update(draw.sample(members, count))
    order = list(range(total))
    draw.shuffle(order)
    held: dict[int, Split] = {}
    start = 0
    for name, share in SPLIT_SHARES.items():
        size = round(total * share)
        held.update(dict.fromkeys(order[start : start + size], name))
        start += size
    cards: list[Card] = []
    for index, (task, domain) in enumerate(zip(tasks, domains, strict=True)):
        think = index in think_slots
        split: Split = held.get(index, "train")
        cards.append(
            Card(
                id=f"card-{index:04d}",
                task=task,
                domain=domain,
                think=think,
                input_language=input_language(draw, task, think),
                system=system_condition(draw),
                user_turns=MULTI_TURN_USERS if draw.random() < MULTI_TURN_SHARE else 1,
                split=split,
            )
        )
    return cards


CORE_CARDS: Final = 120
"""Training cards drawn by task share; rounding gives 119. They were the pilot's cards, and
their batches were written before the set was cut to them."""

EXTRA_THINK: Final = 4
"""The core holds 20 think cards, 16.8%; four more bring the set to the §6 fifth."""

EXTRA_IDENTITY: Final = 2
"""The core holds two identity cards, 20 candidates for a target of 20; two more give room."""

VALIDATION_CARDS: Final = 6


def working_set(cards: Sequence[Card], seed: int = SEED) -> list[Card]:
    """The cards the teacher is asked for: about 1,000 accepted conversations
    (`docs/llm_post_training_design.md` §4.4), each task in its share of the bank, by seed."""
    draw = random.Random(seed + 2)
    chosen: list[Card] = []
    for task, count in TASK_CARDS.items():
        members = [card for card in cards if card.task == task and card.split == "train"]
        take = max(1, round(CORE_CARDS * count / sum(TASK_CARDS.values())))
        chosen.extend(draw.sample(members, min(take, len(members))))
    extra = random.Random(seed + 3)
    taken = {card.id for card in chosen}
    rest = [card for card in cards if card.split == "train" and card.id not in taken]
    chosen.extend(extra.sample([c for c in rest if c.think], EXTRA_THINK))
    taken = {card.id for card in chosen}
    identity = [c for c in rest if c.task == "identity" and c.id not in taken]
    chosen.extend(extra.sample(identity, EXTRA_IDENTITY))
    validation = [card for card in cards if card.split == "validation"]
    chosen.extend(extra.sample(validation, VALIDATION_CARDS))
    return sorted(chosen, key=lambda card: card.id)


TASKS_ROTATION: Final = (
    "reasoning",
    "native_kabyle",
    "code",
    "grounded",
    "hierarchy",
    "non_kabyle_input",
    "translation",
    "identity",
)


def teacher_cards(cards: Sequence[Card], seed: int = SEED) -> list[Card]:
    """All training cards ordered in a balanced, interleaved sequence across all tasks,
    allocating 1 thinking card every 3 cards (~33% think share in new generation) to build
    towards the 25% target."""
    train_cards = [c for c in cards if c.split == "train"]
    rng = random.Random(seed + 42)

    pools: dict[tuple[str, bool], list[Card]] = {}
    for c in train_cards:
        pools.setdefault((c.task, c.think), []).append(c)

    for key in pools:
        rng.shuffle(pools[key])

    scheduled: list[Card] = []
    task_idx = 0

    while any(pools.values()):
        want_think = len(scheduled) % 3 == 0

        found_card: Card | None = None
        for offset in range(len(TASKS_ROTATION)):
            t = TASKS_ROTATION[(task_idx + offset) % len(TASKS_ROTATION)]
            bucket = pools.get((t, want_think), [])
            if bucket:
                found_card = bucket.pop(0)
                task_idx = (task_idx + offset + 1) % len(TASKS_ROTATION)
                break

        if found_card is None:
            for offset in range(len(TASKS_ROTATION)):
                t = TASKS_ROTATION[(task_idx + offset) % len(TASKS_ROTATION)]
                bucket = pools.get((t, not want_think), [])
                if bucket:
                    found_card = bucket.pop(0)
                    task_idx = (task_idx + offset + 1) % len(TASKS_ROTATION)
                    break

        if found_card is None:
            for bucket in pools.values():
                if bucket:
                    found_card = bucket.pop(0)
                    break

        if found_card is not None:
            scheduled.append(found_card)

    return scheduled


def needs_passage(card: Card) -> bool:
    return card.task == "grounded" or (
        card.task == "native_kabyle" and card.domain in GROUNDED_DOMAINS
    )


def needs_pair(card: Card) -> bool:
    return card.task in {"translation", "non_kabyle_input"}


def reservoir(items: Iterable[Passage], size: int, seed: int) -> list[Passage]:
    """`size` items drawn uniformly from the whole stream, never its prefix."""
    draw = random.Random(seed)
    kept: list[Passage] = []
    for seen, item in enumerate(items):
        if len(kept) < size:
            kept.append(item)
        else:
            slot = draw.randrange(seen + 1)
            if slot < size:
                kept[slot] = item
    return kept


def attach(
    cards: Sequence[Card], passages: Iterable[Passage], pairs: Iterable[Passage], seed: int = SEED
) -> list[Card]:
    """Each card that needs a seed text gets its own, drawn from the whole pool."""
    wanting_passage = [card for card in cards if needs_passage(card)]
    wanting_pair = [card for card in cards if needs_pair(card)]
    drawn_passages = reservoir(passages, len(wanting_passage), seed)
    drawn_pairs = reservoir(pairs, len(wanting_pair), seed + 1)
    for label, drawn, wanting in (
        ("passages", drawn_passages, wanting_passage),
        ("pairs", drawn_pairs, wanting_pair),
    ):
        if len(drawn) < len(wanting):
            message = f"{len(wanting)} cards need {label}, the pool holds {len(drawn)}"
            raise SeedError(message)
    given = {card.id: text for card, text in zip(wanting_passage, drawn_passages, strict=False)}
    given.update({card.id: text for card, text in zip(wanting_pair, drawn_pairs, strict=False)})
    return [replace(card, passage=given.get(card.id)) for card in cards]


def wiki_passages(documents: Iterable[tuple[str, str]], licence: str) -> Iterator[Passage]:
    """Article openings long enough to ground a conversation, cut at a paragraph."""
    for record, text in documents:
        opening = text[:PASSAGE_CHARACTERS]
        cut = opening.rfind("\n")
        if cut >= MIN_PASSAGE_CHARACTERS:
            opening = opening[:cut]
        if len(opening.strip()) >= MIN_PASSAGE_CHARACTERS:
            yield Passage(opening.strip(), "wikimedia.kabwiki", record, licence)


PAIR_MIN_WORDS: Final = 8
"""A pair shorter than this seeds nothing a conversation can be built on."""

PAIR_DIRECTIONS: Final = {"kab-eng": "eng", "kab-fra": "fra"}
"""One direction per pair: `mt/train.jsonl` holds every pair twice."""


def pair_passages(
    rows: Iterable[Mapping[str, object]], licence_of: Callable[[str], str]
) -> Iterator[Passage]:
    """Human translation pairs long enough to seed a conversation, Kabyle first."""
    for index, row in enumerate(rows):
        language = PAIR_DIRECTIONS.get(str(row.get("direction")))
        kabyle, other, source = row.get("source"), row.get("target"), row.get("source_id")
        if language is None or not isinstance(kabyle, str) or not isinstance(other, str):
            continue
        if not isinstance(source, str) or len(kabyle.split()) < PAIR_MIN_WORDS:
            continue
        yield Passage(kabyle, source, str(index), licence_of(source), other, language)


def write(cards: Sequence[Card], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(asdict(card), ensure_ascii=False) + "\n")
    partial.replace(path)


def read(path: Path) -> list[Card]:
    if not path.is_file():
        message = f"{path} not found; build it with `make llm TASK=seeds`"
        raise SeedError(message)
    cards: list[Card] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            passage = row.pop("passage")
            cards.append(Card(**row, passage=Passage(**passage) if passage else None))
    return cards


def summary(cards: Sequence[Card]) -> dict[str, object]:
    def count(values: Iterable[object]) -> dict[str, int]:
        return {str(k): v for k, v in sorted(Tally(values).items(), key=lambda kv: str(kv[0]))}

    return {
        "cards": len(cards),
        "conversations_requested": len(cards) * CONVERSATIONS_PER_CARD,
        "task": count(card.task for card in cards),
        "domain": count(card.domain for card in cards),
        "think": count(card.domain for card in cards if card.think),
        "input_language": count(card.input_language for card in cards),
        "system": count(card.system for card in cards),
        "user_turns": count(card.user_turns for card in cards),
        "split": count(card.split for card in cards),
        "seed_source": count(card.seed_source for card in cards),
        "seed": SEED,
    }
