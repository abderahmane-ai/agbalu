"""The file exchange with Gemini through its chat interface (task 11.22).

Out: numbered prompt batches, each a text file to paste and a manifest of the cards it asks
for. In: the operator saves each reply unedited under `data/raw/gemini-sft/`, with one header
line naming the model as the interface shows it and the date. Pasted chat output drops,
truncates and reorders rows without warning, so every reply line is checked here:

- every requested conversation is returned, missing, or rejected with its reason;
- a card id outside its batch, a repeated conversation, or a shape the card did not ask for
  is rejected, never repaired;
- a card short of its conversations is asked for again in a later batch.

The critic pass works the same way: candidates out in batches, structured verdicts in.
"""

from __future__ import annotations

import json
import re
from collections import Counter as Tally
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml

from agbalu.llm.seeds import CONVERSATIONS_PER_CARD, DOMAIN_CARDS, SYSTEM_SHARES, TASK_CARDS, Card
from agbalu.normalise.rules import ASCII_DIGRAPHS, BASE_LETTERS, SPECIFIC_LETTERS, load_rules

PROMPTS: Final = Path("resources/llm/teacher_prompts.yaml")
BATCH_DIR: Final = Path("data/interim/llm/sft/batches")
REPLY_DIR: Final = Path("data/raw/gemini-sft")
CARDS_PER_BATCH: Final = 2
"""Twenty conversations a reply: long enough to be worth a paste, short enough that a chat
reply carries them without being cut off."""

CANDIDATES_PER_CRITIC: Final = 20
INSTRUCTIONS: Final = ("generator", "critic", "translation", "requests", "schema")
PROMPT_KEYS: Final = frozenset(
    {"version", "compatible", *INSTRUCTIONS, "tasks", "domains", "systems"}
)
FLORES_PILOT: Final = 200
FLORES_BATCH: Final = 50
"""§4.2's teacher check: 200 corrected FLORES+ devtest sentences, pasted 50 at a time."""
HEADER: Final = re.compile(
    r"^#\s*model:\s*(?P<model>[^|]+?)\s*\|\s*date:\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$"
)
LANGUAGE_NAMES: Final = {
    "kab": "Kabyle",
    "eng": "English",
    "fra": "French",
    "ara": "Arabic",
    "zho": "Chinese",
}
NATURALNESS: Final = range(1, 6)
"""The critic's 1 to 5 scale."""

CRITIC_FLAGS: Final = (
    "not_kabyle",
    "translationese",
    "orthography",
    "factual_error",
    "reasoning_mismatch",
    "identity_error",
    "broken",
)


class TeacherError(Exception):
    """A prompt file, batch or reply cannot be used as it is."""


@dataclass(frozen=True, slots=True)
class Prompts:
    version: int
    compatible: tuple[int, ...]
    """Earlier versions whose candidates still count: the change after them only adds a rule
    that the filters also enforce on their output."""

    generator: str
    critic: str
    translation: str
    requests: str
    schema: str
    tasks: dict[str, str]
    domains: dict[str, str]
    systems: dict[str, str]

    @property
    def counted(self) -> frozenset[int]:
        return frozenset({self.version, *self.compatible})


def load_prompts(path: Path = PROMPTS) -> Prompts:
    """The prompt file, its keys equal to the seed bank's quota tables exactly."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != PROMPT_KEYS:
        found = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        message = f"{path}: keys {found}, expected {sorted(PROMPT_KEYS)}"
        raise TeacherError(message)
    for name, expected in (
        ("tasks", set(TASK_CARDS)),
        ("domains", set(DOMAIN_CARDS)),
        ("systems", set(SYSTEM_SHARES)),
    ):
        table = raw[name]
        if not isinstance(table, dict) or set(table) != expected:
            message = f"{path}: {name} keys differ from the seed bank's {sorted(expected)}"
            raise TeacherError(message)
    texts = [raw[key] for key in INSTRUCTIONS]
    if not isinstance(raw["version"], int) or not all(isinstance(text, str) for text in texts):
        message = f"{path}: version must be an integer and the instruction blocks text"
        raise TeacherError(message)
    compatible = raw["compatible"]
    if not isinstance(compatible, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) and v < raw["version"] for v in compatible
    ):
        message = f"{path}: compatible must list earlier versions, got {compatible!r}"
        raise TeacherError(message)
    return Prompts(
        raw["version"],
        tuple(compatible),
        raw["generator"],
        raw["critic"],
        raw["translation"],
        raw["requests"],
        raw["schema"],
        {str(k): str(v) for k, v in raw["tasks"].items()},
        {str(k): str(v) for k, v in raw["domains"].items()},
        {str(k): str(v) for k, v in raw["systems"].items()},
    )


def orthography() -> str:
    """The alphabet and the confusables, read from the normaliser's own tables."""
    folds: dict[str, list[str]] = {}
    for wrong, right in load_rules().homoglyphs.items():
        if right.islower():
            folds.setdefault(right, []).append(wrong)
    never = "; ".join(
        f"{right} never as {' or '.join(sorted(w))}" for right, w in sorted(folds.items())
    )
    return (
        f"- Kabyle Latin alphabet: {' '.join(BASE_LETTERS)} and {' '.join(SPECIFIC_LETTERS)};"
        " p, o, v only in loanwords and names; ţ only where established usage has it.\n"
        f"- Never these look-alikes: {never}.\n"
        f"- Never the ASCII digraphs {', '.join(ASCII_DIGRAPHS)} for the special letters.\n"
        "- Never write in Arabic script or include parenthetical Arabic translations or "
        "glosses; write strictly in the Kabyle Latin alphabet.\n"
        "- Hyphens join clitics as standard Kabyle writes them (ad as-d-iniɣ, yenna-yas)."
    )


def fill(template: str, values: Mapping[str, str]) -> str:
    """`{name}` replaced for the names given; every other brace is left as written."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", value)
    return template


def card_block(card: Card, prompts: Prompts) -> str:
    lines = [
        f"CARD: {card.id}",
        f"TASK: {prompts.tasks[card.task]}",
        f"DOMAIN: {prompts.domains[card.domain]}",
        f"INPUT: {LANGUAGE_NAMES[card.input_language]}",
        f"THINK: {'yes' if card.think else 'no'}",
        f"USER TURNS: {card.user_turns}",
        f"SYSTEM: {prompts.systems[card.system]}",
    ]
    if card.passage is not None and card.passage.other is None:
        lines.append(f"SOURCE ({card.passage.source}):\n{card.passage.text}")
    elif card.passage is not None:
        other = LANGUAGE_NAMES.get(card.passage.other_language or "", "the other language")
        lines.append(f"PAIR:\nKabyle: {card.passage.text}\n{other}: {card.passage.other}")
    return "\n".join(lines)


def generator_prompt(cards: Sequence[Card], prompts: Prompts) -> str:
    per_card = str(CONVERSATIONS_PER_CARD)
    return fill(
        prompts.generator,
        {
            "orthography": orthography(),
            "per_card": per_card,
            "schema": fill(prompts.schema, {"per_card": per_card}),
            "cards": "\n\n".join(card_block(card, prompts) for card in cards),
        },
    )


def batch_name(number: int, kind: str = "batch") -> str:
    return f"{kind}-{number:04d}"


def write_batches(
    cards: Sequence[Card], prompts: Prompts, directory: Path, first: int
) -> list[Path]:
    """Prompt files and manifests for `cards`, `CARDS_PER_BATCH` a batch, numbered from `first`."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for offset in range(0, len(cards), CARDS_PER_BATCH):
        group = cards[offset : offset + CARDS_PER_BATCH]
        name = batch_name(first + offset // CARDS_PER_BATCH)
        (directory / f"{name}.txt").write_text(generator_prompt(group, prompts), encoding="utf-8")
        manifest = {"batch": name, "cards": [card.id for card in group], "version": prompts.version}
        (directory / f"{name}.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        written.append(directory / f"{name}.txt")
    return written


@dataclass(frozen=True, slots=True)
class Reply:
    model: str
    date: str
    lines: tuple[str, ...]


def read_reply(path: Path) -> Reply:
    """A saved reply: the operator's header line, then Gemini's output unedited."""
    text = path.read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    header = HEADER.match(first.strip())
    if header is None:
        message = f"{path}: first line must be '# model: <name as shown> | date: YYYY-MM-DD'"
        raise TeacherError(message)
    return Reply(header["model"].strip(), header["date"], tuple(rest.splitlines()))


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    card: str
    batch: str
    n: int
    turns: tuple[dict[str, str], ...]
    model: str
    date: str
    version: int


@dataclass(slots=True)
class BatchTally:
    batch: str
    requested: int
    returned: int = 0
    rejected: Tally[str] = field(default_factory=Tally)
    ignored: int = 0
    """Blank and code-fence lines, which carry no conversation."""

    @property
    def missing(self) -> int:
        return self.requested - self.returned


TURN_FIELDS: Final = frozenset({"role", "content", "reasoning"})


def read_turns(turns: object) -> list[dict[str, str]] | str:
    """Turns as string fields, or why they are not."""
    if not isinstance(turns, list) or not turns:
        return "turns_not_a_list"
    parsed: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict) or not set(turn) <= TURN_FIELDS:
            return "turn_fields"
        if not all(isinstance(value, str) for value in turn.values()):
            return "turn_values"
        parsed.append({str(k): str(v) for k, v in turn.items()})
    return parsed


def turn_problem(turn: Mapping[str, str], think: bool) -> str | None:
    if turn["role"] == "assistant" and ("reasoning" in turn) != think:
        return "reasoning_for_mode"
    if turn["role"] == "user" and "reasoning" in turn:
        return "reasoning_on_user"
    if not turn.get("content", "").strip():
        return "empty_turn"
    return None


def shape_problem(parsed: Sequence[Mapping[str, str]], card: Card) -> str | None:
    """Whether the turns are the shape the card asked for: its system condition, its number
    of user turns, and reasoning exactly on the assistant turns of a think card."""
    body = list(parsed)
    if card.system == "operator":
        if body[0].get("role") != "system":
            return "operator_system_missing"
        body = body[1:]
    if any(turn.get("role") == "system" for turn in body):
        return "unexpected_system"
    if [turn.get("role") for turn in body] != ["user", "assistant"] * card.user_turns:
        return "wrong_turns"
    return next((p for t in body if (p := turn_problem(t, card.think)) is not None), None)


def turn_shape(turns: object, card: Card) -> tuple[dict[str, str], ...] | str:
    """The turns as the card asked for them, or the reason they are not."""
    parsed = read_turns(turns)
    if isinstance(parsed, str):
        return parsed
    problem = shape_problem(parsed, card)
    return problem if problem is not None else tuple(parsed)


def parse_reply(
    reply: Reply, batch: str, cards: Mapping[str, Card], version: int
) -> tuple[list[Candidate], list[dict[str, object]], BatchTally]:
    """Every line accounted: a candidate, a reject with its reason, or a blank or fence."""
    tally = BatchTally(batch, CONVERSATIONS_PER_CARD * len(cards))
    seen: set[tuple[str, int]] = set()
    candidates: list[Candidate] = []
    rejects: list[dict[str, object]] = []

    def reject(number: int, reason: str, line: str) -> None:
        tally.rejected[reason] += 1
        rejects.append({"batch": batch, "line": number, "reason": reason, "text": line[:300]})

    for number, raw in enumerate(reply.lines, start=2):
        line = raw.strip()
        if not line or line.startswith("```"):
            tally.ignored += 1
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            reject(number, "not_json", line)
            continue
        card_id = row.get("card") if isinstance(row, dict) else None
        n = row.get("n") if isinstance(row, dict) else None
        if not isinstance(card_id, str) or card_id not in cards:
            reject(number, "unknown_card", line)
            continue
        if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= CONVERSATIONS_PER_CARD:
            reject(number, "bad_n", line)
            continue
        if (card_id, n) in seen:
            reject(number, "duplicate", line)
            continue
        shape = turn_shape(row.get("turns"), cards[card_id])
        if isinstance(shape, str):
            reject(number, shape, line)
            continue
        seen.add((card_id, n))
        tally.returned += 1
        candidates.append(
            Candidate(
                f"{card_id}-{batch}-{n:02d}",
                card_id,
                batch,
                n,
                shape,
                reply.model,
                reply.date,
                version,
            )
        )
    return candidates, rejects, tally


def coverage(candidates: Iterable[Candidate]) -> Tally[str]:
    """Conversations parsed per card across every reply."""
    return Tally(candidate.card for candidate in candidates)


def wanting(cards: Sequence[Card], covered: Mapping[str, int], outstanding: set[str]) -> list[Card]:
    """Cards still short of their conversations and not already in an unanswered batch."""
    return [
        card
        for card in cards
        if covered.get(card.id, 0) < CONVERSATIONS_PER_CARD and card.id not in outstanding
    ]


@dataclass(frozen=True, slots=True)
class Verdict:
    candidate: str
    flags: dict[str, bool]
    naturalness: int
    note: str


def parse_critic(lines: Sequence[str], expected: set[str]) -> tuple[dict[str, Verdict], Tally[str]]:
    """Critic verdicts for the candidates asked about; anything else is counted as rejected."""
    verdicts: dict[str, Verdict] = {}
    rejected: Tally[str] = Tally()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rejected["not_json"] += 1
            continue
        name = row.get("candidate") if isinstance(row, dict) else None
        flags = row.get("flags") if isinstance(row, dict) else None
        score = row.get("naturalness") if isinstance(row, dict) else None
        if not isinstance(name, str) or name not in expected:
            rejected["unknown_candidate"] += 1
            continue
        if not isinstance(flags, dict) or set(flags) != set(CRITIC_FLAGS):
            rejected["flags"] += 1
            continue
        if not all(isinstance(value, bool) for value in flags.values()):
            rejected["flags"] += 1
            continue
        if not isinstance(score, int) or isinstance(score, bool) or score not in NATURALNESS:
            rejected["naturalness"] += 1
            continue
        if name in verdicts:
            rejected["duplicate"] += 1
            continue
        note = row.get("note")
        verdicts[name] = Verdict(
            name, {str(k): bool(v) for k, v in flags.items()}, score, str(note or "")
        )
    return verdicts, rejected


def write_critic_batches(
    candidates: Sequence[Candidate], prompts: Prompts, directory: Path, first: int
) -> list[Path]:
    """Critic prompts for `candidates`, `CANDIDATES_PER_CRITIC` a batch, numbered from `first`."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for offset in range(0, len(candidates), CANDIDATES_PER_CRITIC):
        group = candidates[offset : offset + CANDIDATES_PER_CRITIC]
        name = batch_name(first + offset // CANDIDATES_PER_CRITIC, "critic")
        (directory / f"{name}.txt").write_text(critic_prompt(group, prompts), encoding="utf-8")
        manifest = {"batch": name, "candidates": [c.id for c in group], "version": prompts.version}
        (directory / f"{name}.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        written.append(directory / f"{name}.txt")
    return written


def critic_prompt(candidates: Sequence[Candidate], prompts: Prompts) -> str:
    rendered = [
        json.dumps({"candidate": c.id, "turns": list(c.turns)}, ensure_ascii=False)
        for c in candidates
    ]
    return fill(prompts.critic, {"candidates": "\n".join(rendered)})


def translation_prompt(sentences: Sequence[str], first: int, prompts: Prompts) -> str:
    """Sentences numbered from `first`, so every reply line names its sentence."""
    numbered = "\n".join(f"{first + i}. {text}" for i, text in enumerate(sentences))
    return fill(prompts.translation, {"orthography": orthography(), "sentences": numbered})


def parse_translations(lines: Sequence[str], expected: range) -> tuple[dict[int, str], Tally[str]]:
    """Translations by sentence number; a line for a number not asked, twice, or empty is
    counted as rejected, and a missing number is simply absent."""
    found: dict[int, str] = {}
    rejected: Tally[str] = Tally()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rejected["not_json"] += 1
            continue
        number = row.get("i") if isinstance(row, dict) else None
        text = row.get("kab") if isinstance(row, dict) else None
        if not isinstance(number, int) or isinstance(number, bool) or number not in expected:
            rejected["unknown_sentence"] += 1
        elif number in found:
            rejected["duplicate"] += 1
        elif not isinstance(text, str) or not text.strip():
            rejected["empty"] += 1
        else:
            found[number] = text.strip()
    return found, rejected
