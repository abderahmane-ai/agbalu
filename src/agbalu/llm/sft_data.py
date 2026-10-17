"""Teacher candidates into the accepted SFT set: gates, flags, review and quotas (task 11.22).

Every candidate ends in exactly one place, with its reason:

- **rejected** by a hard gate: a template marker, a repetition loop, an evaluation overlap,
  an exact or near duplicate of an earlier candidate;
- **flagged** by a soft signal and set aside unless review accepts it: the critic's flags,
  a language identifier that does not read the Kabyle turns as Kabyle, a sibling variety,
  an orthography edit rate above the threshold, fluency below the held-out floor;
- **accepted**: clean, inside its task quota, and passed by review where review is required.

Identity and instruction-hierarchy candidates are accepted only after review, all of them.
A stratified sample of the rest is reviewed too, and a category whose reviewed rejection
rate passes `CATEGORY_FAILURE` stops the build: it is regenerated, not shipped.

The model-based signals arrive as a mapping from candidate id to values, computed where
each model lives; this module decides, and decides the same way every run.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter as Tally
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, Literal

from agbalu.llm.chat import MARKERS, SYSTEM_POLICY
from agbalu.llm.dedupe import NearDuplicates, exact_key
from agbalu.llm.seeds import Card
from agbalu.llm.teacher import Candidate, Verdict

Status = Literal["rejected", "flagged", "accepted", "surplus", "pending_review"]

TASK_TARGETS: Final[dict[str, int]] = {
    "native_kabyle": 450,
    "non_kabyle_input": 150,
    "translation": 100,
    "reasoning": 100,
    "code": 60,
    "hierarchy": 50,
    "identity": 20,
    "grounded": 70,
}
"""`docs/llm_post_training_design.md` §4.4's shares of 1,000 accepted conversations, cut from
5,000 on 2026-10-16."""

ALWAYS_REVIEWED: Final = frozenset({"identity", "hierarchy"})
REVIEW_SAMPLE: Final = 100
CATEGORY_FAILURE: Final = 0.20
"""A task whose reviewed candidates are rejected more often than this is regenerated."""

CATEGORY_MIN_REVIEWS: Final = 20
"""Fewer reviews than this in a task decide nothing about the task: one rejection out of one
is not a rate."""

THINK_SHARE: Final = 0.12
THINK_TOLERANCE: Final = 0.04
"""Operator decision 2026-10-17: the teacher set ends without its reasoning, code and
grounded cards, so the think share is what the closed set holds, projected at 13.5% of the
teacher conversations after the leak filters, not §6's 20%."""

REPEAT_SPAN: Final = 4
REPEAT_SHARE: Final = 0.25
"""A text whose repeated 4-word spans are more than a quarter of its spans is a loop."""

LANGUAGE_EXEMPT: Final = frozenset({"translation"})
"""Tasks whose answers may legitimately be in another language; their language is checked
by review, not by an identifier."""

SIGNALS: Final = ("glotlid_kabyle", "dihya_sibling", "fluency_bpc")

ARABIC_SCRIPT: Final = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻾]")
"""Arabic, its supplements and presentation forms. The pilot's first three replies glossed
Kabyle words in Arabic script mid-sentence (`يوض (yewweḍ)`) in 22 of 98 conversations,
2026-10-16, and GlotLID reads such text as Kabyle."""


class DataError(Exception):
    """The candidates cannot become an accepted set under the fixed rules."""


@dataclass(frozen=True, slots=True)
class Review:
    candidate: str
    accept: bool
    note: str


@dataclass(frozen=True, slots=True)
class Thresholds:
    fluency_bpc: float
    """Bits per character under the CPT model at the 95th percentile of held-out Kabyle."""


def repeated_share(text: str, span: int = REPEAT_SPAN) -> float:
    words = text.split()
    spans = [tuple(words[i : i + span]) for i in range(len(words) - span + 1)]
    if not spans:
        return 0.0
    return 1.0 - len(set(spans)) / len(spans)


def assistant_texts(candidate: Candidate) -> list[str]:
    texts: list[str] = []
    for turn in candidate.turns:
        if turn.get("role") == "assistant":
            texts.append(turn.get("content", ""))
            if "reasoning" in turn:
                texts.append(turn["reasoning"])
    return texts


def all_texts(candidate: Candidate) -> list[str]:
    return [value for turn in candidate.turns for key, value in turn.items() if key != "role"]


def hard_gate(candidate: Candidate, contaminated: bool, duplicate: str | None) -> str | None:
    """The reason a candidate cannot be used at all, or None."""
    if any(marker in text for text in all_texts(candidate) for marker in MARKERS):
        return "template_marker"
    if any(repeated_share(text) > REPEAT_SHARE for text in assistant_texts(candidate)):
        return "repetition"
    if contaminated:
        return "contaminated"
    return duplicate


UNSEEN_TEXT: Final = re.compile(r"\b(?:aḍris|uḍris|iḍrisen|yiḍrisen|tektabt|SOURCE|PAIR)\b")
"""Words for "the text" or "the source". In the v3 set, 81 of 819 answers cited the card's
SOURCE, which the user never sees ("ilmend n uḍris"), measured 2026-10-17."""

GREETING_TALK: Final = re.compile(r"\b(?:azul|asellem)\b", re.IGNORECASE)
"""A reasoning trace that discusses greeting is discussing the teacher's instructions; 37 of
289 v3 traces did."""


def leaks(candidate: Candidate) -> list[str]:
    """What the teacher's instructions left in the conversation: an answer citing a text no
    user turn mentions, or a reasoning trace about greetings."""
    found: list[str] = []
    asked = " ".join(t["content"] for t in candidate.turns if t["role"] != "assistant")
    if not UNSEEN_TEXT.search(asked) and any(
        UNSEEN_TEXT.search(text) for text in assistant_texts(candidate)
    ):
        found.append("cites_unseen_text")
    reasoning = [t["reasoning"] for t in candidate.turns if "reasoning" in t]
    if any(GREETING_TALK.search(text) for text in reasoning):
        found.append("instruction_leak")
    return found


def soft_flags(
    candidate: Candidate,
    card: Card,
    verdict: Verdict | None,
    signals: Mapping[str, object],
    thresholds: Thresholds,
) -> list[str]:
    """Every soft signal that sends a candidate to review instead of straight in."""
    flags: list[str] = []
    if verdict is None:
        flags.append("no_critic_verdict")
    else:
        flags.extend(f"critic_{name}" for name, raised in sorted(verdict.flags.items()) if raised)
    missing = [name for name in SIGNALS if name not in signals]
    if missing:
        message = f"{candidate.id}: signals {missing} were never computed"
        raise DataError(message)
    if card.task not in LANGUAGE_EXEMPT:
        if signals["glotlid_kabyle"] is not True:
            flags.append("not_read_as_kabyle")
        if signals["dihya_sibling"] is not None:
            flags.append(f"sibling_{signals['dihya_sibling']}")
        if any(ARABIC_SCRIPT.search(text) for text in assistant_texts(candidate)):
            flags.append("arabic_script")
    flags.extend(leaks(candidate))
    fluency = signals["fluency_bpc"]
    if isinstance(fluency, int | float) and fluency > thresholds.fluency_bpc:
        flags.append("disfluent")
    return flags


def stable_rank(candidate_id: str) -> str:
    return hashlib.blake2b(candidate_id.encode(), digest_size=8).hexdigest()


@dataclass(slots=True)
class Ledger:
    """Where every candidate ended and why."""

    status: dict[str, Status] = field(default_factory=dict)
    reasons: dict[str, list[str]] = field(default_factory=dict)

    def put(self, candidate: str, status: Status, reasons: Sequence[str] = ()) -> None:
        self.status[candidate] = status
        self.reasons[candidate] = list(reasons)

    def counts(self) -> dict[str, int]:
        return dict(sorted(Tally(self.status.values()).items()))


def review_sample(
    eligible: Sequence[Candidate], cards: Mapping[str, Card], size: int = REVIEW_SAMPLE
) -> list[str]:
    """Candidate ids to review: every identity and hierarchy one, and a sample of the rest
    stratified by task, chosen by a stable hash so it does not move between runs."""
    required = [c.id for c in eligible if cards[c.card].task in ALWAYS_REVIEWED]
    others = [c for c in eligible if cards[c.card].task not in ALWAYS_REVIEWED]
    by_task: dict[str, list[Candidate]] = {}
    for candidate in others:
        by_task.setdefault(cards[candidate.card].task, []).append(candidate)
    chosen: list[str] = []
    for task, members in sorted(by_task.items()):
        share = round(size * len(members) / max(1, len(others)))
        ordered = sorted(members, key=lambda c: stable_rank(c.id))
        chosen.extend(c.id for c in ordered[:share])
        del task
    return required + chosen


def category_failures(
    reviews: Mapping[str, Review], candidates: Mapping[str, Candidate], cards: Mapping[str, Card]
) -> dict[str, float]:
    """Reviewed rejection rate per task, for the tasks above `CATEGORY_FAILURE` on at least
    `CATEGORY_MIN_REVIEWS` reviews."""
    seen: Tally[str] = Tally()
    rejected: Tally[str] = Tally()
    for name, review in reviews.items():
        task = cards[candidates[name].card].task
        seen[task] += 1
        rejected[task] += not review.accept
    return {
        task: round(rejected[task] / seen[task], 4)
        for task in sorted(seen)
        if seen[task] >= CATEGORY_MIN_REVIEWS and rejected[task] / seen[task] > CATEGORY_FAILURE
    }


def select(
    candidates: Sequence[Candidate],
    cards: Mapping[str, Card],
    verdicts: Mapping[str, Verdict],
    reviews: Mapping[str, Review],
    ledger: Ledger,
) -> list[Candidate]:
    """Within each task quota, think-mode candidates first (the set runs short of its fifth),
    then the most natural; a candidate that must be reviewed and has not been is held back as
    pending. Validation candidates are outside the quotas, which count training only."""
    by_task: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_task.setdefault(cards[candidate.card].task, []).append(candidate)
    accepted: list[Candidate] = []
    for task, members in sorted(by_task.items()):
        ordered = sorted(
            members,
            key=lambda c: (
                not cards[c.card].think,
                -(verdicts[c.id].naturalness if c.id in verdicts else 0),
                stable_rank(c.id),
            ),
        )
        for candidate in ordered:
            needs_review = task in ALWAYS_REVIEWED or candidate.id in reviews
            if needs_review and candidate.id not in reviews:
                ledger.put(candidate.id, "pending_review")
                continue
            if candidate.id in reviews and not reviews[candidate.id].accept:
                ledger.put(candidate.id, "rejected", ["review", reviews[candidate.id].note])
                continue
            training = cards[candidate.card].split == "train"
            filled = sum(
                cards[c.card].task == task and cards[c.card].split == "train" for c in accepted
            )
            if training and filled >= TASK_TARGETS[task]:
                ledger.put(candidate.id, "surplus")
                continue
            ledger.put(candidate.id, "accepted")
            accepted.append(candidate)
    return accepted


def with_policy(candidate: Candidate, card: Card) -> list[dict[str, str]]:
    """The turns as trained: the policy system prompt first where the card's condition is
    `policy`, the teacher's own operator prompt kept, nothing added for `none`."""
    turns = [dict(turn) for turn in candidate.turns]
    if card.system == "policy":
        turns.insert(0, {"role": "system", "content": SYSTEM_POLICY})
    return turns


def accepted_row(candidate: Candidate, card: Card, normaliser_version: str) -> dict[str, object]:
    split = "train" if card.split == "train" else "validation"
    return {
        "id": candidate.id,
        "split": split,
        "card": card.id,
        "card_split": card.split,
        "task": card.task,
        "domain": card.domain,
        "think": card.think,
        "system": card.system,
        "turns": with_policy(candidate, card),
        "teacher": {"model": candidate.model, "date": candidate.date, "batch": candidate.batch},
        "prompt_version": candidate.version,
        "normaliser": normaliser_version,
        "source": card.passage.source if card.passage else None,
        "licence": card.passage.licence if card.passage else None,
    }


def check_think_share(accepted: Sequence[Candidate], cards: Mapping[str, Card]) -> float:
    if not accepted:
        message = "no candidate was accepted"
        raise DataError(message)
    share = sum(cards[c.card].think for c in accepted) / len(accepted)
    if abs(share - THINK_SHARE) > THINK_TOLERANCE:
        message = f"think-mode is {share:.3f} of the accepted set, not {THINK_SHARE}"
        raise DataError(message)
    return share


def read_reviews(path: Path) -> dict[str, Review]:
    """The operator's verdicts, one JSON object a line; a malformed line stops the build."""
    if not path.is_file():
        return {}
    reviews: dict[str, Review] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            name, verdict = row.get("candidate"), row.get("verdict")
            if not isinstance(name, str) or verdict not in {"accept", "reject"}:
                message = f"{path}:{number}: needs a candidate id and verdict accept or reject"
                raise DataError(message)
            if name in reviews:
                message = f"{path}:{number}: {name} is reviewed twice"
                raise DataError(message)
            reviews[name] = Review(name, verdict == "accept", str(row.get("note") or ""))
    return reviews


def review_sheet(
    names: Iterable[str], candidates: Mapping[str, Candidate], cards: Mapping[str, Card]
) -> list[dict[str, object]]:
    """What the operator reads to decide: the card's ask and the conversation."""
    return [
        {
            "candidate": name,
            "task": cards[candidates[name].card].task,
            "domain": cards[candidates[name].card].domain,
            "think": cards[candidates[name].card].think,
            "turns": list(candidates[name].turns),
            "verdict": None,
            "note": "",
        }
        for name in names
    ]


@dataclass(frozen=True, slots=True)
class Evidence:
    """What the decisions read besides the candidates and their cards."""

    verdicts: Mapping[str, Verdict]
    signals: Mapping[str, Mapping[str, object]]
    reviews: Mapping[str, Review]
    thresholds: Thresholds
    prompt_versions: frozenset[int]
    """The teacher prompt versions that count (`Prompts.counted`). A candidate from any other
    is rejected, so the set is written under one recipe."""


def normalised(candidate: Candidate, card: Card, normalise: Callable[[str], str]) -> Candidate:
    """Assistant turns always, user turns when the card asks in Kabyle, through the
    versioned normaliser; system turns and other-language prompts as written."""
    turns: list[dict[str, str]] = []
    for turn in candidate.turns:
        kabyle = turn["role"] == "assistant" or (
            turn["role"] == "user" and card.input_language == "kab"
        )
        turns.append({k: normalise(v) if kabyle and k != "role" else v for k, v in turn.items()})
    return replace(candidate, turns=tuple(turns))


@dataclass(frozen=True, slots=True)
class Text:
    """How a candidate's text is checked and cleaned: the evaluation-overlap test and the
    normaliser."""

    contaminated: Callable[[str], bool]
    normalise: Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Screened:
    clean: list[Candidate]
    normalised: dict[str, Candidate]
    ledger: Ledger


def assemble(
    candidates: Sequence[Candidate], cards: Mapping[str, Card], evidence: Evidence, text: Text
) -> tuple[list[Candidate], Ledger]:
    """The accepted candidates, and the ledger that accounts for every other one."""
    screened = screen(candidates, cards, evidence, text)
    ledger = screened.ledger
    failures = category_failures(evidence.reviews, screened.normalised, cards)
    if failures:
        message = f"review rejected too many candidates in {failures}; regenerate these tasks"
        raise DataError(message)
    accepted = select(screened.clean, cards, evidence.verdicts, evidence.reviews, ledger)
    if len(ledger.status) != len(candidates):
        message = f"{len(ledger.status)} candidates accounted for of {len(candidates)}"
        raise DataError(message)
    check_think_share(accepted, cards)
    return accepted, ledger


def screen(
    candidates: Sequence[Candidate], cards: Mapping[str, Card], evidence: Evidence, text: Text
) -> Screened:
    """Normalise, then the hard gates and the soft flags; quotas are not applied here."""
    contaminated, normalise = text.contaminated, text.normalise
    ledger = Ledger()
    exact: set[bytes] = set()
    near = NearDuplicates()
    clean: list[Candidate] = []
    by_id: dict[str, Candidate] = {}
    for raw in sorted(candidates, key=lambda c: c.id):
        card = cards[raw.card]
        candidate = normalised(raw, card, normalise)
        by_id[candidate.id] = candidate
        if candidate.version not in evidence.prompt_versions:
            ledger.put(candidate.id, "rejected", [f"prompt_version_{candidate.version}"])
            continue
        if card.split == "release":
            ledger.put(candidate.id, "rejected", ["release_card"])
            continue
        answer = "\n".join(assistant_texts(candidate))
        key = exact_key(answer)
        duplicate = (
            "duplicate"
            if key in exact
            else "near_duplicate"
            if near.seen(candidate.id, answer)
            else None
        )
        exact.add(key)
        overlap = any(contaminated(text) for text in all_texts(candidate))
        reason = hard_gate(candidate, overlap, duplicate)
        if reason is not None:
            ledger.put(candidate.id, "rejected", [reason])
            continue
        flags = soft_flags(
            candidate,
            card,
            evidence.verdicts.get(candidate.id),
            evidence.signals.get(candidate.id, {}),
            evidence.thresholds,
        )
        review = evidence.reviews.get(candidate.id)
        if flags and (review is None or not review.accept):
            ledger.put(candidate.id, "flagged", flags)
            continue
        clean.append(candidate)
    return Screened(clean, by_id, ledger)
