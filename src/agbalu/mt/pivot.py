"""Select the pivot job: Kabyle sentences whose English or French side can carry a
translation into a third language.

Kabyle has no bitext with Arabic, Spanish or German worth training on, but it has
282,035 English-paired and 199,736 French-paired sentences whose Kabyle side is human.
Translating the *pivot* side into a third language and keeping the human Kabyle gives
synthetic-source, authentic-target pairs — the back-translation arrangement, which is
reported superior to synthesising the target side and is not capped by the teacher.

32,153 of those sentences carry both an English and a French side. Those get two
independent teachers, and `modal_app.synth` keeps their output only where the two agree,
which is what keeps synthetic noise out rather than hoping it is small.

The pivot side is language-identified before it is used. The corpus's own
`foreign-language-mismatch` detector is a hard defect and already drops what it catches,
but it misses Latin-script neighbours: a `fra` field reading *"Tom dijo que nunca comió
suchi"* survives it. Generating Arabic from Spanish and labelling it a French pivot is
exactly the silent corruption this project exists to avoid.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, TypedDict

from agbalu.mt.data import PARALLEL_DIR, is_mined
from agbalu.parallel.quality import HARD_DEFECTS

OUTPUT: Final = Path("data/processed/mt/pivot.jsonl")

PIVOT_CODE: Final[dict[str, str]] = {"eng": "eng_Latn", "fra": "fra_Latn"}
"""The label `nllb-lid218e` must return for a side to be usable as that language."""

LID_BATCH: Final = 10_000

BOTH_TEACHERS: Final = 2

PLACEHOLDER: Final = re.compile(r"%[sdfx@]|%\d|\{\d*\}|\$\{|<[a-zA-Z/][^>]*>|&[a-z]+;")
"""printf, brace, shell and markup placeholders. 16.8% of the corpus comes from Pontoon,
Weblate and translatewiki, whose rows are interface strings rather than prose."""

MIN_CHARS: Final = 12


class Identifier(Protocol):
    """`agbalu.bench.lid.Identifier`, structurally, so the selection is testable without
    a 1.1 GB fastText model on disk."""

    def identify(self, texts: Sequence[str]) -> list[str]: ...


class PivotStats(TypedDict):
    read: int
    mined_excluded: int
    hard_defective: int
    wrong_language: dict[str, int]
    unusable: int
    kept: int
    two_teacher: int
    eng_only: int
    fra_only: int


@dataclass(frozen=True, slots=True)
class PivotRecord:
    """One Kabyle sentence and the pivot sides that survived identification."""

    kab: str
    eng: str | None = None
    fra: str | None = None

    @property
    def teachers(self) -> int:
        return int(self.eng is not None) + int(self.fra is not None)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {"kab": self.kab, "teachers": self.teachers}
        if self.eng is not None:
            row["eng"] = self.eng
        if self.fra is not None:
            row["fra"] = self.fra
        return row


@dataclass(slots=True)
class _Counts:
    read: int = 0
    mined: int = 0
    defective: int = 0
    wrong_language: dict[str, int] = field(default_factory=dict)


def usable(record: PivotRecord) -> bool:
    """Whether a record is prose worth translating.

    The corpus's defect labels do not catch localisation misalignment: the pair
    `("!! tuccḍa: %s", "GPL-2.0-or-later")` carries no defect and is not a translation of
    anything. Two teachers would agree on it perfectly, so the agreement filter cannot
    reject it either — agreement measures teacher error, not alignment error. Measured at
    8.6% of the pivot pool.
    """
    sides = [record.kab, *(s for s in (record.eng, record.fra) if s is not None)]
    return not any(
        PLACEHOLDER.search(side) or " " not in side.strip() or len(side.strip()) < MIN_CHARS
        for side in sides
    )


def _key(kab: str) -> str:
    return unicodedata.normalize("NFC", kab).casefold().strip()


def _read_side(path: Path, language: str, counts: _Counts) -> dict[str, str]:
    """Every non-mined, defect-free pair in one file, keyed on its Kabyle side."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            counts.read += 1
            if is_mined(row["source"]):
                counts.mined += 1
                continue
            if any(defect in HARD_DEFECTS for defect in row["defects"]):
                counts.defective += 1
                continue
            out.setdefault(_key(row["kab"]), row[language])
    return out


def filter_language(
    texts: dict[str, str], language: str, identifier: Identifier, *, batch: int = LID_BATCH
) -> tuple[dict[str, str], int]:
    """Drop entries whose value is not the language it claims to be.

    Returns the survivors and how many were dropped. Identification is batched because a
    per-call round trip over 280,000 strings is the slowest part of this module.
    """
    expected = PIVOT_CODE[language]
    keys = list(texts)
    kept: dict[str, str] = {}
    dropped = 0
    for start in range(0, len(keys), batch):
        window = keys[start : start + batch]
        for key, label in zip(window, identifier.identify([texts[k] for k in window]), strict=True):
            if label == expected:
                kept[key] = texts[key]
            else:
                dropped += 1
    return kept, dropped


def select(
    identifier: Identifier, parallel_dir: Path = PARALLEL_DIR
) -> tuple[list[PivotRecord], PivotStats]:
    """Every Kabyle sentence with at least one identified pivot side."""
    counts = _Counts()
    eng = _read_side(parallel_dir / "agbalu-parallel-v1.kab-eng.jsonl", "eng", counts)
    fra = _read_side(parallel_dir / "agbalu-parallel-v1.kab-fra.jsonl", "fra", counts)

    eng, counts.wrong_language["eng"] = filter_language(eng, "eng", identifier)
    fra, counts.wrong_language["fra"] = filter_language(fra, "fra", identifier)

    candidates = [
        PivotRecord(kab=key, eng=eng.get(key), fra=fra.get(key))
        for key in sorted(set(eng) | set(fra))
    ]
    records = [record for record in candidates if usable(record)]
    two = sum(1 for r in records if r.teachers == BOTH_TEACHERS)
    stats: PivotStats = {
        "read": counts.read,
        "mined_excluded": counts.mined,
        "hard_defective": counts.defective,
        "wrong_language": dict(counts.wrong_language),
        "unusable": len(candidates) - len(records),
        "kept": len(records),
        "two_teacher": two,
        "eng_only": sum(1 for r in records if r.eng is not None and r.fra is None),
        "fra_only": sum(1 for r in records if r.fra is not None and r.eng is None),
    }
    return records, stats


@dataclass(frozen=True, slots=True)
class SynthPair:
    """One synthesised pair: machine-made source, authentic Kabyle target."""

    source: str
    kab: str
    language: str
    teachers: int
    agreement: float | None
    """chrF between the two teachers, or None where only one was available."""

    def as_row(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.kab,
            "language": self.language,
            "teachers": self.teachers,
            "agreement": None if self.agreement is None else round(self.agreement, 2),
        }


class CombineStats(TypedDict):
    language: str
    two_teacher: int
    agreed: int
    disagreed: int
    single_teacher: int
    kept: int
    threshold: float
    mean_agreement: float | None


class Scorer(Protocol):
    def __call__(self, first: str, second: str) -> float: ...


@dataclass(frozen=True, slots=True)
class Policy:
    """What to keep, and how sure the two teachers must be."""

    language: str
    threshold: float = 50.0
    keep_single: bool = True


def combine(
    records: Sequence[PivotRecord],
    english: Mapping[int, str],
    french: Mapping[int, str],
    policy: Policy,
    scorer: Scorer,
) -> tuple[list[SynthPair], CombineStats]:
    """Pick one translation per record, dropping two-teacher disagreements.

    Where both teachers produced output, the French one is kept when they agree — French
    is the closer contact language for Kabyle and its pivot side is the larger of the two
    after identification. A disagreement drops the record rather than demoting it: it means
    one teacher is wrong and nothing here says which.
    """
    pairs: list[SynthPair] = []
    agreed = disagreed = single = 0
    scores: list[float] = []

    for index, record in enumerate(records):
        left, right = english.get(index), french.get(index)
        if left is not None and right is not None:
            score = scorer(left, right)
            if score < policy.threshold:
                disagreed += 1
                continue
            agreed += 1
            scores.append(score)
            pairs.append(SynthPair(right, record.kab, policy.language, BOTH_TEACHERS, score))
        elif policy.keep_single and (chosen := right if right is not None else left) is not None:
            single += 1
            pairs.append(SynthPair(chosen, record.kab, policy.language, 1, None))

    stats: CombineStats = {
        "language": policy.language,
        "two_teacher": agreed + disagreed,
        "agreed": agreed,
        "disagreed": disagreed,
        "single_teacher": single,
        "kept": len(pairs),
        "threshold": policy.threshold,
        "mean_agreement": round(sum(scores) / len(scores), 2) if scores else None,
    }
    return pairs, stats


def write_pairs(pairs: Iterable[SynthPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.as_row(), ensure_ascii=False) + "\n")


def write(records: Iterable[PivotRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_row(), ensure_ascii=False) + "\n")


def read(path: Path) -> list[PivotRecord]:
    records: list[PivotRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(PivotRecord(kab=row["kab"], eng=row.get("eng"), fra=row.get("fra")))
    return records


def build(
    identifier: Identifier, parallel_dir: Path = PARALLEL_DIR, output: Path = OUTPUT
) -> PivotStats:
    records, stats = select(identifier, parallel_dir)
    write(records, output)
    beside = output.with_suffix(".stats.json")
    beside.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats
