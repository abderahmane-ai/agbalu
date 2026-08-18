"""Audit the orthography of a FLORES+ language split, and repair it.

The comparison this is built to support is Oktem et al. 2025
(aclanthology.org/2025.wmt-1.82), who corrected the Tamazight portions of FLORES+
and OLDI Seed: **36% of FLORES+ sentences changed, with 19% mean token divergence
among the changed ones**. Those figures are reported per corrected *sentence*, so
`token_divergence` here is computed the same way — over changed sentences only —
and is directly comparable.

Only the rule-verifiable layer is applied: homoglyphs, invisibles, Unicode form,
whitespace and punctuation. Mistranslation and loanword overuse, which are most of
what Oktem's linguists fixed, need human judgement and are out of scope.

`protected` counts tokens the normaliser refused to rewrite because they are
spelled in another language's orthography — `Chişinău`, not a corrupted `Chiṣinău`.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from agbalu.bench.flores import Sentence
from agbalu.normalise import Normaliser
from agbalu.normalise.models import ChangeKind


@dataclass(frozen=True, slots=True)
class SentenceDiff:
    """One sentence the audit would change."""

    id: int
    split: str
    original: str
    corrected: str
    token_divergence: float
    kinds: tuple[ChangeKind, ...]
    codepoints: tuple[str, ...]


@dataclass
class AuditReport:
    """What the audit found across a whole language."""

    language: str
    sentences: int = 0
    changed: int = 0
    revisions: dict[str, int] = field(default_factory=dict)
    by_kind: Counter[str] = field(default_factory=Counter)
    by_codepoint: Counter[str] = field(default_factory=Counter)
    by_split: Counter[str] = field(default_factory=Counter)
    protected: int = 0
    diffs: list[SentenceDiff] = field(default_factory=list)

    @property
    def changed_rate(self) -> float:
        return self.changed / self.sentences if self.sentences else 0.0

    @property
    def mean_token_divergence(self) -> float:
        """Mean over changed sentences only, matching Oktem et al.'s definition."""
        if not self.diffs:
            return 0.0
        return sum(d.token_divergence for d in self.diffs) / len(self.diffs)


def token_divergence(original: str, corrected: str) -> float:
    """Share of whitespace tokens that differ, aligned by position.

    Length differences count as divergent, so a correction that splits or joins a
    token is not silently scored as agreement.
    """
    before = original.split()
    after = corrected.split()
    if not before and not after:
        return 0.0
    longest = max(len(before), len(after))
    same = sum(1 for a, b in zip(before, after, strict=False) if a == b)
    return (longest - same) / longest


def describe(codepoint: str) -> str:
    """`ε U+03B5 GREEK SMALL LETTER EPSILON`, for a report a linguist can check."""
    try:
        name = unicodedata.name(codepoint)
    except ValueError:
        name = "UNNAMED"
    return f"{codepoint} U+{ord(codepoint):04X} {name}"


def audit(
    sentences: list[Sentence], language: str, normaliser: Normaliser | None = None
) -> AuditReport:
    engine = normaliser if normaliser is not None else Normaliser()
    report = AuditReport(language=language, sentences=len(sentences))

    for sentence in sentences:
        report.revisions[sentence.last_updated] = report.revisions.get(sentence.last_updated, 0) + 1
        result = engine.analyse(sentence.text)
        report.protected += sum(1 for f in result.flags if f.kind == "foreign-proper-noun")
        if not result.changed:
            continue
        corrected = result.text
        report.changed += 1
        report.by_split[sentence.split] += 1
        kinds = tuple(sorted({c.kind for c in result.changes}))
        for kind in kinds:
            report.by_kind[kind] += 1

        codepoints: list[str] = []
        for change in result.changes:
            if change.kind in ("homoglyph", "invisible-removed") and len(change.before) == 1:
                report.by_codepoint[change.before] += 1
                codepoints.append(change.before)

        report.diffs.append(
            SentenceDiff(
                id=sentence.id,
                split=sentence.split,
                original=sentence.text,
                corrected=corrected,
                token_divergence=token_divergence(sentence.text, corrected),
                kinds=kinds,
                codepoints=tuple(sorted(set(codepoints))),
            )
        )

    return report
