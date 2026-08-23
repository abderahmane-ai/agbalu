"""Language identification against Berber siblings — AƔBALU-Bench task 7.4.

The deliverable is the confusion matrix, not the macro-F1: the question worth
answering is which sibling Kabyle is mistaken for and in which direction, because
that is what silently contaminates every dataset labelled "Tamazight"
(docs/benchmark.md §3.3, CLAUDE.md §1).

The held-out set is drawn round-robin across every source a language has, so no
class is one source wearing a language's name. Sampling per language instead of per
source is how the first set came to be four-fifths GlotCC — whose labels are GlotLID
output, which made the ceiling agreement with the system being beaten.

Sources, readers and filters live in `lid_sources`, which the training-set builder
reads from too. One table, because two descriptions of the same corpus drift and the
drift is invisible until both are measured.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from agbalu.bench.lid_sources import (
    BERBER_LANGUAGES,
    KABYLE,
    MIN_WORDS,
    NOT_AMAZIGH,
    Source,
    SourceError,
)
from agbalu.bench.lid_sources import SOURCES as ALL_SOURCES
from agbalu.bench.lid_sources import lines as source_lines

EVAL_LANGUAGES: Final[tuple[str, ...]] = BERBER_LANGUAGES
"""The label set public identifiers are scored over. `NOT_AMAZIGH` is deliberately
absent: it is a decision the trained classifier makes, not a language GlotLID or
`lid218e` can be asked about."""

HELD_OUT_SOURCES: Final[frozenset[str]] = frozenset({"hf.flores-plus-taq"})
"""Sources that exist only to be evaluated on.

FLORES+ Tamasheq is the whole of it. The sibling registry records it as never
ingested, and it was 78.7% of the Tamasheq training rows until `nllb-seed.taq`
replaced it — training on the corpus the evaluation is drawn from is not a
decontamination question, it is the same 2,009 sentences and one translator.
"""


class LidError(Exception):
    """The evaluation set could not be built, or a system returned an unusable label."""


@dataclass(frozen=True, slots=True)
class Example:
    """One evaluation line, the language it is held to be, and where it came from."""

    text: str
    language: str
    source: str
    domain: str

    def to_dict(self) -> dict[str, str]:
        return {
            "text": self.text,
            "language": self.language,
            "source": self.source,
            "domain": self.domain,
        }


@runtime_checkable
class Identifier(Protocol):
    """A language identifier, scored as a closed-set classifier over `EVAL_LANGUAGES`."""

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def labels(self) -> frozenset[str]: ...

    def identify(self, texts: Sequence[str]) -> list[str]: ...


FLORES_TAMASHEQ: Final = Source(
    id="hf.flores-plus-taq",
    language="taq_Latn",
    domain="encyclopaedia",
    trust="gold",
    licence="cc-by-sa-4.0",
    reader="flores",
    location="hf.flores-plus-taq",
    argument="taq_Latn",
)


def evaluation_sources() -> tuple[Source, ...]:
    """Every source the held-out set may draw from, including the held-out-only ones."""
    return (*ALL_SOURCES, FLORES_TAMASHEQ)


def build_pools(raw: Path) -> dict[str, list[Example]]:
    """Every usable line, per language, tagged with the source and domain it came from."""
    pools: dict[str, list[Example]] = {language: [] for language in (*EVAL_LANGUAGES, NOT_AMAZIGH)}
    for source in evaluation_sources():
        if source.language not in pools:
            msg = f"{source.id}: label {source.language!r} is outside the label set"
            raise LidError(msg)
        pools[source.language].extend(
            Example(text=line, language=source.language, source=source.id, domain=source.domain)
            for line in source_lines(source, raw)
        )
    empty = sorted(language for language, pool in pools.items() if not pool)
    if empty:
        msg = f"no usable lines for {', '.join(empty)}"
        raise LidError(msg)
    return pools


def _round_robin(groups: Sequence[Sequence[Example]], total: int) -> list[Example]:
    """Take from each group in turn until `total` is reached or every group is spent.

    Round-robin rather than proportional: a language whose largest source is 71,114
    lines and whose smallest is 177 would otherwise be measured almost entirely on
    the large one, which is the domain confound the set exists to avoid.
    """
    taken: list[Example] = []
    cursors = [0] * len(groups)
    while len(taken) < total:
        progressed = False
        for index, group in enumerate(groups):
            if len(taken) >= total:
                break
            if cursors[index] < len(group):
                taken.append(group[cursors[index]])
                cursors[index] += 1
                progressed = True
        if not progressed:
            break
    return taken


def build_evalset(
    pools: Mapping[str, Sequence[Example]], *, per_language: int, seed: int
) -> tuple[Example, ...]:
    """Sample a balanced, source-stratified evaluation set, one fixed draw per seed.

    Balanced because an unbalanced confusion matrix reports the prior, not the
    confusion: with 10x more Kabyle than Tamasheq, always answering Kabyle scores
    well and the matrix says nothing about either language.

    Raises:
        LidError: a language has fewer usable lines than `per_language`, which
            would silently unbalance the matrix.
    """
    if per_language < 1:
        msg = f"per_language must be positive, got {per_language}"
        raise LidError(msg)
    short = {lang: len(pool) for lang, pool in pools.items() if len(pool) < per_language}
    if short:
        detail = ", ".join(f"{lang} has {count}" for lang, count in sorted(short.items()))
        msg = f"per_language={per_language} exceeds the available lines: {detail}"
        raise LidError(msg)

    examples: list[Example] = []
    for language in sorted(pools):
        by_source: dict[str, list[Example]] = {}
        for example in pools[language]:
            by_source.setdefault(example.source, []).append(example)
        groups = []
        for source in sorted(by_source):
            group = by_source[source]
            random.Random(f"{seed}:{language}:{source}").shuffle(group)
            groups.append(group)
        examples.extend(_round_robin(groups, per_language))
    return tuple(examples)


def write_evalset(examples: Sequence[Example], path: Path) -> None:
    """Write the held-out set, one record per line, provenance included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".part")
    with staging.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    staging.replace(path)


def read_evalset(path: Path) -> tuple[Example, ...]:
    """Read a written held-out set back."""
    if not path.is_file():
        msg = f"evaluation set not found: {path}"
        raise LidError(msg)
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(
                Example(
                    text=record["text"],
                    language=record["language"],
                    source=record.get("source", "?"),
                    domain=record.get("domain", "?"),
                )
            )
    if not examples:
        msg = f"evaluation set carried no records: {path}"
        raise LidError(msg)
    return tuple(examples)


def kabyle_side(record: Mapping[str, object]) -> str | None:
    """The Kabyle half of an MT pair, read from its declared direction.

    `direction` is `src-tgt`, so Kabyle is `source` when the direction starts with
    `kab` and `target` when it ends with it. Reading the wrong field would label
    English or French text and report it as Kabyle contamination.
    """
    direction = record.get("direction")
    if not isinstance(direction, str) or "-" not in direction:
        return None
    src, tgt = direction.split("-", maxsplit=1)
    field = "source" if src.startswith("kab") else "target" if tgt.startswith("kab") else None
    if field is None:
        return None
    text = record.get(field)
    return text if isinstance(text, str) else None


@dataclass(frozen=True, slots=True)
class SourceAudit:
    """What one source's Kabyle side was classified as."""

    source_id: str
    distinct: int
    judged: int
    too_short: int
    labels: Mapping[str, int]

    @property
    def kabyle_share(self) -> float:
        return self.labels.get(KABYLE, 0) / self.judged if self.judged else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "distinct": self.distinct,
            "judged": self.judged,
            "too_short": self.too_short,
            "kabyle_share": self.kabyle_share,
            "labels": dict(self.labels),
        }


def audit_sources(
    corpus: Path, *, identifier: Identifier, per_source: int, seed: int
) -> list[SourceAudit]:
    """Classify the Kabyle side of an MT corpus, grouped by the source it came from.

    Each pair appears once per direction, so the Kabyle side is deduplicated before
    sampling: counting it twice would double a source's weight without adding
    evidence.

    Lines under `MIN_WORDS` are counted but not judged. fastText on a three-word
    string is noise, and reporting that noise as contamination would invent a
    finding (CLAUDE.md §6.3: a measurement over too small a population is not a
    measurement).

    Raises:
        LidError: the corpus is missing.
    """
    if not corpus.is_file():
        msg = f"MT corpus not found: {corpus}"
        raise LidError(msg)

    pools: dict[str, set[str]] = {}
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = kabyle_side(record)
            if text is None:
                continue
            pools.setdefault(str(record.get("source_id", "?")), set()).add(text)

    audits: list[SourceAudit] = []
    for source_id in sorted(pools):
        distinct = sorted(pools[source_id])
        judged = [x for x in distinct if len(x.split()) >= MIN_WORDS]
        random.Random(f"{seed}:{source_id}").shuffle(judged)
        sample = judged[:per_source]
        labels: Counter[str] = Counter(identifier.identify(sample))
        audits.append(
            SourceAudit(
                source_id=source_id,
                distinct=len(distinct),
                judged=len(sample),
                too_short=len(distinct) - len(judged),
                labels=dict(labels.most_common()),
            )
        )
    return sorted(audits, key=lambda audit: audit.kabyle_share)


@dataclass(frozen=True, slots=True)
class MarkerAudit:
    """What the lines carrying a suspect character turned out to be."""

    marker: str
    system: str
    revision: str
    records: int
    occurrences: int
    classified: int
    labels: Mapping[str, int]
    by_source: Mapping[str, int]

    @property
    def kabyle_share(self) -> float:
        return self.labels.get(KABYLE, 0) / self.classified if self.classified else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "system": self.system,
            "revision": self.revision,
            "records": self.records,
            "occurrences": self.occurrences,
            "classified": self.classified,
            "kabyle_share": self.kabyle_share,
            "labels": dict(self.labels),
            "by_source": dict(self.by_source),
        }


def audit_marker(corpus: Path, *, marker: str, identifier: Identifier, limit: int) -> MarkerAudit:
    """Classify the corpus lines carrying `marker`, and say what they are.

    Built for the `ṯ` U+1E6F lead (CLAUDE.md §6.2): a character that is not
    Kabyle orthography appears in AƔBALU-Text v1, and the question is whether the
    lines carrying it are sibling text that survived acquisition.

    Raises:
        LidError: the corpus is missing.
    """
    if not corpus.is_file():
        msg = f"corpus not found: {corpus}"
        raise LidError(msg)
    texts: list[str] = []
    sources: Counter[str] = Counter()
    occurrences = 0
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record.get("text")
            if not isinstance(text, str) or marker not in text:
                continue
            occurrences += text.count(marker)
            sources[str(record.get("source", "?"))] += 1
            if len(texts) < limit:
                texts.append(text)

    labels: Counter[str] = Counter(identifier.identify(texts))
    return MarkerAudit(
        marker=marker,
        system=identifier.name,
        revision=identifier.revision,
        records=sum(sources.values()),
        occurrences=occurrences,
        classified=len(texts),
        labels=dict(labels.most_common()),
        by_source=dict(sources.most_common()),
    )


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Counts of (true language, predicted label), and the metrics derived from them.

    `labels` are the true languages; predictions may fall outside them, and those
    columns are kept so an off-set answer is visible rather than silently dropped.
    """

    labels: tuple[str, ...]
    counts: Mapping[tuple[str, str], int]
    system_labels: frozenset[str]

    @property
    def discriminable(self) -> tuple[str, ...]:
        """Eval languages the system could name at all.

        A language outside the system's label set has a recall of zero by
        construction, not by failing to discriminate. Averaging over it would
        report the label inventory and call it accuracy.
        """
        return tuple(x for x in self.labels if x in self.system_labels)

    @property
    def unnameable(self) -> tuple[str, ...]:
        return tuple(x for x in self.labels if x not in self.system_labels)

    @property
    def predicted_labels(self) -> tuple[str, ...]:
        seen = {predicted for _, predicted in self.counts}
        known = [x for x in self.labels if x in seen]
        extra = sorted(seen - set(self.labels))
        return tuple(known + extra)

    def total(self) -> int:
        return sum(self.counts.values())

    def support(self, language: str) -> int:
        return sum(n for (true, _), n in self.counts.items() if true == language)

    def correct(self, language: str) -> int:
        return self.counts.get((language, language), 0)

    def accuracy(self) -> float:
        total = self.total()
        return sum(self.correct(x) for x in self.labels) / total if total else 0.0

    def precision(self, language: str) -> float:
        predicted = sum(n for (_, pred), n in self.counts.items() if pred == language)
        return self.correct(language) / predicted if predicted else 0.0

    def recall(self, language: str) -> float:
        support = self.support(language)
        return self.correct(language) / support if support else 0.0

    def f1(self, language: str) -> float:
        precision, recall = self.precision(language), self.recall(language)
        denominator = precision + recall
        return 2 * precision * recall / denominator if denominator else 0.0

    def macro_f1(self) -> float:
        """Macro-F1 over the languages the system can actually name.

        Reported alongside the matrix, never instead of it: with a label set that
        omits three of the six siblings, a single averaged number hides exactly
        the thing being measured (docs/benchmark.md §3.3).
        """
        nameable = self.discriminable
        return sum(self.f1(x) for x in nameable) / len(nameable) if nameable else 0.0

    def absorption(self, language: str, *, into: str = KABYLE) -> float:
        """Share of `language` that the system labelled `into`.

        The contamination quantity. For a sibling the system cannot name, this is
        where its text goes instead — and a dataset built by filtering on that
        label inherits it.
        """
        support = self.support(language)
        return self.counts.get((language, into), 0) / support if support else 0.0

    def confusions(self, language: str) -> list[tuple[str, int]]:
        """Where `language` went when it was wrong, heaviest first."""
        wrong = [
            (pred, n)
            for (true, pred), n in self.counts.items()
            if true == language and pred != language and n
        ]
        return sorted(wrong, key=lambda item: (-item[1], item[0]))

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": list(self.labels),
            "predicted_labels": list(self.predicted_labels),
            "discriminable": list(self.discriminable),
            "unnameable": list(self.unnameable),
            "total": self.total(),
            "accuracy": self.accuracy(),
            "macro_f1_discriminable": self.macro_f1(),
            "matrix": {
                true: {pred: self.counts.get((true, pred), 0) for pred in self.predicted_labels}
                for true in self.labels
            },
            "per_language": {
                language: {
                    "support": self.support(language),
                    "in_label_set": language in self.system_labels,
                    "precision": self.precision(language),
                    "recall": self.recall(language),
                    "f1": self.f1(language),
                    "absorbed_into_kabyle": self.absorption(language),
                    "confused_with": dict(self.confusions(language)),
                }
                for language in self.labels
            },
        }


def score(system: Identifier, examples: Sequence[Example]) -> ConfusionMatrix:
    """Run `system` over `examples` and tabulate true against predicted.

    Raises:
        LidError: the system returned a different number of labels than it was given.
    """
    scored = [example for example in examples if example.language in EVAL_LANGUAGES]
    predictions = system.identify([example.text for example in scored])
    if len(predictions) != len(scored):
        msg = f"{system.name} returned {len(predictions)} labels for {len(scored)} inputs"
        raise LidError(msg)
    counts: Counter[tuple[str, str]] = Counter()
    for example, predicted in zip(scored, predictions, strict=True):
        counts[(example.language, predicted)] += 1
    labels = tuple(x for x in EVAL_LANGUAGES if any(true == x for true, _ in counts))
    return ConfusionMatrix(labels=labels, counts=dict(counts), system_labels=system.labels)


__all__ = [
    "EVAL_LANGUAGES",
    "HELD_OUT_SOURCES",
    "KABYLE",
    "NOT_AMAZIGH",
    "ConfusionMatrix",
    "Example",
    "Identifier",
    "LidError",
    "MarkerAudit",
    "SourceAudit",
    "SourceError",
    "audit_marker",
    "audit_sources",
    "build_evalset",
    "build_pools",
    "kabyle_side",
    "read_evalset",
    "score",
    "write_evalset",
]
