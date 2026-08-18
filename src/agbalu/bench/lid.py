"""Language identification against Berber siblings — AƔBALU-Bench task 7.4.

The deliverable is the confusion matrix, not the macro-F1: the question worth
answering is which sibling Kabyle is mistaken for and in which direction, because
that is what silently contaminates every dataset labelled "Tamazight"
(docs/benchmark.md §3.3, CLAUDE.md §1).

Latin script only. Every sibling that FLORES+ or GlotCC carries in Tifinagh is
separable from Latin-script Kabyle on the encoding alone, so including it would
measure the script and report the result as language identification.

Sibling text is never normalised. The reference normaliser encodes *Kabyle*
orthography, and applying it to a sibling would rewrite the very distinctions the
classifier is being asked to find.
"""

from __future__ import annotations

import bz2
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final, Protocol

import pyarrow.parquet as pq

KABYLE: Final = "kab_Latn"

EVAL_LANGUAGES: Final[tuple[str, ...]] = (
    "kab_Latn",
    "shi_Latn",
    "rif_Latn",
    "taq_Latn",
    "tzm_Latn",
    "shy_Latn",
)
"""The label set, as `ISO639-3_ISO15924`.

`zgh` is absent deliberately: the survey of 2026-08-09 found no Latin-script
Standard Moroccan Tamazight anywhere, and its Tifinagh text would be separable on
script alone.
"""

MIN_WORDS: Final = 5
MAX_WORDS: Final = 120
MIN_LATIN_SHARE: Final = 0.80
"""Share of cased letters that must be Latin for a line to count as Latin script.

Below 1.0 because Berber Latin orthography legitimately carries `ɛ ɣ ḥ ḍ ṣ ṭ ẓ`,
some of which normalise to non-Latin script names, and because Arabic or Tifinagh
glosses appear inline in wiki-derived text.
"""

_WIKI_RESIDUE: Final[tuple[str, ...]] = (
    "vignette",
    "Taggayt:",
    "Afaylu:",
    "Template:",
    "Wp/",
    "http://",
    "https://",
    "{{",
    "}}",
    "[[",
    "]]",
)
"""Markup that survives a crude wikitext strip. A line carrying any of it is
dropped rather than cleaned: a half-stripped template is not a sentence of any
language, and letting it through would teach the eval set to reward markup."""

_SENTENCE_SPLIT: Final = re.compile(r"(?<=[.!?])\s+|\n+")
_WS: Final = re.compile(r"\s+")

_TEMPLATE: Final = re.compile(r"\{\{[^{}]*\}\}")
_WIKILINK: Final = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]")
_MARKUP_LINE: Final = re.compile(r"^[#*:;].*$", re.MULTILINE)
_EMPHASIS: Final = re.compile(r"''+|=+")
_HTML: Final = re.compile(r"<[^>]+>")

_MEDIAWIKI_NS: Final = "{http://www.mediawiki.org/xml/export-0.11/}"


class LidError(Exception):
    """The evaluation set could not be built, or a system returned an unusable label."""


@dataclass(frozen=True, slots=True)
class Example:
    """One evaluation line and the language it is held to be."""

    text: str
    language: str
    source: str


class Identifier(Protocol):
    """A language identifier, scored as a closed-set classifier over `EVAL_LANGUAGES`.

    `identify` returns one raw label per input, in order. Labels outside the eval
    set are kept verbatim and counted as errors rather than remapped: a system
    that answers `ber_Latn` for Kabyle has made a real mistake, and folding it
    into a correct answer would hide the confusion being measured.
    """

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def labels(self) -> frozenset[str]: ...

    def identify(self, texts: Sequence[str]) -> list[str]: ...


def latin_share(text: str) -> float:
    """Fraction of alphabetic characters whose Unicode name says LATIN."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if unicodedata.name(c, "").startswith("LATIN"))
    return latin / len(letters)


def is_usable(line: str) -> bool:
    """Whether a candidate line belongs in the evaluation set."""
    if any(marker in line for marker in _WIKI_RESIDUE):
        return False
    words = line.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return False
    return latin_share(line) >= MIN_LATIN_SHARE


def sentences(document: str) -> Iterator[str]:
    """Split a document into candidate lines, without judging them."""
    for chunk in _SENTENCE_SPLIT.split(document):
        cleaned = _WS.sub(" ", chunk).strip()
        if cleaned:
            yield cleaned


def strip_wikitext(raw: str) -> str:
    """Remove the markup a MediaWiki dump carries around its prose."""
    text = _TEMPLATE.sub(" ", raw)
    text = _WIKILINK.sub(r"\1", text)
    text = _MARKUP_LINE.sub(" ", text)
    text = _EMPHASIS.sub(" ", text)
    return _HTML.sub(" ", text)


def read_glotcc(directory: Path) -> Iterator[str]:
    """Every document in a GlotCC language partition."""
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        msg = f"no parquet under {directory}"
        raise LidError(msg)
    for path in files:
        table = pq.read_table(path, columns=["content"])
        for value in table.column("content"):
            content = value.as_py()
            if isinstance(content, str):
                yield content


def read_flores(path: Path) -> Iterator[str]:
    """Sentences from one FLORES+ language split."""
    if not path.is_file():
        msg = f"FLORES+ split not found: {path}"
        raise LidError(msg)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record.get("text")
            if isinstance(text, str):
                yield text


def read_mediawiki(path: Path) -> Iterator[str]:
    """Page text from a `pages-articles` dump, with markup stripped."""
    if not path.is_file():
        msg = f"MediaWiki dump not found: {path}"
        raise LidError(msg)
    with bz2.open(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if not element.tag.endswith("}page"):
                continue
            revision = element.find(f"{_MEDIAWIKI_NS}revision")
            node = revision.find(f"{_MEDIAWIKI_NS}text") if revision is not None else None
            if node is not None and node.text:
                yield strip_wikitext(node.text)
            element.clear()


def candidates(documents: Iterable[str]) -> list[str]:
    """Usable, deduplicated lines from a stream of documents, in first-seen order."""
    seen: set[str] = set()
    kept: list[str] = []
    for document in documents:
        for line in sentences(document):
            if line in seen or not is_usable(line):
                continue
            seen.add(line)
            kept.append(line)
    return kept


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
    return sorted(audits, key=lambda a: a.kabyle_share)


def build_evalset(
    pools: Mapping[str, Sequence[str]], *, per_language: int, seed: int
) -> tuple[Example, ...]:
    """Sample a balanced evaluation set, one fixed draw per seed.

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
        pool = list(pools[language])
        random.Random(f"{seed}:{language}").shuffle(pool)
        examples.extend(
            Example(text=text, language=language, source=language) for text in pool[:per_language]
        )
    return tuple(examples)


def build_pools(raw: Path) -> dict[str, list[str]]:
    """Usable lines per language, from the acquired sibling and Kabyle sources.

    Kabyle is drawn from GlotCC rather than from AƔBALU-Text so that it shares a
    crawl, a cleaning pipeline and a domain with four of the five siblings: a
    matrix built from our curated corpus against web-crawled siblings would
    measure provenance and report it as language.

    Tamasheq is the exception and comes from FLORES+, because its GlotCC
    partition holds five documents. That is a real heterogeneity in the eval set
    and is recorded rather than hidden.
    """
    dumps = sorted((raw / "wikimedia.shywiktionary").glob("*pages-articles*.bz2"))
    if not dumps:
        msg = f"no Shawiya dump under {raw / 'wikimedia.shywiktionary'}"
        raise LidError(msg)
    flores = raw / "hf.flores-plus-taq"
    return {
        "kab_Latn": candidates(read_glotcc(raw / "hf.glotcc-v1-kab")),
        "shi_Latn": candidates(read_glotcc(raw / "hf.glotcc-v1-shi")),
        "rif_Latn": candidates(read_glotcc(raw / "hf.glotcc-v1-rif")),
        "tzm_Latn": candidates(read_glotcc(raw / "hf.glotcc-v1-tzm")),
        "taq_Latn": candidates(
            chain(
                read_flores(flores / "dev" / "taq_Latn.jsonl"),
                read_flores(flores / "devtest" / "taq_Latn.jsonl"),
            )
        ),
        "shy_Latn": candidates(read_mediawiki(dumps[-1])),
    }


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
    predictions = system.identify([example.text for example in examples])
    if len(predictions) != len(examples):
        msg = f"{system.name} returned {len(predictions)} labels for {len(examples)} inputs"
        raise LidError(msg)
    counts: Counter[tuple[str, str]] = Counter()
    for example, predicted in zip(examples, predictions, strict=True):
        counts[(example.language, predicted)] += 1
    labels = tuple(x for x in EVAL_LANGUAGES if any(t == x for t, _ in counts))
    return ConfusionMatrix(labels=labels, counts=dict(counts), system_labels=system.labels)
