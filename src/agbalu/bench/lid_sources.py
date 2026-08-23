"""Every acquired Berber line, labelled with the language, source and domain it came from.

One table, two consumers: `lid.build_pools` cuts the held-out set from it and
`lid_dataset` cuts train and dev from it. They were two independent tables until
2026-08-23, and the drift was not visible until both were measured — a char n-gram
scored 98.26% on a dev split drawn from the training sources and 76.27% on the
held-out set drawn from different ones.

`domain` is load-bearing, not description. It is the group axis the sampler
balances over and the axis the contrastive loss mines negatives along, because the
confusion this benchmark exists to measure is between languages, and a corpus where
each language arrives from one source lets a classifier answer with the source
instead.

Siblings are never put through the Kabyle normaliser: it encodes Kabyle
orthography, and applying it would rewrite the distinctions the model must find.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pyarrow.parquet as pq

KABYLE: Final = "kab_Latn"
NOT_AMAZIGH: Final = "NOT_AMAZIGH"

BERBER_LANGUAGES: Final[tuple[str, ...]] = (
    "kab_Latn",
    "shi_Latn",
    "rif_Latn",
    "taq_Latn",
    "tzm_Latn",
    "shy_Latn",
)
"""The label set, as `ISO639-3_ISO15924`.

Latin script only. Every sibling that FLORES+, GlotCC or translatewiki also carries
in Tifinagh is separable from Latin-script Kabyle on the encoding alone, so
including it would measure the script and report the result as language.

Standard Moroccan Tamazight (`zgh`) is absent for the same reason and it was
checked rather than assumed: of 18,386 `zgh` strings in the Tatoeba export, 18,359
are Tifinagh and 27 are Latin.
"""

CLASSES: Final[tuple[str, ...]] = (*BERBER_LANGUAGES, NOT_AMAZIGH)

Domain = Literal[
    "scripture",
    "web",
    "everyday",
    "encyclopaedia",
    "interface",
    "lexicon",
    "negative-latin",
    "negative-other",
]
Trust = Literal["gold", "silver"]

Reader = Literal[
    "glotcc",
    "flores",
    "gzip_lines",
    "moses_zip",
    "ocr_bible",
    "mediawiki",
    "wiktionary_entries",
    "tatoeba",
    "tatoeba_negatives",
    "beni_mellal",
    "csv_target",
    "tsv_target",
    "agbalu_text",
]

MIN_WORDS: Final = 4
MAX_WORDS: Final = 120
"""Four, measured against five on the two classes that cannot spare a line: it buys
Shawiya 622 lines against 411 and Central Atlas Tamazight 1,128 against 783, and
GlotLID's Berber share over the added lines is unchanged (Shawiya 96.6% against
97.1%). Below four the added lines stop being sentences."""
MIN_LATIN_SHARE: Final = 0.80
"""Share of cased letters that must be Latin for a line to count as Latin script.

Below 1.0 because Berber Latin orthography legitimately carries `ɛ ɣ ḥ ḍ ṣ ṭ ẓ`,
some of which normalise to non-Latin script names, and because Arabic or Tifinagh
glosses appear inline in wiki-derived text.
"""

AGBALU_TEXT: Final = Path("data/processed/text/agbalu-text-v1.jsonl")


class SourceError(Exception):
    """A registered source could not be read."""


@dataclass(frozen=True, slots=True)
class Source:
    """One acquired source, and how its lines are labelled once read."""

    id: str
    language: str
    domain: Domain
    trust: Trust
    licence: str
    reader: Reader
    location: str
    argument: str = ""

    @property
    def group(self) -> str:
        """The `(language, domain)` cell this source contributes to."""
        return f"{self.language}/{self.domain}"


SOURCES: Final[tuple[Source, ...]] = (
    # Kabyle. The one language with no shortage, capped in the builder rather than here.
    Source("agbalu.text", KABYLE, "everyday", "gold", "various", "agbalu_text", str(AGBALU_TEXT)),
    Source(
        "opus.bible-uedin-kab",
        KABYLE,
        "scripture",
        "gold",
        "cc-by-4.0",
        "moses_zip",
        "opus.bible-uedin-kab/en-kab.txt.zip",
        "kab",
    ),
    Source("hf.glotcc-kab", KABYLE, "web", "silver", "cc0-1.0", "glotcc", "hf.glotcc-v1-kab"),
    # Tashelhit.
    Source(
        "opus.bible-uedin-shi",
        "shi_Latn",
        "scripture",
        "gold",
        "cc-by-4.0",
        "gzip_lines",
        "opus.bible-uedin-shi/shi.txt.gz",
    ),
    Source(
        "wikimedia.shiwiki",
        "shi_Latn",
        "encyclopaedia",
        "gold",
        "cc-by-sa-4.0",
        "mediawiki",
        "wikimedia.shiwiki",
    ),
    Source(
        "tatoeba.shi",
        "shi_Latn",
        "everyday",
        "gold",
        "cc-by-2.0",
        "tatoeba",
        "hf.tamazight-nlp.tatoeba/data/train-00000-of-00001.parquet",
        "shi",
    ),
    Source(
        "translatewiki.shi",
        "shi_Latn",
        "interface",
        "gold",
        "cc-by-3.0",
        "csv_target",
        "hf.tamazight-nlp.translatewiki-siblings/en-shi.csv",
    ),
    Source("hf.glotcc-shi", "shi_Latn", "web", "silver", "cc0-1.0", "glotcc", "hf.glotcc-v1-shi"),
    # Tarifit. The OCR New Testament is capped in the builder: at full size it is 35% of
    # every training row and worth 1.2 points of held-out recall.
    Source(
        "archive.tarifit-nt",
        "rif_Latn",
        "scripture",
        "gold",
        "all-rights-reserved",
        "ocr_bible",
        "archive.tarifit-nt/Latin-MERGED_djvu.txt",
    ),
    Source(
        "tatoeba.rif",
        "rif_Latn",
        "everyday",
        "gold",
        "cc-by-2.0",
        "tatoeba",
        "hf.tamazight-nlp.tatoeba/data/train-00000-of-00001.parquet",
        "rif",
    ),
    Source(
        "translatewiki.rif",
        "rif_Latn",
        "interface",
        "gold",
        "cc-by-3.0",
        "csv_target",
        "hf.tamazight-nlp.translatewiki-siblings/en-rif.csv",
    ),
    Source("hf.glotcc-rif", "rif_Latn", "web", "silver", "cc0-1.0", "glotcc", "hf.glotcc-v1-rif"),
    # Tamasheq. FLORES+ is held-out only — it is where the evaluation sentences come from,
    # and the registry records it as never ingested.
    Source(
        "nllb-seed.taq",
        "taq_Latn",
        "encyclopaedia",
        "gold",
        "cc-by-sa-4.0",
        "csv_target",
        "hf.tamazight-nlp.nllb-seed-taq/eng_Latn-taq_Latn.csv",
    ),
    Source("hf.glotcc-taq", "taq_Latn", "web", "silver", "cc0-1.0", "glotcc", "hf.glotcc-v1-taq"),
    # Central Atlas Tamazight. The translatewiki slice is largely Tifinagh and most of it
    # falls out on the Latin-share filter; what survives is still a domain tzm lacked.
    Source(
        "hf.beni-mellal",
        "tzm_Latn",
        "everyday",
        "gold",
        "unknown",
        "beni_mellal",
        "hf.tamazight-nlp.beni-mellal/data/train-00000-of-00001.parquet",
    ),
    Source(
        "translatewiki.tzm",
        "tzm_Latn",
        "interface",
        "gold",
        "cc-by-3.0",
        "csv_target",
        "hf.tamazight-nlp.translatewiki-siblings/en-tzm.csv",
    ),
    Source(
        "pontoon.tzm",
        "tzm_Latn",
        "interface",
        "gold",
        "mpl-2.0",
        "tsv_target",
        "hf.tamazight-nlp.pontoon-siblings/en-tzm.tsv",
    ),
    Source("hf.glotcc-tzm", "tzm_Latn", "web", "silver", "cc0-1.0", "glotcc", "hf.glotcc-v1-tzm"),
    # Shawiya. The thinnest class in the set, and the only Shawiya text that exists:
    # shy.wiktionary is its one Wikimedia project and no crawl carries the language.
    #
    # Read through the `{{langue|shy}}` sections and never as whole pages. Whole pages
    # yield 1,423 lines of which GlotLID calls 40.5% French, 21.5% English and 21%
    # Berber — the wiki's meta and Lua are French. Section-gated, the same dump yields
    # 276 lines at 97.1% Berber.
    Source(
        "wikimedia.shywiktionary",
        "shy_Latn",
        "lexicon",
        "gold",
        "cc-by-sa-4.0",
        "wiktionary_entries",
        "wikimedia.shywiktionary",
    ),
    Source(
        "tatoeba.shy",
        "shy_Latn",
        "everyday",
        "gold",
        "cc-by-2.0",
        "tatoeba",
        "hf.tamazight-nlp.tatoeba/data/train-00000-of-00001.parquet",
        "shy",
    ),
    Source(
        "translatewiki.shy",
        "shy_Latn",
        "interface",
        "gold",
        "cc-by-3.0",
        "csv_target",
        "hf.tamazight-nlp.translatewiki-siblings/en-shy.csv",
    ),
    # The negative class, in two halves. Latin non-Berber is the hard half and the one a
    # deployed filter actually meets; the other-script half is what stops a Tifinagh or
    # Arabic line being absorbed into a Berber label rather than rejected.
    Source(
        "tatoeba.negatives-latin",
        NOT_AMAZIGH,
        "negative-latin",
        "gold",
        "cc-by-2.0",
        "tatoeba_negatives",
        "hf.tamazight-nlp.tatoeba/data/train-00000-of-00001.parquet",
        "latin",
    ),
    Source(
        "tatoeba.negatives-other",
        NOT_AMAZIGH,
        "negative-other",
        "gold",
        "cc-by-2.0",
        "tatoeba_negatives",
        "hf.tamazight-nlp.tatoeba/data/train-00000-of-00001.parquet",
        "other",
    ),
)

NEGATIVE_LANGUAGES: Final[dict[str, tuple[str, ...]]] = {
    "latin": ("fra", "eng", "spa", "epo", "deu", "ita", "por", "nld", "tur"),
    "other": ("ara", "arq", "ary", "zgh", "heb", "rus", "ukr"),
}
"""Tatoeba codes taken as not-Amazigh, split by the script they are written in.

`zgh` is in the `other` half deliberately: it is Amazigh, but the 18,359 Tifinagh
strings in this export are outside the Latin label set, and a model that answered
`kab_Latn` for a Tifinagh line would be wrong in the way that matters.
"""

_SENTENCE_SPLIT: Final = re.compile(r"(?<=[.!?])\s+|\n+")
_WS: Final = re.compile(r"\s+")
_LEADING_VERSE_NUMBER: Final = re.compile(r"^\s*\d+\s+")
_PLACEHOLDER: Final = re.compile(r"%(?:\d+\$)?[sd]|\{\d*\}|\$[A-Za-z_]+|&[a-z]+;")
_MAX_COMMA_RATIO: Final = 0.3
_MEDIAWIKI_NS: Final = "{http://www.mediawiki.org/xml/export-0.11/}"

_TEMPLATE: Final = re.compile(r"\{\{[^{}]*\}\}")
_WIKILINK: Final = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]")
_MARKUP_LINE: Final = re.compile(r"^[#*:;].*$", re.MULTILINE)
_EMPHASIS: Final = re.compile(r"''+|=+")
_HTML: Final = re.compile(r"<[^>]+>")
_TEMPLATE_PASSES: Final = 4

WIKI_RESIDUE: Final[tuple[str, ...]] = (
    "vignette",
    "Taggayt:",
    "Afaylu:",
    "Template:",
    "Module:",
    "Wp/",
    "http://",
    "https://",
    "{{",
    "}}",
    "[[",
    "]]",
    "l['",
    "nom '",
    "{ nom",
    "cellpadding",
    "PAGENAME",
)
"""Markup that survives a crude wikitext strip. A line carrying any of it is
dropped rather than cleaned: a half-stripped template is not a sentence of any
language, and letting it through would teach the set to reward markup."""

_WIKTIONARY_NAMESPACES: Final[tuple[str, ...]] = (
    "Module:",
    "Template:",
    "MediaWiki:",
    "User:",
    "File:",
    "Wiktionary:",
    "Category:",
    "Taggayt:",
    "Afaylu:",
)
_SHY_SECTION: Final = re.compile(r"==\s*\{\{langue\|[^}]+\}\}\s*==")
_SHY_MARKERS: Final = ("{{shy-", "awal|shy", "buḍ|shy", "Isem|shy")


def strip_wikitext(raw: str) -> str:
    """Remove the markup a MediaWiki dump carries around its prose."""
    text = _TEMPLATE.sub(" ", raw)
    text = _WIKILINK.sub(r"\1", text)
    text = _MARKUP_LINE.sub(" ", text)
    text = _EMPHASIS.sub(" ", text)
    return _HTML.sub(" ", text)


def sentences(document: str) -> Iterator[str]:
    """Split a document into candidate lines, without judging them."""
    for chunk in _SENTENCE_SPLIT.split(document):
        cleaned = _WS.sub(" ", chunk).strip()
        if cleaned:
            yield cleaned


def read_glotcc(directory: Path) -> Iterator[str]:
    """Every document in a GlotCC language partition."""
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        msg = f"no parquet under {directory}"
        raise SourceError(msg)
    for path in files:
        table = pq.read_table(path, columns=["content"])
        for value in table.column("content"):
            content = value.as_py()
            if isinstance(content, str):
                yield content


def read_flores(directory: Path, language: str) -> Iterator[str]:
    """Sentences from every FLORES+ split of one language."""
    splits = sorted(directory.glob(f"*/{language}.jsonl"))
    if not splits:
        msg = f"no FLORES+ split for {language} under {directory}"
        raise SourceError(msg)
    for path in splits:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = record.get("text")
                if isinstance(text, str):
                    yield text


def read_gzip_lines(path: Path) -> Iterator[str]:
    """Non-empty lines from a gzipped text file (OPUS mono bundle)."""
    if not path.is_file():
        msg = f"gzip text not found: {path}"
        raise SourceError(msg)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            if line := raw.strip():
                yield line


def read_moses_zip(path: Path, language: str) -> Iterator[str]:
    """One side of an OPUS moses `.zip`, selected by its language extension."""
    if not path.is_file():
        msg = f"moses bundle not found: {path}"
        raise SourceError(msg)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(f".{language}")]
        if not members:
            msg = f"{path}: no member ending .{language}"
            raise SourceError(msg)
        for name in members:
            with archive.open(name) as handle:
                for raw in io.TextIOWrapper(handle, encoding="utf-8"):
                    if line := raw.strip():
                        yield line


def read_ocr_bible(path: Path) -> Iterator[str]:
    """Lines from a DjVu OCR text layer, leading verse numbers stripped."""
    if not path.is_file():
        msg = f"OCR text not found: {path}"
        raise SourceError(msg)
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if line := _LEADING_VERSE_NUMBER.sub("", raw).strip():
                yield line


def _newest_dump(directory: Path) -> Path:
    dumps = sorted(directory.glob("*pages-articles*.bz2"))
    if not dumps:
        msg = f"no MediaWiki dump under {directory}"
        raise SourceError(msg)
    return dumps[-1]


def _pages(directory: Path) -> Iterator[tuple[str, str]]:
    """`(title, wikitext)` for every page in the newest dump under `directory`."""
    with bz2.open(_newest_dump(directory), "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if not element.tag.endswith("}page"):
                continue
            title = element.find(f"{_MEDIAWIKI_NS}title")
            revision = element.find(f"{_MEDIAWIKI_NS}revision")
            node = revision.find(f"{_MEDIAWIKI_NS}text") if revision is not None else None
            if title is not None and title.text and node is not None and node.text:
                yield title.text, node.text
            element.clear()


def read_mediawiki(directory: Path) -> Iterator[str]:
    """Article prose from a `pages-articles` dump, with markup stripped."""
    for title, wikitext in _pages(directory):
        if not title.startswith(_WIKTIONARY_NAMESPACES):
            yield strip_wikitext(wikitext)


def read_wiktionary_entries(directory: Path) -> Iterator[str]:
    """Headwords, plurals and glosses from the Shawiya sections of a Wiktionary dump.

    A separate source from `read_mediawiki` over the same bytes, and separately
    labelled `lexicon`, because entries are one to three words where prose is
    fifteen. Merged into one source they would make length a class feature.
    """
    for title, wikitext in _pages(directory):
        if title.startswith(_WIKTIONARY_NAMESPACES):
            continue
        if "{{langue|shy}}" not in wikitext.replace(" ", ""):
            continue
        for part in _SHY_SECTION.split(wikitext):
            if not any(marker in part for marker in _SHY_MARKERS):
                continue
            yield title
            for match in re.finditer(r"\{\{shy-amtawa-au\|s=([^|]*)\|p=([^|]*)\|", part):
                yield match.group(1).strip()
                yield match.group(2).strip()
            for raw in part.splitlines():
                yield strip_wikitext(raw).strip()


def read_tatoeba(path: Path, language: str) -> Iterator[str]:
    """The `language`-side string of every Tatoeba pair carrying it."""
    if not path.is_file():
        msg = f"Tatoeba parquet not found: {path}"
        raise SourceError(msg)
    table = pq.read_table(
        path, columns=["source_lang", "source_string", "target_lang", "target_string"]
    )
    columns = (
        table.column("source_lang").to_pylist(),
        table.column("source_string").to_pylist(),
        table.column("target_lang").to_pylist(),
        table.column("target_string").to_pylist(),
    )
    for source_language, source_text, target_language, target_text in zip(*columns, strict=True):
        if source_language == language and isinstance(source_text, str):
            yield source_text
        if target_language == language and isinstance(target_text, str):
            yield target_text


def read_tatoeba_negatives(path: Path, half: str) -> Iterator[str]:
    """Non-Amazigh Tatoeba strings from one script half of `NEGATIVE_LANGUAGES`."""
    wanted = NEGATIVE_LANGUAGES.get(half)
    if wanted is None:
        msg = f"unknown negative half {half!r}"
        raise SourceError(msg)
    for language in wanted:
        yield from read_tatoeba(path, language)


def is_conjugation_drill(text: str) -> bool:
    """A learner-deck row that lists inflected forms rather than a sentence."""
    if "//" in text:
        return True
    words = text.split()
    return bool(words) and text.count(",") / len(words) > _MAX_COMMA_RATIO


def read_beni_mellal(path: Path) -> Iterator[str]:
    """The Tamazight column of the Beni-Mellal deck, without its inflection tables."""
    if not path.is_file():
        msg = f"Beni-Mellal parquet not found: {path}"
        raise SourceError(msg)
    table = pq.read_table(path, columns=["Tamazight"])
    for value in table.column("Tamazight").to_pylist():
        if isinstance(value, str) and not is_conjugation_drill(value):
            yield value


def _read_delimited(path: Path, delimiter: str) -> Iterator[str]:
    """The last column of a headed, delimited bitext — the Amazigh side.

    `QUOTE_NONE` is wrong here and right elsewhere: these files quote embedded
    delimiters, and the corpus-wide rule against default quoting was written for
    the Tatoeba exports, which do not.
    """
    if not path.is_file():
        msg = f"delimited text not found: {path}"
        raise SourceError(msg)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            msg = f"{path} is empty"
            raise SourceError(msg)
        for row in reader:
            if row and (value := _PLACEHOLDER.sub(" ", row[-1]).strip()):
                yield value


def read_agbalu_text(path: Path) -> Iterator[str]:
    """Every sentence of the Kabyle corpus, in file order; the builder samples it."""
    if not path.is_file():
        msg = f"AGBALU-Text not found: {path}"
        raise SourceError(msg)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record.get("text")
            if isinstance(text, str):
                yield text


def latin_share(text: str) -> float:
    """Fraction of alphabetic characters whose Unicode name says LATIN."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if unicodedata.name(c, "").startswith("LATIN"))
    return latin / len(letters)


def is_usable(line: str, *, require_latin: bool = True) -> bool:
    """Whether a candidate line is a sentence of the script the label set covers.

    `MIN_WORDS` applies to every source including the Wiktionary entries. Training
    on one-word headwords while evaluating on fifteen-word prose made length a
    feature of Shawiya, which is not a fact about Shawiya.
    """
    if any(marker in line for marker in WIKI_RESIDUE):
        return False
    if not MIN_WORDS <= len(line.split()) <= MAX_WORDS:
        return False
    return not require_latin or latin_share(line) >= MIN_LATIN_SHARE


def candidates(
    stream: Iterable[str],
    *,
    require_latin: bool = True,
    limit: int | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """Usable, deduplicated lines from a stream of documents.

    `limit` is a seeded reservoir over the whole stream, never a prefix. The first
    80,000 sentences of a source-ordered corpus are one source, and the first 8,000
    lines of a New Testament are Matthew.
    """
    if limit is not None and limit < 1:
        msg = f"limit must be positive, got {limit}"
        raise SourceError(msg)
    draw = rng if rng is not None else random.Random(0)
    seen: set[str] = set()
    kept: list[str] = []
    seen_count = 0
    for document in stream:
        for line in sentences(document):
            if line in seen or not is_usable(line, require_latin=require_latin):
                continue
            seen.add(line)
            if limit is None or len(kept) < limit:
                kept.append(line)
            else:
                index = draw.randrange(seen_count + 1)
                if index < limit:
                    kept[index] = line
            seen_count += 1
    return kept


def lines(source: Source, raw: Path, *, limit: int | None = None, seed: int = 0) -> list[str]:
    """Every usable, deduplicated line one source contributes, at most `limit` of them."""
    return candidates(
        documents(source, raw),
        require_latin=source.domain != "negative-other",
        limit=limit,
        rng=random.Random(f"{seed}:{source.id}"),
    )


READERS: Final[dict[Reader, Callable[[Path, str], Iterator[str]]]] = {
    "glotcc": lambda base, _: read_glotcc(base),
    "flores": read_flores,
    "gzip_lines": lambda base, _: read_gzip_lines(base),
    "moses_zip": read_moses_zip,
    "ocr_bible": lambda base, _: read_ocr_bible(base),
    "mediawiki": lambda base, _: read_mediawiki(base),
    "wiktionary_entries": lambda base, _: read_wiktionary_entries(base),
    "tatoeba": read_tatoeba,
    "tatoeba_negatives": read_tatoeba_negatives,
    "beni_mellal": lambda base, _: read_beni_mellal(base),
    "csv_target": lambda base, _: _read_delimited(base, ","),
    "tsv_target": lambda base, _: _read_delimited(base, "\t"),
    "agbalu_text": lambda base, _: read_agbalu_text(base),
}
"""Every `Reader` name mapped to the function that opens it, taking the resolved path
and the source's `argument`. A dict rather than a match statement so a `Reader` added
without a reader is a `KeyError` at the table, not a silently empty source."""


def documents(source: Source, raw: Path) -> Iterator[str]:
    """Raw documents for one source, before sentence splitting and filtering."""
    base = Path(source.location) if source.location.startswith("data/") else raw / source.location
    yield from READERS[source.reader](base, source.argument)
