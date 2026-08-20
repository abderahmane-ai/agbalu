"""Splitting a document into units an NLLB model can actually translate.

NLLB carries 1,024 positions and was trained on *sentence* pairs, so a document has to be
segmented whichever way you look at it: the positions are a hard limit, and the training
distribution is a soft one that punishes long inputs well before the hard limit is reached.
Segmenting is therefore not a workaround for a short context — it is how the model is meant
to be used, and the reason `bench.translate` scores FLORES+ a sentence at a time.

What it prevents is silent truncation. `generate` encodes with `truncation=True`, so an
over-long source is cut and its tail is never translated, with nothing in the output saying
so — the failure mode this project names first: a result that looks like a result.

**A line break is a boundary unless it is a hard wrap.** Splitting on sentence punctuation
alone glues every line that has no punctuation — a heading, a table-of-contents row, a line
of verse, a scene break — onto the paragraph below it, and the break itself is then inside a
segment where no whitespace bookkeeping can restore it. `unwrap` decides which breaks are
presentation by measuring the width the document was wrapped to; the rest become boundaries.

**Reassembly preserves every non-whitespace character.** Segments carry the whitespace that
surrounded them, so paragraphs, blank lines and indentation survive a round trip; the one
thing deliberately not preserved is the position of a hard wrap, which `unwrap` has already
replaced with a space. On a document with no hard wraps `assemble` fed each segment's own
text returns the input byte for byte.

There is no cross-sentence context here, and none is available: an encoder-decoder trained
on independent pairs has no mechanism for it, and concatenating neighbours to manufacture
some moves the input off the distribution the model was fitted to.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

MAX_SOURCE_TOKENS: Final = 510
"""512 less the source-language token and `</s>`.

Not the 1,024 of `max_position_embeddings`. The NLLB paper states the model's own training
ceiling: "the maximum sequence length during training is 512 for both the encoder and the
decoder" ([arXiv 2207.04672](https://arxiv.org/abs/2207.04672) §8.2.4). The positions allow
twice that, so a longer unit runs without error and without ever having been trained for —
the encoder's own position embeddings above 512 saw no gradient. A hard ceiling, and one
the text should never reach: `SOFT_TARGET_TOKENS` is what units are actually built to."""

SOFT_TARGET_TOKENS: Final = 128
"""What the training distribution holds, as opposed to what the positions allow.

FLORES+ English devtest — the sentences every published number in this project was measured
on — runs median 29 tokens, p99 63, longest 84. A unit past this is being asked for
something the model was never shown: the King James genealogies chain verses with semicolons
and no sentence-final punctuation for hundreds of words, and one reached the encoder at 641
tokens, 7.6x the benchmark's longest sentence. It fitted `MAX_SOURCE_TOKENS` and came back
in French.

Set well above p99 rather than at it, because coming under the target costs a clause
boundary and a sentence is worth keeping whole where it can be."""

SENTENCE_END: Final = re.compile(
    r"""
    (?<= [.!?…] )        # sentence-final punctuation
    [)\]"'»”’]*          # any closing bracket or quote that belongs to it
    (?= \s )             # followed by whitespace, which the suffix will absorb
    """,
    re.VERBOSE,
)

CLAUSE_ENDS: Final = (
    re.compile(r"(?<=[;:])(?=\s)"),
    re.compile(r"(?<=[;:,])(?=\s)"),
)
"""Fallback boundaries, strongest first, for a sentence past the target on its own.

A semicolon or a colon ends a clause that stands alone; a comma frequently does not — it
separates a list, or fences an apposition whose two halves have to be translated together.
Commas are therefore only reached when the stronger marks did not bring the unit under the
target, which is what keeps `a, b and c` intact in a sentence that has a semicolon in it."""

LINE_END: Final = re.compile(r"(?=\n)")
"""Every line break that survived `unwrap`, and therefore every block boundary."""

WRAP_MARGIN: Final = 6
"""How far under the wrap width a line may still count as filled. A wrapper breaks before
the word that would overflow, so the last word's length is the slack."""

MIN_WRAPPED_LINES: Final = 8
"""Below this a document has too few lines for a wrap width to be estimated rather than
read off whichever line happens to be longest."""

WRAP_PERCENTILE: Final = 0.95
"""The wrap width, read off the line lengths. Not the maximum: a document usually holds a
handful of lines a wrapper could not break, and they sit above the width."""

NARROWEST_WRAP: Final = 40
WIDEST_WRAP: Final = 100
"""Plain text is wrapped somewhere between these columns. Outside them, lines of a uniform
length are a list, a table or unwrapped paragraphs — all of which look identical to wrapping
by length alone, and none of which may have its line breaks discarded."""

ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        # French and English titles and connectives that end in a period mid-sentence.
        "m",
        "mm",
        "mme",
        "mlle",
        "dr",
        "pr",
        "st",
        "ste",
        "mr",
        "mrs",
        "ms",
        "prof",
        "etc",
        "cf",
        "vs",
        "ca",
        "al",
        "ed",
        "fig",
        "no",
        "nos",
        "p",
        "pp",
        "vol",
        "jr",
        "sr",
        "art",
        "av",
        "bd",
        "env",
        "ex",
        "réf",
        "tél",
    }
)
"""Lower-cased tokens whose trailing period does not end a sentence. Kabyle borrows the
French set: `Mme`, `Dr` and `etc.` appear in Kabyle text unchanged."""

INITIAL: Final = re.compile(r"(?:^|\s)[^\W\d_]\.$")
"""A single *letter* and a period — `J.` in `J. Amrouche` — which is never a sentence end.

Letters rather than `\\w`, which also matches digits: `Chapter 5.` would otherwise be read as
an initial and never end a sentence, and a numbered document is made of them."""


class SegmentError(Exception):
    """A document cannot be split into translatable units."""


@dataclass(frozen=True, slots=True)
class Segment:
    """One translatable unit, with the whitespace that surrounded it in the source."""

    text: str
    prefix: str = ""
    suffix: str = ""

    def restore(self, translation: str) -> str:
        return f"{self.prefix}{translation}{self.suffix}"


def ends_sentence(left: str) -> bool:
    """Whether text ending here is a sentence rather than an abbreviation or an initial."""
    stripped = left.rstrip(")]\"'»”’")
    if not stripped.endswith("."):
        return True
    if INITIAL.search(stripped):
        return False
    word = re.split(r"[\s(\[\"'«“‘]", stripped[:-1])[-1]
    return word.lower() not in ABBREVIATIONS


def wrap_width(text: str) -> int | None:
    """The column a hard-wrapped document was wrapped to, or `None` if it is not wrapped.

    A wrapped document puts most of its lines within a word's length of one column, and that
    column lies in the range plain text is typeset at. An unwrapped one — a paragraph per
    line — fails the second test, a list of equal-length items usually fails it too, and a
    short fragment has too few lines to measure. All three keep every line break.
    """
    lengths = sorted(len(line.rstrip()) for line in text.splitlines() if line.strip())
    if len(lengths) < MIN_WRAPPED_LINES:
        return None
    width = lengths[int(WRAP_PERCENTILE * (len(lengths) - 1))]
    if not NARROWEST_WRAP <= width <= WIDEST_WRAP:
        return None
    filled = sum(1 for length in lengths if length >= width - WRAP_MARGIN)
    return width if filled * 2 >= len(lengths) else None


def unwrap(text: str) -> str:
    """Hard wraps joined into one line each; every other line break left where it is.

    A break is a hard wrap when the next line's first word would not have fitted after it —
    the wrapper's own rule, applied backwards. A heading, a verse line and a table-of-contents
    row all end well short of the width, so the word after them would have fitted and their
    break is kept.
    """
    width = wrap_width(text)
    if width is None:
        return text
    lines = text.splitlines(keepends=True)
    pieces: list[str] = []
    continuing = False
    for index, line in enumerate(lines):
        body = line.rstrip()
        following = lines[index + 1].split() if index + 1 < len(lines) else []
        joined = bool(body) and bool(following) and len(body) + 1 + len(following[0]) > width
        content = body.lstrip() if continuing else body
        pieces.append(content + (" " if joined else line[len(body) :]))
        continuing = joined
    return "".join(pieces)


def split_keeping_space(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split at every match, keeping the whitespace that follows attached to the left piece."""
    pieces: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        cut = match.end()
        if pattern is SENTENCE_END and not ends_sentence(text[start:cut]):
            continue
        run = re.match(r"\s*", text[cut:])
        end = cut + (run.end() if run else 0)
        pieces.append(text[start:end])
        start = end
    if start < len(text):
        pieces.append(text[start:])
    return [piece for piece in pieces if piece]


def pack(pieces: Sequence[str], count: Callable[[str], int], limit: int) -> list[str]:
    """Consecutive pieces joined while they fit, so a split does not become a shattering.

    Emitting each clause alone would trade one over-long unit for a dozen fragments, which
    is off the training distribution in the other direction.
    """
    packed: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if current and count(candidate) > limit:
            packed.append(current)
            current = piece
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def by_words(text: str, count: Callable[[str], int], budget: int) -> list[str]:
    """Last resort: greedy packing of whitespace-separated words.

    Reached only by text with no sentence and no clause punctuation inside the budget. The
    result is not a sentence and the translation of it will not be good — but it is the whole
    text, which truncation is not.
    """
    return pack(re.findall(r"\S+\s*", text), count, budget)


def fit(text: str, count: Callable[[str], int], budget: int, target: int) -> list[str]:
    """`text` in units the model was trained to take, none of them past the hard limit.

    Two thresholds, because there are two constraints. `budget` is what the model was trained
    to encode and breaking it asks for position embeddings that saw no gradient; `target` is
    what the training distribution actually holds, and passing it degrades quietly.

    The boundaries are tried strongest first and only as far as they are needed: a sentence
    that comes under the target on its semicolons keeps its commas. A long unbroken clause is
    still a sentence, so it is handed over whole unless it breaks the hard budget, where
    cutting mid-phrase is the lesser loss against truncation.
    """
    if count(text) <= target:
        return [text]
    grouped = [text]
    for pattern in CLAUSE_ENDS:
        clauses = split_keeping_space(text, pattern)
        if len(clauses) > 1:
            grouped = pack(clauses, count, target)
        if all(count(group) <= target for group in grouped):
            break
    return [
        piece
        for group in grouped
        for piece in ([group] if count(group) <= budget else by_words(group, count, budget))
    ]


def plan(
    text: str,
    count: Callable[[str], int],
    budget: int = MAX_SOURCE_TOKENS,
    target: int = SOFT_TARGET_TOKENS,
) -> list[Segment]:
    """Every unit to translate, in order, carrying the whitespace between them.

    `count` is injected so the plan can be built and tested without a tokenizer, and so the
    budget is measured with the *model's* tokenizer rather than a proxy for it. A target
    above the budget is the budget: the hard limit cannot be relaxed by asking for longer
    units.
    """
    if budget < 1:
        message = f"budget must be positive, got {budget}"
        raise SegmentError(message)
    target = min(target, budget)
    if not text.strip():
        return []

    document = unwrap(text)
    lead = re.match(r"\s*", document)
    prefix = lead.group() if lead else ""
    body = document[len(prefix) :]

    segments: list[Segment] = []
    for block in split_keeping_space(body, LINE_END):
        for sentence in split_keeping_space(block, SENTENCE_END):
            for piece in fit(sentence, count, budget, target):
                stripped = piece.rstrip()
                segments.append(Segment(text=stripped, suffix=piece[len(stripped) :]))
    if not segments:
        return []
    first = segments[0]
    segments[0] = Segment(text=first.text, prefix=prefix, suffix=first.suffix)
    return segments


def assemble(segments: Sequence[Segment], translations: Sequence[str]) -> str:
    """Put the translated units back where their sources were."""
    if len(segments) != len(translations):
        message = f"{len(translations)} translations for {len(segments)} segments"
        raise SegmentError(message)
    return "".join(
        segment.restore(translation)
        for segment, translation in zip(segments, translations, strict=True)
    )
