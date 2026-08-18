"""CoNLL-U reading for `UD_Kabyle-ADPT`, the only Kabyle treebank there is.

Two views of the same file:

- **syntactic words** — the numeric-id rows, one gold tag each. The UD unit.
- **surface tokens** — what a tagger reading running text sees.

They are not the same sequence. 29.6% of syntactic words come out of multiword tokens
(`lbiru-ines` -> `lbiru` + `in` + `as`), so scoring a whitespace-token tagger against
the word sequence requires deciding what to do about the split. `agbalu.bench.pos`
decides; this module reports the structure.

Malformed rows raise (CLAUDE.md §2.1a).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DEFAULT_ROOT: Final = Path("data/raw/git.ud-kabyle-adpt")

Split = Literal["train", "dev", "test"]
SPLITS: Final[tuple[Split, ...]] = ("train", "dev", "test")

COLUMNS: Final = 10

_NO_SPACE_AFTER: Final = "SpaceAfter=No"
_EMPTY: Final = "_"


class TreebankError(Exception):
    """The treebank could not be read."""


@dataclass(frozen=True, slots=True)
class Word:
    """One syntactic word — a row whose ID is a plain integer."""

    id: int
    form: str
    lemma: str
    upos: str
    feats: str
    head: int | None
    deprel: str
    space_after: bool


@dataclass(frozen=True, slots=True)
class Token:
    """One surface token and the syntactic words it spans.

    A token spanning more than one word has no single gold part of speech, which
    is the whole difficulty of scoring a tagger against this treebank.
    """

    form: str
    space_after: bool
    words: tuple[Word, ...]

    @property
    def is_multiword(self) -> bool:
        return len(self.words) > 1


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence in both views."""

    sent_id: str
    text: str
    split: Split
    words: tuple[Word, ...]
    tokens: tuple[Token, ...]

    def surface(self) -> str:
        """The sentence rebuilt from its tokens, honouring `SpaceAfter=No`.

        Equal to `text` for all 1,930 sentences of `UD_Kabyle-ADPT`; a tagger is
        fed this rather than `text` so its input provably matches the tokens it
        is scored on.
        """
        parts: list[str] = []
        for index, token in enumerate(self.tokens):
            parts.append(token.form)
            if token.space_after and index < len(self.tokens) - 1:
                parts.append(" ")
        return "".join(parts)


def split_path(root: Path, split: Split) -> Path:
    return root / f"kab_adpt-ud-{split}.conllu"


def _parse_head(value: str, path: Path, line_number: int) -> int | None:
    if value == _EMPTY:
        return None
    try:
        return int(value)
    except ValueError:
        msg = f"{path}:{line_number}: HEAD is neither an integer nor '_': {value!r}"
        raise TreebankError(msg) from None


def _parse_range(value: str, path: Path, line_number: int) -> tuple[int, int]:
    low, _, high = value.partition("-")
    if not low.isdigit() or not high.isdigit():
        msg = f"{path}:{line_number}: malformed multiword token ID {value!r}"
        raise TreebankError(msg)
    start, end = int(low), int(high)
    if end <= start:
        msg = f"{path}:{line_number}: multiword token ID {value!r} does not span forward"
        raise TreebankError(msg)
    return start, end


def _tokens(
    rows: list[tuple[int, int, str, bool]], words: list[Word], path: Path, sent_id: str
) -> tuple[Token, ...]:
    """Surface tokens, merging each declared range over the words it covers."""
    by_id = {word.id: word for word in words}
    spans = {start: (end, form, space_after) for start, end, form, space_after in rows}

    claimed: set[int] = set()
    for start, end, _, _ in rows:
        span = set(range(start, end + 1))
        missing = sorted(span - by_id.keys())
        if missing:
            msg = (
                f"{path}: sentence {sent_id!r} declares token {start}-{end}, "
                f"missing word(s) {missing}"
            )
            raise TreebankError(msg)
        if span & claimed:
            msg = f"{path}: sentence {sent_id!r} has overlapping multiword ranges at {start}-{end}"
            raise TreebankError(msg)
        claimed |= span

    tokens: list[Token] = []
    consumed: set[int] = set()
    for word in words:
        if word.id in consumed:
            continue
        declared = spans.get(word.id)
        if declared is None:
            tokens.append(Token(form=word.form, space_after=word.space_after, words=(word,)))
            continue
        end, form, space_after = declared
        covered = [by_id[index] for index in range(word.id, end + 1)]
        consumed.update(member.id for member in covered)
        tokens.append(Token(form=form, space_after=space_after, words=tuple(covered)))
    return tuple(tokens)


def _metadata(comment: str, sent_id: str, text: str) -> tuple[str, str]:
    """`sent_id` and `text` after applying one comment line, unchanged by any other."""
    key, separator, value = comment[1:].partition("=")
    if not separator:
        return sent_id, text
    name = key.strip()
    if name == "sent_id":
        return value.strip(), text
    if name == "text":
        return sent_id, value.strip()
    return sent_id, text


def read_conllu(path: Path, split: Split) -> Iterator[Sentence]:
    """Every sentence of one CoNLL-U file, in both views.

    Empty nodes (`8.1`) are dropped: they are enhanced-dependency artifacts with no
    surface realisation and no tag to score. `UD_Kabyle-ADPT` has none.
    """
    if not path.is_file():
        msg = f"treebank not found: {path}"
        raise TreebankError(msg)

    sent_id = ""
    text = ""
    words: list[Word] = []
    ranges: list[tuple[int, int, str, bool]] = []

    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")

            if not stripped.strip():
                if words:
                    yield Sentence(
                        sent_id=sent_id,
                        text=text,
                        split=split,
                        words=tuple(words),
                        tokens=_tokens(ranges, words, path, sent_id),
                    )
                sent_id, text, words, ranges = "", "", [], []
                continue

            if stripped.startswith("#"):
                sent_id, text = _metadata(stripped, sent_id, text)
                continue

            fields = stripped.split("\t")
            if len(fields) != COLUMNS:
                msg = f"{path}:{line_number}: {len(fields)} columns, expected {COLUMNS}"
                raise TreebankError(msg)

            identifier = fields[0]
            space_after = _NO_SPACE_AFTER not in fields[9]

            if "-" in identifier:
                start, end = _parse_range(identifier, path, line_number)
                ranges.append((start, end, fields[1], space_after))
            elif "." in identifier:
                continue
            elif identifier.isdigit():
                words.append(
                    Word(
                        id=int(identifier),
                        form=fields[1],
                        lemma=fields[2],
                        upos=fields[3],
                        feats=fields[5],
                        head=_parse_head(fields[6], path, line_number),
                        deprel=fields[7],
                        space_after=space_after,
                    )
                )
            else:
                msg = f"{path}:{line_number}: unreadable ID field {identifier!r}"
                raise TreebankError(msg)

    if words:
        yield Sentence(
            sent_id=sent_id,
            text=text,
            split=split,
            words=tuple(words),
            tokens=_tokens(ranges, words, path, sent_id),
        )


def read_split(root: Path, split: Split) -> list[Sentence]:
    return list(read_conllu(split_path(root, split), split))


def read_all(root: Path = DEFAULT_ROOT) -> Iterator[Sentence]:
    """Every split present on disk. `UD_Kabyle-ADPT` ships train and test only."""
    found = False
    for split in SPLITS:
        path = split_path(root, split)
        if not path.is_file():
            continue
        found = True
        yield from read_conllu(path, split)
    if not found:
        msg = f"no treebank split found under {root}"
        raise TreebankError(msg)
