"""The reference Kabyle normaliser: deterministic, idempotent, versioned.

Pipeline order is load-bearing and must not be reshuffled casually:

    1. NFC            compose combining marks, so `t`+U+0323 becomes `ṭ` before
                      any table lookup sees it
    2. invisibles     drop ZWSP/ZWJ/ZWNJ/BOM/soft hyphen, which carry no meaning
                      in Kabyle and silently break tokenisation
    3. homoglyphs     Greek/Cyrillic/Turkish/Czech lookalikes to canonical Kabyle
    4. diacritics     foreign accents to base letters — OFF by default, lossy
    5. whitespace     exotic spaces to U+0020, then collapse runs and strip
    6. punctuation    typographic dashes and quotes to ASCII

Steps 1-3 and 5-6 are safe: each is a total function whose output contains no
character the same step would rewrite again (enforced by `rules._check_confluent`).
Step 4 is lossy for proper nouns and must be requested explicitly.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from agbalu.normalise.models import Change, Flag, NormalisationResult
from agbalu.normalise.rules import ALPHABET, ASCII_DIGRAPHS, HYPHEN, RuleSet, load_rules

NORMALISER_VERSION: Final = "1.3.0"
"""Bump on any change to output. Every downstream artifact is keyed to this.

1.3.0 — the foreign-proper-noun guard splits on any whitespace, not only U+0020.
Normalisation was not idempotent when NBSP or a tab run joined a guarded token to a
correctable one: `À\xa0ð` gave `À ð` then `À ḍ`. See `_protect`.
1.2.0 — the foreign-proper-noun guard no longer fires on `ţ`, which decomposes as `ă`
does but is Kabyle; it had blocked repairs in words such as `Yeţwaṛfaε`.
1.1.0 — homoglyph substitution no longer rewrites foreign proper nouns. 1.0.0 turned
Romanian `Chişinău` into `Chiṣinău`; see `is_foreign_token`.
"""

INVISIBLE: Final = frozenset(
    {
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
        "\u00ad",  # SOFT HYPHEN
        "\u2060",  # WORD JOINER
    }
)

_WHITESPACE_RUN: Final = re.compile(r"[ \t]{2,}")
_TOKEN: Final = re.compile(r"\S+")
"""A maximal run of non-whitespace. The only definition of a token in this module:
`_protect`, `_protected_spans` and `flag` must agree on where words end."""

_PUNCTUATION_KEEP: Final = frozenset('.,;:?!«»"()[]…%&/\\+=*#@_')


class Normaliser:
    """Applies the rule table. Construct once and reuse; it holds no per-call state."""

    def __init__(
        self,
        rules: RuleSet | None = None,
        *,
        fold_diacritics: bool = False,
        collapse_whitespace: bool = True,
    ) -> None:
        self.rules = rules if rules is not None else load_rules()
        self.fold_diacritics = fold_diacritics
        self.collapse_whitespace = collapse_whitespace
        self.version = f"{NORMALISER_VERSION}+rules{self.rules.version}"

    def is_foreign_orthography_mark(self, char: str) -> bool:
        """A letter belonging to another language's alphabet, not to a corruption.

        Decomposition is the test: European orthography composes (`ă` is `a`+U+0306)
        while mojibake and IPA are atomic (`Ʃ`, `ð`, `þ`, `ŋ`). Accepting any
        out-of-inventory letter instead blocked the legitimate `γ`->`ɣ` repair in
        `Ʃemdeγ-am`. `ALPHABET` and `preserved_chars` are both excluded — `ţ` is Kabyle
        (`docs/orthography.md` §4) yet decomposes like a European letter.

        Two gaps needing a lexicon: `ø` and `ł` are real orthography yet atomic, and a
        name whose only foreign letter is itself a rewrite source (`Škoda` via `š`->`ṣ`)
        is still rewritten.
        """
        if not char.isalpha() or char in ALPHABET or char in self.rules.homoglyphs:
            return False
        if char.lower() in self.rules.preserved_chars:
            return False
        decomposed = unicodedata.normalize("NFD", char)
        return len(decomposed) > 1 and decomposed[0].isascii()

    def is_foreign_token(self, token: str) -> bool:
        """A capitalised token spelled in another language's orthography.

        `Chişinău` is the discriminating case: `ă` U+0103 belongs to no Kabyle word
        and to no rewrite rule, which marks the whole token as foreign. Its `ş` is
        then Romanian, not a corrupted Kabyle `ṣ`, and rewriting it invents a word.
        Lowercase Kabyle such as `uεqal` carries no such letter and is untouched.
        """
        if not token[:1].isupper():
            return False
        return any(self.is_foreign_orthography_mark(c) for c in token)

    def _protect(self, original: str, rewritten: str) -> str:
        """Restore tokens the homoglyph pass should not have touched.

        Token-wise, so one protected proper noun does not veto genuine repairs
        elsewhere in the same string.

        A token is a maximal run of non-whitespace, not a `" "`-delimited field. NBSP
        and a tab separate two words exactly as U+0020 does, and a guard that
        disagreed made the result depend on which space character stood between them:
        `À\xa0ð` normalised to `À ð`, then to `À ḍ` — the guard shielded `ð` while the
        NBSP still joined it to `À`, and stopped shielding it once the NBSP became a
        space. Substitution is length-preserving, so spans carry across unchanged.
        """
        if len(original) != len(rewritten):
            return rewritten
        out = list(rewritten)
        for match in _TOKEN.finditer(original):
            token = match.group()
            if token != rewritten[match.start() : match.end()] and self.is_foreign_token(token):
                out[match.start() : match.end()] = token
        return "".join(out)

    def normalise(self, text: str) -> str:
        """Return the canonical form of `text`."""
        if not text:
            return text
        out = unicodedata.normalize("NFC", text)
        out = "".join(ch for ch in out if ch not in INVISIBLE)
        substituted = "".join(self.rules.homoglyphs.get(ch, ch) for ch in out)
        out = self._protect(out, substituted)
        if self.fold_diacritics:
            out = "".join(self.rules.diacritics.get(ch, ch) for ch in out)
        out = "".join(self.rules.whitespace.get(ch, ch) for ch in out)
        out = "".join(self.rules.punctuation.get(ch, ch) for ch in out)
        if self.collapse_whitespace:
            out = _WHITESPACE_RUN.sub(" ", out)
            out = "\n".join(line.strip() for line in out.split("\n"))
            out = out.strip()
        return out

    def analyse(self, text: str) -> NormalisationResult:
        """Normalise `text` and account for every edit and every deliberate refusal."""
        changes: list[Change] = []
        composed = unicodedata.normalize("NFC", text)
        if composed != text:
            changes.append(Change(kind="nfc", position=0, before=text, after=composed))

        for index, char in enumerate(composed):
            if char in INVISIBLE:
                changes.append(
                    Change(kind="invisible-removed", position=index, before=char, after="")
                )
            elif char in self.rules.homoglyphs:
                changes.append(
                    Change(
                        kind="homoglyph",
                        position=index,
                        before=char,
                        after=self.rules.homoglyphs[char],
                    )
                )
            elif self.fold_diacritics and char in self.rules.diacritics:
                changes.append(
                    Change(
                        kind="diacritic-fold",
                        position=index,
                        before=char,
                        after=self.rules.diacritics[char],
                    )
                )
            elif char in self.rules.whitespace:
                changes.append(
                    Change(
                        kind="whitespace",
                        position=index,
                        before=char,
                        after=self.rules.whitespace[char],
                    )
                )
            elif char in self.rules.punctuation:
                changes.append(
                    Change(
                        kind="punctuation",
                        position=index,
                        before=char,
                        after=self.rules.punctuation[char],
                    )
                )

        protected = self._protected_spans(composed)
        kept = [
            c
            for c in changes
            if c.kind != "homoglyph"
            or not any(start <= c.position < end for start, end in protected)
        ]
        flags = list(self.flag(composed))
        flags.extend(
            Flag(
                kind="foreign-proper-noun",
                position=start,
                text=composed[start:end],
                detail="spelled in another language's orthography; homoglyph rules not applied",
            )
            for start, end in protected
        )
        return NormalisationResult(
            text=self.normalise(text),
            original=text,
            version=self.version,
            changes=tuple(kept),
            flags=tuple(flags),
        )

    def _protected_spans(self, text: str) -> list[tuple[int, int]]:
        """The same tokens `_protect` shields, so the report matches the output."""
        return [
            (match.start(), match.end())
            for match in _TOKEN.finditer(text)
            if self.is_foreign_token(match.group())
            and any(c in self.rules.homoglyphs for c in match.group())
        ]

    def flag(self, text: str) -> list[Flag]:
        """Spans a human should look at. Never modifies anything."""
        flags: list[Flag] = []
        for index, char in enumerate(text):
            if char in self.rules.preserved_chars:
                flags.append(
                    Flag(
                        kind="legacy-t-cedilla",
                        position=index,
                        text=char,
                        detail=(
                            "Dallet-tradition spirantised t; modern form is ambiguous "
                            "between t and tt, so it is never rewritten"
                        ),
                    )
                )
            elif char in self.rules.rejected_chars:
                flags.append(
                    Flag(
                        kind="rejected-character",
                        position=index,
                        text=char,
                        detail="not in the Kabyle inventory and not a known homoglyph",
                    )
                )
            elif char.isalpha() and char not in ALPHABET and char not in self.rules.homoglyphs:
                flags.append(
                    Flag(
                        kind="out-of-inventory",
                        position=index,
                        text=char,
                        detail=f"U+{ord(char):04X} {unicodedata.name(char, '<unnamed>')}",
                    )
                )

        for match in _TOKEN.finditer(text):
            word = match.group().lower().strip("".join(_PUNCTUATION_KEEP))
            for digraph in ASCII_DIGRAPHS:
                if digraph in word:
                    flags.append(
                        Flag(
                            kind="ascii-digraph",
                            position=match.start(),
                            text=match.group(),
                            detail=(
                                f"{digraph!r} may stand for a Kabyle letter but collides "
                                "with loanwords — review, never auto-substitute"
                            ),
                        )
                    )
                    break
        return flags


_DEFAULT: Normaliser | None = None


def normalise(text: str) -> str:
    """Normalise with the default rule table, loaded once per process."""
    global _DEFAULT  # noqa: PLW0603 — a process-wide cache of an immutable table
    if _DEFAULT is None:
        _DEFAULT = Normaliser()
    return _DEFAULT.normalise(text)


__all__ = ["HYPHEN", "INVISIBLE", "NORMALISER_VERSION", "Normaliser", "normalise"]
