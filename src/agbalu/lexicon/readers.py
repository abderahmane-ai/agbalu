"""One reader per lexical source format.

Each yields `Entry` objects; the pipeline attaches provenance and normalisation.
No reader invents a lemma or a part of speech the source did not record.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pyarrow.parquet as pq

from agbalu.lexicon.models import Entry, Gloss, LexiconError, Upos, features_of
from agbalu.lexicon.pos import upos_for

csv.field_size_limit(10**9)

FRENCH: Final = "fra"
SPANISH: Final = "spa"
ARABIC: Final = "ara"

GLOSS_LANGUAGES: Final[tuple[str, ...]] = (FRENCH, SPANISH, ARABIC)
"""Amawal writes its translations positionally: `livre - libro - كتاب`."""

MIN_GLOSSES: Final = 2
"""A single dash-separated field is ordinary prose, not a translation triple."""

_HTML_TAG: Final = re.compile(r"<[^>]+>")
_PLURAL: Final = re.compile(r"^\(([^)]+)\)")
_WHITESPACE: Final = re.compile(r"\s+")
_ANNEXED: Final = re.compile(r"Addad amaruz:\s*</strong>\s*([^\n<(]+)")
"""*Addad amaruz*, the annexed state; recorded for 4,465 of Amawal's 10,958 entries."""

TENSE_FEATURE: Final[dict[str, str]] = {
    "aoriste": "Aor",
    "aoriste intensif": "AorInt",
    "prétérit": "Perf",
    "prétérit négatif": "PerfNeg",
    "impératif": "Imp",
    "impératif intensif": "ImpInt",
    "participe aoriste": "PartAor",
    "participe aoriste intensif": "PartAorInt",
    "participe prétérit": "PartPerf",
    "participe prétérit négatif": "PartPerfNeg",
}
"""French tense names in `kabyle-verbs` to a FEATS value.

Opaque codes, not UD `Tense=` values: `aoriste`/`prétérit` are aspects and the intensive
is a derived stem, so `Past`/`Pres` would assert a tense system Kabyle does not have.
"""

PERSON_FEATURES: Final[dict[str, tuple[str | None, str | None, str | None]]] = {
    "1s": ("1", "Sing", None),
    "2s": ("2", "Sing", None),
    "2s_m": ("2", "Sing", "Masc"),
    "2s_f": ("2", "Sing", "Fem"),
    "3s_m": ("3", "Sing", "Masc"),
    "3s_f": ("3", "Sing", "Fem"),
    "1p": ("1", "Plur", None),
    "2p": ("2", "Plur", None),
    "2p_m": ("2", "Plur", "Masc"),
    "2p_f": ("2", "Plur", "Fem"),
    "3p_m": ("3", "Plur", "Masc"),
    "3p_f": ("3", "Plur", "Fem"),
    "participe": (None, None, None),
}


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", text)).strip()


def read_hunspell(path: Path) -> Iterator[Entry]:
    """`kab.dic`: `form/flags po:label st:stem is:inflection`.

    The first line is a declared entry count, not an entry. The ~400 entries with no
    morphology are bare function words (`d`, `ur`, `ara`) and are kept unlabelled.
    """
    if not path.is_file():
        msg = f"hunspell dictionary not found: {path}"
        raise LexiconError(msg)

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        msg = f"hunspell dictionary is empty: {path}"
        raise LexiconError(msg)

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        form = parts[0].split("/", 1)[0]
        if not form:
            continue
        upos: Upos | None = None
        stem: str | None = None
        inflection: str | None = None
        for field in parts[1:]:
            key, _, value = field.partition(":")
            if not value:
                continue
            if key == "po":
                upos = upos_for(value)
            elif key == "st":
                stem = value
            elif key == "is":
                inflection = value
        yield Entry(
            form=form,
            lemma=stem,
            upos=upos,
            # `is:asget` is Kabyle for "plural"; it is the only inflection value the
            # file uses that maps onto a UD feature.
            features=features_of(Number="Plur" if inflection == "asget" else None),
            glosses=(),
            source="",
            licence="",
            redistribution="",
        )


def read_verb_forms(directory: Path) -> Iterator[Entry]:
    """`kabyle-verbs/lemmatizer/*.parquet`: one row per inflected form.

    `seq2seq` holds these same 344,745 rows with input and target swapped; reading both
    double-counts every form. The registry's 695,688 is that sum plus the 6,198 sources.
    """
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        msg = f"no verb parquet shards under {directory}"
        raise LexiconError(msg)

    for shard in shards:
        table = pq.read_table(shard, columns=["form", "infinitif", "tense", "person"])
        for row in table.to_pylist():
            form = str(row.get("form") or "")
            lemma = str(row.get("infinitif") or "")
            if not form or not lemma:
                continue
            person, number, gender = PERSON_FEATURES.get(str(row.get("person") or ""), (None,) * 3)
            yield Entry(
                form=form,
                lemma=lemma,
                upos="VERB",
                features=features_of(
                    Aspect=TENSE_FEATURE.get(str(row.get("tense") or "")),
                    Person=person,
                    Number=number,
                    Gender=gender,
                ),
                glosses=(),
                source="",
                licence="",
                redistribution="",
            )


def read_verb_lemmas(directory: Path) -> Iterator[Entry]:
    """`kabyle-verbs/conjugation-tables/*.parquet`: the 6,198 paradigms.

    Carries the French gloss and irregularity flags, which the expanded table drops.
    """
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        msg = f"no conjugation parquet shards under {directory}"
        raise LexiconError(msg)

    for shard in shards:
        table = pq.read_table(shard, columns=["name", "translation", "isIrregular", "isDerived"])
        for row in table.to_pylist():
            name = str(row.get("name") or "")
            if not name:
                continue
            translation = str(row.get("translation") or "")
            yield Entry(
                form=name,
                lemma=name,
                upos="VERB",
                features=features_of(
                    VerbForm="Inf",
                    Irregular="Yes" if str(row.get("isIrregular")) == "True" else None,
                    Derived="Yes" if str(row.get("isDerived")) == "True" else None,
                ),
                glosses=(Gloss(FRENCH, translation),) if translation else (),
                source="",
                licence="",
                redistribution="",
            )


MIN_TOPONYM_LENGTH: Final = 3
"""Below this, an OSM `name` is a road reference code, not a place name.

All 35 shorter entries are codes (`2a`, `D`, `3A`); every 3-character entry is real
(`Sig`, `Tiṭ`, `Ɛuf`, `SAA`). A digit test would be wrong — `20 Ɣuct 1955` is a street.
"""


def read_toponyms(directory: Path) -> Iterator[Entry]:
    """`kabyle-toponyms`: OSM place names, Kabyle beside French."""
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        msg = f"no toponym parquet shards under {directory}"
        raise LexiconError(msg)

    for shard in shards:
        table = pq.read_table(shard, columns=["kabyle", "french", "category"])
        for row in table.to_pylist():
            kabyle = str(row.get("kabyle") or "")
            if len(kabyle) < MIN_TOPONYM_LENGTH:
                continue
            french = str(row.get("french") or "")
            category = str(row.get("category") or "")
            yield Entry(
                form=kabyle,
                lemma=kabyle,
                upos="PROPN",
                features=features_of(NameType="Geo", GeoCategory=category or None),
                glosses=(Gloss(FRENCH, french),) if french else (),
                source="",
                licence="",
                redistribution="",
            )


def read_tafsut(path: Path) -> Iterator[Entry]:
    """`tafsut_math_lexicon.jsonl`: French↔Kabyle mathematical terminology, 1984.

    `main_entry` rows are set in full capitals by the printed lexicon's typography, not
    its orthography, so they are lower-cased. The 2,561 multi-word entries are kept.
    """
    if not path.is_file():
        msg = f"tafsut lexicon not found: {path}"
        raise LexiconError(msg)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            kabyle = str(row.get("kab") or "").strip()
            if not kabyle:
                continue
            if str(row.get("type")) == "main_entry" and kabyle.isupper():
                kabyle = kabyle.lower()
            french = str(row.get("fr") or "").strip()
            yield Entry(
                form=kabyle,
                lemma=None,
                upos=None,
                features=features_of(Domain="Math"),
                glosses=(Gloss(FRENCH, french.lower() if french.isupper() else french),)
                if french
                else (),
                source="",
                licence="",
                redistribution="",
            )


def _amawal_glosses(content: str) -> tuple[Gloss, ...]:
    """Pull `french - spanish - arabic` out of an Amawal entry body.

    Entries read `<strong>Adlis</strong> (idlisen), livre - libro - كتاب`.
    """
    for segment in content.split("\n"):
        cleaned = _clean(segment)
        if " - " not in cleaned:
            continue
        _, _, tail = cleaned.partition("),")
        parts = [p.strip() for p in (tail or cleaned).split(" - ")]
        glosses = tuple(
            Gloss(language, text)
            for language, text in zip(GLOSS_LANGUAGES, parts, strict=False)
            if text
        )
        if len(glosses) >= MIN_GLOSSES:
            return glosses
    return ()


def _amawal_plural(content: str) -> str | None:
    for segment in content.split("\n"):
        cleaned = _clean(segment)
        _, _, tail = cleaned.partition("(")
        if not tail:
            continue
        match = _PLURAL.match("(" + tail)
        if match:
            plural = match.group(1).strip()
            # Bracketed dialect markers such as `[ṬB][ZW]` ride along with the plural.
            plural = re.sub(r"\[[^\]]*\]", "", plural).strip()
            if plural and " " not in plural:
                return plural
    return None


def _amawal_annexed(content: str) -> str | None:
    match = _ANNEXED.search(content)
    if not match:
        return None
    annexed = _clean(match.group(1)).strip(" .,;:")
    return annexed if annexed and " " not in annexed else None


def read_amawal(path: Path) -> Iterator[Entry]:
    """`amawal_net.csv`: a WordPress export, one post per headword.

    Only the positionally fixed facts are taken — gloss triple, plural, annexed state.
    The rest of `post_content` is free prose in three languages.
    """
    if not path.is_file():
        msg = f"amawal export not found: {path}"
        raise LexiconError(msg)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            headword = _clean(row.get("post_title") or "")
            if not headword or " " in headword:
                continue
            content = row.get("post_content") or ""
            plural = _amawal_plural(content)
            annexed = _amawal_annexed(content)
            yield Entry(
                form=headword,
                lemma=headword,
                upos=None,
                features=features_of(
                    Number="Sing" if plural else None,
                    State="Free" if annexed else None,
                ),
                glosses=_amawal_glosses(content),
                source="",
                licence="",
                redistribution="",
            )
            if plural:
                yield Entry(
                    form=plural,
                    lemma=headword,
                    upos=None,
                    features=features_of(Number="Plur"),
                    glosses=(),
                    source="",
                    licence="",
                    redistribution="",
                )
            if annexed:
                yield Entry(
                    form=annexed,
                    lemma=headword,
                    upos=None,
                    features=features_of(State="Cons"),
                    glosses=(),
                    source="",
                    licence="",
                    redistribution="",
                )


def read_g2p(path: Path) -> Iterator[tuple[str, str]]:
    """`kab_g2p_train.tsv`: orthography and IPA, one pair per line.

    The pairs are sentences, not words; `agbalu.g2p` aligns them.
    """
    if not path.is_file():
        msg = f"g2p training data not found: {path}"
        raise LexiconError(msg)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            orthography, tab, ipa = stripped.partition("\t")
            if not tab or not orthography.strip() or not ipa.strip():
                continue
            yield orthography.strip(), ipa.strip()


def rebrand(entry: Entry, source: str, licence: str, redistribution: str) -> Entry:
    """Attach provenance, which the readers do not know."""
    return Entry(
        form=entry.form,
        lemma=entry.lemma,
        upos=entry.upos,
        features=entry.features,
        glosses=entry.glosses,
        source=source,
        licence=licence,
        redistribution=redistribution,
    )
