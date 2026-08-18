"""Typed schema for the AƔBALU corpus registry.

The registry is the single source of truth for where every byte of AƔBALU data
came from, under which licence, and how to fetch it again. It is validated
strictly: an unknown field or a missing licence is an error, not a warning.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Modality = Literal["text", "speech", "parallel", "lexicon", "annotation", "multimodal", "tool"]
"""What kind of data a source holds.

`parallel` is text aligned across languages; `annotation` is text carrying
linguistic labels (POS, dependencies, NER); `lexicon` is word- or lemma-level;
`tool` is software, a model, or a specification rather than data.
"""

Access = Literal["hf", "opus", "http", "git", "wikimedia", "manual"]
"""How the source is retrieved. `manual` means a human step is required."""

Tier = Literal["core", "supplementary", "reference", "excluded"]
"""Role in the corpus.

`core` enters the released corpus; `supplementary` enters only licence-compatible
cuts; `reference` is consulted but never ingested (benchmarks, specs); `excluded`
is recorded so the decision to leave it out is auditable.
"""

Redistribution = Literal["permissive", "share-alike", "non-commercial", "unclear", "none"]
"""Coarse redistribution class derived from the licence, used to cut releases."""

# Kabyle is `kab`; siblings that must never be silently merged are listed in
# CLAUDE.md §1. Codes are ISO 639-3, optionally with an ISO 15924 script suffix.
_LANG_RE: Final = re.compile(r"^[a-z]{2,3}(_[A-Z][a-z]{3})?$")

SIBLING_LANGUAGES: Final = frozenset({"shi", "tzm", "rif", "taq", "zgh", "shy"})
"""Northern and Southern Berber languages that are *not* Kabyle (CLAUDE.md §1).

Registered separately from the Kabyle corpus so that sibling text can be acquired
for the task-7.4 language-identification benchmark without ever becoming eligible
for AƔBALU-Text. `Source` refuses these; `SiblingSource` requires one.
"""

_PERMISSIVE: Final = frozenset(
    {"cc0-1.0", "mit", "apache-2.0", "cc-by-2.0", "cc-by-3.0", "cc-by-4.0", "odc-by"}
)
_SHARE_ALIKE: Final = frozenset({"cc-by-sa-3.0", "cc-by-sa-4.0", "gfdl", "mpl-2.0", "odbl"})
"""`odbl` is share-alike, not permissive: ODbL §4.4 requires a publicly used *derivative
database* to be offered under ODbL too. Classing it permissive would let a permissive-only
release be cut by code (§2.1 rule 2) and come out mislabelled — which is precisely the
guarantee that classing exists to provide. `odc-by` is attribution-only and stays."""
_NON_COMMERCIAL: Final = frozenset({"cc-by-nc-4.0", "cc-by-nc-sa-4.0"})

LangCode = Annotated[str, Field(pattern=_LANG_RE.pattern)]
SourceId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", max_length=96)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
"""Lowercase hex SHA-256. Set for any source whose exact bytes we hold."""


def redistribution_class(licence: str) -> Redistribution:
    """Map a licence identifier to its coarse redistribution class."""
    normalised = licence.strip().lower()
    if normalised in _PERMISSIVE:
        return "permissive"
    if normalised in _SHARE_ALIKE:
        return "share-alike"
    if normalised in _NON_COMMERCIAL:
        return "non-commercial"
    if normalised in {"unknown", "other", ""}:
        return "unclear"
    return "unclear"


class SourceSize(BaseModel):
    """Measured size of a source. Every field is optional but at least one is required.

    Counts are what the provider reports at `Source.retrieved`; they are recorded
    so drift is detectable on re-fetch, not because they are authoritative.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: int | None = Field(default=None, ge=0)
    bytes: int | None = Field(default=None, ge=0)
    sentences: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    hours: float | None = Field(default=None, ge=0.0)
    articles: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_one_measure(self) -> Self:
        if not any(
            v is not None
            for v in (self.rows, self.bytes, self.sentences, self.tokens, self.hours, self.articles)
        ):
            msg = "SourceSize requires at least one measured field"
            raise ValueError(msg)
        return self


def _base_language(code: str) -> str:
    """Strip an ISO 15924 script suffix: `shi_Latn` -> `shi`."""
    return code.split("_", maxsplit=1)[0]


class SourceBase(BaseModel):
    """Fields and invariants shared by every registered source.

    The language rule is deliberately *not* here: it is what separates the Kabyle
    registry from the sibling registry, and each subclass states its own.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: SourceId
    name: str = Field(min_length=1, max_length=200)
    modality: Modality
    tier: Tier
    access: Access
    uri: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    languages: tuple[LangCode, ...] = Field(min_length=1)
    size: SourceSize
    retrieved: date
    checksum: Sha256 | None = None
    notes: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _check_tier_and_access(self) -> Self:
        if self.tier == "excluded" and not self.notes:
            msg = f"source {self.id!r} is excluded but gives no reason in notes"
            raise ValueError(msg)
        # A repo-local file is bytes we hold; its integrity must be pinnable.
        if self.access == "manual" and self.checksum is None:
            msg = f"source {self.id!r} is a local file and must declare a checksum"
            raise ValueError(msg)
        return self

    @property
    def redistribution(self) -> Redistribution:
        return redistribution_class(self.licence)

    @property
    def is_parallel(self) -> bool:
        return self.modality == "parallel" or len(self.languages) > 1


class Source(SourceBase):
    """One external Kabyle data source."""

    @model_validator(mode="after")
    def _require_kabyle(self) -> Self:
        # `kab`, `kab_Latn` and `kab_Tfng` all count; `kabyle` or a sibling does not.
        if not any(_base_language(lang) == "kab" for lang in self.languages):
            msg = f"source {self.id!r} does not include 'kab'; it does not belong in this registry"
            raise ValueError(msg)
        return self


class SiblingSource(SourceBase):
    """One external source of a Berber language that is *not* Kabyle.

    Acquired for the task-7.4 LID benchmark. The Kabyle exclusion is the point:
    it is what makes accidental ingestion into AƔBALU-Text a validation error
    rather than a silent merge (CLAUDE.md §1).
    """

    @model_validator(mode="after")
    def _require_sibling_and_refuse_kabyle(self) -> Self:
        bases = {_base_language(lang) for lang in self.languages}
        if "kab" in bases:
            msg = f"sibling source {self.id!r} includes 'kab'; it belongs in the Kabyle registry"
            raise ValueError(msg)
        if not bases & SIBLING_LANGUAGES:
            known = ", ".join(sorted(SIBLING_LANGUAGES))
            msg = f"sibling source {self.id!r} declares no Berber sibling language ({known})"
            raise ValueError(msg)
        return self


def _ensure_unique_ids(sources: Sequence[SourceBase]) -> None:
    seen: set[str] = set()
    for source in sources:
        if source.id in seen:
            msg = f"duplicate source id: {source.id!r}"
            raise ValueError(msg)
        seen.add(source.id)


class Registry(BaseModel):
    """The complete catalogue of Kabyle data sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    surveyed: date
    sources: tuple[Source, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        _ensure_unique_ids(self.sources)
        return self

    def by_id(self, source_id: str) -> Source:
        for source in self.sources:
            if source.id == source_id:
                return source
        msg = f"no source with id {source_id!r}"
        raise KeyError(msg)

    def filter(
        self,
        *,
        modality: Modality | None = None,
        tier: Tier | None = None,
        redistribution: Redistribution | None = None,
    ) -> tuple[Source, ...]:
        """Select sources matching every supplied criterion."""
        return tuple(
            s
            for s in self.sources
            if (modality is None or s.modality == modality)
            and (tier is None or s.tier == tier)
            and (redistribution is None or s.redistribution == redistribution)
        )


class SiblingRegistry(BaseModel):
    """Berber sibling-language sources, held apart from the Kabyle registry.

    Separate files rather than a `tier` on one file: a tier is a policy that code
    must remember to apply, whereas a separate root model makes ingesting a
    sibling into AƔBALU-Text a type error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    surveyed: date
    sources: tuple[SiblingSource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        _ensure_unique_ids(self.sources)
        return self

    def by_id(self, source_id: str) -> SiblingSource:
        for source in self.sources:
            if source.id == source_id:
                return source
        msg = f"no sibling source with id {source_id!r}"
        raise KeyError(msg)

    def by_language(self, language: str) -> tuple[SiblingSource, ...]:
        """Every source declaring `language`, matched on the base code."""
        wanted = _base_language(language)
        return tuple(
            s for s in self.sources if any(_base_language(x) == wanted for x in s.languages)
        )
