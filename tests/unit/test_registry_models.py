from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from agbalu.registry.models import (
    Registry,
    Source,
    SourceSize,
    redistribution_class,
)


def make_source(**overrides: Any) -> Source:  # noqa: ANN401 — test factory takes arbitrary fields
    base: dict[str, Any] = {
        "id": "hf.example.kab",
        "name": "Example",
        "modality": "text",
        "tier": "core",
        "access": "hf",
        "uri": "https://example.invalid/kab",
        "licence": "cc0-1.0",
        "languages": ("kab",),
        "size": {"rows": 10},
        "retrieved": dt.date(2026, 8, 5),
    }
    base.update(overrides)
    return Source.model_validate(base)


class TestSourceSize:
    def test_accepts_a_single_measure(self) -> None:
        assert SourceSize(rows=1).rows == 1

    def test_rejects_no_measures(self) -> None:
        with pytest.raises(ValidationError, match="at least one measured field"):
            SourceSize()

    def test_rejects_all_explicit_nones(self) -> None:
        with pytest.raises(ValidationError, match="at least one measured field"):
            SourceSize(rows=None, bytes=None, hours=None)

    @pytest.mark.parametrize("field", ["rows", "bytes", "sentences", "tokens", "articles"])
    def test_rejects_negative_counts(self, field: str) -> None:
        with pytest.raises(ValidationError):
            SourceSize.model_validate({field: -1})

    def test_rejects_negative_hours(self) -> None:
        with pytest.raises(ValidationError):
            SourceSize(hours=-0.1)

    def test_zero_is_a_valid_measure(self) -> None:
        # A source measured at zero rows is a fact worth recording, not an error.
        assert SourceSize(rows=0).rows == 0

    def test_is_frozen(self) -> None:
        size = SourceSize(rows=1)
        with pytest.raises(ValidationError):
            size.rows = 2

    def test_forbids_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            SourceSize.model_validate({"rows": 1, "megabytes": 3})


class TestSource:
    def test_requires_kab_in_languages(self) -> None:
        with pytest.raises(ValidationError, match="does not include 'kab'"):
            make_source(languages=("shi", "eng"))

    @pytest.mark.parametrize("code", ["kab", "kab_Latn", "kab_Tfng"])
    def test_script_suffixed_kabyle_satisfies_the_requirement(self, code: str) -> None:
        assert make_source(languages=(code,)).languages == (code,)

    @pytest.mark.parametrize("sibling", ["shi", "tzm", "rif", "taq"])
    def test_berber_siblings_alone_are_rejected(self, sibling: str) -> None:
        # Scope discipline: related Berber languages are not Kabyle.
        with pytest.raises(ValidationError, match="does not include 'kab'"):
            make_source(languages=(sibling,))

    def test_rejects_empty_languages(self) -> None:
        with pytest.raises(ValidationError):
            make_source(languages=())

    def test_excluded_source_must_explain_itself(self) -> None:
        with pytest.raises(ValidationError, match="gives no reason"):
            make_source(tier="excluded", notes="")

    def test_excluded_source_with_notes_is_valid(self) -> None:
        assert make_source(tier="excluded", notes="too small").tier == "excluded"

    @pytest.mark.parametrize("bad", ["KAB", "k", "kabyle", "kab_latn", "kab-Latn", "", "kab_LATN"])
    def test_rejects_malformed_language_codes(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_source(languages=("kab", bad))

    @pytest.mark.parametrize("good", ["kab_Latn", "kab_Tfng", "en", "eng", "fra"])
    def test_accepts_wellformed_language_codes(self, good: str) -> None:
        assert good in make_source(languages=("kab", good)).languages

    @pytest.mark.parametrize("bad", ["-leading", ".dot", "Upper", "has space", "", "a" * 97])
    def test_rejects_malformed_ids(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_source(id=bad)

    @pytest.mark.parametrize("bad", ["text ", "TEXT", "audio", ""])
    def test_rejects_unknown_modality(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_source(modality=bad)

    def test_rejects_empty_licence(self) -> None:
        with pytest.raises(ValidationError):
            make_source(licence="")

    def test_forbids_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            make_source(downloads=42)

    def test_is_frozen(self) -> None:
        source = make_source()
        with pytest.raises(ValidationError):
            source.name = "renamed"

    def test_is_parallel_follows_modality(self) -> None:
        assert make_source(modality="parallel", languages=("kab",)).is_parallel

    def test_is_parallel_follows_multiple_languages(self) -> None:
        assert make_source(modality="text", languages=("kab", "eng")).is_parallel

    def test_monolingual_text_is_not_parallel(self) -> None:
        assert not make_source(modality="text", languages=("kab",)).is_parallel

    def test_notes_length_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            make_source(notes="x" * 1001)


class TestChecksum:
    HEX = "e" * 64

    def test_local_source_requires_a_checksum(self) -> None:
        with pytest.raises(ValidationError, match="must declare a checksum"):
            make_source(access="manual", uri="data/raw/x.tsv")

    def test_local_source_with_checksum_is_valid(self) -> None:
        source = make_source(access="manual", uri="data/raw/x.tsv", checksum=self.HEX)
        assert source.checksum == self.HEX

    def test_remote_source_may_omit_a_checksum(self) -> None:
        assert make_source(access="hf").checksum is None

    @pytest.mark.parametrize(
        "bad",
        [
            "E" * 64,  # uppercase hex
            "e" * 63,  # too short
            "e" * 65,  # too long
            "g" * 64,  # not hex
            "",
            "sha256:" + "e" * 64,
            " " + "e" * 64,
        ],
    )
    def test_rejects_malformed_checksums(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            make_source(checksum=bad)


class TestRedistributionClass:
    @pytest.mark.parametrize(
        ("licence", "expected"),
        [
            ("cc0-1.0", "permissive"),
            ("mit", "permissive"),
            ("apache-2.0", "permissive"),
            ("odc-by", "permissive"),
            # ODbL §4.4 obliges a publicly used derivative database to be ODbL as well, so
            # it belongs beside CC-BY-SA. Classed permissive it would have put 8,554
            # toponym entries into a "permissive-only" lexicon release.
            ("odbl", "share-alike"),
            ("cc-by-sa-4.0", "share-alike"),
            ("gfdl", "share-alike"),
            ("mpl-2.0", "share-alike"),
            ("cc-by-nc-4.0", "non-commercial"),
            ("other", "unclear"),
            ("unknown", "unclear"),
            ("cc", "unclear"),
            ("", "unclear"),
        ],
    )
    def test_mapping(self, licence: str, expected: str) -> None:
        assert redistribution_class(licence) == expected

    @pytest.mark.parametrize("variant", ["  MIT  ", "MIT", "Mit", "\tmit\n"])
    def test_is_case_and_whitespace_insensitive(self, variant: str) -> None:
        assert redistribution_class(variant) == "permissive"

    def test_unrecognised_licence_is_unclear_not_permissive(self) -> None:
        # Failing open here would leak restricted data into a permissive release.
        assert redistribution_class("some-bespoke-eula") == "unclear"

    def test_exposed_on_source(self) -> None:
        assert make_source(licence="cc-by-nc-4.0").redistribution == "non-commercial"


class TestRegistry:
    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(ValidationError, match="duplicate source id"):
            Registry(
                version="1.0.0",
                surveyed=dt.date(2026, 8, 5),
                sources=(make_source(), make_source(name="Other")),
            )

    def test_rejects_empty_sources(self) -> None:
        with pytest.raises(ValidationError):
            Registry(version="1.0.0", surveyed=dt.date(2026, 8, 5), sources=())

    @pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0-rc1", "1", ""])
    def test_rejects_malformed_version(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            Registry(version=bad, surveyed=dt.date(2026, 8, 5), sources=(make_source(),))

    def test_by_id_returns_the_source(self) -> None:
        registry = Registry(version="1.0.0", surveyed=dt.date(2026, 8, 5), sources=(make_source(),))
        assert registry.by_id("hf.example.kab").name == "Example"

    def test_by_id_raises_on_miss(self) -> None:
        registry = Registry(version="1.0.0", surveyed=dt.date(2026, 8, 5), sources=(make_source(),))
        with pytest.raises(KeyError, match="absent"):
            registry.by_id("absent")

    def test_filter_combines_criteria_conjunctively(self) -> None:
        registry = Registry(
            version="1.0.0",
            surveyed=dt.date(2026, 8, 5),
            sources=(
                make_source(id="a", modality="text", tier="core", licence="mit"),
                make_source(id="b", modality="text", tier="supplementary", licence="mit"),
                make_source(id="c", modality="speech", tier="core", licence="cc-by-nc-4.0"),
            ),
        )
        assert [s.id for s in registry.filter(modality="text")] == ["a", "b"]
        assert [s.id for s in registry.filter(modality="text", tier="core")] == ["a"]
        assert [s.id for s in registry.filter(redistribution="non-commercial")] == ["c"]
        assert registry.filter(modality="speech", tier="supplementary") == ()

    def test_filter_without_criteria_returns_everything(self) -> None:
        registry = Registry(version="1.0.0", surveyed=dt.date(2026, 8, 5), sources=(make_source(),))
        assert len(registry.filter()) == 1
