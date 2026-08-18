from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agbalu.registry.loader import RegistryError, load_sibling_registry
from agbalu.registry.models import (
    SIBLING_LANGUAGES,
    SiblingRegistry,
    SiblingSource,
    Source,
)

SIBLING_REGISTRY: Path = Path("resources/sibling_registry.yaml")


def make_sibling(**overrides: Any) -> SiblingSource:  # noqa: ANN401 — test factory
    base: dict[str, Any] = {
        "id": "hf.example.shi",
        "name": "Example",
        "modality": "text",
        "tier": "reference",
        "access": "hf",
        "uri": "https://example.invalid/shi",
        "licence": "cc0-1.0",
        "languages": ["shi_Latn"],
        "size": {"bytes": 1},
        "retrieved": dt.date(2026, 8, 9),
    }
    return SiblingSource.model_validate(base | overrides)


def test_sibling_source_accepts_every_declared_sibling() -> None:
    for language in sorted(SIBLING_LANGUAGES):
        assert make_sibling(languages=[language]).languages == (language,)


def test_sibling_source_accepts_a_script_suffix() -> None:
    assert make_sibling(languages=["tzm_Latn"]).languages == ("tzm_Latn",)


@pytest.mark.parametrize("language", ["kab", "kab_Latn", "kab_Tfng"])
def test_sibling_source_refuses_kabyle(language: str) -> None:
    with pytest.raises(ValidationError, match="belongs in the Kabyle registry"):
        make_sibling(languages=[language])


def test_sibling_source_refuses_kabyle_even_beside_a_sibling() -> None:
    # The dangerous shape: a "Tamazight" source that quietly carries both.
    with pytest.raises(ValidationError, match="belongs in the Kabyle registry"):
        make_sibling(languages=["shi_Latn", "kab_Latn"])


@pytest.mark.parametrize("language", ["eng", "fra", "ber"])
def test_sibling_source_refuses_a_non_sibling(language: str) -> None:
    with pytest.raises(ValidationError, match="no Berber sibling language"):
        make_sibling(languages=[language])


@pytest.mark.parametrize("language", ["kabyle", "SHI", "shi_latn", "s"])
def test_sibling_source_refuses_a_malformed_language_code(language: str) -> None:
    # Caught by the LangCode pattern, before the sibling rule is ever reached.
    with pytest.raises(ValidationError, match="should match pattern"):
        make_sibling(languages=[language])


def test_kabyle_source_still_refuses_a_sibling_only_source() -> None:
    with pytest.raises(ValidationError, match="does not belong in this registry"):
        Source.model_validate(
            {
                "id": "hf.example.shi",
                "name": "Example",
                "modality": "text",
                "tier": "reference",
                "access": "hf",
                "uri": "https://example.invalid/shi",
                "licence": "cc0-1.0",
                "languages": ["shi_Latn"],
                "size": {"bytes": 1},
                "retrieved": dt.date(2026, 8, 9),
            }
        )


def test_sibling_registry_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate source id"):
        SiblingRegistry.model_validate(
            {
                "version": "1.0.0",
                "surveyed": dt.date(2026, 8, 9),
                "sources": [
                    make_sibling().model_dump(mode="json"),
                    make_sibling().model_dump(mode="json"),
                ],
            }
        )


def test_sibling_registry_requires_at_least_one_source() -> None:
    with pytest.raises(ValidationError):
        SiblingRegistry.model_validate(
            {"version": "1.0.0", "surveyed": dt.date(2026, 8, 9), "sources": []}
        )


def test_by_language_matches_on_the_base_code() -> None:
    registry = SiblingRegistry.model_validate(
        {
            "version": "1.0.0",
            "surveyed": dt.date(2026, 8, 9),
            "sources": [
                make_sibling(id="a", languages=["shi_Latn"]).model_dump(mode="json"),
                make_sibling(id="b", languages=["rif_Latn"]).model_dump(mode="json"),
            ],
        }
    )
    assert [s.id for s in registry.by_language("shi")] == ["a"]
    assert [s.id for s in registry.by_language("shi_Latn")] == ["a"]
    assert registry.by_language("taq") == ()


def test_loader_reports_a_missing_file() -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_sibling_registry(Path("resources/does-not-exist.yaml"))


def test_loader_rejects_a_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="must be a YAML mapping"):
        load_sibling_registry(path)


def test_loader_tolerates_a_bom(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    body = (
        "version: '1.0.0'\n"
        "surveyed: 2026-08-09\n"
        "sources:\n"
        "  - id: hf.example.shi\n"
        "    name: Example\n"
        "    modality: text\n"
        "    tier: reference\n"
        "    access: hf\n"
        "    uri: https://example.invalid/shi\n"
        "    licence: cc0-1.0\n"
        "    languages: [shi_Latn]\n"
        "    size: { bytes: 1 }\n"
        "    retrieved: 2026-08-09\n"
    )
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    assert len(load_sibling_registry(path).sources) == 1


@pytest.mark.integration
def test_the_shipped_sibling_registry_validates() -> None:
    registry = load_sibling_registry(SIBLING_REGISTRY)
    assert registry.sources
    for source in registry.sources:
        assert not any(language.startswith("kab") for language in source.languages)
        # Sibling bytes exist to be classified against, never released.
        assert source.tier == "reference"
