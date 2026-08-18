from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.registry.loader import RegistryError, load_registry

VALID = """
version: "1.0.0"
surveyed: 2026-08-05
sources:
  - id: hf.example.kab
    name: Example
    modality: text
    tier: core
    access: hf
    uri: https://example.invalid/kab
    licence: cc0-1.0
    languages: [kab]
    size: { rows: 10 }
    retrieved: 2026-08-05
"""


def write(tmp_path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(text, encoding=encoding)
    return path


def test_loads_a_valid_registry(tmp_path: Path) -> None:
    registry = load_registry(write(tmp_path, VALID))
    assert registry.version == "1.0.0"
    assert registry.by_id("hf.example.kab").modality == "text"


def test_tolerates_a_utf8_bom(tmp_path: Path) -> None:
    # The seed Tatoeba export ships with a BOM; assume any input may.
    registry = load_registry(write(tmp_path, VALID, encoding="utf-8-sig"))
    assert len(registry.sources) == 1


def test_preserves_non_ascii_kabyle_orthography(tmp_path: Path) -> None:
    text = VALID.replace("name: Example", "name: Taqbaylit — ɣ ɛ ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ")
    registry = load_registry(write(tmp_path, text))
    assert registry.sources[0].name.endswith("ţ")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="registry not found"):
        load_registry(tmp_path / "nope.yaml")


def test_directory_instead_of_file_raises(tmp_path: Path) -> None:
    with pytest.raises((RegistryError, IsADirectoryError, PermissionError)):
        load_registry(tmp_path)


def test_non_utf8_bytes_raise(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_bytes(b"version: \xff\xfe not utf-8")
    with pytest.raises(RegistryError, match="not valid UTF-8"):
        load_registry(path)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not valid YAML"):
        load_registry(write(tmp_path, "version: '1.0.0'\n  bad: [unclosed\n"))


@pytest.mark.parametrize(
    ("content", "kind"),
    [("", "NoneType"), ("- a\n- b\n", "list"), ("just a string\n", "str"), ("42\n", "int")],
)
def test_non_mapping_document_raises(tmp_path: Path, content: str, kind: str) -> None:
    with pytest.raises(RegistryError, match=f"must be a YAML mapping, got {kind}"):
        load_registry(write(tmp_path, content))


def test_schema_violation_raises_with_count(tmp_path: Path) -> None:
    text = VALID.replace("languages: [kab]", "languages: [shi]")
    with pytest.raises(RegistryError, match="failed validation"):
        load_registry(write(tmp_path, text))


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    text = VALID.replace("    retrieved: 2026-08-05", "    retrieved: 2026-08-05\n    stars: 3")
    with pytest.raises(RegistryError, match="failed validation"):
        load_registry(write(tmp_path, text))


def test_yaml_bomb_is_not_expanded(tmp_path: Path) -> None:
    # safe_load must refuse aliases-as-code; a billion-laughs payload must not hang.
    bomb = "a: &a [x,x,x,x,x,x,x,x,x]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]\nc: [*b,*b,*b]\n"
    with pytest.raises(RegistryError, match="failed validation"):
        load_registry(write(tmp_path, bomb))


def test_arbitrary_python_tags_are_refused(tmp_path: Path) -> None:
    payload = "version: !!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(RegistryError, match="not valid YAML"):
        load_registry(write(tmp_path, payload))
