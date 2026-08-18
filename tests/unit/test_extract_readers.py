from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from agbalu.extract.readers import ReadError, read_records


def test_tsv_with_header(tmp_path: Path) -> None:
    path = tmp_path / "a.tsv"
    path.write_text("id\tkab\n1\tazul\n2\tthanemirth\n", encoding="utf-8")
    assert list(read_records(path)) == [
        {"id": "1", "kab": "azul"},
        {"id": "2", "kab": "thanemirth"},
    ]


def test_tsv_without_header_gets_positional_names(tmp_path: Path) -> None:
    path = tmp_path / "a.tsv"
    path.write_text("Azul fell-awen ay imdukkal\tHello there friends\n", encoding="utf-8")
    rows = list(read_records(path))
    assert rows == [{"col0": "Azul fell-awen ay imdukkal", "col1": "Hello there friends"}]


def test_quote_none_keeps_lines_after_an_embedded_quote(tmp_path: Path) -> None:
    """The defect that silently loses 5,904 seed sentences (CLAUDE.md 2.1a)."""
    path = tmp_path / "a.tsv"
    path.write_text('id\tkab\n1\tyenna-d "azul"\n2\ttaqbaylit\n3\tazekka\n', encoding="utf-8")
    rows = list(read_records(path))
    assert len(rows) == 3
    assert rows[0]["kab"] == 'yenna-d "azul"'


def test_bom_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "a.tsv"
    path.write_bytes("﻿id\tkab\n1\tazul\n".encode())
    assert next(iter(read_records(path))) == {"id": "1", "kab": "azul"}


def test_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "a.tsv"
    path.write_bytes(b"id\tkab\r\n1\tazul\r\n2\tazekka\r\n")
    rows = list(read_records(path))
    assert [r["kab"] for r in rows] == ["azul", "azekka"]


def test_jsonl_nested_objects_are_flattened(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps({"translation": {"kab": "azul", "en": "hello"}, "id": 7}) + "\n",
        encoding="utf-8",
    )
    assert list(read_records(path)) == [
        {"translation.kab": "azul", "translation.en": "hello", "id": "7"}
    ]


def test_jsonl_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"kab": "azul"}\nnot json at all\n{"kab": "azekka"}\n', encoding="utf-8")
    assert [r["kab"] for r in read_records(path)] == ["azul", "azekka"]


def test_jsonl_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('\n\n{"kab": "azul"}\n\n', encoding="utf-8")
    assert list(read_records(path)) == [{"kab": "azul"}]


def test_json_array_and_single_object(tmp_path: Path) -> None:
    array = tmp_path / "a.json"
    array.write_text(json.dumps([{"kab": "azul"}, {"kab": "azekka"}]), encoding="utf-8")
    assert len(list(read_records(array))) == 2

    single = tmp_path / "b.json"
    single.write_text(json.dumps({"kab": "azul"}), encoding="utf-8")
    assert list(read_records(single)) == [{"kab": "azul"}]


def test_zip_reads_only_the_kab_member(tmp_path: Path) -> None:
    path = tmp_path / "en-kab.txt.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("X.en-kab.en", "hello\nworld\n")
        archive.writestr("X.en-kab.kab", "azul\nddunit\n")
        archive.writestr("README", "ignore me")
    assert [r["kab"] for r in read_records(path)] == ["azul", "ddunit"]


def test_zip_without_a_kab_member_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("only.en", "hello\n")
    assert list(read_records(path)) == []


def test_corrupt_zip_raises_read_error(tmp_path: Path) -> None:
    path = tmp_path / "a.zip"
    path.write_bytes(b"this is not a zip file")
    with pytest.raises(ReadError):
        list(read_records(path))


def test_txt_yields_one_record_per_non_empty_line(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("azul\n\n  ddunit  \n", encoding="utf-8")
    assert list(read_records(path)) == [{"text": "azul"}, {"text": "ddunit"}]


def test_invalid_utf8_is_replaced_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"azul\n\xff\xfe bad bytes\n")
    assert len(list(read_records(path))) == 2


def test_unknown_suffix_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(b"\x89PNG\r\n")
    assert list(read_records(path)) == []


def test_empty_file_yields_nothing(tmp_path: Path) -> None:
    for name in ("a.tsv", "a.jsonl", "a.txt", "a.json"):
        path = tmp_path / name
        path.write_text("", encoding="utf-8")
        assert list(read_records(path)) == []
