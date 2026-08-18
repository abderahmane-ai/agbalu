"""Split construction, and the exclusions that make the evaluation honest.

The filter and the decontamination are tested on the shapes the real corpus has and the
fixtures do not: a sentence that appears on both sides, a source whose boundaries a miner
chose, and a transcript that carries no final mark at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.punctuation.corpus import (
    WRITTEN_SPLITS,
    CorpusError,
    build,
    is_wellformed,
    read_split,
    scan_text_corpus,
)
from agbalu.punctuation.labels import collation_key

TRUSTED = "local.tatoeba-kab-mono"
UNTRUSTED = "opus.nllb-kab"


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.mark.parametrize(
    "text",
    [
        "Ɣef wannect-a i d-nusa.",
        "Ţ is a real Kabyle letter here.",
        "Ṯamurt-nneɣ tettwali-t akken iwata.",
        "Azul fell-awen, amek tellid?",
    ],
)
def test_wellformed_accepts_kabyle_sentences(text: str) -> None:
    assert is_wellformed(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Short.",
        "azul fell-awen amek tellid?",
        "Azul fell-awen amek tellid",
        "Yettwalqem %1$d wass aya n tikkelt.",
        "Awal Aɛeggun Aɛeggun",
        "Mḥemmed mmi-s n taklit [yeqqel-d] yuɣal.",
        "Ad nruḥ 12 34 56 78 90 12 34.",
        "A b.",
        "Ɣ " + "ay " * 61 + ".",
    ],
)
def test_wellformed_rejects_non_sentences(text: str) -> None:
    assert not is_wellformed(text)


def test_scan_keeps_every_key_but_trains_only_on_trusted_sentences(tmp_path: Path) -> None:
    path = tmp_path / "text.jsonl"
    write_jsonl(
        path,
        [
            {"text": "Azul fell-awen, amek tellid?", "source": TRUSTED},
            {"text": "Ẓriɣ belli ad d-yas azekka.", "source": UNTRUSTED},
            {"text": "azul fell-awen", "source": TRUSTED},
            {"text": "", "source": TRUSTED},
        ],
    )
    result = scan_text_corpus(path)

    assert result.records == 4
    assert result.wellformed == 2
    assert result.untrusted == 1
    assert [row.text for row in result.rows] == ["Azul fell-awen, amek tellid?"]
    assert result.per_source[TRUSTED] == 3
    assert result.kept_per_source[UNTRUSTED] == 1
    assert collation_key("azul fell-awen") in result.keys


def test_scan_reports_a_missing_corpus_by_name(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="no text corpus"):
        scan_text_corpus(tmp_path / "absent.jsonl")


def build_fixture(tmp_path: Path) -> Path:
    """A text corpus and three audio splits that exercise every exclusion."""
    write_jsonl(
        tmp_path / "text.jsonl",
        [
            {"text": "Yiwen wass ad nemlil deg temdint.", "source": TRUSTED},
            {"text": "Tazmert-ik tegguma ad tban.", "source": TRUSTED},
            {"text": "Ur yessaweḍ ad d-yawey akayad-nni.", "source": TRUSTED},
            {"text": "Ẓriɣ belli ad d-yas azekka ɣer da.", "source": UNTRUSTED},
        ],
    )
    speech = tmp_path / "speech"
    write_jsonl(speech / "train.jsonl", [{"text": "Tegzem yiwen n useklu."}])
    write_jsonl(
        speech / "dev.jsonl",
        [{"text": "Anida tellid ass-a?"}, {"text": "Tegzem yiwen n useklu."}],
    )
    write_jsonl(
        speech / "test.jsonl",
        [
            {"text": "D acu i txedmed assa ɣer da?"},
            {"text": "Ur yessaweḍ ad d-yawey akayad-nni."},
            {"text": "Tegzem yiwen n useklu."},
            {"text": "ur tt-id-iṣaḥ ad terr awal"},
        ],
    )
    return speech


def test_build_excludes_for_each_reason_separately(tmp_path: Path) -> None:
    speech = build_fixture(tmp_path)
    stats = build(tmp_path / "text.jsonl", speech, tmp_path / "out")

    test = stats["excluded"]["test"]
    assert test["clips"] == 4
    assert test["no_final_mark"] == 1
    assert test["in_text_corpus"] == 1
    assert test["in_speech_train"] == 1
    assert test["kept"] == 1

    dev = stats["excluded"]["dev"]
    assert dev["in_speech_train"] == 1
    assert dev["kept"] == 1


def test_build_never_trains_on_a_held_out_sentence(tmp_path: Path) -> None:
    speech = build_fixture(tmp_path)
    build(tmp_path / "text.jsonl", speech, tmp_path / "out")

    train = {collation_key(row.text) for row in read_split(tmp_path / "out" / "train.jsonl")}
    for split in ("dev", "test"):
        held = {collation_key(row.text) for row in read_split(tmp_path / "out" / f"{split}.jsonl")}
        assert held, f"{split} is empty, so the check cannot fail"
        assert not train & held


def test_build_adds_speech_train_and_drops_the_untrusted_source(tmp_path: Path) -> None:
    speech = build_fixture(tmp_path)
    stats = build(tmp_path / "text.jsonl", speech, tmp_path / "out")

    sources = {row.source for row in read_split(tmp_path / "out" / "train.jsonl")}
    assert "speech.train" in sources
    assert UNTRUSTED not in sources
    assert stats["speech_added"] == 1
    assert stats["untrusted_dropped"] == 1


def test_build_writes_exactly_the_declared_splits(tmp_path: Path) -> None:
    """`WRITTEN_SPLITS` is what `modal_app.punctuation.upload_punctuation` stages.

    A second copy of this list omitted `ood` and the evaluation failed on the volume, so the
    file set on disk is asserted against the constant rather than against a literal.
    """
    speech = build_fixture(tmp_path)
    build(tmp_path / "text.jsonl", speech, tmp_path / "out")

    written = {path.name for path in (tmp_path / "out").glob("*.jsonl")}
    assert written == {f"{split}.jsonl" for split in WRITTEN_SPLITS}


def test_build_writes_the_label_distributions(tmp_path: Path) -> None:
    speech = build_fixture(tmp_path)
    stats = build(tmp_path / "text.jsonl", speech, tmp_path / "out")

    train = stats["splits"]["train"]
    assert train["rows"] > 0
    assert train["words"] == sum(train["punctuation"].values())
    assert set(train["case"]) <= {"LOWER", "UPPER_INIT"}


def test_read_split_names_the_command_that_creates_it(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="make punctuation TASK=corpus"):
        read_split(tmp_path / "absent.jsonl")


def test_missing_speech_split_is_reported_by_name(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "text.jsonl", [{"text": "Yiwen wass ad nemlil.", "source": TRUSTED}])
    with pytest.raises(CorpusError, match="no speech split"):
        build(tmp_path / "text.jsonl", tmp_path / "empty", tmp_path / "out")
