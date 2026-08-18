"""Flattening a trimmed MT checkpoint into a publishable directory.

The failure that matters is a staging that half-succeeds: `hf upload` publishes whatever a
directory holds, so a refused staging must leave nothing behind, and a nested source must
never survive into the output where it would be uploaded a second time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from tools.stage_mt_release import StagingError, resolve_source, stage

KEPT: tuple[int, ...] = (0, 1, 2, 3, 17, 256203)


def write_checkpoint(root: Path, *, kept: tuple[int, ...] = KEPT, vocab: int | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "keep.json").write_text(json.dumps({"keep": list(kept)}), encoding="utf-8")
    (root / "config.json").write_text(
        json.dumps({"vocab_size": len(kept) if vocab is None else vocab}), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "generation_config.json").write_text("{}", encoding="utf-8")
    (root / "training_args.bin").write_bytes(b"trainer state")
    return root


@pytest.fixture
def card(tmp_path: Path) -> Path:
    path = tmp_path / "card.md"
    path.write_text("# Amrouche-1.3B\n", encoding="utf-8")
    return path


def test_flattens_the_final_nesting(tmp_path: Path, card: Path) -> None:
    write_checkpoint(tmp_path / "source" / "final")
    kept, staged = stage(tmp_path / "source", tmp_path / "out", card)

    assert kept == len(KEPT)
    assert {path.name for path in staged} == {
        "model.safetensors",
        "keep.json",
        "config.json",
        "tokenizer.json",
        "generation_config.json",
        "README.md",
    }
    assert (tmp_path / "out" / "model.safetensors").read_bytes() == b"weights"


def test_accepts_an_unnested_source(tmp_path: Path, card: Path) -> None:
    write_checkpoint(tmp_path / "source")
    stage(tmp_path / "source", tmp_path / "out", card)
    assert (tmp_path / "out" / "model.safetensors").is_file()


def test_card_becomes_readme(tmp_path: Path, card: Path) -> None:
    write_checkpoint(tmp_path / "source")
    stage(tmp_path / "source", tmp_path / "out", card)
    assert (tmp_path / "out" / "README.md").read_text(encoding="utf-8") == "# Amrouche-1.3B\n"


def test_training_state_is_dropped(tmp_path: Path, card: Path) -> None:
    write_checkpoint(tmp_path / "source")
    stage(tmp_path / "source", tmp_path / "out", card)
    assert not (tmp_path / "out" / "training_args.bin").exists()


def test_weights_are_linked_not_copied(tmp_path: Path, card: Path) -> None:
    source = write_checkpoint(tmp_path / "source")
    stage(tmp_path / "source", tmp_path / "out", card)
    assert (tmp_path / "out" / "model.safetensors").stat().st_ino == (
        source / "model.safetensors"
    ).stat().st_ino


def test_staging_twice_is_idempotent(tmp_path: Path, card: Path) -> None:
    write_checkpoint(tmp_path / "source")
    _, first = stage(tmp_path / "source", tmp_path / "out", card)
    _, second = stage(tmp_path / "source", tmp_path / "out", card)
    assert [p.name for p in first] == [p.name for p in second]


def test_vocabulary_disagreement_is_refused(tmp_path: Path, card: Path) -> None:
    write_checkpoint(tmp_path / "source", vocab=len(KEPT) + 1)
    with pytest.raises(StagingError, match="vocab_size"):
        stage(tmp_path / "source", tmp_path / "out", card)


def test_missing_keep_is_refused(tmp_path: Path, card: Path) -> None:
    source = write_checkpoint(tmp_path / "source")
    (source / "keep.json").unlink()
    with pytest.raises(StagingError, match=re.escape("keep.json")):
        stage(tmp_path / "source", tmp_path / "out", card)


def test_missing_weights_is_refused(tmp_path: Path, card: Path) -> None:
    source = write_checkpoint(tmp_path / "source")
    (source / "model.safetensors").unlink()
    with pytest.raises(StagingError, match=re.escape("model.safetensors")):
        stage(tmp_path / "source", tmp_path / "out", card)


def test_missing_card_is_refused(tmp_path: Path) -> None:
    write_checkpoint(tmp_path / "source")
    with pytest.raises(StagingError, match="card not found"):
        stage(tmp_path / "source", tmp_path / "out", tmp_path / "absent.md")


def test_out_containing_source_is_refused(tmp_path: Path, card: Path) -> None:
    """The layout as downloaded. Staging in place would upload the weights twice."""
    write_checkpoint(tmp_path / "out" / "final")
    with pytest.raises(StagingError, match="must not contain"):
        stage(tmp_path / "out" / "final", tmp_path / "out", card)


def test_a_refused_staging_writes_nothing(tmp_path: Path, card: Path) -> None:
    write_checkpoint(tmp_path / "source", vocab=len(KEPT) + 1)
    with pytest.raises(StagingError):
        stage(tmp_path / "source", tmp_path / "out", card)
    assert not (tmp_path / "out").exists()


def test_resolve_source_names_both_candidates(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    with pytest.raises(StagingError, match="final"):
        resolve_source(tmp_path / "source")
