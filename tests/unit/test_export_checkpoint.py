"""Staging a checkpoint as a release: what is dropped, and what is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tools.export_checkpoint import (
    ExportError,
    config_of,
    drop_derived,
    load,
    stage,
    untie,
    weights,
)


def card(tmp_path: Path) -> Path:
    path = tmp_path / "card.md"
    path.write_text("# Model\n", encoding="utf-8")
    return path


def checkpoint(tmp_path: Path, **extra: object) -> Path:
    shared = torch.randn(4, 3)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {"embed.weight": shared, "lm_head.weight": shared},
            "config": {"hidden_size": 3},
            **extra,
        },
        path,
    )
    return path


def test_a_tied_tensor_is_written_once_and_recorded_as_dropped() -> None:
    shared = torch.randn(2, 2)
    kept, dropped = untie({"embed.weight": shared, "lm_head.weight": shared})
    assert list(kept) == ["embed.weight"]
    assert dropped == ["lm_head.weight"]


def test_two_tensors_that_merely_have_equal_values_are_both_kept() -> None:
    """Aliasing is storage identity, not equality: two independent copies of the same
    numbers are two weights, and dropping one would publish a broken model."""
    kept, dropped = untie({"a": torch.zeros(2, 2), "b": torch.zeros(2, 2)})
    assert set(kept) == {"a", "b"}
    assert dropped == []


def test_a_derived_table_is_dropped_by_suffix_and_recorded() -> None:
    """A table the module rebuilds from its config is not weights. Twelve of them are 25.2 MB
    against a 125.6 MB model, and `from_pretrained` reports every one as unexpected."""
    tensors = {
        "layers.0.position_indices": torch.zeros(4, 4),
        "layers.1.position_indices": torch.zeros(4, 4),
        "layers.0.weight": torch.zeros(2, 2),
    }
    kept, dropped = drop_derived(tensors, ["position_indices"])

    assert list(kept) == ["layers.0.weight"]
    assert dropped == ["layers.0.position_indices", "layers.1.position_indices"]


def test_nothing_is_dropped_when_no_suffix_is_given() -> None:
    """The default must publish every tensor: a silent drop is a silently broken model."""
    tensors = {"a.position_indices": torch.zeros(2, 2), "b.weight": torch.zeros(2, 2)}
    kept, dropped = drop_derived(tensors, [])

    assert kept == tensors
    assert dropped == []


def test_a_suffix_matches_the_end_of_a_key_not_the_middle() -> None:
    tensors = {"position_indices.weight": torch.zeros(2, 2)}
    kept, dropped = drop_derived(tensors, ["position_indices"])

    assert list(kept) == ["position_indices.weight"]
    assert dropped == []


def test_a_truncated_checkpoint_is_refused_with_its_size_and_path(tmp_path: Path) -> None:
    """`artifacts/asr/best.pt` is 833 MB of a partial volume download and fails exactly
    here. `torch.load` names neither the path nor the cause."""
    path = tmp_path / "partial.pt"
    torch.save({"model_state_dict": {"w": torch.zeros(2)}, "config": {}}, path)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
    with pytest.raises(ExportError, match="not a readable checkpoint"):
        load(path)


def test_a_missing_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="no checkpoint"):
        load(tmp_path / "nowhere.pt")


def test_a_checkpoint_with_no_weights_lists_what_it_did_hold() -> None:
    with pytest.raises(ExportError, match="config"):
        weights({"config": {}, "val_acc": 0.9})


def test_a_checkpoint_with_no_config_cannot_say_what_shape_it_is() -> None:
    with pytest.raises(ExportError, match="no config"):
        config_of({"model_state_dict": {}})


def test_staging_writes_the_weights_the_config_and_the_card(tmp_path: Path) -> None:
    out = tmp_path / "release"
    staged = stage(checkpoint(tmp_path), out, card(tmp_path))
    assert {entry.name for entry in staged} == {
        "README.md",
        "config.json",
        "model.safetensors",
    }
    assert json.loads((out / "config.json").read_text(encoding="utf-8")) == {"hidden_size": 3}


def test_the_manifest_records_what_the_source_actually_contained(tmp_path: Path) -> None:
    """So a card claiming a checkpoint holds an optimizer state can be checked against it."""
    out = tmp_path / "release"
    stage(checkpoint(tmp_path, val_acc=0.99), out, card(tmp_path))
    stats = json.loads((out / "export.stats.json").read_text(encoding="utf-8"))
    assert stats["source_contents"] == ["config", "model_state_dict", "val_acc"]
    assert stats["dropped_tied"] == ["lm_head.weight"]
    assert stats["parameters"] == 12


def test_resume_state_is_dropped_and_named(tmp_path: Path) -> None:
    out = tmp_path / "release"
    stage(
        checkpoint(tmp_path, optimizer_state_dict={"step": 1}, scheduler={"last": 2}),
        out,
        card(tmp_path),
    )
    stats = json.loads((out / "export.stats.json").read_text(encoding="utf-8"))
    assert stats["dropped_state"] == ["optimizer_state_dict", "scheduler"]


def test_extras_are_copied_verbatim(tmp_path: Path) -> None:
    extra = tmp_path / "vocab.json"
    extra.write_text('{"a": 0}', encoding="utf-8")
    out = tmp_path / "release"
    staged = stage(checkpoint(tmp_path), out, card(tmp_path), [extra])
    assert "vocab.json" in {entry.name for entry in staged}
    assert (out / "vocab.json").read_text(encoding="utf-8") == '{"a": 0}'


def test_a_refused_export_leaves_no_directory_that_looks_published(tmp_path: Path) -> None:
    out = tmp_path / "release"
    with pytest.raises(ExportError, match="card not found"):
        stage(checkpoint(tmp_path), out, tmp_path / "missing.md")
    assert not out.exists()


def test_a_missing_extra_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    out = tmp_path / "release"
    with pytest.raises(ExportError, match="extra files not found"):
        stage(checkpoint(tmp_path), out, card(tmp_path), [tmp_path / "absent.klm"])
    assert not out.exists()
