"""Unit tests for byte sequence transformation, collation, and cell sampling."""

from __future__ import annotations

import pytest

from agbalu.dihya.config import BYTE_OFFSET, ModelConfig
from agbalu.dihya.dataset import (
    CellSampler,
    Example,
    TextDataset,
    collate_examples,
    text_to_bytes,
)


def test_text_to_bytes_mapping() -> None:
    """Verify byte token offset and truncation."""
    text = "Azul"
    tokens = text_to_bytes(text, max_length=10)
    assert len(tokens) == 4
    # 'A' is ASCII 65 -> token ID 65 + BYTE_OFFSET (67)
    assert tokens[0] == ord("A") + BYTE_OFFSET


def test_collate_examples_padding() -> None:
    """Verify collation dynamically pads to batch maximum and computes masks."""
    config = ModelConfig(max_length=64)
    examples = [
        Example(text="Azul", label="kab_Latn", domain="everyday", source="test"),
        Example(
            text="Mayemmi ur teccid ticemrarin?",
            label="rif_Latn",
            domain="everyday",
            source="test",
        ),
    ]

    batch = collate_examples(examples, config, domain_map={"everyday": 0}, group_map={})
    assert batch.input_ids.shape[0] == 2
    assert batch.pad_mask.shape == batch.input_ids.shape
    assert batch.labels.shape == (2,)
    assert batch.script_labels.shape == (2,)
    assert batch.filter_labels.shape == (2,)
    assert batch.filter_labels[0] == 1.0  # Amazigh


def test_cell_sampler_batch_generation() -> None:
    """Verify CellSampler produces stratified batches across cells."""
    examples = [
        Example(text=f"kab_{i}", label="kab_Latn", domain="web", source="s1") for i in range(20)
    ] + [
        Example(text=f"shy_{i}", label="shy_Latn", domain="scripture", source="s2")
        for i in range(20)
    ]
    dataset = TextDataset(examples)
    sampler = CellSampler(dataset, groups_per_batch=2, samples_per_group=4)

    assert len(sampler) > 0
    first_batch = next(iter(sampler))
    assert len(first_batch) == 8  # 2 groups * 4 samples


def test_cells_use_declared_composite_label_not_digit_heuristics() -> None:
    ex = Example("Azul 23 57", "kab_Latn", "web", "test")
    assert ex.group_key == "kab_Latn/web"


@pytest.mark.parametrize(
    ("text", "label"), [("Azul", "unknown"), ("ⴰⵣⵓⵍ", "shi_Latn"), ("", "kab_Latn")]
)
def test_invalid_example_fails(text: str, label: str) -> None:
    with pytest.raises(ValueError, match="source='test'"):
        Example(text, label, "web", "test")


def test_collation_respects_config_class_order_and_rejects_mapping_typos() -> None:
    config = ModelConfig(classes=("rif_Latn", "kab_Latn"))
    example = Example("Azul", "kab_Latn", "web", "test")
    assert collate_examples([example], config, {}, {}).labels.tolist() == [1]
    with pytest.raises(KeyError, match="web"):
        collate_examples([example], config, {"typo": 0}, {})


def test_epoch_replay_and_actual_batch_size() -> None:
    dataset = TextDataset([Example(f"Azul {i}", "kab_Latn", "web", "test") for i in range(20)])
    sampler = CellSampler(dataset, groups_per_batch=16, samples_per_group=2)
    assert sampler.batch_size == 2
    assert len(sampler) == 10
    sampler.set_epoch(1, 42)
    batches = list(sampler)
    sampler.set_epoch(1, 42)
    assert list(sampler) == batches
    sampler.set_epoch(2, 42)
    assert list(sampler) != batches


def test_tiny_cells_share_sampling_mass_without_changing_original_domains() -> None:
    rows = [Example(f"Azul fell awen {index}", "kab_Latn", "web", "test") for index in range(100)]
    rows.append(Example("Azul fell awen interface", "kab_Latn", "interface", "test"))
    sampler = CellSampler(TextDataset(rows), 2, 4, min_group_support=20)
    assert sampler.cells == ["kab_Latn/web"]
    assert sampler.cell_aliases["kab_Latn/interface"] == "kab_Latn/web"
    assert len(sampler.cell_indices["kab_Latn/web"]) == 101
    assert rows[-1].domain == "interface"
