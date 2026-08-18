"""The architecture record a restorer checkpoint does not carry.

The checkpoint holds `model`, `optimizer`, `state` and `rng` and no config, so these fields
are the only thing standing between the published weights and a repository nothing can
construct. Each is asserted against the constant the trainer actually built with, not against
a literal copied from the config.
"""

from __future__ import annotations

import json
from pathlib import Path

from agbalu.hub.belaid.configuration_belaid import BelaidConfig
from agbalu.model.config import PRESETS
from agbalu.punctuation.labels import CASE, PUNCTUATION
from agbalu.punctuation.release import PRESET, config, write_config


def test_the_shape_is_the_preset_the_trainer_builds_with() -> None:
    """A hand-copied width would publish weights that load into the wrong architecture."""
    written = config()
    preset = PRESETS[PRESET]

    assert written["hidden_size"] == preset.hidden_size
    assert written["num_hidden_layers"] == preset.num_hidden_layers
    assert written["num_attention_heads"] == preset.num_attention_heads
    assert written["intermediate_size"] == preset.intermediate_size
    assert written["vocab_size"] == preset.vocab_size
    assert written["position_bucket_size"] == preset.position_bucket_size


def test_the_label_sets_are_the_ones_the_heads_are_sized_by() -> None:
    written = config()
    assert written["punctuation_labels"] == list(PUNCTUATION)
    assert written["case_labels"] == list(CASE)


def test_the_config_constructs_the_published_class() -> None:
    """The published `BelaidConfig` must accept every field this writes, by name."""
    loaded = BelaidConfig.from_dict(config())

    assert loaded.model_type == "belaid"
    assert len(loaded.punctuation_labels) == len(PUNCTUATION)
    assert len(loaded.case_labels) == len(CASE)
    assert loaded.hidden_size % loaded.num_attention_heads == 0


def test_writing_creates_the_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deeper" / "config.json"
    assert write_config(out) == out
    assert json.loads(out.read_text(encoding="utf-8"))["model_type"] == "belaid"
