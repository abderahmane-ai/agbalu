"""The licence composition of a built artifact.

Tested where the other `tools/` scripts are not, because this one's output is a public
claim: it is what the model card states about text the weights were trained on, and an
over-counted permissive share is a licence assertion nobody can retract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.licence_composition import compose

REGISTRY = """
version: "1.0.0"
surveyed: 2026-08-09
sources:
  - id: hf.permissive.kab
    name: Permissive
    modality: text
    tier: core
    access: hf
    uri: https://example.invalid/a
    licence: cc0-1.0
    languages: [kab]
    size: { rows: 10 }
    retrieved: 2026-08-09
  - id: hf.unclear.kab
    name: Unclear
    modality: text
    tier: supplementary
    access: hf
    uri: https://example.invalid/b
    licence: other
    languages: [kab]
    size: { rows: 10 }
    retrieved: 2026-08-09
  - id: hf.noncommercial.kab
    name: Non-commercial
    modality: text
    tier: core
    access: hf
    uri: https://example.invalid/c
    licence: cc-by-nc-4.0
    languages: [kab]
    size: { rows: 10 }
    retrieved: 2026-08-09
"""


def write_pair(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    registry = tmp_path / "registry.yaml"
    registry.write_text(REGISTRY, encoding="utf-8")
    stats = tmp_path / "artifact.stats.json"
    stats.write_text(json.dumps({"sources": rows}), encoding="utf-8")
    return stats, registry


def test_kept_counts_are_grouped_by_the_source_licence(tmp_path: Path) -> None:
    stats, registry = write_pair(
        tmp_path,
        [
            {"source_id": "hf.permissive.kab", "kept": 100},
            {"source_id": "hf.unclear.kab", "kept": 250},
            {"source_id": "hf.noncommercial.kab", "kept": 50},
        ],
    )
    result = compose(stats, registry)
    assert result["total"] == 400
    assert result["by_redistribution"] == {"unclear": 250, "permissive": 100, "non-commercial": 50}
    assert result["by_tier"] == {"core": 150, "supplementary": 250}


def test_two_sources_under_one_licence_are_summed(tmp_path: Path) -> None:
    stats, registry = write_pair(
        tmp_path,
        [
            {"source_id": "hf.permissive.kab", "kept": 7},
            {"source_id": "hf.permissive.kab", "kept": 5},
        ],
    )
    assert compose(stats, registry)["by_licence"] == {"cc0-1.0": 12}


def test_a_source_contributing_nothing_does_not_inflate_a_tier(tmp_path: Path) -> None:
    """`kept` is zero for a source whose every row deduplicated away, and counting its
    declared size instead would credit the corpus with text it does not contain."""
    stats, registry = write_pair(
        tmp_path,
        [
            {"source_id": "hf.permissive.kab", "kept": 0},
            {"source_id": "hf.unclear.kab", "kept": 10},
        ],
    )
    result = compose(stats, registry)
    assert result["total"] == 10
    assert result["by_redistribution"]["permissive"] == 0


def test_a_source_absent_from_the_registry_is_named_not_dropped(tmp_path: Path) -> None:
    """Silently skipping it would shrink the denominator and overstate every share."""
    stats, registry = write_pair(
        tmp_path,
        [
            {"source_id": "hf.permissive.kab", "kept": 10},
            {"source_id": "hf.ghost.kab", "kept": 90},
        ],
    )
    result = compose(stats, registry)
    assert result["unregistered"] == ["hf.ghost.kab"]
    assert result["total"] == 10


def test_an_artifact_with_no_sources_does_not_divide_by_zero(tmp_path: Path) -> None:
    stats, registry = write_pair(tmp_path, [])
    result = compose(stats, registry)
    assert result["total"] == 0
    assert result["by_licence"] == {}


def test_groups_are_ordered_largest_first(tmp_path: Path) -> None:
    """The card reads the first row as the dominant term."""
    stats, registry = write_pair(
        tmp_path,
        [
            {"source_id": "hf.permissive.kab", "kept": 1},
            {"source_id": "hf.unclear.kab", "kept": 99},
        ],
    )
    assert next(iter(compose(stats, registry)["by_redistribution"])) == "unclear"


@pytest.mark.integration
def test_the_real_corpus_composition_sums_to_its_own_total() -> None:
    """The published numbers come from this path; a drift between the stats file and the
    registry would surface here rather than in a model card."""
    stats = Path("data/processed/text/agbalu-text-v1.stats.json")
    if not stats.is_file():
        pytest.skip("AƔBALU-Text v1 not built; run `make extract`")
    result = compose(stats)
    assert not result["unregistered"]
    assert sum(result["by_redistribution"].values()) == result["total"]
    assert sum(result["by_licence"].values()) == result["total"]
    assert sum(result["by_tier"].values()) == result["total"]
