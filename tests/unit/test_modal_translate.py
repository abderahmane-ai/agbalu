"""The translate entrypoint's argument handling.

Both checks exist because the alternative is a failure minutes after launch, on a GPU: an
unchecked direction reaches `NLLB_CODE` as a `KeyError` naming a dict rather than the flag
that was mistyped, and a weights default one directory off reaches `from_pretrained`, which
reads an unresolvable local path as a Hub repo id.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from modal_app.translate import (
    DOCUMENTS,
    TRANSLATIONS,
    default_output,
    default_weights,
    parse_direction,
)

from agbalu.bench.mt import DIRECTIONS
from agbalu.mt.finetune import DEFAULT_RUN_NAME as MT_RUN_NAME


class TestParseDirection:
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_every_harness_direction_is_accepted(self, direction: str) -> None:
        """Asserted against the harness's own list rather than a copy: the two were
        separate tuples kept in step by hand, so adding a language to one left the other
        refusing a direction the benchmark scores."""
        assert parse_direction(direction) == direction

    @pytest.mark.parametrize(
        "value", ["kab-deu", "kab-arb", "eng_kab", "KAB-ENG", "", "kab", "eng-kab "]
    )
    def test_anything_else_is_refused_by_name(self, value: str) -> None:
        with pytest.raises(ValueError, match="unknown direction"):
            parse_direction(value)


class TestDefaultWeights:
    def test_it_is_where_the_fine_tune_writes(self) -> None:
        """`modal_app.mt.finetune` saves to `<checkpoints>/mt/<run>/final`, one directory
        deeper than the run — the shape of default that broke every documented reproduce
        command once before.

        The invariant is the shape, not the run name: pinning the literal made a routine
        bump from v1 to v2 look like a defect.
        """
        assert default_weights() == f"/checkpoints/mt/{MT_RUN_NAME}/final"


class TestDefaultOutput:
    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_the_direction_is_the_directory(self, direction: str) -> None:
        chosen = parse_direction(direction)
        assert default_output(DOCUMENTS / "eng" / "dracula.txt", chosen).parent == (
            TRANSLATIONS / direction
        )

    def test_the_stem_carries_across_unchanged(self) -> None:
        """One work translated from two source languages keeps one basename, which is what
        makes `arb-kab/quran.txt` and `eng-kab/quran.txt` comparable."""
        for language in ("eng", "arb"):
            source = DOCUMENTS / language / "quran.txt"
            assert default_output(source, "eng-kab").name == "quran.txt"
            assert source.stem == "quran"

    def test_an_output_never_lands_beside_its_source(self) -> None:
        source = DOCUMENTS / "fra" / "voyage-au-centre-de-la-terre.txt"
        assert DOCUMENTS not in default_output(source, "fra-kab").parents

    def test_a_source_outside_the_documents_tree_still_resolves(self) -> None:
        """`--file` accepts any path; the destination is decided by the direction and the
        stem, never by where the source happened to sit."""
        assert default_output(Path("somewhere/else/notes.txt"), "eng-kab") == (
            TRANSLATIONS / "eng-kab" / "notes.txt"
        )

    @pytest.mark.parametrize("name", ["a.b.txt", "no-extension", "dotted.name.eng.txt"])
    def test_only_the_final_extension_is_dropped(self, name: str) -> None:
        produced = default_output(DOCUMENTS / "eng" / name, "eng-kab")
        assert produced.suffix == ".txt"
        assert produced.parent == TRANSLATIONS / "eng-kab"
