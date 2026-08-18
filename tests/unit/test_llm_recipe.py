"""The CPT recipe: what the corpus reader refuses, how blocks are cut, and the schedule.

The three functions here decide what the run trains on and at what rate, and each has a
failure that looks like a healthy run: a document stream that silently yields nothing, a
packer that drops the tail it should keep, and a schedule whose first step is at lr 0.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterator
from pathlib import Path

import pytest

from agbalu.llm.recipe import (
    FINAL_LR_FRACTION,
    MIN_BLOCK,
    PEAK_LR,
    WARMUP_FRACTION,
    RecipeError,
    documents,
    learning_rate,
    pack,
)


def _corpus(path: Path, rows: list[object]) -> Path:
    target = path / "cpt.jsonl"
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return target


class TestDocuments:
    def test_a_missing_corpus_names_the_target_that_builds_it(self, tmp_path: Path) -> None:
        with pytest.raises(RecipeError, match="make llm TASK=mixture"):
            list(documents(tmp_path / "absent.jsonl"))

    def test_a_directory_is_not_a_corpus(self, tmp_path: Path) -> None:
        with pytest.raises(RecipeError):
            list(documents(tmp_path))

    def test_an_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "cpt.jsonl"
        target.write_text("", encoding="utf-8")
        assert list(documents(target)) == []

    def test_it_yields_the_text_of_each_row_in_file_order(self, tmp_path: Path) -> None:
        target = _corpus(tmp_path, [{"text": "azul"}, {"text": "aman"}, {"text": "aɣbalu"}])
        assert list(documents(target)) == ["azul", "aman", "aɣbalu"]

    def test_blank_lines_are_skipped_rather_than_parsed(self, tmp_path: Path) -> None:
        target = tmp_path / "cpt.jsonl"
        target.write_text('{"text": "azul"}\n\n   \n{"text": "aman"}\n', encoding="utf-8")
        assert list(documents(target)) == ["azul", "aman"]

    def test_a_row_without_text_names_the_line_it_is_on(self, tmp_path: Path) -> None:
        target = _corpus(tmp_path, [{"text": "azul"}, {"id": 2}])
        with pytest.raises(RecipeError, match=r":2 is not a document"):
            list(documents(target))

    def test_a_bare_string_is_not_a_document(self, tmp_path: Path) -> None:
        """A file of raw sentences reads as valid JSON line by line, so the type is what
        separates it from the corpus this function expects."""
        target = _corpus(tmp_path, ["azul"])
        with pytest.raises(RecipeError, match=r":1 is not a document"):
            list(documents(target))

    def test_a_null_row_is_refused(self, tmp_path: Path) -> None:
        target = _corpus(tmp_path, [None])
        with pytest.raises(RecipeError, match=r":1 is not a document"):
            list(documents(target))

    def test_it_refuses_the_bad_line_only_after_yielding_the_good_ones(
        self, tmp_path: Path
    ) -> None:
        """Streamed, not read whole: the caller has already packed two documents when the
        third is refused."""
        target = _corpus(tmp_path, [{"text": "a"}, {"text": "b"}, {"id": 3}])
        stream = documents(target)
        assert [next(stream), next(stream)] == ["a", "b"]
        with pytest.raises(RecipeError):
            next(stream)

    def test_malformed_json_raises_the_decoder_error_not_a_recipe_error(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "cpt.jsonl"
        target.write_text('{"text": "azul"}\n{"text": \n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            list(documents(target))

    def test_kabyle_orthography_and_a_homoglyph_survive_unchanged(self, tmp_path: Path) -> None:
        """The reader is not a normaliser: `ɣ` and a Greek `ε` must reach the tokenizer as
        written, so corruption stays measurable downstream rather than being half-repaired
        here."""
        text = "aɣbalu tamaziɣt ḥ ḍ ṣ ṭ ẓ ṛ č ǧ ţ εps"
        target = _corpus(tmp_path, [{"text": text}])
        assert list(documents(target)) == [text]

    def test_whitespace_inside_a_document_is_not_stripped(self, tmp_path: Path) -> None:
        """Only the blank-line test strips, and it strips a copy: a document that is one
        newline-separated paragraph keeps its shape."""
        target = _corpus(tmp_path, [{"text": "  azul\naman  "}])
        assert list(documents(target)) == ["  azul\naman  "]

    def test_a_document_that_is_only_whitespace_is_still_a_document(self, tmp_path: Path) -> None:
        target = _corpus(tmp_path, [{"text": "   "}])
        assert list(documents(target)) == ["   "]

    def test_a_non_string_text_is_coerced_rather_than_refused(self, tmp_path: Path) -> None:
        target = _corpus(tmp_path, [{"text": 42}])
        assert list(documents(target)) == ["42"]

    def test_an_empty_document_is_yielded_not_dropped(self, tmp_path: Path) -> None:
        """`pack` separates on it, so dropping it here would silently join two documents."""
        target = _corpus(tmp_path, [{"text": ""}, {"text": "azul"}])
        assert list(documents(target)) == ["", "azul"]


class TestPack:
    def test_a_length_below_two_teaches_nothing_and_is_refused(self) -> None:
        for length in (-1, 0, 1):
            with pytest.raises(RecipeError, match="at least 2"):
                list(pack([[1, 2, 3]], length, 0))

    def test_the_shortest_legal_block_is_one_context_and_one_target(self) -> None:
        assert list(pack([[1, 2]], MIN_BLOCK, 0)) == [[1, 2]]

    def test_documents_are_concatenated_with_a_separator_between_them(self) -> None:
        assert list(pack([[1, 2], [3]], 3, 0)) == [[1, 2, 0]]

    def test_a_document_longer_than_the_block_spans_blocks(self) -> None:
        assert list(pack([[1, 2, 3, 4, 5]], 2, 0)) == [[1, 2], [3, 4], [5, 0]]

    def test_the_remainder_is_dropped_rather_than_padded(self) -> None:
        """A short final block would be padding the loss has to mask, and the corpus is
        5.25M documents — the tail is one block."""
        assert list(pack([[1, 2, 3, 4]], 3, 0)) == [[1, 2, 3]]

    def test_an_empty_stream_yields_no_block(self) -> None:
        assert list(pack([], 4, 0)) == []

    def test_empty_documents_still_advance_the_buffer_by_their_separator(self) -> None:
        assert list(pack([[], [], [], []], 2, 9)) == [[9, 9], [9, 9]]

    def test_every_block_is_exactly_the_requested_length(self) -> None:
        blocks = list(pack([list(range(7)), list(range(5))], 4, 0))
        assert [len(block) for block in blocks] == [4, 4, 4]

    def test_it_consumes_lazily_so_the_corpus_is_never_held_whole(self) -> None:
        """The corpus does not fit in memory: taking one block must not have read the
        second document."""
        read: list[int] = []

        def stream() -> Iterator[list[int]]:
            for index, ids in enumerate([[1, 2], [3, 4]]):
                read.append(index)
                yield ids

        first = next(iter(pack(stream(), 2, 0)))
        assert first == [1, 2]
        assert read == [0]

    def test_no_token_of_the_input_is_lost_or_reordered(self) -> None:
        blocks = list(pack([[1, 2, 3], [4, 5]], 2, 0))
        flat = [token for block in blocks for token in block]
        assert flat == [1, 2, 3, 0, 4, 5]


class TestLearningRate:
    def test_a_run_of_no_steps_has_no_schedule(self) -> None:
        for total in (-1, 0):
            with pytest.raises(RecipeError, match="total steps must be positive"):
                learning_rate(0, total)

    def test_a_step_outside_the_run_is_refused(self) -> None:
        for step in (-1, 10, 11):
            with pytest.raises(RecipeError, match="outside a run"):
                learning_rate(step, 10)

    def test_the_first_step_is_not_at_zero(self) -> None:
        """A first step at lr 0 wastes an optimizer step, and on a smoke it is a measurable
        share of the budget."""
        assert learning_rate(0, 1000) > 0.0

    def test_the_warmup_is_a_fraction_of_the_schedule_not_a_fixed_count(self) -> None:
        """A fixed 500-step warmup makes a 20-step smoke train at 4% of peak throughout. As
        a fraction the whole warmup is one step, so the smoke reaches peak immediately."""
        assert learning_rate(0, 20) == PEAK_LR

    def test_a_single_step_run_trains_at_peak(self) -> None:
        assert learning_rate(0, 1) == PEAK_LR

    def test_peak_is_reached_at_the_end_of_warmup_and_not_before(self) -> None:
        total = 1000
        warmup = int(total * WARMUP_FRACTION)
        assert learning_rate(warmup - 1, total) == pytest.approx(PEAK_LR)
        assert learning_rate(warmup - 2, total) < PEAK_LR

    def test_the_ramp_is_linear_in_the_step(self) -> None:
        total = 1000
        rates = [learning_rate(step, total) for step in range(10)]
        gaps = [round(b - a, 12) for a, b in itertools.pairwise(rates)]
        assert len(set(gaps)) == 1

    def test_it_decays_to_the_floor_and_never_below(self) -> None:
        total = 1000
        last = learning_rate(total - 1, total)
        assert last == pytest.approx(PEAK_LR * FINAL_LR_FRACTION, rel=0.01)
        assert last > PEAK_LR * FINAL_LR_FRACTION

    def test_it_never_rises_after_warmup(self) -> None:
        total = 500
        warmup = max(1, int(total * WARMUP_FRACTION))
        rates = [learning_rate(step, total) for step in range(warmup, total)]
        assert all(b <= a for a, b in itertools.pairwise(rates))

    def test_the_midpoint_of_the_decay_sits_halfway_between_peak_and_floor(self) -> None:
        """Cosine, not linear: the schedule spends longer near peak than a line would."""
        total = 1002
        warmup = max(1, int(total * WARMUP_FRACTION))
        middle = warmup + (total - warmup) // 2
        expected = PEAK_LR * (FINAL_LR_FRACTION + (1.0 - FINAL_LR_FRACTION) * 0.5)
        assert learning_rate(middle, total) == pytest.approx(expected, rel=1e-3)

    def test_the_peak_is_a_parameter_the_caller_can_lower(self) -> None:
        assert learning_rate(0, 1, peak=1e-4) == 1e-4

    def test_every_rate_in_a_run_is_finite_and_positive(self) -> None:
        rates = [learning_rate(step, 64) for step in range(64)]
        assert all(math.isfinite(rate) and rate > 0.0 for rate in rates)
