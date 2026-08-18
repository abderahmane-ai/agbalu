"""Tokens spent per Kabyle word (task 11.1).

The number decides whether vocabulary expansion is attempted at all, so the arithmetic is
asserted against hand-computed values rather than against whatever the code returns, and
the sample is asserted to be a sample of the file rather than of its first rows.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

from agbalu.llm.fertility import (
    Fertility,
    FertilityError,
    measure,
    report,
    sample,
    texts,
)

KABYLE = "Aɣbalu n tmaziɣt d ayen ɣef nettmeslay, ṭṭfeɣ aḍar-iw ṣṣbeḥ."


class FakeEncoder:
    """A whitespace tokenizer that splits `split_on` into characters."""

    def __init__(self, name: str, *, split_on: str = "", unknown: int | None = 0) -> None:
        self._name = name
        self._split_on = split_on
        self._unknown = unknown

    @property
    def name(self) -> str:
        return self._name

    @property
    def unknown_id(self) -> int | None:
        return self._unknown

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        rows: list[list[int]] = []
        for text in texts:
            ids: list[int] = []
            for word in text.split():
                pieces = 1 + sum(word.count(c) for c in self._split_on)
                ids.extend(range(1, pieces + 1))
            rows.append(ids)
        return rows


def corpus(path: Path, rows: int, *, text: str = KABYLE) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for i in range(rows):
            handle.write(json.dumps({"text": f"{text} {i:05d}"}, ensure_ascii=False) + "\n")
    return path


def test_tokens_per_word_is_the_stated_arithmetic() -> None:
    result = measure(FakeEncoder("plain"), ["a b c", "d e"])
    assert result.words == 5
    assert result.tokens == 5
    assert result.tokens_per_word == 1.0


def test_a_splitting_vocabulary_costs_more_tokens() -> None:
    # 4 words; "ɣ" occurs once in "aɣbalu" and once in "tmaziɣt".
    sentences = ["aɣbalu n tmaziɣt", "azul"]
    plain = measure(FakeEncoder("plain"), sentences)
    split = measure(FakeEncoder("split", split_on="ɣ"), sentences)
    assert plain.tokens == 4
    assert split.tokens == 4 + 2
    assert split.tokens_per_word > plain.tokens_per_word


def test_unknown_share_counts_only_the_unknown_id() -> None:
    result = measure(FakeEncoder("plain", unknown=1), ["a b", "c"])
    assert result.unknown == 3  # every word's first id is 1
    assert result.unknown_share == 1.0


def test_a_vocabulary_without_an_unknown_token_reports_zero() -> None:
    assert measure(FakeEncoder("plain", unknown=None), ["a b"]).unknown == 0


def test_empty_input_is_refused() -> None:
    with pytest.raises(FertilityError, match="nothing to measure"):
        measure(FakeEncoder("plain"), [])


def test_a_tokenizer_that_drops_rows_is_caught() -> None:
    class Dropping(FakeEncoder):
        def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
            return super().encode_batch(texts)[:-1]

    with pytest.raises(FertilityError, match="returned 1 rows for 2"):
        measure(Dropping("dropping"), ["a b", "c d"])


def test_zero_words_is_refused() -> None:
    with pytest.raises(FertilityError, match="0 words"):
        Fertility(name="x", sentences=1, words=0, tokens=3, unknown=0)


def test_report_ratios_are_against_the_named_reference() -> None:
    sentences = ["aɣbalu n tmaziɣt"]
    result = report(
        [FakeEncoder("split", split_on="ɣ"), FakeEncoder("ours")], sentences, reference="ours"
    )
    rows = result.by_name()
    assert rows["ours"].ratio_to_reference == 1.0
    # 5 tokens for 3 words against 3 for 3, rounded to the 4 places `Report` writes.
    assert rows["split"].ratio_to_reference == pytest.approx(5 / 3, abs=1e-4)


def test_an_unknown_reference_is_refused() -> None:
    with pytest.raises(FertilityError, match="is not among"):
        report([FakeEncoder("a")], ["x y"], reference="b")


def test_report_records_the_population() -> None:
    result = report([FakeEncoder("a")], ["x y"])
    assert result.as_dict()["population"] == {"sentences": 1, "min_chars": 40, "max_chars": 300}


def test_without_a_reference_no_ratio_is_written() -> None:
    result = report([FakeEncoder("a")], ["x y"])
    assert result.vocabularies[0].ratio_to_reference is None
    assert "ratio_to_reference" not in result.vocabularies[0].as_dict()


def test_sample_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    path = corpus(tmp_path / "c.jsonl", 500)
    assert sample(path, 20, seed=7) == sample(path, 20, seed=7)


def test_a_different_seed_draws_differently(tmp_path: Path) -> None:
    path = corpus(tmp_path / "c.jsonl", 500)
    assert sample(path, 20, seed=7) != sample(path, 20, seed=8)


def test_sample_reaches_the_end_of_the_file(tmp_path: Path) -> None:
    """A prefix is ordered by source id, so it is a different corpus."""
    path = corpus(tmp_path / "c.jsonl", 4000)
    drawn = sample(path, 200, seed=3)
    tails = [int(re.search(r"(\d{5})$", s).group(1)) for s in drawn]  # type: ignore[union-attr]
    assert max(tails) > 3000


def test_sample_smaller_than_requested_returns_everything(tmp_path: Path) -> None:
    path = corpus(tmp_path / "c.jsonl", 5)
    assert len(sample(path, 100, seed=1)) == 5


def test_length_bounds_are_applied(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"text": "short"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"text": "x" * 400}, ensure_ascii=False)
        + "\n"
        + json.dumps({"text": KABYLE}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    assert sample(path, 10, seed=1) == [KABYLE]


def test_a_corpus_with_no_eligible_row_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"text": "short"}) + "\n", encoding="utf-8")
    with pytest.raises(FertilityError, match="no rows"):
        sample(path, 10, seed=1)


def test_a_non_positive_count_is_refused(tmp_path: Path) -> None:
    path = corpus(tmp_path / "c.jsonl", 5)
    with pytest.raises(FertilityError, match="count must be positive"):
        sample(path, 0, seed=1)


def test_rows_without_the_field_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"other": KABYLE}) + "\n" + json.dumps({"text": KABYLE}) + "\n",
        encoding="utf-8",
    )
    assert list(texts(path)) == [KABYLE]


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("\n" + json.dumps({"text": KABYLE}) + "\n\n", encoding="utf-8")
    assert list(texts(path)) == [KABYLE]


def test_a_malformed_row_names_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"text": KABYLE}) + "\nnot json\n", encoding="utf-8")
    with pytest.raises(FertilityError, match=re.escape("c.jsonl:2")):
        list(texts(path))
