"""Building the continued-pretraining corpus (task 11.4).

The mixture's ratio is the experiment, so it is asserted by tokens rather than by rows —
an equal number of rows is a 1:1.8 token mixture, which is the defect this file exists to
prevent. Since task 11.2 the corpus also owes an exclusion: every record the evaluation
split selects must be absent from what is written.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from agbalu.llm.corpus import Source, is_held_out
from agbalu.llm.mixture import (
    KABYLE_CODE,
    Mixture,
    MixtureError,
    Tally,
    aligned_document,
    build,
    documents,
)

KAB = "Aɣbalu n tmaziɣt d ayen ɣef nettmeslay."
ENG = "The source of Tamazight is what we speak about."

NEVER_HELD_OUT = 10**9
"""A rate no sample of this size reaches, so an arithmetic test is not at the mercy of a
digest. The exclusion itself is asserted at rate 1 and against the predicate directly."""


class WordCounter:
    """Whitespace tokens, so the arithmetic in a test is checkable by eye."""

    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def mono(path: Path, rows: int) -> Source:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(json.dumps({"text": f"{KAB} {i}"}, ensure_ascii=False) + "\n")
    return Source(name="mono", path=path, kind="kabyle", fields=("text",))


def pairs(path: Path, rows: int) -> Source:
    with path.open("w", encoding="utf-8") as fh:
        for i in range(rows):
            fh.write(
                json.dumps({"kab": f"{KAB} {i}", "eng": f"{ENG} {i}"}, ensure_ascii=False) + "\n"
            )
    return Source(name="pairs", path=path, kind="aligned", fields=("kab", "eng"), code="eng_Latn")


def test_aligned_document_tags_both_sides() -> None:
    doc = aligned_document(KAB, ENG, "eng_Latn", kabyle_first=True)
    assert doc == f"{KABYLE_CODE}: {KAB}\neng_Latn: {ENG}"


def test_direction_reverses() -> None:
    doc = aligned_document(KAB, ENG, "eng_Latn", kabyle_first=False)
    assert doc.startswith("eng_Latn: ")
    assert KABYLE_CODE in doc


def test_both_directions_appear_across_rows(tmp_path: Path) -> None:
    source = pairs(tmp_path / "p.jsonl", 10)
    heads = {text.split(":")[0] for text, _ in documents(source)}
    assert heads == {KABYLE_CODE, "eng_Latn"}


def test_alignment_is_deterministic(tmp_path: Path) -> None:
    source = pairs(tmp_path / "p.jsonl", 10)
    assert list(documents(source)) == list(documents(source))


def test_build_counts_tokens_by_kind(tmp_path: Path) -> None:
    sources = [mono(tmp_path / "m.jsonl", 4), pairs(tmp_path / "p.jsonl", 4)]
    result = build(sources, WordCounter(), tmp_path / "out.jsonl", epochs=1, rate=NEVER_HELD_OUT)
    # mono row: 7-word sentence + index = 8
    assert result.tokens("kabyle") == 4 * 8
    # aligned row: tag + 7 + index = 9, then tag + 9 + index = 11
    assert result.tokens("aligned") == 4 * 20
    assert result.total_tokens == 4 * 28


def test_aligned_share_is_by_tokens_not_rows(tmp_path: Path) -> None:
    """Equal rows is not a 1:1 mixture; this is the whole point of the module."""
    sources = [mono(tmp_path / "m.jsonl", 100), pairs(tmp_path / "p.jsonl", 100)]
    result = build(sources, WordCounter(), tmp_path / "out.jsonl", epochs=1, rate=NEVER_HELD_OUT)
    assert result.aligned_share == pytest.approx(20 / 28, abs=1e-3)
    assert result.aligned_share > 0.5


def test_epochs_multiply_the_total_not_the_file(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    result = build(
        [mono(tmp_path / "m.jsonl", 10)], WordCounter(), out, epochs=4, rate=NEVER_HELD_OUT
    )
    payload = result.as_dict()
    assert payload["tokens_total"] == 4 * result.total_tokens
    assert len(out.read_text(encoding="utf-8").splitlines()) == 10


def test_every_document_is_written(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    build(
        [mono(tmp_path / "m.jsonl", 7), pairs(tmp_path / "p.jsonl", 5)],
        WordCounter(),
        out,
        epochs=1,
        rate=NEVER_HELD_OUT,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12
    assert all(set(r) == {"text", "kind"} for r in rows)


def test_each_row_records_the_kind_the_replay_ratio_is_measured_over(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    build(
        [mono(tmp_path / "m.jsonl", 7), pairs(tmp_path / "p.jsonl", 5)],
        WordCounter(),
        out,
        epochs=1,
        rate=NEVER_HELD_OUT,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["kabyle"] * 7 + ["aligned"] * 5


def test_batching_does_not_change_the_result(tmp_path: Path) -> None:
    def src() -> list[Source]:
        return [mono(tmp_path / "m.jsonl", 50), pairs(tmp_path / "p.jsonl", 50)]

    a = build(src(), WordCounter(), tmp_path / "a.jsonl", epochs=1, batch=3)
    b = build(src(), WordCounter(), tmp_path / "b.jsonl", epochs=1, batch=1000)
    assert a.as_dict() == b.as_dict()
    assert (tmp_path / "a.jsonl").read_text(encoding="utf-8") == (tmp_path / "b.jsonl").read_text(
        encoding="utf-8"
    )


class TestHeldOutRecordsAreExcluded:
    def test_nothing_is_written_when_everything_is_held_out(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        result = build([mono(tmp_path / "m.jsonl", 12)], WordCounter(), out, epochs=1, rate=1)
        assert out.read_text(encoding="utf-8") == ""
        assert result.held_out == 12
        assert result.total_tokens == 0

    def test_the_exclusion_uses_the_same_predicate_as_the_evaluation_split(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.jsonl"
        rows = 400
        build([mono(tmp_path / "m.jsonl", rows)], WordCounter(), out, epochs=1, rate=5)
        lines = out.read_text(encoding="utf-8").splitlines()
        written = {json.loads(line)["text"] for line in lines}
        expected = {f"{KAB} {i}" for i in range(rows) if not is_held_out(f"{KAB} {i}", rate=5)}
        assert written == expected

    def test_an_aligned_pair_is_held_out_by_its_kabyle_side(self, tmp_path: Path) -> None:
        """Both mirrored copies of a pair carry the same Kabyle sentence, so both leave."""
        path = tmp_path / "mt.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for direction, first, second in (("kab-eng", KAB, ENG), ("eng-kab", ENG, KAB)):
                fh.write(
                    json.dumps(
                        {"source": first, "target": second, "direction": direction},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        source = Source(
            name="mt",
            path=path,
            kind="aligned",
            fields=("source", "target"),
            direction_field="direction",
        )
        out = tmp_path / "out.jsonl"
        result = build([source], WordCounter(), out, epochs=1, rate=1)
        assert result.held_out == 2
        assert out.read_text(encoding="utf-8") == ""

    def test_the_rate_is_recorded_in_the_statistics(self, tmp_path: Path) -> None:
        result = build(
            [mono(tmp_path / "m.jsonl", 3)], WordCounter(), tmp_path / "o.jsonl", epochs=1, rate=7
        )
        assert result.as_dict()["held_out_rate"] == 7


def test_zero_epochs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MixtureError, match="epochs must be positive"):
        build([mono(tmp_path / "m.jsonl", 2)], WordCounter(), tmp_path / "o.jsonl", epochs=0)


def test_a_zero_batch_is_refused(tmp_path: Path) -> None:
    sources = [mono(tmp_path / "m.jsonl", 2)]
    with pytest.raises(MixtureError, match="batch must be positive"):
        build(sources, WordCounter(), tmp_path / "o.jsonl", epochs=1, batch=0)


def test_a_counter_that_drops_rows_is_caught(tmp_path: Path) -> None:
    class Dropping(WordCounter):
        def count(self, texts: Sequence[str]) -> list[int]:
            return super().count(texts)[:-1]

    with pytest.raises(MixtureError, match="counts for"):
        build([mono(tmp_path / "m.jsonl", 3)], Dropping(), tmp_path / "o.jsonl", epochs=1)


def test_an_empty_mixture_has_no_share() -> None:
    assert Mixture(tallies=(), epochs=1).aligned_share == 0.0


def test_tally_round_trips() -> None:
    tally = Tally("x", "kabyle", 3, 30, 1)
    assert tally.as_dict() == {
        "name": "x",
        "kind": "kabyle",
        "documents": 3,
        "tokens": 30,
        "held_out": 1,
    }
