"""Signals per candidate: language labels, fluency, and their merge."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from agbalu.llm import signals
from agbalu.llm.evaluate import Scorer
from agbalu.llm.teacher import Candidate
from tests.unit import tiny_lm


class Labels:
    def __init__(self, label: str) -> None:
        self.label = label
        self.seen: list[str] = []

    def identify(self, texts: Sequence[str]) -> list[str]:
        self.seen.extend(texts)
        return [self.label] * len(texts)


def candidate(answer: str, reasoning: str | None = None) -> Candidate:
    turn = {"role": "assistant", "content": answer}
    if reasoning is not None:
        turn["reasoning"] = reasoning
    return Candidate("c1", "k", "b", 1, ({"role": "user", "content": "Azul"}, turn), "m", "d", 1)


class TestLanguage:
    def test_the_signals_read_the_answer_and_the_reasoning(self) -> None:
        glot, dihya = Labels("kab_Latn"), Labels("kab_Latn")
        found = signals.language_signals([candidate("Tiririt.", reasoning="Axemmem.")], glot, dihya)
        assert set(found["c1"]) == {"glotlid_kabyle", "glotlid_label", "dihya_sibling"}
        assert found["c1"]["glotlid_kabyle"] is True
        assert found["c1"]["dihya_sibling"] is None
        assert "Axemmem." in glot.seen[0]
        assert "Azul" not in glot.seen[0]

    def test_another_language_and_a_sibling_are_reported(self) -> None:
        found = signals.language_signals(
            [candidate("Hello there.")], Labels("eng_Latn"), Labels("shi_Latn")
        )
        assert found["c1"]["glotlid_kabyle"] is False
        assert found["c1"]["dihya_sibling"] == "shi_Latn"

    @pytest.mark.parametrize(
        ("labels", "expected"),
        [
            (["kab_Latn", "shi_Latn", "kab_Latn"], None),
            (["rif_Latn", "rif_Latn", "kab_Latn"], "rif_Latn"),
            (["NOT_AMAZIGH"], None),
            ([], None),
        ],
    )
    def test_the_sibling_is_the_majority_label(
        self, labels: list[str], expected: str | None
    ) -> None:
        assert signals.sibling(labels) == expected


@pytest.fixture(scope="module")
def scorer(tmp_path_factory: pytest.TempPathFactory) -> Scorer:
    model = tiny_lm.load(tiny_lm.save(tmp_path_factory.mktemp("tiny")), torch.float32)
    return Scorer.of(model, tiny_lm.EOS, torch.device("cpu"))


class TestFluency:
    def test_each_text_is_scored_alone(self, scorer: Scorer) -> None:
        tokenizer = tiny_lm.CharTokenizer()
        both = signals.bits_per_character(scorer, tokenizer, ["azul", "fell-ak"])
        alone = signals.bits_per_character(scorer, tokenizer, ["fell-ak"])
        assert both[1] == pytest.approx(alone[0], rel=1e-5)
        assert all(value > 0 for value in both)

    def test_an_empty_text_is_refused(self, scorer: Scorer) -> None:
        with pytest.raises(signals.SignalError):
            signals.bits_per_character(scorer, tiny_lm.CharTokenizer(), [""])

    def test_the_percentile_is_nearest_rank(self) -> None:
        assert signals.percentile([float(i) for i in range(1, 101)], 0.95) == 95.0
        with pytest.raises(signals.SignalError):
            signals.percentile([], 0.95)


def test_passes_merge_per_candidate() -> None:
    merged = signals.merged({"a": {"x": 1}}, {"a": {"y": 2}, "b": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 2}, "b": {"y": 3}}
