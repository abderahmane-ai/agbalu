"""Translation as a prompt (task 11.2).

A prompted model answers in prose, so the extraction below is part of the metric: a
hypothesis that keeps the model's preamble scores the preamble. The batching tests exist
because reordering for speed is only safe while input order is restored exactly.
"""

from __future__ import annotations

import pytest

from agbalu.bench.mt import LANGUAGE_CODE
from agbalu.llm.prompting import (
    LANGUAGE_NAME,
    MAX_NEW_TOKENS,
    PromptError,
    Shot,
    hypothesis,
    instruction,
    length_ordered_batches,
    messages,
    new_tokens,
    require_str,
    require_tensor,
    shots,
)

KAB = "Azul fell-awen."
ENG = "Hello to you all."


class TestInstruction:
    def test_both_languages_are_named(self) -> None:
        text = instruction("eng-kab")
        assert "English" in text
        assert "Kabyle" in text

    def test_the_direction_is_not_symmetric(self) -> None:
        assert instruction("eng-kab") != instruction("kab-eng")

    @pytest.mark.parametrize("direction", ["eng_kab", "eng-kab-fra", "", "eng"])
    def test_a_malformed_direction_is_refused(self, direction: str) -> None:
        with pytest.raises(PromptError, match="src-tgt"):
            instruction(direction)

    def test_an_unnamed_language_is_refused(self) -> None:
        with pytest.raises(PromptError, match="no language name"):
            instruction("eng-zgh")


class TestMessages:
    def test_the_instruction_leads(self) -> None:
        turns = messages("eng-kab", ENG)
        assert turns[0]["role"] == "system"
        assert turns[-1] == {"role": "user", "content": ENG}

    def test_shots_alternate_user_and_assistant(self) -> None:
        """The base's template raises `TemplateError` on any other role name."""
        turns = messages("eng-kab", ENG, [Shot(ENG, KAB), Shot(ENG, KAB)])
        assert [t["role"] for t in turns] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]

    def test_every_role_is_one_the_template_accepts(self) -> None:
        turns = messages("eng-kab", ENG, [Shot(ENG, KAB)])
        assert {t["role"] for t in turns} <= {"system", "user", "assistant"}

    def test_no_shots_still_asks_the_question(self) -> None:
        assert len(messages("eng-kab", ENG)) == 2


class TestShots:
    def test_the_draw_is_reproducible(self) -> None:
        sources = [f"s{i}" for i in range(50)]
        targets = [f"t{i}" for i in range(50)]
        assert shots(sources, targets, 5) == shots(sources, targets, 5)

    def test_the_pairs_stay_aligned(self) -> None:
        sources = [f"s{i}" for i in range(50)]
        targets = [f"t{i}" for i in range(50)]
        assert all(shot.source[1:] == shot.target[1:] for shot in shots(sources, targets, 5))

    def test_the_head_of_the_split_is_not_taken(self) -> None:
        """FLORES+ is ordered by document, so the first rows are one article."""
        sources = [f"s{i}" for i in range(200)]
        drawn = shots(sources, sources, 5)
        assert [s.source for s in drawn] != sources[:5]

    def test_a_seed_change_changes_the_draw(self) -> None:
        sources = [f"s{i}" for i in range(200)]
        assert shots(sources, sources, 5, seed=1) != shots(sources, sources, 5, seed=2)

    def test_zero_shots_is_allowed(self) -> None:
        assert shots(["a"], ["b"], 0) == ()

    def test_more_shots_than_pairs_is_refused(self) -> None:
        with pytest.raises(PromptError, match="cannot draw"):
            shots(["a"], ["b"], 5)

    def test_a_misaligned_split_is_refused(self) -> None:
        with pytest.raises(PromptError, match="against"):
            shots(["a", "b"], ["c"], 1)


class TestHypothesis:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("Azul fell-awen.", "Azul fell-awen."),
            ("  Azul fell-awen.  ", "Azul fell-awen."),
            ("Translation: Azul fell-awen.", "Azul fell-awen."),
            ("translation : Azul fell-awen.", "Azul fell-awen."),
            ("Kabyle: Azul fell-awen.", "Azul fell-awen."),
            ('"Azul fell-awen."', "Azul fell-awen."),
            ("«Azul fell-awen.»", "Azul fell-awen."),
            ("Azul fell-awen.\n\nThis means hello.", "Azul fell-awen."),
            ("\n\nAzul fell-awen.", "Azul fell-awen."),
        ],
    )
    def test_the_translation_is_taken_off_the_reply(self, reply: str, expected: str) -> None:
        assert hypothesis(reply) == expected

    @pytest.mark.parametrize("reply", ["", "   ", "\n\n"])
    def test_an_empty_reply_scores_as_empty_rather_than_raising(self, reply: str) -> None:
        """A refusal is a legitimate baseline result and must not end the run."""
        assert hypothesis(reply) == ""

    def test_kabyle_letters_survive(self) -> None:
        assert hypothesis("Aɣbalu n tmaziɣt, ḥemmleɣ-t.") == "Aɣbalu n tmaziɣt, ḥemmleɣ-t."


class TestNewTokens:
    def test_the_budget_scales_with_the_source(self) -> None:
        assert new_tokens([10]) == 52
        assert new_tokens([10, 30]) == 92

    def test_the_budget_is_capped(self) -> None:
        assert new_tokens([10_000]) == MAX_NEW_TOKENS

    def test_an_empty_batch_is_refused(self) -> None:
        with pytest.raises(PromptError, match="empty batch"):
            new_tokens([])


class TestLengthOrderedBatches:
    def test_every_index_appears_exactly_once(self) -> None:
        batched = length_ordered_batches([5, 1, 9, 3, 7], 2)
        assert sorted(i for batch in batched for i in batch) == [0, 1, 2, 3, 4]

    def test_batches_are_sized_as_asked(self) -> None:
        assert [len(b) for b in length_ordered_batches([1] * 7, 3)] == [3, 3, 1]

    def test_the_longest_come_first(self) -> None:
        batched = length_ordered_batches([1, 9, 5], 1)
        assert [b[0] for b in batched] == [1, 2, 0]

    def test_equal_lengths_keep_input_order(self) -> None:
        assert length_ordered_batches([4, 4, 4], 3) == [[0, 1, 2]]

    def test_an_empty_population_yields_no_batch(self) -> None:
        assert length_ordered_batches([], 4) == []

    def test_a_zero_size_is_refused(self) -> None:
        with pytest.raises(PromptError, match="batch size must be positive"):
            length_ordered_batches([1, 2], 0)


def test_every_benchmark_language_can_be_named() -> None:
    """A direction the harness scores but the prompt cannot name would fail on the GPU."""
    assert set(LANGUAGE_CODE) <= set(LANGUAGE_NAME)


class TestNarrowing:
    """`transformers` returns a union decided by the call's own flags. These narrow it
    where the flag is known, so flipping one is a named error rather than a `TypeError`
    raised somewhere further down."""

    def test_a_string_passes_through(self) -> None:
        assert require_str("azul", "apply_chat_template") == "azul"

    def test_a_tokenised_template_is_refused_by_name(self) -> None:
        """What `apply_chat_template` returns when `tokenize` is left at its default."""
        with pytest.raises(PromptError, match="apply_chat_template returned list"):
            require_str([1, 2, 3], "apply_chat_template")

    def test_a_batch_decode_is_refused(self) -> None:
        with pytest.raises(PromptError, match="decode returned list, not a string"):
            require_str(["a", "b"], "decode")

    def test_a_tensor_passes_through(self) -> None:
        torch = pytest.importorskip("torch")
        value = torch.zeros(2, 3)
        assert require_tensor(value, "generate") is value

    def test_a_dict_output_is_refused(self) -> None:
        """What `generate` returns once `return_dict_in_generate` is set."""
        pytest.importorskip("torch")
        with pytest.raises(PromptError, match="generate returned dict, not a tensor"):
            require_tensor({"sequences": []}, "generate")
