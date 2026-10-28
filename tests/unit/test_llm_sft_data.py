"""Candidates into the accepted SFT set: each gate, flag, review rule and quota, and a ledger
that accounts for every candidate."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agbalu.llm import sft_data
from agbalu.llm.chat import SYSTEM_POLICY
from agbalu.llm.seeds import Card, System
from agbalu.llm.sft_data import Evidence, Review, Thresholds
from agbalu.llm.teacher import Candidate

CLEAN = {
    "glotlid_kabyle": True,
    "dihya_sibling": None,
    "fluency_bpc": 1.0,
}
THRESHOLDS = Thresholds(fluency_bpc=2.0)


def card(
    identifier: str,
    task: str = "native_kabyle",
    *,
    think: bool = False,
    system: System = "none",
) -> Card:
    return Card(identifier, task, "culture", think, "kab", system, 1, "train")


def candidate(
    identifier: str, card_id: str, answer: str, *, reasoning: str | None = None
) -> Candidate:
    assistant = {"role": "assistant", "content": answer}
    if reasoning is not None:
        assistant["reasoning"] = reasoning
    turns = ({"role": "user", "content": "Azul, amek telliḍ?"}, assistant)
    return Candidate(identifier, card_id, "batch-0001", 1, turns, "Gemini 4 Argon", "2026-10-17", 1)


def answer(index: int) -> str:
    return (
        f"Nekk d Jugurtha, ad k-ɛawneɣ deg wawal wis {index} s tmeslayt taqbaylit tamaynut ass-a."
    )


def world(count: int = 10, thinking: int = 2) -> tuple[list[Candidate], dict[str, Card]]:
    cards = {f"k{i}": card(f"k{i}", think=i < thinking) for i in range(count)}
    found = [
        candidate(f"c{i}", f"k{i}", answer(i), reasoning="Axemmem." if i < thinking else None)
        for i in range(count)
    ]
    return found, cards


def evidence(
    found: list[Candidate],
    *,
    signals: dict[str, dict[str, object]] | None = None,
    reviews: dict[str, Review] | None = None,
) -> Evidence:
    """Clean evidence for every candidate, with the given entries replaced."""
    all_signals: dict[str, dict[str, object]] = {c.id: dict(CLEAN) for c in found}
    all_signals |= signals or {}
    return Evidence(all_signals, reviews or {}, THRESHOLDS, prompt_versions=frozenset({1}))


def run(
    found: list[Candidate], cards: dict[str, Card], ev: Evidence, *, contaminated: str = ""
) -> tuple[list[Candidate], sft_data.Ledger]:
    checks = sft_data.Text(
        contaminated=lambda text: bool(contaminated) and contaminated in text,
        normalise=lambda text: text.replace("ε", "ɛ"),
    )
    return sft_data.assemble(found, cards, ev, checks)


class TestGates:
    def test_a_clean_world_is_accepted_and_accounted(self) -> None:
        found, cards = world()
        accepted, ledger = run(found, cards, evidence(found))
        assert len(accepted) == 10
        assert ledger.counts() == {"accepted": 10}

    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("Azul <|im_end|> fell-ak", "template_marker"),
            ("a b c d " * 20, "repetition"),
        ],
    )
    def test_a_hard_gate_rejects_with_its_reason(self, text: str, reason: str) -> None:
        found, cards = world(11)
        found[10] = candidate("c10", "k10", text)
        _, ledger = run(found, cards, evidence(found))
        assert (ledger.status["c10"], ledger.reasons["c10"]) == ("rejected", [reason])

    def test_an_evaluation_overlap_is_rejected(self) -> None:
        found, cards = world(11)
        found[10] = candidate("c10", "k10", "Tafyirt n FLORES tella dagi.")
        _, ledger = run(found, cards, evidence(found), contaminated="FLORES")
        assert ledger.reasons["c10"] == ["contaminated"]

    def test_exact_and_near_copies_are_rejected_after_the_first(self) -> None:
        found, cards = world(12)
        found[10] = candidate("z1", "k10", answer(3))
        found[11] = candidate("z2", "k11", answer(3) + " Ih.")
        _, ledger = run(found, cards, evidence(found))
        assert ledger.reasons["z1"] == ["duplicate"]
        assert ledger.reasons["z2"] == ["near_duplicate"]
        assert ledger.status["c3"] == "accepted"

    def test_an_older_prompt_version_is_rejected_and_blocks_no_newer_copy(self) -> None:
        found, cards = world(11)
        found[10] = replace(candidate("a10", "k10", answer(10)), version=0)
        found.append(candidate("c10", "k10", answer(10)))
        _, ledger = run(found, cards, evidence(found))
        assert (ledger.status["a10"], ledger.reasons["a10"]) == ("rejected", ["prompt_version_0"])
        assert ledger.status["c10"] == "accepted"

    def test_a_release_card_never_enters_training_or_validation(self) -> None:
        found, cards = world(11)
        cards["k10"] = replace(card("k10"), split="release")
        _, ledger = run(found, cards, evidence(found))
        assert (ledger.status["c10"], ledger.reasons["c10"]) == ("rejected", ["release_card"])

    def test_text_is_normalised_before_it_is_compared_or_kept(self) -> None:
        found, cards = world()
        found[5] = candidate("c5", "k5", answer(5).replace("ɛ", "ε"))
        accepted, _ = run(found, cards, evidence(found))
        kept = next(c for c in accepted if c.id == "c5")
        assert "ε" not in kept.turns[1]["content"]


class TestFlags:
    @pytest.mark.parametrize(
        ("signal", "value", "flag"),
        [
            ("glotlid_kabyle", False, "not_read_as_kabyle"),
            ("dihya_sibling", "shi", "sibling_shi"),
            ("fluency_bpc", 3.0, "disfluent"),
        ],
    )
    def test_a_soft_signal_sets_a_candidate_aside(
        self, signal: str, value: object, flag: str
    ) -> None:
        found, cards = world(11)
        ev = evidence(found, signals={"c10": {**CLEAN, signal: value}})
        _, ledger = run(found, cards, ev)
        assert (ledger.status["c10"], ledger.reasons["c10"]) == ("flagged", [flag])

    def test_a_flagged_candidate_enters_when_review_accepts_it(self) -> None:
        found, cards = world(11)
        ev = evidence(
            found,
            signals={"c10": {**CLEAN, "fluency_bpc": 9.0}},
            reviews={"c10": Review("c10", True, "")},
        )
        _, ledger = run(found, cards, ev)
        assert ledger.status["c10"] == "accepted"

    def test_translation_answers_are_not_held_to_kabyle(self) -> None:
        found, cards = world(11)
        cards["k10"] = card("k10", "translation")
        ev = evidence(found, signals={"c10": {**CLEAN, "glotlid_kabyle": False}})
        _, ledger = run(found, cards, ev)
        assert ledger.status["c10"] == "accepted"

    @pytest.mark.parametrize(
        ("content", "reasoning"),
        [
            ("Deg 2008, يوض (yewweḍ) umḍan n yimezdaɣ.", None),
            ("Azul fell-ak.", "Ad xemmemeɣ ﻻ ɣef waya."),
        ],
    )
    def test_arabic_script_in_an_answer_or_its_reasoning_sets_it_aside(
        self, content: str, reasoning: str | None
    ) -> None:
        found, cards = world(11)
        found[10] = candidate("c10", "k10", content, reasoning=reasoning)
        cards["k10"] = card("k10", think=reasoning is not None)
        _, ledger = run(found, cards, evidence(found))
        assert (ledger.status["c10"], ledger.reasons["c10"]) == ("flagged", ["arabic_script"])

    @pytest.mark.parametrize(
        ("content", "reasoning", "flag"),
        [
            ("Ilmend n uḍris, Bern tettwabna s yiẓra.", None, "cites_unseen_text"),
            ("Bern tettwabna s yiẓra.", "Ad rreɣ war azul imi ur d-yenni ara.", "instruction_leak"),
            ("Bern tettwabna s yiẓra.", "Ad rreɣ srid acku ulac sslam.", "instruction_leak"),
            ("Deg uɣbalu, Bern tettwabna s yiẓra.", None, "cites_unseen_text"),
            ("Deg umagrad n wikipedya, Bern d tamaneɣt.", None, "cites_unseen_text"),
            ("Bniɣ ɣef umuddel Qwen3.5-4B-Base.", None, "names_base_model"),
            ("Mačči d Alibaba i iyi-d-yesnulfan.", None, "names_base_model"),
        ],
    )
    def test_an_instruction_leak_sets_a_candidate_aside(
        self, content: str, reasoning: str | None, flag: str
    ) -> None:
        found, cards = world(11)
        found[10] = candidate("c10", "k10", content, reasoning=reasoning)
        cards["k10"] = card("k10", think=reasoning is not None)
        _, ledger = run(found, cards, evidence(found))
        assert (ledger.status["c10"], ledger.reasons["c10"]) == ("flagged", [flag])

    def test_a_text_the_user_pasted_may_be_cited(self) -> None:
        found, cards = world(10)
        found[9] = Candidate(
            "c9",
            "k9",
            "batch-0001",
            1,
            (
                {"role": "user", "content": "Ɣer aḍris-a: Bern d tamaneɣt n Swis."},
                {"role": "assistant", "content": "Deg uḍris-nni, Bern d tamaneɣt."},
            ),
            "Gemini 3.8 Flash",
            "2026-10-17",
            1,
        )
        _, ledger = run(found, cards, evidence(found))
        assert ledger.status["c9"] == "accepted"

    def test_arabic_script_in_a_user_turn_or_a_translation_is_not_flagged(self) -> None:
        found, cards = world(10)
        user_arabic = Candidate(
            "c8",
            "k8",
            "batch-0001",
            1,
            (
                {"role": "user", "content": "ما هي عاصمة الجزائر؟"},
                {"role": "assistant", "content": answer(8)},
            ),
            "Gemini 4 Argon",
            "2026-10-17",
            1,
        )
        found[8] = user_arabic
        found[9] = candidate("c9", "k9", "الجزائر عاصمة الجزائر.")
        cards["k9"] = card("k9", "translation")
        _, ledger = run(found, cards, evidence(found))
        assert (ledger.status["c8"], ledger.status["c9"]) == ("accepted", "accepted")

    def test_an_uncomputed_signal_stops_the_build(self) -> None:
        found, cards = world()
        ev = evidence(found, signals={"c3": {"glotlid_kabyle": True}})
        with pytest.raises(sft_data.DataError, match="never computed"):
            run(found, cards, ev)


class TestReviewAndQuotas:
    def test_identity_waits_for_review_and_enters_after_it(self) -> None:
        found, cards = world(11)
        cards["k10"] = card("k10", "identity")
        _, waiting = run(found, cards, evidence(found))
        assert waiting.status["c10"] == "pending_review"
        _, reviewed = run(found, cards, evidence(found, reviews={"c10": Review("c10", True, "")}))
        assert reviewed.status["c10"] == "accepted"

    def test_a_review_rejection_is_final(self) -> None:
        found, cards = world(11)
        _, ledger = run(
            found, cards, evidence(found, reviews={"c10": Review("c10", False, "calque")})
        )
        assert ledger.status["c10"] == "rejected"

    def test_a_task_over_quota_cuts_the_last_in_stable_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sft_data.TASK_TARGETS, "native_kabyle", 10)
        found, cards = world(11)
        plain = [c.id for c in found if not cards[c.card].think]
        last = max(plain, key=sft_data.stable_rank)
        _, ledger = run(found, cards, evidence(found))
        assert [n for n, s in ledger.status.items() if s == "surplus"] == [last]

    def test_a_task_over_quota_keeps_think_mode_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sft_data.TASK_TARGETS, "native_kabyle", 10)
        found, cards = world(11, thinking=1)
        last = max(found, key=lambda c: sft_data.stable_rank(c.id))
        index = found.index(last)
        found[index] = candidate(last.id, last.card, answer(index), reasoning="Axemmem.")
        cards[last.card] = card(last.card, think=True)
        _, ledger = run(found, cards, evidence(found))
        assert ledger.status[last.id] == "accepted"
        assert sum(status == "surplus" for status in ledger.status.values()) == 1

    def test_validation_candidates_are_outside_the_quota(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sft_data.TASK_TARGETS, "native_kabyle", 10)
        found, cards = world(11)
        cards["k10"] = replace(card("k10"), split="validation")
        accepted, ledger = run(found, cards, evidence(found))
        assert ledger.status["c10"] == "accepted"
        assert len(accepted) == 11

    def test_a_category_review_failure_stops_the_build(self) -> None:
        found, cards = world(30, thinking=7)
        reviews = {f"c{i}": Review(f"c{i}", i % 4 != 0, "") for i in range(30)}
        with pytest.raises(sft_data.DataError, match=r"native_kabyle.*regenerate"):
            run(found, cards, evidence(found, reviews=reviews))

    def test_too_few_reviews_decide_nothing_about_a_category(self) -> None:
        found, cards = world(30, thinking=7)
        reviews = {"c10": Review("c10", False, "")}
        _, ledger = run(found, cards, evidence(found, reviews=reviews))
        assert ledger.status["c10"] == "rejected"

    def test_the_think_share_is_held_to_its_target(self) -> None:
        found, cards = world(10, thinking=5)
        with pytest.raises(sft_data.DataError, match=r"think-mode is 0\.500"):
            run(found, cards, evidence(found))

    def test_the_review_sample_takes_every_identity_candidate(self) -> None:
        found, cards = world(40, thinking=10)
        cards["k39"] = card("k39", "identity")
        chosen = sft_data.review_sample(found, cards, size=10)
        assert "c39" in chosen
        assert len(chosen) == 11
        assert chosen == sft_data.review_sample(found, cards, size=10)


class TestRows:
    def test_the_policy_is_added_only_for_its_condition(self) -> None:
        found, _ = world(1)
        with_policy = sft_data.with_policy(found[0], card("k0", system="policy"))
        without = sft_data.with_policy(found[0], card("k0", system="none"))
        assert with_policy[0] == {"role": "system", "content": SYSTEM_POLICY}
        assert without[0]["role"] == "user"

    def test_a_row_carries_its_provenance(self) -> None:
        found, _ = world(1)
        row = sft_data.accepted_row(found[0], card("k0"), "1.3.0")
        assert row["teacher"] == {
            "model": "Gemini 4 Argon",
            "date": "2026-10-17",
            "batch": "batch-0001",
        }
        assert (row["split"], row["normaliser"]) == ("train", "1.3.0")

    def test_reviews_are_read_strictly(self, tmp_path: Path) -> None:
        path = tmp_path / "reviews.jsonl"
        path.write_text(
            json.dumps({"candidate": "c1", "verdict": "maybe"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(sft_data.DataError, match="accept or reject"):
            sft_data.read_reviews(path)
        path.write_text(
            json.dumps({"candidate": "c1", "verdict": "accept"})
            + "\n"
            + json.dumps({"candidate": "c1", "verdict": "reject"})
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(sft_data.DataError, match="twice"):
            sft_data.read_reviews(path)
        assert sft_data.read_reviews(tmp_path / "absent.jsonl") == {}
