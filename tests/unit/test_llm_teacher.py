"""The Gemini exchange: prompts from the seed bank, every reply line accounted for."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agbalu.llm import seeds, teacher
from agbalu.llm.seeds import CONVERSATIONS_PER_CARD, Card, Passage

PROMPTS = teacher.load_prompts()


def card(
    identifier: str = "card-0001",
    *,
    think: bool = False,
    system: seeds.System = "none",
    turns: int = 1,
) -> Card:
    return Card(identifier, "native_kabyle", "culture", think, "kab", system, turns, "train")


def conversation(card_id: str, n: int, *, think: bool = False, system: bool = False) -> str:
    answer = {"role": "assistant", "content": "Azul fell-ak."}
    if think:
        answer["reasoning"] = "Ad d-rreɣ tiririt."
    turns = [{"role": "user", "content": "Azul!"}, answer]
    if system:
        turns.insert(0, {"role": "system", "content": "Kečč d amɛiwen n uɣerbaz."})
    return json.dumps({"card": card_id, "n": n, "turns": turns}, ensure_ascii=False)


def reply(lines: list[str]) -> teacher.Reply:
    return teacher.Reply("Gemini 4 Argon", "2026-10-17", tuple(lines))


class TestPrompts:
    def test_the_prompt_file_matches_the_seed_bank(self) -> None:
        assert set(PROMPTS.tasks) == set(seeds.TASK_CARDS)
        assert set(PROMPTS.domains) == set(seeds.DOMAIN_CARDS)
        assert set(PROMPTS.systems) == set(seeds.SYSTEM_SHARES)

    def test_a_prompt_names_every_card_and_leaves_no_placeholder(self) -> None:
        cards = [card("card-0001"), card("card-0002", think=True)]
        text = teacher.generator_prompt(cards, PROMPTS)
        assert "CARD: card-0001" in text
        assert "CARD: card-0002" in text
        assert "THINK: yes" in text
        for name in ("{orthography}", "{per_card}", "{schema}", "{cards}"):
            assert name not in text
        assert f"Write exactly {CONVERSATIONS_PER_CARD} conversations" in text

    def test_the_orthography_comes_from_the_normaliser(self) -> None:
        lines = teacher.orthography()
        assert "ɛ never as" in lines
        assert "ε" in lines
        assert "č ḍ ɛ ǧ ɣ ḥ ṛ ṣ ṭ ẓ" in lines

    def test_a_grounded_card_carries_its_source_and_a_pair_card_its_pair(self) -> None:
        source = Passage("Tizi Wezzu d tamdint.", "wikimedia.kabwiki", "r1", "cc-by-sa-4.0")
        pair = Passage("Azul.", "mt-train", "3", "cc-by-4.0", "Hello.", "eng")
        grounded = teacher.card_block(replace(card(), passage=source), PROMPTS)
        paired = teacher.card_block(replace(card(), passage=pair), PROMPTS)
        assert "SOURCE (wikimedia.kabwiki):\nTizi Wezzu d tamdint." in grounded
        assert "English: Hello." in paired

    def test_a_prompt_file_with_a_missing_task_is_refused(self, tmp_path: Path) -> None:
        text = Path("resources/llm/teacher_prompts.yaml").read_text(encoding="utf-8")
        broken = tmp_path / "p.yaml"
        broken.write_text(
            text.replace("  identity: The user asks", "  someone: The user asks"), encoding="utf-8"
        )
        with pytest.raises(teacher.TeacherError, match="tasks keys differ"):
            teacher.load_prompts(broken)

    def test_batches_hold_their_cards_and_a_manifest(self, tmp_path: Path) -> None:
        cards = [card(f"card-{i:04d}") for i in range(5)]
        written = teacher.write_batches(cards, PROMPTS, tmp_path, first=7)
        assert [p.name for p in written] == ["batch-0007.txt", "batch-0008.txt", "batch-0009.txt"]
        manifest = json.loads((tmp_path / "batch-0009.json").read_text(encoding="utf-8"))
        assert manifest["cards"] == ["card-0004"]


class TestReplies:
    def test_the_header_is_required(self, tmp_path: Path) -> None:
        path = tmp_path / "batch-0001.txt"
        path.write_text(conversation("card-0001", 1) + "\n", encoding="utf-8")
        with pytest.raises(teacher.TeacherError, match="first line must be"):
            teacher.read_reply(path)

    def test_a_saved_reply_is_read_with_its_model_and_date(self, tmp_path: Path) -> None:
        path = tmp_path / "batch-0001.txt"
        path.write_text(
            "# model: Gemini 4 Argon | date: 2026-10-17\n" + conversation("card-0001", 1),
            encoding="utf-8",
        )
        read = teacher.read_reply(path)
        assert (read.model, read.date, len(read.lines)) == ("Gemini 4 Argon", "2026-10-17", 1)

    def test_every_line_is_accounted_for(self) -> None:
        cards = {"card-0001": card()}
        lines = [
            "```json",
            conversation("card-0001", 1),
            conversation("card-0001", 1),
            conversation("card-0009", 2),
            "not json at all",
            json.dumps({"card": "card-0001", "n": 99, "turns": []}),
            conversation("card-0001", 3, think=True),
            "",
            "```",
        ]
        found, rejects, tally = teacher.parse_reply(reply(lines), "batch-0001", cards, 1)
        assert [c.n for c in found] == [1]
        assert dict(tally.rejected) == {
            "duplicate": 1,
            "unknown_card": 1,
            "not_json": 1,
            "bad_n": 1,
            "reasoning_for_mode": 1,
        }
        assert tally.ignored == 3
        assert len(found) + sum(tally.rejected.values()) + tally.ignored == len(lines)
        assert tally.missing == CONVERSATIONS_PER_CARD - 1
        assert {r["reason"] for r in rejects} == set(tally.rejected)

    @pytest.mark.parametrize(
        ("target", "line", "reason"),
        [
            (card(system="operator"), conversation("card-0001", 1), "operator_system_missing"),
            (card(), conversation("card-0001", 1, system=True), "unexpected_system"),
            (card(turns=3), conversation("card-0001", 1), "wrong_turns"),
            (card(think=True), conversation("card-0001", 1), "reasoning_for_mode"),
        ],
    )
    def test_a_shape_the_card_did_not_ask_for_is_rejected(
        self, target: Card, line: str, reason: str
    ) -> None:
        _, _, tally = teacher.parse_reply(reply([line]), "b", {"card-0001": target}, 1)
        assert dict(tally.rejected) == {reason: 1}

    def test_an_operator_card_keeps_its_system_turn(self) -> None:
        cards = {"card-0001": card(system="operator")}
        found, _, _ = teacher.parse_reply(
            reply([conversation("card-0001", 1, system=True)]), "b", cards, 1
        )
        assert found[0].turns[0]["role"] == "system"
        assert found[0].id == "card-0001-b-01"

    def test_short_cards_are_asked_for_again_unless_outstanding(self) -> None:
        cards = [card("a"), card("b"), card("c")]
        again = teacher.wanting(cards, {"a": CONVERSATIONS_PER_CARD, "b": 3}, outstanding={"c"})
        assert [c.id for c in again] == ["b"]


class TestTranslationPilot:
    def test_sentences_are_numbered_from_their_offset(self) -> None:
        text = teacher.translation_prompt(["Hello.", "Good night."], 50, PROMPTS)
        assert "50. Hello.\n51. Good night." in text
        assert "{sentences}" not in text

    def test_every_line_is_a_translation_or_a_counted_rejection(self) -> None:
        lines = [
            json.dumps({"i": 50, "kab": "Azul."}),
            json.dumps({"i": 50, "kab": "Azul fell-ak."}),
            json.dumps({"i": 99, "kab": "x"}),
            json.dumps({"i": 51, "kab": " "}),
            "garbage",
        ]
        found, rejected = teacher.parse_translations(lines, range(50, 52))
        assert found == {50: "Azul."}
        assert dict(rejected) == {
            "duplicate": 1,
            "unknown_sentence": 1,
            "empty": 1,
            "not_json": 1,
        }


def test_the_prompt_file_holds_no_critic() -> None:
    assert "critic" not in teacher.PROMPT_KEYS
    assert not hasattr(PROMPTS, "critic")
