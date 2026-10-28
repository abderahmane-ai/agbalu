"""The seed bank: every quota is met exactly, the draw is seeded, seed texts come from the pool."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from agbalu.llm import seeds
from agbalu.llm.seeds import Card, Passage


@pytest.fixture(scope="module")
def cards() -> list[Card]:
    return seeds.allocate()


def passage(index: int, *, other: str | None = None) -> Passage:
    return Passage(
        f"Aḍris {index} " * 50, "src", f"r{index}", "cc-by-sa-4.0", other, "eng" if other else None
    )


class TestQuotas:
    def test_tasks_and_domains_are_exact(self, cards: list[Card]) -> None:
        assert Counter(card.task for card in cards) == seeds.TASK_CARDS
        assert Counter(card.domain for card in cards) == seeds.DOMAIN_CARDS

    def test_thinking_is_exact_per_domain(self, cards: list[Card]) -> None:
        assert Counter(card.domain for card in cards if card.think) == seeds.THINK_CARDS
        assert sum(card.think for card in cards) == 240

    def test_a_fifth_of_the_conversations_think(self, cards: list[Card]) -> None:
        assert sum(card.think for card in cards) / len(cards) == pytest.approx(0.20)

    def test_splits_are_five_percent_each(self, cards: list[Card]) -> None:
        split = Counter(card.split for card in cards)
        assert (split["validation"], split["release"]) == (60, 60)

    def test_system_conditions_and_turns_are_near_their_shares(self, cards: list[Card]) -> None:
        system = Counter(card.system for card in cards)
        assert abs(system["policy"] / len(cards) - 0.60) < 0.05
        assert abs(system["none"] / len(cards) - 0.25) < 0.05
        multi = sum(card.user_turns == seeds.MULTI_TURN_USERS for card in cards)
        assert abs(multi / len(cards) - 0.25) < 0.05

    def test_non_kabyle_input_cards_are_never_asked_in_kabyle(self, cards: list[Card]) -> None:
        assert all(
            card.input_language in seeds.OTHER_LANGUAGES
            for card in cards
            if card.task == "non_kabyle_input"
        )

    def test_the_allocation_is_seeded(self, cards: list[Card]) -> None:
        assert seeds.allocate() == cards
        assert seeds.allocate(seed=1) != cards

    def test_ids_are_unique(self, cards: list[Card]) -> None:
        assert len({card.id for card in cards}) == len(cards)

    def test_the_working_set_covers_every_task_a_fifth_thinks(self, cards: list[Card]) -> None:
        chosen = seeds.working_set(cards)
        assert len(chosen) == 131
        assert len({card.id for card in chosen}) == len(chosen)
        assert {card.task for card in chosen} == set(seeds.TASK_CARDS)
        splits = [card.split for card in chosen]
        assert (splits.count("train"), splits.count("validation")) == (125, 6)
        train = [card for card in chosen if card.split == "train"]
        assert 0.18 <= sum(card.think for card in train) / len(train) <= 0.22
        assert sum(card.task == "identity" for card in chosen) == 4
        assert chosen == seeds.working_set(cards)

    def test_teacher_cards_covers_targeted_tasks_and_is_balanced(self, cards: list[Card]) -> None:
        asked = seeds.teacher_cards(cards)
        assert all(card.split == "train" and card.task in seeds.TASKS_ROTATION for card in asked)
        assert set(card.task for card in asked) == set(seeds.TASKS_ROTATION)
        think_share = sum(card.think for card in asked) / len(asked)
        assert 0.18 <= think_share <= 0.35


class TestSeedTexts:
    def test_every_card_that_needs_a_seed_text_gets_its_own(self, cards: list[Card]) -> None:
        pool = [passage(i) for i in range(2000)]
        pairs = [passage(i, other=f"text {i}") for i in range(2000)]
        attached = seeds.attach(cards, pool, pairs)
        for card in attached:
            if seeds.needs_passage(card):
                assert card.passage is not None
                assert card.passage.other is None
            elif seeds.needs_pair(card):
                assert card.passage is not None
                assert card.passage.other is not None
            else:
                assert card.passage is None
        records = [
            card.passage.record for card in attached if card.passage and not card.passage.other
        ]
        assert len(records) == len(set(records))

    def test_a_pool_too_small_is_refused(self, cards: list[Card]) -> None:
        with pytest.raises(seeds.SeedError, match="cards need passages"):
            seeds.attach(cards, [passage(1)], [passage(i, other="x") for i in range(2000)])

    def test_the_reservoir_reaches_past_the_prefix(self) -> None:
        drawn = seeds.reservoir([passage(i) for i in range(1000)], 20, seed=2)
        assert max(int(p.record[1:]) for p in drawn) > 500

    def test_wiki_openings_are_cut_at_a_paragraph_and_short_ones_dropped(self) -> None:
        long = ("Amezwaru. " * 50) + "\n" + ("Wis sin. " * 200)
        found = list(seeds.wiki_passages([("a", long), ("b", "Awal.")], "cc-by-sa-4.0"))
        assert [p.record for p in found] == ["a"]
        assert len(found[0].text) <= seeds.PASSAGE_CHARACTERS


class TestPairs:
    def test_one_direction_of_each_long_pair_is_kept_with_its_licence(self) -> None:
        long = "Taswiɛt kan tebbejneq d imeṭṭawen deg uxxam n baba-s ass-a"
        rows = [
            {
                "source": long,
                "target": "Suddenly she burst into tears.",
                "direction": "kab-eng",
                "source_id": "opus.tatoeba-kab",
            },
            {
                "source": "Suddenly she burst into tears.",
                "target": long,
                "direction": "eng-kab",
                "source_id": "opus.tatoeba-kab",
            },
            {"source": "Gabryola", "target": "Gabriola", "direction": "kab-eng", "source_id": "x"},
        ]
        found = list(seeds.pair_passages(rows, lambda source: f"licence of {source}"))
        assert len(found) == 1
        assert (found[0].text, found[0].other_language) == (long, "eng")
        assert found[0].licence == "licence of opus.tatoeba-kab"


class TestFiles:
    def test_cards_round_trip(self, cards: list[Card], tmp_path: Path) -> None:
        attached = seeds.attach(
            cards, [passage(i) for i in range(2000)], [passage(i, other="x") for i in range(2000)]
        )
        seeds.write(attached, tmp_path / "seeds.jsonl")
        assert seeds.read(tmp_path / "seeds.jsonl") == attached

    def test_the_summary_counts_every_card(self, cards: list[Card]) -> None:
        stats = seeds.summary(cards)
        assert stats["cards"] == 1200
        assert stats["conversations_requested"] == 12000

    def test_a_missing_file_names_its_command(self, tmp_path: Path) -> None:
        with pytest.raises(seeds.SeedError, match="make llm TASK=seeds"):
            seeds.read(tmp_path / "absent.jsonl")
