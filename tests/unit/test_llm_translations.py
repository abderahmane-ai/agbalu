"""Translation conversations from human pairs: templates, filters, the draw and the rows."""

from __future__ import annotations

import json

import pytest

from agbalu.llm import teacher, translations
from agbalu.llm.chat import SYSTEM_POLICY
from agbalu.llm.translations import Pair, Provenance, Template, TranslationError


def licence_of(source: str) -> str:
    return f"cc-by-2.0-fr ({source})"


PROVENANCE = Provenance(licence_of, lambda text: text.replace("ε", "ɛ"), "n1")


def templates(per_pair: int = translations.MIN_TEMPLATES) -> list[Template]:
    return [
        Template(ask, into, f"{ask}->{into} #{n}: {{text}}")
        for ask, into in translations.language_pairs()
        for n in range(per_pair)
    ]


def row(
    direction: str, kabyle: str, other: str, source: str = "opus.tatoeba-kab"
) -> dict[str, str]:
    text, target = (kabyle, other) if direction.startswith("kab") else (other, kabyle)
    return {"direction": direction, "source": text, "target": target, "source_id": source}


def sentence(index: int, language: str) -> str:
    return f"{language} tafyirt tis {index} i d-yettwarun s wawal ilaqen akka"


def corpus(count: int = 120) -> list[dict[str, str]]:
    """`count` Kabyle sentences, each paired in all four directions, as the file holds them."""
    rows: list[dict[str, str]] = []
    for i in range(count):
        for direction in translations.DIRECTIONS:
            other = "eng" if "eng" in direction else "fra"
            rows.append(row(direction, sentence(i, "kab"), sentence(i, other)))
    return rows


class TestRequests:
    def test_the_prompt_asks_for_every_language_pair(self) -> None:
        text = translations.requests_prompt(teacher.load_prompts())
        for ask, into in translations.language_pairs():
            assert f"ask: {ask} ({teacher.LANGUAGE_NAMES[ask]}), into: {into} " in text
        assert "{text}" in text
        assert "{pairs}" not in text
        assert "{per_pair}" not in text

    def test_each_bad_line_is_counted_by_its_reason(self) -> None:
        lines = [
            '{"ask": "eng", "into": "kab", "template": "Translate into Kabyle: {text}"}',
            '{"ask": "eng", "into": "kab", "template": "Translate into Kabyle: {text}"}',
            '{"ask": "zho", "into": "eng", "template": "{text}"}',
            '{"ask": "eng", "into": "kab", "template": "no placeholder"}',
            '{"ask": "eng", "into": "kab", "template": "{text} and {name}"}',
            "not json",
            "```",
            "",
        ]
        found, rejected = translations.parse_requests(lines)
        assert [t.text for t in found] == ["Translate into Kabyle: {text}"]
        assert rejected == {
            "duplicate": 1,
            "unknown_languages": 1,
            "placeholder": 1,
            "stray_brace": 1,
            "not_json": 1,
        }

    def test_a_language_pair_short_of_templates_stops_the_build(self) -> None:
        short = [t for t in templates() if (t.ask, t.into) != ("zho", "kab")]
        with pytest.raises(TranslationError, match=r"zho->kab"):
            translations.check_templates(short)


class TestPairs:
    @pytest.mark.parametrize(
        ("pair", "reason"),
        [
            (
                row("kab-eng", "Tom yella da ass-a s tidet akk", "Tom was here today truly indeed"),
                "stock_names",
            ),
            (row("kab-eng", "Azul", "Hello"), "length"),
            (row("kab-spa", sentence(1, "kab"), sentence(1, "spa")), "direction"),
            (row("kab-eng", sentence(1, "kab"), sentence(1, "eng"), "hf.tutlaytai.x"), "source"),
        ],
    )
    def test_a_filtered_pair_is_counted_under_its_reason(
        self, pair: dict[str, str], reason: str
    ) -> None:
        tally = translations.PairTally()
        found = translations.eligible_pairs([pair], lambda _: True, set(), tally)
        assert not any(found.values())
        assert (tally.read, dict(tally.dropped)) == (1, {reason: 1})

    def test_evaluation_text_and_card_sentences_are_kept_out(self) -> None:
        rows = [row("kab-eng", sentence(1, "kab"), sentence(1, "eng"))]
        rows.append(row("kab-eng", sentence(2, "kab"), sentence(2, "eng")))
        tally = translations.PairTally()
        found = translations.eligible_pairs(
            rows, lambda text: "tis 1 " not in text, {sentence(2, "kab")}, tally
        )
        assert not found["kab-eng"]
        assert tally.dropped == {"evaluation_or_held_out": 1, "used_by_a_card": 1}

    def test_no_kabyle_sentence_is_drawn_in_two_directions(self) -> None:
        tally = translations.PairTally()
        eligible = translations.eligible_pairs(corpus(), lambda _: True, set(), tally)
        chosen = translations.draw_pairs(eligible)
        kabyle = [pair.kabyle for pairs in chosen.values() for pair in pairs]
        assert {d: len(p) for d, p in chosen.items()} == translations.DIRECTIONS
        assert len(set(kabyle)) == len(kabyle)

    def test_too_few_pairs_stop_the_build(self) -> None:
        tally = translations.PairTally()
        eligible = translations.eligible_pairs(corpus(20), lambda _: True, set(), tally)
        with pytest.raises(TranslationError, match="eligible pairs"):
            translations.draw_pairs(eligible)


class TestConversations:
    def test_every_pair_read_is_accounted_and_every_row_is_well_formed(self) -> None:
        rows, stats = translations.build(corpus(), templates(), lambda _: True, set(), PROVENANCE)
        assert len(rows) == sum(translations.DIRECTIONS.values())
        eligible, dropped = stats["eligible"], stats["dropped"]
        assert isinstance(eligible, dict)
        assert isinstance(dropped, dict)
        assert sum(eligible.values()) + sum(dropped.values()) == stats["pairs_read"]
        assert len({r["id"] for r in rows}) == len(rows)
        for conversation in rows:
            turns = conversation["turns"]
            assert isinstance(turns, list)
            roles = [turn["role"] for turn in turns]
            assert roles in (["user", "assistant"], ["system", "user", "assistant"])
            assert (conversation["system"] == "policy") == (roles[0] == "system")
            if roles[0] == "system":
                assert turns[0]["content"] == SYSTEM_POLICY
            assert "{text}" not in turns[-2]["content"]
            assert conversation["split"] == "train"
            assert conversation["think"] is False
        json.dumps(rows, ensure_ascii=False)

    def test_the_answer_is_the_human_translation_into_the_asked_language(self) -> None:
        pair = Pair("eng-kab", "The mill turns all day long.", "Tasirt tezzi akk ass ε.", "s", 7)
        rows = translations.conversations({"eng-kab": [pair]}, templates(), PROVENANCE)
        turns = rows[0]["turns"]
        assert isinstance(turns, list)
        assert turns[-1] == {"role": "assistant", "content": "Tasirt tezzi akk ass ɛ."}
        assert "The mill turns all day long." in turns[-2]["content"]
        request = rows[0]["request"]
        assert isinstance(request, dict)
        assert request["into"] == "kab"
        assert (rows[0]["record"], rows[0]["licence"]) == (7, "cc-by-2.0-fr (s)")
