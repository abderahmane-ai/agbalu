from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from agbalu.tokenizer.seed import (
    LEXICON_FLOOR,
    SeedPool,
    build_pool,
    clitic_pieces,
    lexicon_pieces,
    state_pieces,
    substring_counts,
    write_seed_file,
)
from agbalu.tokenizer.spec import MAX_PIECE_LENGTH, METASPACE, TokenizerError, required_chars


def write_lexicon(path: Path, forms: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"form": f}, ensure_ascii=False) + "\n" for f in forms),
        encoding="utf-8",
    )
    return path


class TestSubstringCounts:
    def test_marks_word_initial_pieces_with_the_metaspace(self) -> None:
        """`▁azul` and `azul` are different candidates and SentencePiece sees both: the
        second is what a non-initial occurrence elsewhere in the corpus would use."""
        counts = substring_counts(Counter({"azul": 5}))
        assert counts["▁azul"] == 5
        assert counts["azul"] == 5
        assert counts["zul"] == 5

    def test_weights_by_type_frequency(self) -> None:
        counts = substring_counts(Counter({"azul": 7}))
        assert counts["▁a"] == 7

    def test_accumulates_a_piece_across_types(self) -> None:
        counts = substring_counts(Counter({"axxam": 3, "wexxam": 4}))
        assert counts["xxam"] == 7

    def test_drops_types_below_the_frequency_floor(self) -> None:
        counts = substring_counts(Counter({"azul": 1}), min_type_count=2)
        assert counts == {}

    def test_honours_the_floor_boundary(self) -> None:
        counts = substring_counts(Counter({"azul": 2}), min_type_count=2)
        assert counts["▁azul"] == 2

    def test_never_exceeds_the_maximum_piece_length(self) -> None:
        long_word = "a" * 40
        counts = substring_counts(Counter({long_word: 2}))
        assert max(len(p) for p in counts) == MAX_PIECE_LENGTH

    def test_empty_table_gives_no_candidates(self) -> None:
        assert substring_counts(Counter()) == {}


class TestStatePieces:
    def test_offers_both_sides_of_the_alternation(self) -> None:
        pieces = state_pieces()
        assert "▁we" in pieces
        assert "▁a" in pieces
        assert "▁ta" in pieces
        assert "▁ti" in pieces

    def test_every_piece_is_word_initial(self) -> None:
        assert all(p.startswith(METASPACE) for p in state_pieces())


class TestLexiconPieces:
    def test_missing_lexicon_names_the_rebuild_command(self, tmp_path: Path) -> None:
        with pytest.raises(TokenizerError, match="make lexicon"):
            lexicon_pieces(tmp_path / "absent.jsonl")

    def test_offers_the_whole_form_and_its_stem(self, tmp_path: Path) -> None:
        pieces = lexicon_pieces(write_lexicon(tmp_path / "l.jsonl", ["axxam"]))
        assert "▁axxam" in pieces
        assert "xxam" in pieces

    def test_strips_the_longest_prefix_first(self, tmp_path: Path) -> None:
        pieces = lexicon_pieces(write_lexicon(tmp_path / "l.jsonl", ["tamurt"]))
        assert "murt" in pieces
        assert "amurt" not in pieces

    def test_skips_stems_too_short_to_be_evidence(self, tmp_path: Path) -> None:
        pieces = lexicon_pieces(write_lexicon(tmp_path / "l.jsonl", ["ass"]))
        assert "▁ass" in pieces
        assert "ss" not in pieces

    def test_skips_multiword_forms(self, tmp_path: Path) -> None:
        pieces = lexicon_pieces(write_lexicon(tmp_path / "l.jsonl", ["sider tiddi"]))
        assert pieces == set()

    def test_skips_forms_that_could_never_be_one_piece(self, tmp_path: Path) -> None:
        pieces = lexicon_pieces(write_lexicon(tmp_path / "l.jsonl", ["a" * MAX_PIECE_LENGTH]))
        assert pieces == set()

    def test_skips_empty_forms(self, tmp_path: Path) -> None:
        assert lexicon_pieces(write_lexicon(tmp_path / "l.jsonl", [""])) == set()

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        path.write_text('\n{"form": "azul"}\n\n', encoding="utf-8")
        assert "▁azul" in lexicon_pieces(path)


class TestCliticPieces:
    def test_takes_only_what_follows_a_hyphen(self) -> None:
        pieces = clitic_pieces(Counter({"ɣur-s": 10, "azul": 99}))
        assert pieces == {"s"}

    def test_collects_every_segment_of_a_cluster(self) -> None:
        pieces = clitic_pieces(Counter({"yefka-yas-t-id": 5}))
        assert pieces == {"yas", "t", "id"}

    def test_keeps_only_the_most_frequent(self) -> None:
        freq = Counter({"a-x": 100, "a-y": 50, "a-z": 1})
        assert clitic_pieces(freq, top=2) == {"x", "y"}

    def test_no_hyphens_gives_nothing(self) -> None:
        assert clitic_pieces(Counter({"azul": 10})) == set()


class TestBuildPool:
    def test_rejects_a_non_positive_cap(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["azul"])
        with pytest.raises(TokenizerError, match="must be positive"):
            build_pool(Counter({"azul": 5}), lexicon, seed_size=0)

    def test_forces_lexical_pieces_past_the_cap(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["taqbaylit"])
        pool = build_pool(Counter({"azul": 50}), lexicon, seed_size=1)
        pieces = {p for p, _ in pool.pieces}
        assert pool.from_corpus == 1
        assert "▁taqbaylit" in pieces
        assert "qbaylit" in pieces

    def test_a_piece_the_corpus_never_shows_still_enters(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["tafelwit"])
        pool = build_pool(Counter({"azul": 5}), lexicon)
        scores = dict(pool.pieces)
        assert scores["▁tafelwit"] == float(LEXICON_FLOOR * len("▁tafelwit"))

    def test_a_lexical_piece_the_corpus_shows_keeps_its_real_count(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["axxam"])
        pool = build_pool(Counter({"axxam": 9}), lexicon)
        scores = dict(pool.pieces)
        assert scores["xxam"] == float(9 * len("xxam"))

    def test_every_required_character_is_a_candidate(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["azul"])
        pool = build_pool(Counter({"azul": 5}), lexicon)
        pieces = {p for p, _ in pool.pieces}
        assert set(required_chars()) <= pieces

    def test_counts_what_seeding_alone_contributed(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["azul"])
        pool = build_pool(Counter({"azul": 5}), lexicon)
        assert pool.lexicon_only > 0
        assert pool.from_corpus + pool.lexicon_only == len(pool)

    def test_is_ordered_by_descending_gain(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["azul"])
        pool = build_pool(Counter({"azul": 50, "aqbayli": 3}), lexicon)
        gains = [score for _, score in pool.pieces]
        assert gains == sorted(gains, reverse=True)

    def test_holds_no_duplicate_pieces(self, tmp_path: Path) -> None:
        lexicon = write_lexicon(tmp_path / "l.jsonl", ["axxam", "wexxam"])
        pool = build_pool(Counter({"axxam": 4, "wexxam": 6}), lexicon)
        pieces = [p for p, _ in pool.pieces]
        assert len(pieces) == len(set(pieces))


class TestWriteSeedFile:
    def test_refuses_an_empty_pool(self, tmp_path: Path) -> None:
        with pytest.raises(TokenizerError, match="empty seed pool"):
            write_seed_file(SeedPool((), 0, 0, 0), tmp_path / "seed.tsv")

    def test_writes_piece_tab_score(self, tmp_path: Path) -> None:
        dest = tmp_path / "nested" / "seed.tsv"
        write_seed_file(SeedPool((("▁azul", 12.0), ("zul", 3.5)), 2, 0, 0), dest)
        rows = [line.split("\t") for line in dest.read_text(encoding="utf-8").splitlines()]
        assert rows == [["▁azul", "12.000000"], ["zul", "3.500000"]]

    def test_leaves_no_staging_file_behind(self, tmp_path: Path) -> None:
        dest = tmp_path / "seed.tsv"
        write_seed_file(SeedPool((("▁azul", 1.0),), 1, 0, 0), dest)
        assert list(tmp_path.glob("*.partial")) == []
