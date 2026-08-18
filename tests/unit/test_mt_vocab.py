"""Vocabulary trimming.

A trimmed model and an untrimmed tokenizer disagree silently — the model answers fluently
in the wrong tokens rather than raising — so the id map is where the correctness lives and
is tested without needing a checkpoint.
"""

from __future__ import annotations

from typing import Final

import pytest
import torch

from agbalu.mt.vocab import LANGUAGE_CODES, Coverage, Remap, TrimError, coverage, required_ids

SPECIALS: Final = (0, 1, 2, 3)


class FakeTokenizer:
    """The slice of the tokenizer API `vocab` drives.

    `convert_tokens_to_ids` returns `unk_token_id` for an unknown token rather than
    raising, which is the behaviour the language-code check exists to catch.
    """

    def __init__(self, codes: dict[str, int] | None = None, size: int = 100) -> None:
        self.all_special_ids = list(SPECIALS)
        self.unk_token_id = 3
        self.size = size
        self.codes = (
            codes if codes is not None else {code: 10 + i for i, code in enumerate(LANGUAGE_CODES)}
        )

    def __len__(self) -> int:
        return self.size

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.codes.get(token, self.unk_token_id)

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        assert not add_special_tokens
        return {"input_ids": [[20 + len(t), 21 + len(t)] for t in texts]}


class TestRequiredIds:
    def test_specials_and_language_codes_all_survive(self) -> None:
        ids = required_ids(FakeTokenizer())
        assert set(SPECIALS) <= ids
        assert {10, 11, 12} <= ids

    def test_a_missing_language_code_raises_rather_than_translating_wrongly(self) -> None:
        """`convert_tokens_to_ids` answers unk for an unknown code, so an unchecked trim
        would produce fluent output in the wrong language."""
        tokenizer = FakeTokenizer(codes={"eng_Latn": 11})
        with pytest.raises(TrimError, match="kab_Latn"):
            required_ids(tokenizer)


class TestCoverage:
    def test_it_counts_what_the_text_reaches_plus_what_must_survive(self) -> None:
        found = coverage(["ab", "cde"], FakeTokenizer())
        assert {22, 23, 24} <= set(found.keep)
        assert set(SPECIALS) <= set(found.keep)
        assert found.scanned == 2

    def test_keep_is_ascending_so_the_new_id_is_its_index(self) -> None:
        found = coverage(["x"], FakeTokenizer())
        assert list(found.keep) == sorted(found.keep)

    def test_empty_text_still_keeps_the_specials(self) -> None:
        found = coverage([], FakeTokenizer())
        assert set(found.keep) == set(SPECIALS) | {10, 11, 12}
        assert found.scanned == 0

    def test_the_batch_boundary_does_not_lose_anything(self) -> None:
        small = coverage([str(i) for i in range(10)], FakeTokenizer(), batch_size=1)
        large = coverage([str(i) for i in range(10)], FakeTokenizer(), batch_size=1_000)
        assert small.keep == large.keep

    def test_share_is_of_the_original_vocabulary(self) -> None:
        found = coverage(["a"], FakeTokenizer(size=200))
        assert found.share == pytest.approx(found.kept / 200)

    def test_share_of_an_empty_vocabulary_is_zero_not_a_crash(self) -> None:
        assert Coverage(keep=(), scanned=0, original_size=0).share == 0.0


class TestRemap:
    def test_new_ids_are_positions_in_keep(self) -> None:
        remap = Remap((0, 1, 2, 3, 10, 250))
        assert remap.to_new([0, 10, 250]) == [0, 4, 5]

    def test_it_round_trips(self) -> None:
        keep = (0, 1, 2, 3, 7, 99, 1_000)
        remap = Remap(keep)
        assert remap.to_old(remap.to_new(keep)) == list(keep)

    def test_a_trimmed_away_token_raises_instead_of_becoming_unk(self) -> None:
        """Silently mapping to unk would train the model on text it cannot represent."""
        remap = Remap((0, 1, 2))
        with pytest.raises(TrimError, match="trimmed away"):
            remap.to_new([0, 55])

    def test_decoding_ignores_ids_outside_the_trimmed_range(self) -> None:
        """Generation can emit an id past the table before the model has learned not to."""
        remap = Remap((0, 1, 2))
        assert remap.to_old([0, 2, 99, -1]) == [0, 2]

    def test_an_empty_keep_maps_nothing(self) -> None:
        assert Remap(()).to_old([0, 1]) == []


class TestVocabTables:
    """The generation path, whose contract is deliberately the opposite of `Remap`'s: it
    meets text the trim never scanned, so an unrepresentable id is out-of-vocabulary
    rather than a bug."""

    def test_kept_ids_map_to_their_positions(self) -> None:
        tables = Remap((0, 1, 2, 3, 10, 250)).tables(original_size=300, unk_id=3)
        assert tables.to_new(torch.tensor([[0, 10, 250]])).tolist() == [[0, 4, 5]]

    def test_a_trimmed_away_id_becomes_unk(self) -> None:
        """Raising here ends a scoring run over one token in one sentence of a thousand."""
        tables = Remap((0, 1, 2, 3, 10)).tables(original_size=300, unk_id=3)
        assert tables.to_new(torch.tensor([[10, 55, 0]])).tolist() == [[4, 3, 0]]

    def test_it_round_trips_what_it_kept(self) -> None:
        keep = (0, 1, 2, 3, 7, 99)
        tables = Remap(keep).tables(original_size=300, unk_id=3)
        ids = torch.tensor([list(keep)])
        assert tables.to_old(tables.to_new(ids)).tolist() == [list(keep)]

    def test_a_trimmed_away_unknown_token_is_refused(self) -> None:
        """Without an `unk` row there is nothing for `to_new` to fall back to."""
        with pytest.raises(TrimError, match="cannot be used"):
            Remap((0, 1, 2)).tables(original_size=300, unk_id=3)

    def test_an_unknown_token_outside_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(TrimError, match="outside the untrimmed vocabulary"):
            Remap((0, 1, 2, 3)).tables(original_size=4, unk_id=9)

    def test_a_forced_token_that_was_trimmed_away_still_raises(self) -> None:
        """A missing language token would translate fluently into the wrong language, so
        it must never quietly become unk."""
        tables = Remap((0, 1, 2, 3)).tables(original_size=300, unk_id=3)
        with pytest.raises(TrimError, match="cannot be used"):
            tables.new_id(250)

    def test_a_keep_longer_than_the_vocabulary_is_refused(self) -> None:
        with pytest.raises(TrimError, match="smaller than"):
            Remap((0, 1, 2, 3)).tables(original_size=2, unk_id=3)
