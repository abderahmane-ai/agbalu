"""Kokoro's token table, the three rows Kabyle takes, and the refusal on everything else."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from agbalu.tts.g2p import LENGTH, PLAIN, SPIRANTS, phonemize_word, unsupported
from agbalu.tts.kokoro import fold
from agbalu.tts.vocabulary import PLBERT_LIMIT, TABLE, Vocabulary, VocabularyError

if TYPE_CHECKING:
    from pathlib import Path


def table(
    base: dict[str, int],
    assigned: dict[str, int],
    *,
    n_token: int = 16,
    pad: int = 0,
) -> dict[str, Any]:
    return {
        "n_token": n_token,
        "pad": {"symbol": "$", "index": pad},
        "base": base,
        "assigned": assigned,
    }


def written(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestVendoredTable:
    def test_loads_the_repository_table(self) -> None:
        loaded = Vocabulary.load()
        assert loaded.n_token == 178
        assert loaded.pad_index == 0

    def test_kokoro_maps_one_hundred_and_fourteen_symbols(self) -> None:
        """114 mapped, one pad, three assigned to Kabyle — 118 of 178 rows claimed."""
        loaded = Vocabulary.load()
        assert len(loaded.symbols) == 118
        assert len(loaded.free()) == 60

    def test_kabyle_takes_exactly_three_rows(self) -> None:
        assert dict(Vocabulary.load().assigned) == {"ħ": 7, "ʕ": 8, "ˤ": 26}

    def test_the_assigned_rows_were_free(self) -> None:
        """Each assigned index must have been unclaimed by Kokoro, or the fine-tune would
        overwrite a trained embedding."""
        payload = json.loads(TABLE.read_text(encoding="utf-8"))
        kokoro = set(payload["base"].values()) | {payload["pad"]["index"]}
        assert not kokoro & set(payload["assigned"].values())

    def test_covers_everything_the_g2p_can_emit_once_folded(self) -> None:
        """The three-rows-not-four claim, over whole emissions rather than characters."""
        loaded = Vocabulary.load()
        emitted = {LENGTH, "ɑ", *PLAIN.values()}
        for short, stop in SPIRANTS.values():
            emitted.update({short, stop})

        unmapped = {s for emission in emitted for s in loaded.unmapped(fold(emission))}
        assert unmapped == set()

    def test_the_character_inventory_reports_a_fourth_symbol_that_is_not_one(self) -> None:
        """`g2p.inventory()` splits multi-character outputs, so the tie bar appears in it
        standalone and counts as a fourth gap. It is only ever emitted inside `t͡ʃ` and
        `d͡ʒ`, which `fold` rewrites onto rows Kokoro already has. Four unfolded, three
        folded — pinned here so the two figures are never reconciled the wrong way."""
        loaded = Vocabulary.load()
        kokoro_only = {s: i for s, i in loaded.symbols.items() if s not in loaded.assigned}

        assert unsupported(kokoro_only) == frozenset({"ħ", "ʕ", "ˤ", "͡"})


class TestSymbolList:
    def test_one_entry_per_embedding_row(self) -> None:
        assert len(Vocabulary.load().symbol_list()) == 178

    def test_the_three_kabyle_rows_hold_their_symbols(self) -> None:
        """The recipe's own table has PUA placeholders here and no entry for these three,
        so this is the difference between training them and deleting them."""
        rendered = Vocabulary.load().symbol_list()
        assert (rendered[7], rendered[8], rendered[26]) == ("ħ", "ʕ", "ˤ")

    def test_every_mapped_symbol_lands_on_its_own_index(self) -> None:
        loaded = Vocabulary.load()
        rendered = loaded.symbol_list()
        assert all(rendered[index] == symbol for symbol, index in loaded.symbols.items())

    def test_no_row_repeats(self) -> None:
        """A repeated entry would silently alias two ids onto one embedding."""
        rendered = Vocabulary.load().symbol_list()
        assert len(set(rendered)) == len(rendered)

    def test_filler_never_collides_with_a_phoneme(self) -> None:
        loaded = Vocabulary.load()
        rendered = loaded.symbol_list()
        filler = {rendered[index] for index in loaded.free()}
        assert not filler & set(loaded.symbols)

    def test_round_trips_back_to_the_same_ids(self) -> None:
        loaded = Vocabulary.load()
        recovered = {s: i for i, s in enumerate(loaded.symbol_list())}
        assert all(recovered[symbol] == index for symbol, index in loaded.symbols.items())

    def test_a_table_with_no_free_rows_needs_no_filler(self, tmp_path: Path) -> None:
        payload = table({"a": 1, "b": 2}, {}, n_token=3)
        loaded = Vocabulary.load(written(tmp_path / "t.json", payload))
        assert loaded.symbol_list() == ("$", "a", "b")


class TestEncoding:
    @pytest.fixture
    def kokoro(self) -> Vocabulary:
        return Vocabulary.load()

    def test_empty_string_encodes_to_nothing(self, kokoro: Vocabulary) -> None:
        assert kokoro.encode("") == ()

    def test_a_single_symbol(self, kokoro: Vocabulary) -> None:
        assert kokoro.encode("a") == (kokoro.symbols["a"],)

    def test_the_three_assigned_symbols_encode(self, kokoro: Vocabulary) -> None:
        assert kokoro.encode("ħʕˤ") == (7, 8, 26)

    def test_an_unmapped_symbol_raises_and_names_it(self, kokoro: Vocabulary) -> None:
        with pytest.raises(VocabularyError, match=r"U\+1E93"):
            kokoro.encode("aẓu")

    def test_every_unmapped_symbol_is_named_not_just_the_first(self, kokoro: Vocabulary) -> None:
        with pytest.raises(VocabularyError) as caught:
            kokoro.encode("ẓṭ")
        assert "U+1E93" in str(caught.value)
        assert "U+1E6D" in str(caught.value)

    def test_a_repeated_unmapped_symbol_is_named_once(self, kokoro: Vocabulary) -> None:
        assert kokoro.unmapped("ẓẓẓ") == ("ẓ",)

    def test_zero_width_space_is_refused_rather_than_ignored(self, kokoro: Vocabulary) -> None:
        """A ZWSP is invisible in a filelist and would otherwise be dropped silently."""
        with pytest.raises(VocabularyError, match=r"U\+200B"):
            kokoro.encode("a\u200bb")

    def test_a_non_breaking_space_is_not_the_space_the_table_has(self, kokoro: Vocabulary) -> None:
        with pytest.raises(VocabularyError, match=r"U\+00A0"):
            kokoro.encode("a b")

    def test_a_decomposed_form_is_refused(self, kokoro: Vocabulary) -> None:
        """`ç` NFC is a row; `c` plus U+0327 is two characters and the second has none."""
        with pytest.raises(VocabularyError, match=r"U\+0327"):
            kokoro.encode("ç")

    def test_the_tie_bar_is_refused_unfolded_and_encodes_folded(self, kokoro: Vocabulary) -> None:
        """This is the three-rows-not-four arithmetic, executed rather than asserted."""
        with pytest.raises(VocabularyError, match=r"U\+0361"):
            kokoro.encode("t͡ʃ")
        assert kokoro.encode(fold("t͡ʃ")) == (kokoro.symbols["ʧ"],)

    def test_a_real_kabyle_reading_round_trips(self, kokoro: Vocabulary) -> None:
        ids = kokoro.encode(fold(phonemize_word("aɣbalu")))
        assert len(ids) > 0
        assert all(0 <= i < kokoro.n_token for i in ids)


class TestTableInvariants:
    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VocabularyError, match="not found"):
            Vocabulary.load(tmp_path / "absent.json")

    def test_an_assigned_row_kokoro_already_uses_is_refused(self, tmp_path: Path) -> None:
        payload = table({"a": 1}, {"ħ": 1})
        with pytest.raises(VocabularyError, match="already Kokoro's"):
            Vocabulary.load(written(tmp_path / "t.json", payload))

    def test_an_assigned_symbol_kokoro_already_has_is_refused(self, tmp_path: Path) -> None:
        payload = table({"a": 1}, {"a": 2})
        with pytest.raises(VocabularyError, match="already in the base table"):
            Vocabulary.load(written(tmp_path / "t.json", payload))

    def test_an_index_past_the_embedding_is_refused(self, tmp_path: Path) -> None:
        payload = table({"a": 99}, {}, n_token=16)
        with pytest.raises(VocabularyError, match=r"outside 0\.\.15"):
            Vocabulary.load(written(tmp_path / "t.json", payload))

    def test_two_symbols_on_one_row_is_refused(self, tmp_path: Path) -> None:
        payload = table({"a": 1, "b": 1}, {})
        with pytest.raises(VocabularyError, match="share an index"):
            Vocabulary.load(written(tmp_path / "t.json", payload))

    def test_a_table_with_no_assignments_is_valid(self, tmp_path: Path) -> None:
        loaded = Vocabulary.load(written(tmp_path / "t.json", table({"a": 1}, {})))
        assert loaded.assigned == {}
        assert loaded.encode("a") == (1,)

    def test_free_excludes_pad_base_and_assigned(self, tmp_path: Path) -> None:
        payload = table({"a": 1, "b": 2}, {"ħ": 3}, n_token=6)
        assert Vocabulary.load(written(tmp_path / "t.json", payload)).free() == (4, 5)

    def test_a_full_table_has_no_free_rows(self, tmp_path: Path) -> None:
        payload = table({"a": 1, "b": 2}, {}, n_token=3)
        assert Vocabulary.load(written(tmp_path / "t.json", payload)).free() == ()


def test_the_plbert_cap_leaves_room_for_the_special_positions() -> None:
    """PL-BERT is a 512-position encoder and the recipe filters above 510."""
    assert PLBERT_LIMIT == 510
