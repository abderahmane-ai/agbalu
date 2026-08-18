"""Assembling the recipe's filelists: what is kept, what is dropped, what stops the run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agbalu.tts.g2p import PhonemeError
from agbalu.tts.training import (
    OVERLONG,
    UNMAPPED,
    Row,
    Selection,
    TrainingError,
    assign_speaker_ids,
    merge,
    parse,
    read_list,
    renumber,
    require_encodable,
    select,
    select_ood,
    voice_list,
    write_list,
    write_ood,
)
from agbalu.tts.vocabulary import Vocabulary, VocabularyError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def kokoro() -> Vocabulary:
    return Vocabulary.load()


def row(ipa: str = "azul", speaker: str = "kab_male", audio: str = "a/1.wav") -> Row:
    return Row(audio=audio, ipa=ipa, speaker=speaker)


class TestParsing:
    def test_the_three_fields(self) -> None:
        parsed = parse("voices/kab_male/restored/x.wav|azul|kab_male")
        assert parsed.audio == "voices/kab_male/restored/x.wav"
        assert parsed.ipa == "azul"
        assert parsed.speaker == "kab_male"

    def test_a_trailing_newline_is_not_part_of_the_speaker(self) -> None:
        assert parse("a.wav|azul|kab_male\n").speaker == "kab_male"

    def test_too_few_fields_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="got 2"):
            parse("a.wav|azul")

    def test_too_many_fields_is_refused(self) -> None:
        """A `|` inside the phoneme string would silently shift the speaker column."""
        with pytest.raises(TrainingError, match="got 4"):
            parse("a.wav|az|ul|kab_male")

    def test_a_missing_audio_path_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="must name its audio"):
            parse("|azul|kab_male")

    def test_a_missing_speaker_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="must name its audio"):
            parse("a.wav|azul|")

    def test_an_empty_phoneme_string_is_parsed_not_refused(self) -> None:
        """Emptiness is the selector's decision, not the parser's — it reports a count."""
        assert parse("a.wav||kab_male").ipa == ""


class TestReadingLists:
    def test_reads_every_row(self, tmp_path: Path) -> None:
        path = tmp_path / "l.txt"
        path.write_text("a.wav|az|m\nb.wav|ul|f\n", encoding="utf-8")
        assert len(read_list(path)) == 2

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "l.txt"
        path.write_text("a.wav|az|m\n\n\nb.wav|ul|f\n", encoding="utf-8")
        assert len(read_list(path)) == 2

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TrainingError, match="not found"):
            read_list(tmp_path / "absent.txt")

    def test_a_bad_row_names_its_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "l.txt"
        path.write_text("a.wav|az|m\nbroken\n", encoding="utf-8")
        with pytest.raises(TrainingError, match=r"l\.txt:2:"):
            read_list(path)

    def test_crlf_endings_do_not_leak_into_the_speaker(self, tmp_path: Path) -> None:
        path = tmp_path / "l.txt"
        path.write_bytes(b"a.wav|az|kab_male\r\nb.wav|ul|kab_male\r\n")
        assert {r.speaker for r in read_list(path)} == {"kab_male"}


class TestSelection:
    def test_keeps_encodable_rows(self, kokoro: Vocabulary) -> None:
        assert len(select([row("azul"), row("ħaʕˤ")], kokoro).rows) == 2

    def test_an_unmapped_phoneme_is_rejected_and_counted(self, kokoro: Vocabulary) -> None:
        chosen = select([row("azul"), row("aẓu")], kokoro)
        assert len(chosen.rows) == 1
        assert chosen.rejected[UNMAPPED] == 1
        assert chosen.unmapped == {"ẓ": 1}

    def test_a_bad_symbol_is_counted_once_per_row_not_per_occurrence(
        self, kokoro: Vocabulary
    ) -> None:
        """Rows affected is the actionable number: it says how much of the corpus a
        missing embedding row would cost, where occurrences say how often it is written."""
        chosen = select([row("aẓ"), row("ẓẓu")], kokoro)
        assert chosen.unmapped == {"ẓ": 2}

    def test_a_sequence_past_the_window_is_dropped_as_data(self, kokoro: Vocabulary) -> None:
        chosen = select([row("a" * 600)], kokoro, limit=510)
        assert chosen.rows == ()
        assert chosen.rejected[OVERLONG] == 1
        assert chosen.unmapped == {}

    def test_exactly_at_the_window_is_kept(self, kokoro: Vocabulary) -> None:
        assert len(select([row("a" * 510)], kokoro, limit=510).rows) == 1

    def test_one_past_the_window_is_dropped(self, kokoro: Vocabulary) -> None:
        assert select([row("a" * 511)], kokoro, limit=510).rows == ()

    def test_longest_is_measured_before_the_window_drops_it(self, kokoro: Vocabulary) -> None:
        """The report must say how far past the limit the corpus reaches, or a cap that
        silently removes half the data reads as a clean build."""
        assert select([row("a" * 900)], kokoro, limit=510).longest == 900

    def test_an_unmapped_row_does_not_contribute_to_longest(self, kokoro: Vocabulary) -> None:
        assert select([row("ẓ" * 900)], kokoro, limit=510).longest == 0

    def test_an_empty_selection_reports_zero_not_an_error(self, kokoro: Vocabulary) -> None:
        chosen = select([], kokoro)
        assert chosen.rows == ()
        assert chosen.longest == 0

    def test_speakers_are_reported_sorted_and_unique(self, kokoro: Vocabulary) -> None:
        chosen = select([row(speaker="kab_male"), row(speaker="kab_female")], kokoro)
        assert chosen.speakers == ("kab_female", "kab_male")

    def test_the_payload_names_bad_symbols_by_codepoint(self, kokoro: Vocabulary) -> None:
        payload = select([row("aẓ")], kokoro).as_dict()
        assert payload["unmapped_symbols"] == {"ẓ U+1E93": 1}


class TestRequireEncodable:
    def test_a_clean_selection_passes(self, kokoro: Vocabulary) -> None:
        require_encodable(select([row("azul")], kokoro))

    def test_an_unmapped_phoneme_stops_the_run(self, kokoro: Vocabulary) -> None:
        with pytest.raises(VocabularyError, match=r"U\+1E93 in 1 rows"):
            require_encodable(select([row("aẓ")], kokoro))

    def test_overlong_rows_alone_do_not_stop_the_run(self, kokoro: Vocabulary) -> None:
        """Dropping a long clip is a data property; dropping a phoneme is corruption."""
        require_encodable(select([row("a" * 900)], kokoro, limit=510))


class TestMerge:
    def test_carries_every_row(self) -> None:
        left = Selection((row(speaker="m"),) * 3, {}, {}, 4)
        right = Selection((row(speaker="f"),) * 2, {}, {}, 4)
        assert len(merge([left, right])) == 5

    def test_interleaves_rather_than_concatenating(self) -> None:
        """Voice order out of the merge would put one speaker's whole corpus first."""
        left = Selection(tuple(row(speaker="m", audio=f"m{i}") for i in range(50)), {}, {}, 4)
        right = Selection(tuple(row(speaker="f", audio=f"f{i}") for i in range(50)), {}, {}, 4)

        merged = merge([left, right])

        assert [r.speaker for r in merged[:50]] != ["m"] * 50

    def test_is_deterministic_under_its_seed(self) -> None:
        made = [Selection(tuple(row(audio=f"{i}") for i in range(40)), {}, {}, 4)]
        assert [r.audio for r in merge(made, seed=7)] == [r.audio for r in merge(made, seed=7)]

    def test_a_different_seed_gives_a_different_order(self) -> None:
        made = [Selection(tuple(row(audio=f"{i}") for i in range(40)), {}, {}, 4)]
        assert [r.audio for r in merge(made, seed=1)] != [r.audio for r in merge(made, seed=2)]

    def test_merging_nothing_gives_nothing(self) -> None:
        assert merge([]) == ()


class TestOodSelection:
    def test_keeps_a_long_encodable_line(self, kokoro: Vocabulary) -> None:
        chosen = select_ood(["a" * 60], kokoro, phonemize=str, exclude=set(), size=10, minimum=50)
        assert chosen.lines == ("a" * 60,)

    def test_a_held_out_text_is_rejected(self, kokoro: Vocabulary) -> None:
        """The positive control for contamination: a line the evaluation uses must not
        reach the adversarial branch, which reads it every step."""
        text = "a" * 60
        chosen = select_ood([text], kokoro, phonemize=str, exclude={text}, size=10, minimum=50)
        assert chosen.lines == ()
        assert chosen.rejected == {"held-out-elsewhere": 1}

    def test_the_same_text_passes_without_the_exclusion(self, kokoro: Vocabulary) -> None:
        """The other direction of the same join — without it the rejection above proves
        nothing, because a filter that matches everything looks identical."""
        text = "a" * 60
        chosen = select_ood([text], kokoro, phonemize=str, exclude=set(), size=10, minimum=50)
        assert chosen.lines == (text,)

    def test_a_short_line_is_rejected(self, kokoro: Vocabulary) -> None:
        chosen = select_ood(["ab"], kokoro, phonemize=str, exclude=set(), size=10, minimum=50)
        assert chosen.rejected == {"too-short": 1}

    def test_an_unmapped_phoneme_is_rejected(self, kokoro: Vocabulary) -> None:
        chosen = select_ood(["ẓ" * 60], kokoro, phonemize=str, exclude=set(), size=10, minimum=50)
        assert chosen.rejected == {UNMAPPED: 1}

    def test_an_unphonemisable_line_is_rejected_not_raised(self, kokoro: Vocabulary) -> None:
        def refuse(_: str) -> str:
            message = "no rule"
            raise PhonemeError(message)

        chosen = select_ood(["x" * 60], kokoro, phonemize=refuse, exclude=set(), size=1, minimum=1)
        assert chosen.rejected == {"no-phoneme-rule": 1}

    def test_duplicates_are_dropped(self, kokoro: Vocabulary) -> None:
        chosen = select_ood(
            ["a" * 60] * 3, kokoro, phonemize=str, exclude=set(), size=10, minimum=50
        )
        assert len(chosen.lines) == 1
        assert chosen.rejected == {"duplicate": 2}

    def test_it_stops_at_the_requested_size(self, kokoro: Vocabulary) -> None:
        texts = ["a" * (60 + i) for i in range(50)]
        assert len(select_ood(texts, kokoro, phonemize=str, exclude=set(), size=5).lines) == 5

    def test_shortest_reports_the_floor_actually_achieved(self, kokoro: Vocabulary) -> None:
        chosen = select_ood(
            ["a" * 60, "b" * 51], kokoro, phonemize=str, exclude=set(), size=10, minimum=50
        )
        assert chosen.shortest == 51


class TestOodWriting:
    def test_writes_every_line(self, tmp_path: Path) -> None:
        path = tmp_path / "OOD_texts.txt"
        assert write_ood(path, ["a" * 60, "b" * 60], minimum=50) == 2
        assert path.read_text(encoding="utf-8").splitlines() == ["a" * 60, "b" * 60]

    def test_one_line_is_refused(self, tmp_path: Path) -> None:
        """The recipe draws `randint(0, len - 1)`, which raises at a single line."""
        with pytest.raises(TrainingError, match="cannot draw from 1 lines"):
            write_ood(tmp_path / "o.txt", ["a" * 60], minimum=50)

    def test_no_lines_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TrainingError, match="cannot draw from 0 lines"):
            write_ood(tmp_path / "o.txt", [], minimum=50)

    def test_a_short_line_is_refused_because_it_hangs_the_sampler(self, tmp_path: Path) -> None:
        """`while len(ps) < min_length` never exits if it can never draw a long enough
        line — an infinite loop inside a paid container."""
        with pytest.raises(TrainingError, match="loop until it happens to draw"):
            write_ood(tmp_path / "o.txt", ["a" * 60, "short"], minimum=50)


class TestSpeakerIds:
    def test_names_become_integers_in_sorted_order(self) -> None:
        assert assign_speaker_ids(["kab_male", "kab_female"]) == {
            "kab_female": 0,
            "kab_male": 1,
        }

    def test_repeats_collapse(self) -> None:
        assert assign_speaker_ids(["a", "a", "b"]) == {"a": 0, "b": 1}

    def test_renumber_replaces_the_column(self) -> None:
        ids = {"kab_male": 1}
        assert renumber([row(speaker="kab_male")], ids)[0].speaker == "1"

    def test_renumber_keeps_everything_else(self) -> None:
        renamed = renumber([row("ħaʕˤ", audio="x.wav", speaker="kab_male")], {"kab_male": 0})[0]
        assert (renamed.audio, renamed.ipa) == ("x.wav", "ħaʕˤ")

    def test_an_unknown_speaker_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="no speaker id assigned"):
            renumber([row(speaker="kab_male")], {"kab_female": 0})


class TestVoiceListNames:
    def test_the_voice_goes_before_the_suffix(self) -> None:
        assert voice_list("train_list.txt", "kab_male") == "train_list.kab_male.txt"
        assert voice_list("val_list.txt", "kab_female") == "val_list.kab_female.txt"

    def test_two_voices_never_collide(self) -> None:
        assert voice_list("train_list.txt", "kab_male") != voice_list(
            "train_list.txt", "kab_female"
        )

    def test_the_merged_name_is_never_returned(self) -> None:
        """A per-voice list that resolved back to the merged one is the whole defect."""
        assert voice_list("train_list.txt", "kab_male") != "train_list.txt"

    def test_a_name_with_no_suffix_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="no suffix"):
            voice_list("train_list", "kab_male")


class TestWriting:
    def test_writes_one_row_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "train_list.txt"
        rows = renumber([row("az"), row("ul")], {"kab_male": 0})

        written = write_list(path, rows)

        assert written == 2
        assert path.read_text(encoding="utf-8").splitlines() == [
            "a/1.wav|az|0",
            "a/1.wav|ul|0",
        ]

    def test_a_named_speaker_column_is_refused(self, tmp_path: Path) -> None:
        """`meldataset` casts this field with `int()`, so a name raises on the first batch
        of a container that has already paid for its GPU."""
        with pytest.raises(TrainingError, match="must hold integers"):
            write_list(tmp_path / "l.txt", [row(speaker="kab_male")])

    def test_round_trips_through_the_reader(self, tmp_path: Path) -> None:
        path = tmp_path / "l.txt"
        rows = renumber(
            [row("ħaʕˤ", speaker="kab_female"), row("azul", speaker="kab_male")],
            {"kab_female": 0, "kab_male": 1},
        )
        write_list(path, rows)
        assert list(read_list(path)) == list(rows)

    def test_an_empty_list_is_refused(self, tmp_path: Path) -> None:
        """The recipe reads an empty filelist as a dataset of nothing and trains anyway."""
        with pytest.raises(TrainingError, match="empty filelist"):
            write_list(tmp_path / "l.txt", [])

    def test_creates_the_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "train_list.txt"
        write_list(path, renumber([row()], {"kab_male": 0}))
        assert path.is_file()
