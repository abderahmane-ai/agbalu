from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.treebank import (
    DEFAULT_ROOT,
    Sentence,
    TreebankError,
    Word,
    read_all,
    read_conllu,
    read_split,
    split_path,
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


COLUMN_ORDER = ("lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc")


def row(identifier: str, form: str, **columns: str) -> str:
    """One CoNLL-U row; every column not named is `_`."""
    unknown = set(columns) - set(COLUMN_ORDER)
    assert not unknown, unknown
    return "\t".join([identifier, form, *(columns.get(name, "_") for name in COLUMN_ORDER)])


def block(*rows: str) -> str:
    return "".join(line + "\n" for line in rows)


SIMPLE = "# sent_id = A2\n# text = Azadaɣ, d win.\n" + block(
    row("1", "Azadaɣ", lemma="azadaɣ", upos="NOUN", head="4", deprel="nsubj", misc="SpaceAfter=No"),
    row("2", ",", lemma=",", upos="PUNCT", head="1", deprel="punct"),
    row("3", "d", lemma="ad", upos="PART", head="4", deprel="advmod"),
    row("4", "win", lemma="wina", upos="PRON", head="0", deprel="root", misc="SpaceAfter=No"),
    row("5", ".", lemma=".", upos="PUNCT", head="4", deprel="punct"),
)

MULTIWORD = "# sent_id = A3\n# text = ɣur-s aksum\n" + block(
    row("1-2", "ɣur-s"),
    row("1", "ɣur", lemma="ɣur", upos="ADP", head="3", deprel="case"),
    row("2", "s", lemma="netta", upos="PRON", head="1", deprel="nmod"),
    row("3", "aksum", lemma="aksum", upos="NOUN", head="0", deprel="root"),
)


class TestParsing:
    def test_a_sentence_is_read_in_both_views(self, tmp_path: Path) -> None:
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", SIMPLE), "test"))
        assert sentence.sent_id == "A2"
        assert sentence.text == "Azadaɣ, d win."
        assert sentence.split == "test"
        assert [w.form for w in sentence.words] == ["Azadaɣ", ",", "d", "win", "."]
        assert [t.form for t in sentence.tokens] == ["Azadaɣ", ",", "d", "win", "."]

    def test_word_fields_are_read_off_the_right_columns(self, tmp_path: Path) -> None:
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", SIMPLE), "test"))
        assert sentence.words[0] == Word(
            id=1,
            form="Azadaɣ",
            lemma="azadaɣ",
            upos="NOUN",
            feats="_",
            head=4,
            deprel="nsubj",
            space_after=False,
        )

    def test_a_multiword_token_spans_its_words(self, tmp_path: Path) -> None:
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", MULTIWORD), "train"))
        assert len(sentence.words) == 3
        assert [t.form for t in sentence.tokens] == ["ɣur-s", "aksum"]
        assert sentence.tokens[0].is_multiword
        assert [w.upos for w in sentence.tokens[0].words] == ["ADP", "PRON"]
        assert not sentence.tokens[1].is_multiword

    def test_a_final_sentence_without_a_trailing_blank_line_is_kept(self, tmp_path: Path) -> None:
        text = SIMPLE.rstrip("\n")
        assert len(list(read_conllu(write(tmp_path / "a.conllu", text), "test"))) == 1

    def test_blank_line_runs_do_not_emit_empty_sentences(self, tmp_path: Path) -> None:
        text = SIMPLE + "\n\n   \n" + MULTIWORD + "\n\n"
        assert len(list(read_conllu(write(tmp_path / "a.conllu", text), "test"))) == 2

    def test_an_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        assert list(read_conllu(write(tmp_path / "a.conllu", ""), "test")) == []

    def test_a_comment_only_file_yields_nothing(self, tmp_path: Path) -> None:
        text = "# newdoc\n# newpar\n"
        assert list(read_conllu(write(tmp_path / "a.conllu", text), "test")) == []

    def test_a_comment_without_an_equals_sign_is_ignored(self, tmp_path: Path) -> None:
        text = "# newdoc\n" + SIMPLE
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", text), "test"))
        assert sentence.sent_id == "A2"

    def test_metadata_does_not_leak_into_the_next_sentence(self, tmp_path: Path) -> None:
        text = (
            SIMPLE
            + "\n"
            + row("1", "Azul", lemma="azul", upos="INTJ", head="0", deprel="root")
            + "\n"
        )
        first, second = list(read_conllu(write(tmp_path / "a.conllu", text), "test"))
        assert first.sent_id == "A2"
        assert (second.sent_id, second.text) == ("", "")

    def test_empty_nodes_are_dropped(self, tmp_path: Path) -> None:
        text = SIMPLE.rstrip("\n") + "\n" + row("5.1", "ellipsis") + "\n"
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", text), "test"))
        assert [w.id for w in sentence.words] == [1, 2, 3, 4, 5]

    def test_an_unspecified_head_reads_as_none(self, tmp_path: Path) -> None:
        text = row("1", "Azul", lemma="azul", upos="INTJ") + "\n"
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", text), "test"))
        assert sentence.words[0].head is None

    def test_a_byte_order_mark_is_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "a.conllu"
        path.write_bytes(b"\xef\xbb\xbf" + SIMPLE.encode("utf-8"))
        [sentence] = list(read_conllu(path, "test"))
        assert sentence.sent_id == "A2"

    def test_crlf_line_endings_do_not_reach_the_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "a.conllu"
        path.write_bytes(SIMPLE.replace("\n", "\r\n").encode("utf-8"))
        [sentence] = list(read_conllu(path, "test"))
        assert sentence.words[-1].deprel == "punct"
        assert sentence.words[-1].space_after is True


class TestMalformedInput:
    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(TreebankError, match="treebank not found"):
            list(read_conllu(tmp_path / "absent.conllu", "test"))

    def test_a_wrong_column_count_is_an_error(self, tmp_path: Path) -> None:
        text = "1\tAzul\tazul\tINTJ\n"
        with pytest.raises(TreebankError, match="4 columns, expected 10"):
            list(read_conllu(write(tmp_path / "a.conllu", text), "test"))

    def test_an_unreadable_id_is_an_error(self, tmp_path: Path) -> None:
        text = row("x", "Azul") + "\n"
        with pytest.raises(TreebankError, match="unreadable ID field"):
            list(read_conllu(write(tmp_path / "a.conllu", text), "test"))

    def test_a_non_integer_head_is_an_error(self, tmp_path: Path) -> None:
        text = row("1", "Azul", head="root") + "\n"
        with pytest.raises(TreebankError, match="HEAD is neither"):
            list(read_conllu(write(tmp_path / "a.conllu", text), "test"))

    def test_a_malformed_range_is_an_error(self, tmp_path: Path) -> None:
        text = row("1-x", "ɣur-s") + "\n" + row("1", "ɣur") + "\n"
        with pytest.raises(TreebankError, match="malformed multiword token ID"):
            list(read_conllu(write(tmp_path / "a.conllu", text), "test"))

    def test_a_range_that_does_not_span_forward_is_an_error(self, tmp_path: Path) -> None:
        text = row("2-1", "ɣur-s") + "\n" + row("1", "ɣur") + "\n" + row("2", "s") + "\n"
        with pytest.raises(TreebankError, match="does not span forward"):
            list(read_conllu(write(tmp_path / "a.conllu", text), "test"))

    def test_a_range_missing_a_member_word_is_an_error(self, tmp_path: Path) -> None:
        """Silently truncating a multiword token loses a gold tag without a trace."""
        text = row("1-2", "ɣur-s") + "\n" + row("1", "ɣur") + "\n"
        with pytest.raises(TreebankError, match=r"missing word\(s\) \[2\]"):
            list(read_conllu(write(tmp_path / "a.conllu", text), "test"))

    def test_overlapping_ranges_are_an_error(self, tmp_path: Path) -> None:
        text = (
            row("1-2", "ab")
            + "\n"
            + row("2-3", "bc")
            + "\n"
            + row("1", "a")
            + "\n"
            + row("2", "b")
            + "\n"
            + row("3", "c")
            + "\n"
        )
        with pytest.raises(TreebankError, match="overlapping multiword ranges at 2-3"):
            list(read_conllu(write(tmp_path / "a.conllu", text), "test"))


class TestSurface:
    def test_space_after_no_joins_the_next_token(self, tmp_path: Path) -> None:
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", SIMPLE), "test"))
        assert sentence.surface() == sentence.text == "Azadaɣ, d win."

    def test_a_multiword_token_contributes_its_own_form(self, tmp_path: Path) -> None:
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", MULTIWORD), "test"))
        assert sentence.surface() == "ɣur-s aksum"

    def test_a_trailing_space_after_no_does_not_add_a_space(self, tmp_path: Path) -> None:
        text = row("1", "Azul", misc="SpaceAfter=No") + "\n"
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", text), "test"))
        assert sentence.surface() == "Azul"

    def test_a_single_word_sentence_round_trips(self, tmp_path: Path) -> None:
        text = row("1", "Azul") + "\n"
        [sentence] = list(read_conllu(write(tmp_path / "a.conllu", text), "test"))
        assert sentence.surface() == "Azul"


class TestSplits:
    def test_split_path_follows_the_ud_naming_convention(self) -> None:
        assert split_path(Path("x"), "test").name == "kab_adpt-ud-test.conllu"

    def test_read_split_reads_one_named_file(self, tmp_path: Path) -> None:
        write(tmp_path / "kab_adpt-ud-train.conllu", SIMPLE)
        assert [s.split for s in read_split(tmp_path, "train")] == ["train"]

    def test_read_all_labels_each_sentence_with_its_split(self, tmp_path: Path) -> None:
        write(tmp_path / "kab_adpt-ud-train.conllu", SIMPLE)
        write(tmp_path / "kab_adpt-ud-test.conllu", MULTIWORD)
        assert [s.split for s in read_all(tmp_path)] == ["train", "test"]

    def test_read_all_over_an_empty_directory_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(TreebankError, match="no treebank split found"):
            list(read_all(tmp_path))


@pytest.mark.integration
class TestRealTreebank:
    @pytest.fixture(scope="class")
    def sentences(self) -> list[Sentence]:
        if not split_path(DEFAULT_ROOT, "test").is_file():
            pytest.skip(f"treebank not present under {DEFAULT_ROOT}")
        return list(read_all(DEFAULT_ROOT))

    def test_the_documented_size_is_what_is_on_disk(self, sentences: list[Sentence]) -> None:
        words = sum(len(s.words) for s in sentences)
        assert len(sentences) == 1930
        assert words == 23761

    def test_every_sentence_rebuilds_its_own_text(self, sentences: list[Sentence]) -> None:
        """The surface view is what a tagger is fed; if it does not reproduce `# text`
        the tagger is scored on input the treebank never contained."""
        assert [s.sent_id for s in sentences if s.surface() != s.text] == []

    def test_multiword_tokens_cover_the_measured_share_of_words(
        self, sentences: list[Sentence]
    ) -> None:
        words = sum(len(s.words) for s in sentences)
        covered = sum(len(t.words) for s in sentences for t in s.tokens if t.is_multiword)
        assert covered / words == pytest.approx(0.296, abs=0.001)

    def test_no_sentence_is_empty(self, sentences: list[Sentence]) -> None:
        assert all(s.words and s.tokens for s in sentences)
