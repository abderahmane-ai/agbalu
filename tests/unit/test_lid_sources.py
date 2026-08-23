from __future__ import annotations

import gzip
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from agbalu.bench.lid_sources import (
    BERBER_LANGUAGES,
    CLASSES,
    MIN_WORDS,
    NEGATIVE_LANGUAGES,
    NOT_AMAZIGH,
    SOURCES,
    Source,
    SourceError,
    candidates,
    documents,
    is_conjugation_drill,
    is_usable,
    lines,
    read_gzip_lines,
    read_moses_zip,
    read_ocr_bible,
)


def sentence(word: str, count: int = MIN_WORDS) -> str:
    return " ".join([word] * count)


class TestTheSourceTable:
    def test_every_source_carries_a_label_in_the_class_list(self) -> None:
        assert {source.language for source in SOURCES} <= set(CLASSES)

    def test_source_ids_are_unique(self) -> None:
        repeated = [name for name, n in Counter(s.id for s in SOURCES).items() if n > 1]
        assert repeated == []

    def test_every_berber_class_has_at_least_two_sources(self) -> None:
        """One source per class is the defect the rebuild exists to fix: with one source
        the classifier answers with the source and the label follows for free."""
        per_language = Counter(source.language for source in SOURCES)
        assert all(per_language[language] >= 2 for language in BERBER_LANGUAGES), per_language

    def test_every_berber_class_has_at_least_two_domains(self) -> None:
        per_language: dict[str, set[str]] = {}
        for source in SOURCES:
            per_language.setdefault(source.language, set()).add(source.domain)
        thin = {lang: doms for lang, doms in per_language.items() if len(doms) < 2}
        assert thin == {}

    def test_flores_is_not_a_training_source(self) -> None:
        """It is where the Tamasheq evaluation sentences come from."""
        assert all(source.reader != "flores" for source in SOURCES)
        assert all("flores" not in source.id for source in SOURCES)

    def test_no_source_claims_a_language_from_a_reader_of_another(self) -> None:
        for source in SOURCES:
            if source.reader == "tatoeba" and source.language != NOT_AMAZIGH:
                assert source.argument == source.language.split("_")[0]

    def test_tahaggart_tamahaq_is_not_mapped_onto_tamasheq(self) -> None:
        """`thv` and `taq` are different ISO codes; merging them is the silent merge
        CLAUDE.md forbids, and it was in the previous table."""
        assert all(source.argument != "thv" for source in SOURCES)

    def test_the_negative_halves_do_not_overlap(self) -> None:
        latin = set(NEGATIVE_LANGUAGES["latin"])
        other = set(NEGATIVE_LANGUAGES["other"])
        assert latin & other == set()

    def test_no_negative_language_is_a_berber_class(self) -> None:
        bases = {language.split("_")[0] for language in BERBER_LANGUAGES}
        every = {*NEGATIVE_LANGUAGES["latin"], *NEGATIVE_LANGUAGES["other"]}
        assert every & bases == set()

    def test_group_names_the_language_and_the_domain(self) -> None:
        source = SOURCES[0]
        assert source.group == f"{source.language}/{source.domain}"


class TestFilters:
    def test_a_line_at_the_threshold_is_kept_and_one_below_is_not(self) -> None:
        assert is_usable(sentence("awal", MIN_WORDS))
        assert not is_usable(sentence("awal", MIN_WORDS - 1))

    def test_the_negative_other_half_may_be_non_latin(self) -> None:
        tifinagh = sentence("ⵜⴰⵎⴰⵣⵉⵖⵜ")
        assert not is_usable(tifinagh)
        assert is_usable(tifinagh, require_latin=False)

    def test_a_null_byte_and_a_bom_do_not_crash_the_filter(self) -> None:
        assert is_usable("﻿awal awal awal awal awal")
        assert is_usable("awal\x00 awal awal awal awal")

    def test_zero_width_and_nbsp_do_not_invent_words(self) -> None:
        assert not is_usable("awal\u200bawal\u200bawal\u200bawal")
        assert is_usable("awal awal awal awal awal")


class TestTheReservoir:
    def test_a_limit_samples_the_whole_stream_and_not_a_prefix(self) -> None:
        """A prefix of a source-ordered file is one source; the first 8,000 lines of a
        New Testament are Matthew."""
        stream = [sentence(f"w{index}") for index in range(4_000)]
        drawn = candidates(stream, limit=200)
        assert len(drawn) == 200
        positions = [int(line.split()[0].removeprefix("w")) for line in drawn]
        assert max(positions) > 3_000

    def test_the_reservoir_is_seeded(self) -> None:
        stream = [sentence(f"w{index}") for index in range(2_000)]
        first = candidates(stream, limit=50, rng=random.Random("a"))
        assert first == candidates(stream, limit=50, rng=random.Random("a"))
        assert first != candidates(stream, limit=50, rng=random.Random("b"))

    def test_a_limit_above_the_stream_keeps_everything(self) -> None:
        stream = [sentence(f"w{index}") for index in range(10)]
        assert len(candidates(stream, limit=1_000)) == 10

    def test_a_non_positive_limit_is_refused(self) -> None:
        with pytest.raises(SourceError, match="limit must be positive"):
            candidates([sentence("a")], limit=0)

    def test_duplicates_are_collapsed_before_the_reservoir_counts_them(self) -> None:
        repeated = [sentence("awal")] * 500
        assert candidates(repeated, limit=10) == [sentence("awal")]


class TestReaders:
    def test_gzip_lines_skips_blanks_and_reports_a_missing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "mono.txt.gz"
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            handle.write("first line here\n\n  \nsecond line here\n")
        assert list(read_gzip_lines(target)) == ["first line here", "second line here"]
        with pytest.raises(SourceError, match="gzip text not found"):
            list(read_gzip_lines(tmp_path / "absent.gz"))

    def test_moses_zip_selects_the_named_side(self, tmp_path: Path) -> None:
        target = tmp_path / "en-kab.txt.zip"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("bundle.en", "english one\nenglish two\n")
            archive.writestr("bundle.kab", "kabyle one\nkabyle two\n")
        assert list(read_moses_zip(target, "kab")) == ["kabyle one", "kabyle two"]
        with pytest.raises(SourceError, match=r"no member ending \.rif"):
            list(read_moses_zip(target, "rif"))

    def test_ocr_bible_strips_leading_verse_numbers_only(self, tmp_path: Path) -> None:
        target = tmp_path / "nt.txt"
        target.write_text("12 mara necc remdey\n2026 was a year\nno number here\n", "utf-8")
        assert list(read_ocr_bible(target)) == [
            "mara necc remdey",
            "was a year",
            "no number here",
        ]

    def test_a_delimited_bitext_takes_the_last_column_and_strips_placeholders(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "en-tzm.tsv"
        target.write_text(
            "source_string\ttarget_string\n"
            "%1$s cannot download\tur izmir %1$s ad d-yagem anaw\n"
            "empty target\t%1$s %2$d\n",
            encoding="utf-8",
        )
        source = Source("t", "tzm_Latn", "interface", "gold", "mpl-2.0", "tsv_target", "en-tzm.tsv")
        read = list(documents(source, tmp_path))
        assert read == ["ur izmir   ad d-yagem anaw"]

    def test_a_quoted_delimiter_does_not_split_the_row(self, tmp_path: Path) -> None:
        target = tmp_path / "en-rif.csv"
        target.write_text(
            'source_string,target_string\nhello,"Ur ten-teg, ca, da yenna-yas"\n', encoding="utf-8"
        )
        source = Source(
            "t", "rif_Latn", "interface", "gold", "cc-by-3.0", "csv_target", "en-rif.csv"
        )
        assert list(documents(source, tmp_path)) == ["Ur ten-teg, ca, da yenna-yas"]

    def test_an_empty_delimited_file_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "x.csv").write_text("", encoding="utf-8")
        source = Source("t", "rif_Latn", "interface", "gold", "cc-by-3.0", "csv_target", "x.csv")
        with pytest.raises(SourceError, match="is empty"):
            list(documents(source, tmp_path))

    def test_a_missing_flores_directory_names_the_language(self, tmp_path: Path) -> None:
        source = Source(
            "f", "taq_Latn", "encyclopaedia", "gold", "cc-by-sa-4.0", "flores", "flores", "taq_Latn"
        )
        with pytest.raises(SourceError, match="no FLORES\\+ split for taq_Latn"):
            list(documents(source, tmp_path))

    def test_flores_reads_every_split_of_the_language(self, tmp_path: Path) -> None:
        root = tmp_path / "flores"
        for split, text in (("dev", "dev sentence"), ("devtest", "devtest sentence")):
            (root / split).mkdir(parents=True)
            (root / split / "taq_Latn.jsonl").write_text(
                json.dumps({"text": text}) + "\n", encoding="utf-8"
            )
            (root / split / "kab_Latn.jsonl").write_text(
                json.dumps({"text": "not this one"}) + "\n", encoding="utf-8"
            )
        source = Source(
            "f", "taq_Latn", "encyclopaedia", "gold", "cc-by-sa-4.0", "flores", "flores", "taq_Latn"
        )
        assert sorted(documents(source, tmp_path)) == ["dev sentence", "devtest sentence"]


class TestTheAnkiDeckFilter:
    @pytest.mark.parametrize(
        "text",
        ["ddu // tddut // nttddu", "ddiɣ, tddid, idda, nedda, tddam"],
    )
    def test_a_conjugation_drill_is_dropped(self, text: str) -> None:
        assert is_conjugation_drill(text)

    def test_a_sentence_with_one_comma_survives(self) -> None:
        assert not is_conjugation_drill("ha ttumubil nna-g, ndda a-nttddu ghr marrakec")

    def test_an_empty_string_is_not_a_drill(self) -> None:
        assert not is_conjugation_drill("")


def test_lines_refuses_a_source_whose_bytes_are_absent(tmp_path: Path) -> None:
    source = Source("s", "shi_Latn", "web", "silver", "cc0-1.0", "glotcc", "missing")
    with pytest.raises(SourceError, match="no parquet under"):
        lines(source, tmp_path)


def test_lines_relaxes_the_latin_filter_only_for_the_other_negative_half(
    tmp_path: Path,
) -> None:
    body = io.StringIO()
    body.write("ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵏⴰⵡⴰⵢⵜ ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⴰⵏⴰⵡⴰⵢⵜ ⵜⴰⵎⴰⵣⵉⵖⵜ\n")
    (tmp_path / "x.csv").write_text("h\n" + body.getvalue(), encoding="utf-8")
    latin = Source("a", NOT_AMAZIGH, "negative-latin", "gold", "cc-by-2.0", "csv_target", "x.csv")
    other = Source("b", NOT_AMAZIGH, "negative-other", "gold", "cc-by-2.0", "csv_target", "x.csv")
    assert lines(latin, tmp_path) == []
    assert len(lines(other, tmp_path)) == 1
