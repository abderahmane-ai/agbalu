from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from agbalu.bench.lid import Example, LidError, write_evalset
from agbalu.bench.lid_dataset import (
    MIN_GROUP_ROWS,
    Record,
    assign_groups,
    build_records,
    decontaminate,
    deduplicate,
    grouped,
    normalise_key,
    split,
    statistics,
    write_jsonl,
)
from agbalu.bench.lid_sources import Source


def record(
    text: str,
    language: str = "kab_Latn",
    domain: str = "web",
    source: str = "s",
    trust: str = "gold",
) -> Record:
    return Record(
        text=text,
        language=language,
        domain=domain,
        source=source,
        label_trust=trust,
        licence="cc0-1.0",
        redistribution="permissive",
    )


def written_evalset(tmp_path: Path, examples: list[Example]) -> Path:
    destination = tmp_path / "lid-eval.jsonl"
    write_evalset(examples, destination)
    return destination


class TestNormalisationKey:
    def test_case_and_whitespace_fold_together(self) -> None:
        assert normalise_key("  Ur   TEN-teg\tca ") == normalise_key("ur ten-teg ca")

    def test_a_diacritic_is_not_folded_away(self) -> None:
        """The emphatics carry meaning; folding them would merge two languages' spellings."""
        assert normalise_key("aḍu") != normalise_key("adu")

    def test_composed_and_decomposed_forms_remain_distinct(self) -> None:
        assert normalise_key("é") != normalise_key("é")


class TestDeduplication:
    def test_a_repeat_within_a_language_collapses_to_one(self) -> None:
        kept, stats = deduplicate([record("Ur ten-teg ca"), record("ur  TEN-teg ca")])
        assert len(kept) == 1
        assert stats.duplicate_dropped == 1

    def test_gold_wins_over_silver_for_the_same_line(self) -> None:
        kept, _ = deduplicate(
            [
                record("shared line here", source="b", trust="silver"),
                record("shared line here", source="a", trust="gold"),
            ]
        )
        assert kept[0].label_trust == "gold"

    def test_a_line_claimed_by_two_languages_is_dropped_entirely(self) -> None:
        """Not assigned to the more likely one: which language owns it is exactly the
        question, and guessing it puts the answer in the training data."""
        kept, stats = deduplicate(
            [record("ambiguous line", language="kab_Latn"), record("ambiguous line", "rif_Latn")]
        )
        assert kept == []
        assert stats.cross_language_dropped == 2

    def test_deduplication_of_nothing_is_empty_not_an_error(self) -> None:
        kept, stats = deduplicate([])
        assert kept == []
        assert stats.kept == 0

    def test_the_result_is_ordered_deterministically(self) -> None:
        rows = [record(f"line number {index}", source=f"s{index % 3}") for index in range(20)]
        assert deduplicate(rows)[0] == deduplicate(list(reversed(rows)))[0]


class TestDecontamination:
    def test_a_held_out_line_is_removed_from_training(self, tmp_path: Path) -> None:
        path = written_evalset(tmp_path, [Example("held out line", "kab_Latn", "s", "web")])
        kept, removed = decontaminate([record("Held  Out LINE"), record("other line")], path)
        assert removed == 1
        assert [row.text for row in kept] == ["other line"]

    def test_the_same_string_in_another_language_is_kept(self, tmp_path: Path) -> None:
        """Exclusion is keyed on `(language, text)`: the string is only contamination for
        the class it was held out for."""
        path = written_evalset(tmp_path, [Example("shared line", "kab_Latn", "s", "web")])
        kept, removed = decontaminate([record("shared line", language="rif_Latn")], path)
        assert removed == 0
        assert len(kept) == 1

    def test_the_check_is_positive_controlled(self, tmp_path: Path) -> None:
        """A check that cannot match returns zero, and zero reads as a clean result."""
        path = written_evalset(tmp_path, [Example("planted line", "kab_Latn", "s", "web")])
        _, removed = decontaminate([record("planted line")], path)
        assert removed == 1

    def test_a_missing_held_out_set_is_refused_not_skipped(self, tmp_path: Path) -> None:
        with pytest.raises(LidError, match="not found"):
            decontaminate([record("line")], tmp_path / "absent.jsonl")


class TestGrouping:
    def test_a_cell_at_the_threshold_keeps_its_own_group(self) -> None:
        rows = [record(f"a{i}", domain="web") for i in range(MIN_GROUP_ROWS)]
        rows += [record(f"b{i}", domain="everyday") for i in range(50)]
        assert assign_groups(rows)["kab_Latn/web"] == "kab_Latn/web"

    def test_a_cell_below_the_threshold_folds_into_the_language_s_largest(self) -> None:
        rows = [record(f"a{i}", domain="interface") for i in range(MIN_GROUP_ROWS - 1)]
        rows += [record(f"b{i}", domain="everyday") for i in range(50)]
        assert assign_groups(rows)["kab_Latn/interface"] == "kab_Latn/everyday"

    def test_a_tiny_cell_never_folds_into_another_language(self) -> None:
        rows = [record(f"a{i}", language="tzm_Latn", domain="interface") for i in range(3)]
        rows += [record(f"b{i}", language="tzm_Latn", domain="everyday") for i in range(40)]
        rows += [record(f"c{i}", language="kab_Latn", domain="web") for i in range(900)]
        assert assign_groups(rows)["tzm_Latn/interface"].startswith("tzm_Latn/")


class TestSplitting:
    def test_the_split_is_per_group_so_every_cell_reaches_dev(self) -> None:
        """A split drawn per language leaves dev 98.9% scripture for Tarifit, so it scores
        a scripture model and calls it Tarifit."""
        rows = [record(f"s{i}", language="rif_Latn", domain="scripture") for i in range(1000)]
        rows += [record(f"w{i}", language="rif_Latn", domain="web") for i in range(40)]
        _, dev = split(rows, dev_fraction=0.05, seed=1)
        assert {row.domain for row in dev} == {"scripture", "web"}

    def test_train_and_dev_are_disjoint_and_exhaustive(self) -> None:
        rows = [record(f"line {i}", domain="web") for i in range(200)]
        train, dev = split(rows, dev_fraction=0.1, seed=1)
        assert len(train) + len(dev) == 200
        assert {r.text for r in train}.isdisjoint(r.text for r in dev)

    def test_the_split_is_seeded(self) -> None:
        rows = [record(f"line {i}", domain="web") for i in range(200)]
        assert split(rows, dev_fraction=0.1, seed=1) == split(rows, dev_fraction=0.1, seed=1)
        assert split(rows, dev_fraction=0.1, seed=1) != split(rows, dev_fraction=0.1, seed=2)

    def test_a_single_row_group_stays_in_training(self) -> None:
        """Holding out the only row of a cell leaves nothing to learn it from."""
        rows = [record("only line here", domain="web")]
        train, dev = split(rows, dev_fraction=0.5, seed=1)
        assert len(train) == 1
        assert dev == []

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_an_impossible_fraction_is_refused(self, fraction: float) -> None:
        with pytest.raises(LidError, match="dev_fraction must be"):
            split([record("a")], dev_fraction=fraction, seed=1)


class TestReadingSources:
    def test_a_label_outside_the_class_list_is_refused(self, tmp_path: Path) -> None:
        rogue = Source("x", "zgh_Latn", "web", "gold", "cc0-1.0", "csv_target", "x.csv")
        with pytest.raises(LidError, match="outside the label set"):
            build_records(tmp_path, [rogue])

    def test_the_cap_bounds_every_source(self, tmp_path: Path) -> None:
        body = "h\n" + "".join(f"line number {index} of many here\n" for index in range(500))
        (tmp_path / "x.csv").write_text(body, encoding="utf-8")
        source = Source("x", "rif_Latn", "interface", "gold", "cc-by-3.0", "csv_target", "x.csv")
        assert len(build_records(tmp_path, [source], cap=25, seed=1)) == 25


class TestStatistics:
    def test_every_reported_number_comes_from_the_rows_that_are_written(
        self, tmp_path: Path
    ) -> None:
        path = written_evalset(tmp_path, [Example("held out", "kab_Latn", "s", "web")])
        train = [record(f"train {i}", domain="web") for i in range(20)]
        train += [record(f"other {i}", language="rif_Latn", domain="scripture") for i in range(9)]
        dev = [record("dev one", domain="web")]
        kept, dedup = deduplicate(train)
        stats = statistics(kept, dev, eval_path=path, dedup=dedup, decontaminated=3)
        assert stats["train_total"] == 29
        assert stats["dev_total"] == 1
        assert stats["eval_total"] == 1
        assert stats["decontaminated_against_eval"] == 3
        per_language = stats["per_language"]
        assert isinstance(per_language, dict)
        assert per_language["kab_Latn"]["train"] == 20
        assert per_language["rif_Latn"]["domains"] == 1

    def test_the_group_report_counts_folded_groups_not_raw_cells(self, tmp_path: Path) -> None:
        path = written_evalset(tmp_path, [Example("held out", "kab_Latn", "s", "web")])
        rows = [record(f"a{i}", domain="everyday") for i in range(60)]
        rows += [record(f"b{i}", domain="interface") for i in range(MIN_GROUP_ROWS - 1)]
        rows = grouped(rows)
        _, dedup = deduplicate(rows)
        stats = statistics(rows, [], eval_path=path, dedup=dedup, decontaminated=0)
        assert stats["train_groups"] == 1
        assert stats["train_by_group"] == {"kab_Latn/everyday": len(rows)}

    def test_an_absent_class_reports_zero_rather_than_a_missing_key(self, tmp_path: Path) -> None:
        path = written_evalset(tmp_path, [Example("held out", "kab_Latn", "s", "web")])
        rows = [record("only kabyle here")]
        _, dedup = deduplicate(rows)
        stats = statistics(rows, [], eval_path=path, dedup=dedup, decontaminated=0)
        per_language = stats["per_language"]
        assert isinstance(per_language, dict)
        assert per_language["shy_Latn"] == {
            "train": 0,
            "dev": 0,
            "eval": 0,
            "sources": 0,
            "domains": 0,
        }
        assert stats["train_imbalance_ratio"] is None


def test_write_jsonl_is_atomic_and_round_trips(tmp_path: Path) -> None:
    rows = [record("first line here"), record("second line here", language="rif_Latn")]
    destination = tmp_path / "nested" / "train.jsonl"
    write_jsonl(rows, destination)
    assert not list(tmp_path.rglob("*.part"))
    read = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["text"] for row in read] == ["first line here", "second line here"]
    assert read[1]["group"] == "rif_Latn/web"


class TestWrittenGroups:
    def test_a_folded_cell_is_written_and_not_only_reported(self) -> None:
        """The fold has to reach the file: the trainer groups on the written column, and a
        cell left unfolded there gives nine rows a whole batch slot."""
        rows = [record(f"a{i}", domain="everyday") for i in range(60)]
        rows += [record(f"b{i}", domain="interface") for i in range(MIN_GROUP_ROWS - 1)]
        written = {row.to_dict()["group"] for row in grouped(rows)}
        assert written == {"kab_Latn/everyday"}

    def test_a_large_enough_cell_keeps_its_own_name(self) -> None:
        rows = [record(f"a{i}", language="taq_Latn", domain="encyclopaedia") for i in range(40)]
        assert Counter(row.group for row in grouped(rows)) == {"taq_Latn/encyclopaedia": 40}

    def test_an_unstamped_record_falls_back_to_its_own_cell(self) -> None:
        assert record("a", language="taq_Latn", domain="web").to_dict()["group"] == "taq_Latn/web"
