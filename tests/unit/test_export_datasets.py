"""Staging the publishable dataset repositories.

The load-bearing tests are the two that catch a defect nothing else would: a card whose
declared configs disagree with what is staged — which the Hub turns into a load error for
whoever downloads it — and the permissive filter, which is the only thing standing between a
share-alike source and a CC-BY-4.0 release.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.export_datasets import (
    Config,
    ExportError,
    Repo,
    check_tasks,
    declared_configs,
    partition,
    permissive,
    project,
    read_jsonl,
    stage,
)

from agbalu.registry.models import redistribution_class

FRONT = """---
license: cc-by-4.0
configs:
  - config_name: alpha
    data_files:
      - split: train
        path: alpha/train.jsonl
---

# Card
"""


def card(tmp_path: Path, text: str = FRONT) -> Path:
    path = tmp_path / "card.md"
    path.write_text(text, encoding="utf-8")
    return path


def alpha(rows: list[dict[str, object]] | None = None) -> Config:
    return Config(
        name="alpha",
        splits={"train": rows if rows is not None else [{"form": "azul", "lemma": "azul"}]},
        fields=("form", "lemma"),
    )


class TestReadJsonl:
    def test_a_missing_file_points_at_its_make_target(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="Run its `make` target"):
            read_jsonl(tmp_path / "absent.jsonl")

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ExportError, match="is empty"):
            read_jsonl(path)

    def test_a_malformed_line_names_its_number(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
        with pytest.raises(ExportError, match=":2 is not JSON"):
            read_jsonl(path)

    def test_a_json_scalar_is_not_a_row(self, tmp_path: Path) -> None:
        path = tmp_path / "a.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")
        with pytest.raises(ExportError, match="not a JSON object"):
            read_jsonl(path)


class TestProject:
    def test_only_the_declared_fields_survive(self) -> None:
        rows = [{"a": 1, "secret": 2}]
        assert project(rows, ("a",)) == [{"a": 1}]

    def test_an_absent_field_becomes_null_so_the_schema_stays_rectangular(self) -> None:
        """`glosses` and `lemma` are genuinely optional; a reader must not have to tell an
        absent key from an unknown value."""
        assert project([{"a": 1}], ("a", "b")) == [{"a": 1, "b": None}]

    def test_field_order_is_the_declared_order(self) -> None:
        assert list(project([{"b": 1, "a": 2}], ("a", "b"))[0]) == ["a", "b"]


class TestPartition:
    def test_rows_group_by_their_own_key(self) -> None:
        rows = [{"split": "dev"}, {"split": "devtest"}, {"split": "dev"}]
        assert {k: len(v) for k, v in partition(rows, "split").items()} == {"dev": 2, "devtest": 1}

    def test_a_row_without_the_key_is_refused(self) -> None:
        with pytest.raises(ExportError, match="no usable 'split'"):
            partition([{"text": "a"}], "split")

    def test_an_empty_key_is_refused(self) -> None:
        with pytest.raises(ExportError, match="no usable 'split'"):
            partition([{"split": ""}], "split")


class TestPermissive:
    def test_only_permissive_rows_survive(self) -> None:
        rows = [
            {"redistribution": "permissive"},
            {"redistribution": "share-alike"},
            {"redistribution": "unclear"},
            {"redistribution": "non-commercial"},
        ]
        assert permissive(rows) == [{"redistribution": "permissive"}]

    def test_an_odbl_row_does_not_reach_a_permissive_release(self) -> None:
        """ODbL §4.4 is share-alike for derivative databases. Classed permissive, 8,554
        toponym entries would have been published under CC-BY-4.0."""
        assert redistribution_class("odbl") == "share-alike"
        assert (
            permissive([{"licence": "odbl", "redistribution": redistribution_class("odbl")}]) == []
        )

    def test_a_row_with_no_class_is_excluded_rather_than_assumed(self) -> None:
        assert permissive([{"licence": "mit"}]) == []


class TestConfig:
    def test_a_split_with_no_rows_is_refused(self) -> None:
        with pytest.raises(ExportError, match="no rows"):
            Config(name="a", splits={"train": []}, fields=("x",))

    def test_a_config_with_no_split_is_refused(self) -> None:
        with pytest.raises(ExportError, match="publishes nothing"):
            Config(name="a", splits={}, fields=("x",))

    def test_a_field_no_row_carries_is_a_misspelling(self) -> None:
        with pytest.raises(ExportError, match=r"no row carries \['glosess'\]"):
            Config(name="a", splits={"train": [{"glosses": 1}]}, fields=("glosess",))

    def test_a_field_only_some_rows_carry_is_allowed(self) -> None:
        config = Config(name="a", splits={"train": [{"x": 1}, {"x": None}]}, fields=("x",))
        assert config.rows == 2


class TestDeclaredConfigs:
    def test_the_config_names_come_off_the_front_matter(self, tmp_path: Path) -> None:
        assert declared_configs(card(tmp_path)) == {"alpha"}

    def test_a_card_without_front_matter_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="no YAML front matter"):
            declared_configs(card(tmp_path, "# Card\n"))


class TestStage:
    def test_a_repo_writes_its_data_card_and_report(self, tmp_path: Path) -> None:
        repo = Repo(name="R", card=card(tmp_path), configs=(alpha(),))
        report = stage(repo, tmp_path / "out")
        target = tmp_path / "out" / "R"
        assert (target / "alpha" / "train.jsonl").is_file()
        assert (target / "README.md").read_text(encoding="utf-8").startswith("---")
        assert report.rows == 1
        assert json.loads((target / "dataset.stats.json").read_text(encoding="utf-8")) == {
            "repo": "R",
            "files": [
                {
                    "config": "alpha",
                    "split": "train",
                    "rows": 1,
                    "path": "alpha/train.jsonl",
                    "bytes": (target / "alpha" / "train.jsonl").stat().st_size,
                }
            ],
            "rows": 1,
        }

    def test_the_rows_round_trip_as_jsonl(self, tmp_path: Path) -> None:
        rows: list[dict[str, object]] = [
            {"form": "aɣbalu", "lemma": "aɣbalu"},
            {"form": "ţ", "lemma": None},
        ]
        repo = Repo(name="R", card=card(tmp_path), configs=(alpha(rows),))
        stage(repo, tmp_path / "out")
        path = tmp_path / "out" / "R" / "alpha" / "train.jsonl"
        assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == rows

    def test_non_ascii_is_written_unescaped(self, tmp_path: Path) -> None:
        """A Kabyle dataset whose file is full of `\\u0263` is unreadable to a human."""
        rows: list[dict[str, object]] = [{"form": "aɣbalu", "lemma": "aɣbalu"}]
        repo = Repo(name="R", card=card(tmp_path), configs=(alpha(rows),))
        stage(repo, tmp_path / "out")
        raw = (tmp_path / "out" / "R" / "alpha" / "train.jsonl").read_text(encoding="utf-8")
        assert "aɣbalu" in raw

    def test_a_card_declaring_a_config_that_is_not_staged_is_refused(self, tmp_path: Path) -> None:
        """The failure this prevents happens on the *downloader's* machine, not ours."""
        text = FRONT.replace("config_name: alpha", "config_name: beta")
        repo = Repo(name="R", card=card(tmp_path, text), configs=(alpha(),))
        with pytest.raises(ExportError, match="the card declares"):
            stage(repo, tmp_path / "out")

    def test_a_stale_directory_does_not_survive_a_restage(self, tmp_path: Path) -> None:
        """A staging directory publishes whatever it last held; the org card reached the Hub
        pre-fix exactly this way."""
        out = tmp_path / "out"
        (out / "R").mkdir(parents=True)
        (out / "R" / "stale.jsonl").write_text("{}\n", encoding="utf-8")
        stage(Repo(name="R", card=card(tmp_path), configs=(alpha(),)), out)
        assert not (out / "R" / "stale.jsonl").exists()

    def test_a_repo_with_no_config_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="nothing to publish"):
            Repo(name="R", card=card(tmp_path), configs=())

    def test_a_missing_card_is_refused_before_anything_is_written(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="card not found"):
            Repo(name="R", card=tmp_path / "absent.md", configs=(alpha(),))
        assert not (tmp_path / "out").exists()


class TestCheckTasks:
    """`task_categories` is a closed vocabulary the Hub does not enforce at upload: a bad
    name publishes fine and warns only on the rendered card. `text2text-generation` reached
    a live public dataset that way."""

    def with_tasks(self, tmp_path: Path, *tasks: str) -> Path:
        listed = "\n".join(f"  - {t}" for t in tasks)
        return card(tmp_path, FRONT.replace("license: cc-by-4.0", f"task_categories:\n{listed}"))

    def test_valid_categories_pass(self, tmp_path: Path) -> None:
        check_tasks(self.with_tasks(tmp_path, "token-classification", "translation"))

    def test_the_category_that_reached_the_hub_is_now_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ExportError, match="text2text-generation"):
            check_tasks(self.with_tasks(tmp_path, "token-classification", "text2text-generation"))

    def test_a_card_declaring_no_task_passes(self, tmp_path: Path) -> None:
        check_tasks(card(tmp_path))

    def test_the_shipped_cards_declare_only_valid_categories(self) -> None:
        """The two cards that are actually published, checked against the vocabulary."""
        for name in ("kabbench.md", "kablex.md"):
            check_tasks(Path("docs/cards") / name)

    def test_staging_refuses_before_writing_anything(self, tmp_path: Path) -> None:
        bad = FRONT.replace("license: cc-by-4.0", "task_categories:\n  - text2text-generation")
        repo = Repo(name="R", card=card(tmp_path, bad), configs=(alpha(),))
        with pytest.raises(ExportError, match="not valid task_categories"):
            stage(repo, tmp_path / "out")
        assert not (tmp_path / "out" / "R").exists()
