from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from agbalu.lid.config import BYTE_OFFSET, PAD_ID, ModelConfig
from agbalu.lid.dataset import (
    Augmentation,
    Corpus,
    CorpusError,
    Record,
    encode,
    group_batches,
    read_split,
    sequential_batches,
)


def row(text: str, language: str = "kab_Latn", domain: str = "web") -> Record:
    return Record(
        text=text, language=language, domain=domain, group=f"{language}/{domain}", source="s"
    )


@pytest.fixture(name="config")
def fixture_config() -> ModelConfig:
    return ModelConfig(embed_dim=32, num_heads=4, num_layers=1, intermediate_dim=64, conv_dim=8)


class TestEncoding:
    def test_bytes_are_shifted_past_the_reserved_ids(self) -> None:
        assert encode("A", 16) == [ord("A") + BYTE_OFFSET]
        assert min(encode("aɣbalu", 32)) >= BYTE_OFFSET

    def test_a_multi_byte_character_becomes_more_than_one_token(self) -> None:
        """`ɣ` is two UTF-8 bytes; the model sees the sequence, which is the point of a
        byte model for orthographies that differ in exactly those characters."""
        assert len(encode("ɣ", 16)) == 2
        assert len(encode("ẓ", 16)) == 3

    def test_truncation_is_on_bytes_and_may_cut_a_character(self) -> None:
        assert len(encode("ɣɣɣ", 3)) == 3

    def test_an_empty_string_encodes_to_nothing(self) -> None:
        assert encode("", 16) == []

    @pytest.mark.parametrize("text", ["﻿abc", "a b", "a\u200bb", "a\x00b", "🙂"])
    def test_awkward_unicode_encodes_without_raising(self, text: str) -> None:
        assert all(token >= BYTE_OFFSET for token in encode(text, 64))


class TestReadingASplit:
    def test_a_split_round_trips_its_provenance(self, tmp_path: Path) -> None:
        target = tmp_path / "train.jsonl"
        target.write_text(
            json.dumps(
                {
                    "text": "awal",
                    "language": "rif_Latn",
                    "domain": "interface",
                    "group": "rif_Latn/interface",
                    "source": "translatewiki.rif",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        records = read_split(target)
        assert records[0].group == "rif_Latn/interface"
        assert records[0].source == "translatewiki.rif"

    def test_a_missing_split_names_the_command_that_builds_it(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="make lid-dataset"):
            read_split(tmp_path / "absent.jsonl")

    def test_an_empty_split_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "train.jsonl").write_text("\n \n", encoding="utf-8")
        with pytest.raises(CorpusError, match="carried no rows"):
            read_split(tmp_path / "train.jsonl")


class TestCorpus:
    def test_labels_and_groups_are_indexed_against_the_class_list(
        self, config: ModelConfig
    ) -> None:
        corpus = Corpus([row("a", "kab_Latn"), row("b", "shy_Latn")], config)
        assert corpus.labels == [0, 5]
        assert corpus.groups == ("kab_Latn/web", "shy_Latn/web")

    def test_a_label_outside_the_class_list_is_refused(self, config: ModelConfig) -> None:
        with pytest.raises(CorpusError, match="outside the class list"):
            Corpus([row("a", "zgh_Latn")], config)

    def test_a_shared_group_index_keeps_dev_aligned_with_train(self, config: ModelConfig) -> None:
        """A group only dev contains would otherwise shift every index by one and make the
        group-DRO weights point at the wrong cells."""
        train = Corpus([row("a", domain="web"), row("b", domain="everyday")], config)
        dev = Corpus([row("c", domain="everyday")], config, groups=train.groups)
        assert dev.groups == train.groups
        assert dev.group_ids == [train.groups.index("kab_Latn/everyday")]

    def test_an_unseen_group_falls_back_within_its_own_language(self, config: ModelConfig) -> None:
        train = Corpus([row("a", "rif_Latn", "scripture")], config)
        dev = Corpus([row("b", "rif_Latn", "interface")], config, groups=train.groups)
        assert dev.groups[dev.group_ids[0]].startswith("rif_Latn/")

    def test_an_unseen_language_cannot_fall_back(self, config: ModelConfig) -> None:
        train = Corpus([row("a", "rif_Latn", "scripture")], config)
        with pytest.raises(CorpusError, match="no group for language 'tzm_Latn'"):
            Corpus([row("b", "tzm_Latn", "web")], config, groups=train.groups)

    def test_class_counts_report_absent_classes_as_zero(self, config: ModelConfig) -> None:
        corpus = Corpus([row("a", "kab_Latn")], config)
        assert corpus.class_counts[0] == 1
        assert corpus.class_counts[5] == 0
        assert len(corpus.class_counts) == config.num_classes


class TestCollation:
    def test_padding_is_masked_and_the_batch_is_the_longest_row(self, config: ModelConfig) -> None:
        corpus = Corpus([row("ab"), row("abcdefgh")], config)
        batch = corpus.collate([0, 1])
        assert batch.input_ids.shape == (2, 8)
        assert batch.pad_mask[0].tolist() == [True, True, *([False] * 6)]
        assert batch.input_ids[0, 2:].eq(PAD_ID).all()

    def test_an_empty_text_still_yields_one_masked_position(self, config: ModelConfig) -> None:
        corpus = Corpus([row("")], config)
        batch = corpus.collate([0])
        assert batch.input_ids.shape == (1, 1)
        assert len(batch) == 1

    def test_a_batch_moves_to_a_device_as_a_whole(self, config: ModelConfig) -> None:
        corpus = Corpus([row("abc")], config)
        moved = corpus.collate([0]).to(torch.device("cpu"))
        assert moved.labels.device.type == "cpu"

    def test_augmentation_is_seeded_and_off_by_default(self, config: ModelConfig) -> None:
        records = [row(" ".join(f"w{i}" for i in range(12)))]
        plain = Corpus(records, config)
        assert plain.encoded(0) == encode(records[0].text, config.max_length)
        first = Corpus(records, config, augmentation=Augmentation(phrase_prob=1.0, seed=5))
        second = Corpus(records, config, augmentation=Augmentation(phrase_prob=1.0, seed=5))
        assert [first.encoded(0) for _ in range(4)] == [second.encoded(0) for _ in range(4)]

    def test_case_augmentation_lowercases(self, config: ModelConfig) -> None:
        corpus = Corpus(
            [row("ABCD EFGH IJKL MNOP")], config, augmentation=Augmentation(case_prob=1.0, seed=1)
        )
        assert corpus.encoded(0) == encode("abcd efgh ijkl mnop", config.max_length)


class TestGroupBatches:
    def test_every_batch_holds_the_declared_shape(self, config: ModelConfig) -> None:
        records = [row(f"a{i}", domain=f"d{i % 5}") for i in range(200)]
        corpus = Corpus(records, config)
        batches = list(
            group_batches(corpus, groups_per_batch=3, samples_per_group=4, batches=6, seed=1)
        )
        assert len(batches) == 6
        assert all(len(batch) == 12 for batch in batches)

    def test_a_batch_never_draws_the_same_group_twice(self, config: ModelConfig) -> None:
        records = [row(f"a{i}", domain=f"d{i % 4}") for i in range(80)]
        corpus = Corpus(records, config)
        for batch in group_batches(
            corpus, groups_per_batch=4, samples_per_group=3, batches=20, seed=2
        ):
            counts = Counter(corpus.group_ids[index] for index in batch)
            assert set(counts.values()) == {3}

    def test_the_tail_group_is_drawn_as_often_as_the_head(self, config: ModelConfig) -> None:
        """The point of grouping: 900 Kabyle rows and 9 Tamazight rows must not give the
        head 100x the gradient."""
        records = [row(f"k{i}", "kab_Latn", "web") for i in range(900)]
        records += [row(f"t{i}", "tzm_Latn", "interface") for i in range(9)]
        corpus = Corpus(records, config)
        seen: Counter[str] = Counter()
        for batch in group_batches(
            corpus, groups_per_batch=2, samples_per_group=5, batches=100, seed=3
        ):
            seen.update(corpus.records[index].language for index in batch)
        assert seen["tzm_Latn"] == seen["kab_Latn"]

    def test_more_groups_requested_than_exist_uses_all_of_them(self, config: ModelConfig) -> None:
        corpus = Corpus([row(f"a{i}", domain="web") for i in range(10)], config)
        batch = next(
            iter(group_batches(corpus, groups_per_batch=8, samples_per_group=2, batches=1, seed=1))
        )
        assert len(batch) == 2

    def test_the_sampler_is_seeded(self, config: ModelConfig) -> None:
        corpus = Corpus([row(f"a{i}", domain=f"d{i % 3}") for i in range(60)], config)
        first = list(
            group_batches(corpus, groups_per_batch=2, samples_per_group=3, batches=5, seed=9)
        )
        assert first == list(
            group_batches(corpus, groups_per_batch=2, samples_per_group=3, batches=5, seed=9)
        )
        assert first != list(
            group_batches(corpus, groups_per_batch=2, samples_per_group=3, batches=5, seed=10)
        )

    @pytest.mark.parametrize(("groups", "samples"), [(0, 4), (4, 0)])
    def test_a_non_positive_shape_is_refused(
        self, config: ModelConfig, groups: int, samples: int
    ) -> None:
        corpus = Corpus([row("a")], config)
        with pytest.raises(CorpusError, match="batch shape must be positive"):
            list(
                group_batches(
                    corpus, groups_per_batch=groups, samples_per_group=samples, batches=1, seed=1
                )
            )


class TestSequentialBatches:
    def test_every_row_appears_exactly_once(self, config: ModelConfig) -> None:
        corpus = Corpus([row(f"line {i}") for i in range(37)], config)
        seen = [index for batch in sequential_batches(corpus, 8) for index in batch]
        assert sorted(seen) == list(range(37))

    def test_rows_are_grouped_by_length_so_padding_does_not_dominate(
        self, config: ModelConfig
    ) -> None:
        corpus = Corpus([row("x" * 100), row("y"), row("z" * 99), row("w"), row("v")], config)
        batches = list(sequential_batches(corpus, 2))
        lengths = [len(corpus.records[index].text) for index in batches[0]]
        assert max(lengths) <= 2
