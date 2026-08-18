"""Fine-tuning configuration and example encoding.

`Dataset` is exercised against a stand-in tokenizer: the encoding contract that matters is
that each example carries its own direction's language tags and that input order survives
the grouping, neither of which needs transformers to check.
"""

from __future__ import annotations

import pytest

from agbalu.mt.data import Example
from agbalu.mt.finetune import BASE_MODEL, SMALL_MODEL, Config, Dataset, nllb_code
from agbalu.mt.vocab import Remap


class FakeTokenizer:
    """Records the `src_lang`/`tgt_lang` in force when each batch was encoded."""

    def __init__(self) -> None:
        self.src_lang = ""
        self.tgt_lang = ""
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def __call__(
        self, texts: list[str], *, text_target: list[str], max_length: int, truncation: bool
    ) -> dict[str, list[list[int]]]:
        self.calls.append((self.src_lang, self.tgt_lang, tuple(texts)))

        def ids(text: str) -> list[int]:
            length = min(len(text), max_length) if truncation else len(text)
            return [1, length]

        return {
            "input_ids": [ids(t) for t in texts],
            "attention_mask": [[1, 1] for _ in texts],
            "labels": [ids(t) for t in text_target],
        }


class TestNllbCode:
    def test_each_side_of_each_direction(self) -> None:
        assert nllb_code("kab-eng", "source") == "kab_Latn"
        assert nllb_code("kab-eng", "target") == "eng_Latn"
        assert nllb_code("fra-kab", "source") == "fra_Latn"
        assert nllb_code("fra-kab", "target") == "kab_Latn"


class TestConfig:
    def test_it_fine_tunes_the_1_3b(self) -> None:
        """Chosen on measurement: the 1.3B is 4-5 chrF++ ahead of the 600M zero-shot on
        every direction, which is more than the recipe can recover."""
        assert Config().model == BASE_MODEL
        assert "1.3B" in Config().model

    def test_the_1_3b_carries_both_memory_workarounds(self) -> None:
        """Neither is free, and both exist only because a 24 GiB card cannot hold Adam's
        moments for 1,162M parameters."""
        config = Config()
        assert config.optimizer == "adafactor"
        assert config.gradient_checkpointing

    def test_the_600m_ablation_drops_them(self) -> None:
        small = Config().small()
        assert small.model == SMALL_MODEL
        assert small.optimizer == "adamw_torch"
        assert not small.gradient_checkpointing

    def test_both_arms_hold_the_same_effective_batch(self) -> None:
        """Otherwise the ablation would compare two recipes, not two models."""
        assert Config().effective_batch == Config().small().effective_batch == 2_048

    def test_the_effective_batch_is_the_published_one(self) -> None:
        """arXiv 2602.04442, single-language full fine-tuning of NLLB-200. The batch sets
        the step count: at 2,048 one epoch is 530 steps, at 64 it is 16,960."""
        assert Config().effective_batch == 2_048
        assert Config().learning_rate == 2e-4

    def test_the_effective_batch_is_the_product_of_both_knobs(self) -> None:
        assert Config(batch_size=8, gradient_accumulation=4).effective_batch == 32


class TestRunLengthCadence:
    """A cadence is a claim about the run it is set against. `eval_every=1_000` on a
    ~1,060-step run evaluated once, so early stopping could never fire."""

    TRAIN_EXAMPLES = 1_085_458
    """`data/processed/mt/train.jsonl`, per `agbalu-mt-v1.stats.json`."""

    def planned_steps(self, config: Config) -> int:
        per_epoch = self.TRAIN_EXAMPLES // config.effective_batch
        return int(per_epoch * config.epochs)

    def test_the_planned_run_is_about_a_thousand_steps(self) -> None:
        assert 900 <= self.planned_steps(Config()) <= 1_200

    def test_evaluation_happens_often_enough_for_early_stopping_to_fire(self) -> None:
        config = Config()
        evaluations = self.planned_steps(config) // config.eval_every
        assert evaluations > config.patience * 3

    def test_saving_is_a_multiple_of_evaluating(self) -> None:
        """`load_best_model_at_end` refuses a `save_steps` that is not."""
        config = Config()
        assert config.save_every % config.eval_every == 0

    def test_logging_fires_many_times_over_the_run(self) -> None:
        config = Config()
        assert self.planned_steps(config) // config.log_every >= 50


class TestDataset:
    def test_each_direction_is_encoded_under_its_own_language_tags(self) -> None:
        examples = [
            Example("azul", "hello", "kab-eng", "s"),
            Example("bonjour", "azul", "fra-kab", "s"),
        ]
        tokenizer = FakeTokenizer()
        Dataset(examples, tokenizer, Config())

        tags = {(src, tgt) for src, tgt, _ in tokenizer.calls}
        assert tags == {("kab_Latn", "eng_Latn"), ("fra_Latn", "kab_Latn")}

    def test_input_order_survives_the_grouping(self) -> None:
        """Grouping by direction reorders the encoding, and the features must come back in
        the order the examples were given."""
        examples = [
            Example("a", "bb", "kab-eng", "s"),
            Example("ccc", "d", "fra-kab", "s"),
            Example("eeeee", "f", "kab-eng", "s"),
        ]
        dataset = Dataset(examples, FakeTokenizer(), Config())
        assert [f["input_ids"][1] for f in dataset.features] == [1, 3, 5]

    def test_one_batch_per_direction_not_one_per_example(self) -> None:
        examples = [Example(f"a{i}", "b", "kab-eng", "s") for i in range(20)]
        tokenizer = FakeTokenizer()
        Dataset(examples, tokenizer, Config())
        assert len(tokenizer.calls) == 1

    def test_ids_are_remapped_when_the_model_is_trimmed(self) -> None:
        examples = [Example("a", "b", "kab-eng", "s")]
        remap = Remap((0, 1, 2))
        dataset = Dataset(examples, FakeTokenizer(), Config(), remap)
        assert dataset[0]["input_ids"] == [1, 1]
        assert dataset[0]["labels"] == [1, 1]

    def test_without_a_remap_the_ids_are_the_tokenizers_own(self) -> None:
        dataset = Dataset([Example("abc", "d", "kab-eng", "s")], FakeTokenizer(), Config())
        assert dataset[0]["input_ids"] == [1, 3]

    def test_an_empty_corpus_makes_an_empty_dataset(self) -> None:
        dataset = Dataset([], FakeTokenizer(), Config())
        assert len(dataset) == 0

    def test_every_feature_carries_the_three_keys_the_collator_needs(self) -> None:
        dataset = Dataset([Example("a", "b", "kab-eng", "s")], FakeTokenizer(), Config())
        assert set(dataset[0]) == {"input_ids", "attention_mask", "labels"}

    def test_an_out_of_range_index_raises(self) -> None:
        dataset = Dataset([], FakeTokenizer(), Config())
        with pytest.raises(IndexError):
            dataset[0]
