"""Sentiment scoring, batching, and the two settings' shapes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from agbalu.bench.sentiment import (
    FINETUNE,
    LABEL_NAMES,
    MAX_PIECES,
    NUM_CLASSES,
    PROBE,
    SETTINGS,
    Head,
    Item,
    SentimentError,
    batches,
    confusion_of,
    encode,
    macro_f1_of,
    read_split,
)
from agbalu.tokenizer.spec import CLS_ID, SEP_ID


class FakeTokenizer:
    """One id per character, offset past the specials."""

    def encode(self, text: str, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [ord(character) % 100 + 5 for character in text]


def items() -> list[Item]:
    return [
        Item(id="a", text="azul", label=2),
        Item(id="b", text="ur zmireɣ ara", label=0),
        Item(id="c", text="d ayen", label=1),
    ]


def test_a_perfect_prediction_scores_one_everywhere() -> None:
    matrix = confusion_of([0, 1, 2], [0, 1, 2])
    macro, per_class = macro_f1_of(matrix)
    assert macro == 1.0
    assert all(per_class[name]["f1"] == 1.0 for name in LABEL_NAMES)


def test_a_class_the_system_never_predicts_scores_zero_rather_than_dividing_by_zero() -> None:
    matrix = confusion_of([0, 1, 2], [0, 0, 0])
    macro, per_class = macro_f1_of(matrix)
    assert per_class["neutral"]["f1"] == 0.0
    assert per_class["positive"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert 0.0 < macro < 1.0


def test_macro_f1_is_not_accuracy_on_an_unbalanced_prediction() -> None:
    """Micro F1 would equal accuracy on this balanced split and report it twice."""
    matrix = confusion_of([0, 0, 1, 2], [0, 0, 0, 0])
    macro, _ = macro_f1_of(matrix)
    accuracy = sum(matrix[i][i] for i in range(NUM_CLASSES)) / 4
    assert accuracy == 0.5
    assert macro != accuracy


def test_the_confusion_matrix_is_gold_by_predicted() -> None:
    matrix = confusion_of([0, 0], [0, 2])
    assert matrix[0][0] == 1
    assert matrix[0][2] == 1
    assert matrix[2][0] == 0


def test_encoding_brackets_each_sentence_and_bounds_its_length() -> None:
    long_item = [Item(id="x", text="a" * (MAX_PIECES + 50), label=0)]
    encoded = encode(long_item, FakeTokenizer())[0]
    assert encoded[0] == CLS_ID
    assert encoded[-1] == SEP_ID
    assert len(encoded) == MAX_PIECES + 2


def test_batches_pad_to_the_longest_member_and_mask_the_padding() -> None:
    rows = items()
    encoded = encode(rows, FakeTokenizer())
    labels = [row.label for row in rows]
    input_ids, mask, gold = next(
        iter(batches(encoded, labels, batch_size=3, order=[0, 1, 2], device=torch.device("cpu")))
    )
    assert input_ids.shape == mask.shape
    assert input_ids.shape[0] == 3
    assert gold.tolist() == labels
    for row, pieces in enumerate(encoded):
        assert mask[row].sum().item() == len(pieces)
        assert input_ids[row, len(pieces) :].sum().item() == 0


def test_the_order_is_the_callers_so_a_shuffle_is_a_seeded_decision() -> None:
    rows = items()
    encoded = encode(rows, FakeTokenizer())
    labels = [row.label for row in rows]
    _, _, gold = next(
        iter(batches(encoded, labels, batch_size=3, order=[2, 0, 1], device=torch.device("cpu")))
    )
    assert gold.tolist() == [labels[2], labels[0], labels[1]]


def test_the_probe_head_is_one_linear_layer_and_the_finetune_head_is_not() -> None:
    """The probe measures what pretraining put in the representation. A hidden layer
    would let the head learn the task instead, which is a different question."""
    probe = Head(8, bottleneck=False)
    assert probe.dense is None
    assert sum(1 for _ in probe.modules() if isinstance(_, torch.nn.Linear)) == 1

    tuned = Head(8, bottleneck=True)
    assert tuned.dense is not None
    assert tuned(torch.zeros((2, 8))).shape == (2, NUM_CLASSES)


def test_the_two_settings_do_not_share_a_recipe() -> None:
    assert SETTINGS["probe"] is PROBE
    assert SETTINGS["finetune"] is FINETUNE
    assert PROBE.learning_rate > FINETUNE.learning_rate
    assert PROBE.seed == FINETUNE.seed == 42


def test_a_split_reads_back_the_fields_the_corpus_carries(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": f"k{i}", "text_kab": "azul", "label": i}) for i in range(NUM_CLASSES)
        )
        + "\n\n",
        encoding="utf-8",
    )
    loaded = read_split("dev", tmp_path)
    assert [item.label for item in loaded] == [0, 1, 2]


def test_a_missing_split_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(SentimentError, match="no test split"):
        read_split("test", tmp_path)
