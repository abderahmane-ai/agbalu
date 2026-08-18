"""Tokenisation, label alignment and batching.

The alignment is the part that can be wrong silently: a label on the wrong subword still
trains, still validates and still reports a number. Every test here checks the position, not
just the shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from agbalu.punctuation.corpus import Row
from agbalu.punctuation.dataset import (
    collate,
    encode_annotation,
    encode_corpus,
    encode_words,
    iter_batches,
    label_counts,
)
from agbalu.punctuation.labels import IGNORE_INDEX, PUNCTUATION_INDEX, annotate
from agbalu.tokenizer.spec import CLS_ID, PAD_ID, SEP_ID


class FakeTokenizer:
    """One piece per character, so a word's subword count is its length and positions are
    checkable by hand. Ids avoid the reserved range."""

    def encode(self, text: str, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [10 + (ord(char) % 100) for char in text]


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def test_encode_words_marks_the_first_subword_of_each_word(tokenizer: FakeTokenizer) -> None:
    ids, first = encode_words(("ab", "cde", "f"), tokenizer)

    assert ids[0] == CLS_ID
    assert ids[-1] == SEP_ID
    assert len(ids) == 1 + 2 + 3 + 1 + 1
    assert first == [1, 3, 6]


def test_encode_words_drops_whole_words_rather_than_splitting_one(
    tokenizer: FakeTokenizer,
) -> None:
    ids, first = encode_words(("aaaa", "bbbb", "cccc"), tokenizer, max_length=8)

    assert len(first) == 1
    assert len(ids) == 1 + 4 + 1
    assert ids[-1] == SEP_ID


def test_encode_words_on_no_words(tokenizer: FakeTokenizer) -> None:
    assert encode_words((), tokenizer) == ([CLS_ID, SEP_ID], [])


def test_encode_annotation_puts_labels_where_the_words_start(tokenizer: FakeTokenizer) -> None:
    annotation = annotate("azul, d acu?")
    ids, punctuation, case = encode_annotation(annotation, tokenizer)

    assert len(ids) == len(punctuation) == len(case)
    labelled = [index for index, value in enumerate(punctuation) if value != IGNORE_INDEX]
    assert labelled == encode_words(annotation.words, tokenizer)[1]
    assert [punctuation[index] for index in labelled] == list(annotation.punctuation)
    assert punctuation[0] == punctuation[-1] == IGNORE_INDEX


def test_encode_annotation_labels_the_question(tokenizer: FakeTokenizer) -> None:
    annotation = annotate("d acu?")
    _, punctuation, _ = encode_annotation(annotation, tokenizer)
    assert PUNCTUATION_INDEX["QUESTION"] in punctuation


def test_encode_corpus_offsets_partition_the_flat_arrays(tokenizer: FakeTokenizer) -> None:
    rows = [Row("Azul fell-awen.", "s"), Row("D acu?", "s"), Row("Ihi, ad nruḥ.", "s")]
    corpus = encode_corpus(rows, tokenizer)

    assert len(corpus) == 3
    assert corpus.offsets[0] == 0
    assert corpus.offsets[-1] == corpus.ids.size
    assert corpus.ids.size == corpus.punctuation.size == corpus.case.size
    for index in range(3):
        ids, punctuation, case = corpus.example(index)
        assert ids[0] == CLS_ID
        assert ids[-1] == SEP_ID
        assert ids.size == punctuation.size == case.size


def test_label_counts_ignores_the_unlabelled_positions(tokenizer: FakeTokenizer) -> None:
    corpus = encode_corpus([Row("Azul, d acu?", "s")], tokenizer)
    punctuation, case = label_counts(corpus)

    assert sum(punctuation.values()) == 3
    assert punctuation["COMMA"] == 1
    assert punctuation["QUESTION"] == 1
    assert punctuation["NONE"] == 1
    assert case["UPPER_INIT"] == 1
    assert case["LOWER"] == 2


def test_collate_pads_to_the_longest_row_only(tokenizer: FakeTokenizer) -> None:
    corpus = encode_corpus([Row("Azul fell-awen n tmurt.", "s"), Row("D acu?", "s")], tokenizer)
    batch = collate(corpus, np.array([0, 1], dtype=np.int64))

    widths = [corpus.example(index)[0].size for index in (0, 1)]
    assert batch.input_ids.shape == (2, max(widths))
    assert int(batch.attention_mask[0].sum()) == widths[0]
    assert int(batch.attention_mask[1].sum()) == widths[1]
    assert int(batch.input_ids[1, widths[1]]) == PAD_ID
    assert int(batch.punctuation[1, widths[1]]) == IGNORE_INDEX
    assert batch.labelled == sum(
        int((corpus.example(index)[1] != IGNORE_INDEX).sum()) for index in (0, 1)
    )


def test_padding_is_never_labelled(tokenizer: FakeTokenizer) -> None:
    corpus = encode_corpus([Row("Azul fell-awen n tmurt-nneɣ.", "s"), Row("Ih.", "s")], tokenizer)
    batch = collate(corpus, np.array([0, 1], dtype=np.int64))

    assert not bool(((batch.punctuation != IGNORE_INDEX) & ~batch.attention_mask).any())


def test_batches_cover_every_row_exactly_once(tokenizer: FakeTokenizer) -> None:
    corpus = encode_corpus(
        [Row(f"Azul{'a' * index} d acu.", "s") for index in range(17)], tokenizer
    )
    generator = np.random.default_rng(0)

    for batches in (
        iter_batches(corpus, 5, shuffle=False),
        iter_batches(corpus, 5, shuffle=True, generator=generator),
    ):
        seen = np.concatenate(batches)
        assert sorted(seen.tolist()) == list(range(17))
        assert len(batches) == 4


def test_batches_are_length_bucketed(tokenizer: FakeTokenizer) -> None:
    rows = [Row("a " * length + "b.", "s") for length in (1, 30, 2, 29, 3, 28)]
    corpus = encode_corpus(rows, tokenizer)
    lengths = np.diff(corpus.offsets)

    for batch in iter_batches(corpus, 2, shuffle=True, generator=np.random.default_rng(1)):
        spread = lengths[batch].max() - lengths[batch].min()
        assert spread < lengths.max() - lengths.min()


def test_shuffling_without_a_generator_is_refused(tokenizer: FakeTokenizer) -> None:
    corpus = encode_corpus([Row("Azul d acu.", "s")], tokenizer)
    with pytest.raises(ValueError, match="needs a generator"):
        iter_batches(corpus, 2, shuffle=True)
