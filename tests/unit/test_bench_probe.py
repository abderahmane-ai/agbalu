"""The linear probe: word-to-position alignment, and the Tagger contract it must meet.

Alignment is the part that can fail silently. A tagger that returns the wrong number of
labels, or reads a word at the wrong position, still produces a plausible accuracy.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from agbalu.bench.probe import (
    UPOS_TAGS,
    Head,
    ProbeConfig,
    ProbeTagger,
    encode_words,
    examples_for,
    train_head,
)
from agbalu.model.config import PRESETS
from agbalu.tokenizer.spec import CLS_ID, SEP_ID
from agbalu.treebank import Sentence, Token, Word

HIDDEN = 12
"""Divisible by the preset's 6 attention heads, which `ModelConfig` validates."""


class FakeTokenizer:
    """One piece per character, so a word's subword count is its length."""

    def encode(self, text: str, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [ord(character) % 100 + 5 for character in text]

    def unk_id(self) -> int:
        return 1


class EmptyTokenizer(FakeTokenizer):
    """A vocabulary that can produce nothing for a word — the `unk` fallback path."""

    def encode(self, _text: str, out_type: type[int]) -> list[int]:
        assert out_type is int
        return []


class FakeEncoder:
    """Deterministic vectors keyed on the piece id, so the head has a signal to learn."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        self.config = replace(PRESETS["kab"], hidden_size=hidden)
        self._hidden = hidden

    def contextualise(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert input_ids.shape == attention_mask.shape
        base = input_ids.unsqueeze(-1).float()
        columns = torch.arange(self._hidden, dtype=torch.float32)
        return torch.sin(base * (columns + 1) * 0.1)


def word(identifier: int, form: str, upos: str) -> Word:
    return Word(
        id=identifier,
        form=form,
        lemma="_",
        upos=upos,
        feats="_",
        head=0,
        deprel="root",
        space_after=True,
    )


def sentence(*words: Word) -> Sentence:
    return Sentence(
        sent_id="s1",
        text=" ".join(w.form for w in words),
        split="train",
        words=words,
        tokens=tuple(Token(form=w.form, space_after=True, words=(w,)) for w in words),
    )


class TestEncodeWords:
    def test_one_first_subword_position_per_word(self) -> None:
        pieces, first = encode_words(["axxam", "d", "ameqqran"], FakeTokenizer())
        assert len(first) == 3
        assert pieces[0].item() == CLS_ID
        assert pieces[-1].item() == SEP_ID

    def test_positions_point_at_the_start_of_each_word(self) -> None:
        words = ["ab", "cde", "f"]
        pieces, first = encode_words(words, FakeTokenizer())
        assert first == [1, 3, 6]
        assert pieces[first[1]].item() == FakeTokenizer().encode("cde", int)[0]

    def test_a_word_producing_no_pieces_still_gets_one_position(self) -> None:
        """Otherwise the labels shift by one from that word onward."""
        _, first = encode_words(["a", "b"], EmptyTokenizer())
        assert len(first) == 2

    def test_an_empty_sentence_is_just_the_two_specials(self) -> None:
        pieces, first = encode_words([], FakeTokenizer())
        assert first == []
        assert pieces.tolist() == [CLS_ID, SEP_ID]

    def test_a_long_sentence_is_truncated_to_the_window(self) -> None:
        pieces, _ = encode_words(["a" * 50] * 40, FakeTokenizer(), max_pieces=32)
        assert pieces.shape[0] == 34


class TestExamplesFor:
    def test_labels_line_up_with_words(self) -> None:
        examples, skipped = examples_for(
            [sentence(word(1, "axxam", "NOUN"), word(2, "d", "AUX"))], FakeTokenizer()
        )
        assert skipped == 0
        _, first, labels = examples[0]
        assert len(first) == len(labels) == 2
        assert labels[0] == UPOS_TAGS.index("NOUN")

    def test_an_unknown_tag_is_ignored_rather_than_mislabelled(self) -> None:
        examples, _ = examples_for([sentence(word(1, "x", "WEIRD"))], FakeTokenizer())
        assert examples[0][2] == [-100]

    def test_an_empty_corpus_yields_nothing(self) -> None:
        assert examples_for([], FakeTokenizer()) == ([], 0)


class TestTraining:
    def test_the_head_learns_something_from_a_frozen_encoder(self) -> None:
        sentences = [
            sentence(word(1, "axxam", "NOUN"), word(2, "d", "AUX")),
            sentence(word(1, "argaz", "NOUN"), word(2, "ur", "PART")),
        ] * 8
        examples, _ = examples_for(sentences, FakeTokenizer())
        head = Head(HIDDEN)
        history = train_head(
            FakeEncoder(),
            head,
            examples,
            device=torch.device("cpu"),
            config=ProbeConfig(epochs=8, batch_size=4),
        )
        assert len(history) == 8
        assert history[-1] < history[0]

    def test_the_same_seed_gives_the_same_head(self) -> None:
        """A reported accuracy is a claim someone can check. Seeding the batch order but
        not the initialisation left ~0.5 points of drift between runs of one checkpoint."""
        sentences = [
            sentence(word(1, "axxam", "NOUN"), word(2, "d", "AUX")),
            sentence(word(1, "argaz", "NOUN"), word(2, "ur", "PART")),
        ] * 4
        examples, _ = examples_for(sentences, FakeTokenizer())

        def fit(seed: int) -> torch.Tensor:
            head = Head(HIDDEN, seed=seed)
            train_head(
                FakeEncoder(),
                head,
                examples,
                device=torch.device("cpu"),
                config=ProbeConfig(epochs=3, batch_size=4, seed=seed),
            )
            return head.linear.weight.detach().clone()

        assert torch.equal(fit(0), fit(0))
        assert not torch.equal(fit(0), fit(1))

    def test_the_encoder_receives_no_gradients(self) -> None:
        """The probe measures the representation, not a fine-tune of it."""
        encoder = FakeEncoder()
        examples, _ = examples_for([sentence(word(1, "axxam", "NOUN"))] * 4, FakeTokenizer())
        head = Head(HIDDEN)
        train_head(
            encoder,
            head,
            examples,
            device=torch.device("cpu"),
            config=ProbeConfig(epochs=1, batch_size=2),
        )
        assert all(p.grad is not None for p in head.parameters())


class TestProbeTagger:
    @staticmethod
    def fitted() -> ProbeTagger:
        sentences = [sentence(word(1, "axxam", "NOUN"), word(2, "d", "AUX"))] * 4
        return ProbeTagger.fit(
            FakeEncoder(),
            FakeTokenizer(),
            sentences,
            device=torch.device("cpu"),
            step=4500,
            config=ProbeConfig(epochs=1, batch_size=2),
        )

    def test_one_label_per_input_word(self) -> None:
        tagged = self.fitted().tag([["axxam", "d", "ameqqran"], ["yiwen"]])
        assert [len(row) for row in tagged] == [3, 1]

    def test_every_label_is_a_universal_tag(self) -> None:
        tagged = self.fitted().tag([["axxam", "d"]])
        assert all(label in UPOS_TAGS for row in tagged for label in row)

    def test_an_empty_sentence_yields_an_empty_row(self) -> None:
        assert self.fitted().tag([[]]) == [[]]

    def test_the_revision_names_the_checkpoint_step(self) -> None:
        """Otherwise two runs of different length report under the same identity."""
        assert self.fitted().revision == "step=4500"

    def test_the_name_carries_no_model_nickname(self) -> None:
        assert self.fitted().name == "encoder-probe"

    def test_fitting_on_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no alignable training sentences"):
            ProbeTagger.fit(
                FakeEncoder(),
                FakeTokenizer(),
                [],
                device=torch.device("cpu"),
                step=1,
            )
