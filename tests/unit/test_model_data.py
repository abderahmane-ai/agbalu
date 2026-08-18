from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch

from agbalu.model.config import ModelError, TrainConfig
from agbalu.model.data import (
    VALIDATION_SHARE,
    PackedDataset,
    _is_validation,
    iter_batches,
    tokenise_corpus,
)
from agbalu.model.modeling import IGNORE_INDEX
from agbalu.tokenizer.spec import CLS_ID

VOCAB = 512


def config(**overrides: object) -> TrainConfig:
    settings: dict[str, object] = {
        "seq_length": 16,
        "local_batch_size": 4,
        "global_batch_size": 8,
        "max_steps": 10,
    }
    settings.update(overrides)
    return TrainConfig(**settings)  # type: ignore[arg-type]


def stream(tokens: int = 2_000) -> npt.NDArray[np.uint16]:
    return np.random.default_rng(0).integers(5, VOCAB, size=tokens, dtype=np.uint16)


def dataset(tokens: int = 2_000, **overrides: object) -> PackedDataset:
    return PackedDataset(
        stream(tokens), config(**overrides), VOCAB, mask_token_id=4, n_special_tokens=5
    )


class TestValidationSplit:
    def test_the_split_is_a_hash_not_a_slice(self) -> None:
        """The corpus is ordered by source, so any contiguous split is a split by source."""
        sentences = [f"sentence number {i}" for i in range(20_000)]
        held = [s for s in sentences if _is_validation(s, VALIDATION_SHARE)]
        assert 30 < len(held) < 180
        positions = [sentences.index(s) for s in held]
        assert min(positions) < len(sentences) // 4
        assert max(positions) > 3 * len(sentences) // 4

    def test_the_same_sentence_always_lands_in_the_same_split(self) -> None:
        assert _is_validation("azul", 0.5) == _is_validation("azul", 0.5)

    def test_a_zero_share_holds_nothing_out(self) -> None:
        assert not any(_is_validation(f"s{i}", 0.0) for i in range(500))

    def test_a_full_share_holds_everything_out(self) -> None:
        assert all(_is_validation(f"s{i}", 1.0) for i in range(500))


@pytest.mark.integration
class TestTokeniseCorpus:
    def test_writes_both_splits_and_their_stats(self, tmp_path: Path) -> None:
        model = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
        if not model.is_file():
            pytest.skip("tokenizer not built; run `make tokenizer-sweep`")
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            "".join(
                json.dumps({"text": f"Azul fell-awen tikkelt {i}"}, ensure_ascii=False) + "\n"
                for i in range(500)
            ),
            encoding="utf-8",
        )
        result = tokenise_corpus(corpus, model, tmp_path / "out", validation_share=0.2)
        assert result.train.is_file()
        stats = json.loads(result.stats.read_text(encoding="utf-8"))
        assert stats["tokens"]["train"] > 0
        assert stats["sentences"]["train"] + stats["sentences"]["validation"] == 500

    def test_an_empty_corpus_is_refused(self, tmp_path: Path) -> None:
        model = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
        if not model.is_file():
            pytest.skip("tokenizer not built; run `make tokenizer-sweep`")
        corpus = tmp_path / "empty.jsonl"
        corpus.write_text("", encoding="utf-8")
        with pytest.raises(ModelError, match="no training tokens"):
            tokenise_corpus(corpus, model, tmp_path / "out")


class TestPackedDataset:
    def test_refuses_a_stream_shorter_than_one_window(self) -> None:
        with pytest.raises(ModelError, match="shorter than one"):
            PackedDataset(stream(8), config(), VOCAB, mask_token_id=4, n_special_tokens=5)

    def test_every_window_starts_with_cls(self) -> None:
        data = dataset()
        for index in (0, 1, len(data) - 1):
            assert int(data[index][0][0]) == CLS_ID

    def test_windows_are_the_configured_length(self) -> None:
        data = dataset()
        ids, attention, labels = data[0]
        assert ids.numel() == attention.numel() == labels.numel() == 16

    def test_windows_do_not_overlap(self) -> None:
        data = dataset()
        assert len(data) == 2_000 // 15

    def test_masking_follows_the_step_schedule(self) -> None:
        """`set_step` drives the inverse schedule; without it the rate never anneals.

        That is exactly what keeps the validation loss comparable across steps: the trainer
        advances the training dataset only, so the held-out split stays at step 0 and is
        always masked at `mask_p_start`. Advance it and every eval measures an easier task
        than the one before, making the curve improve on its own.
        """
        data = dataset(max_steps=100)
        data.set_step(0)
        early = sum(int((data[i][2] != IGNORE_INDEX).sum()) for i in range(40))
        data.set_step(100)
        late = sum(int((data[i][2] != IGNORE_INDEX).sum()) for i in range(40))
        assert early > late

    def test_access_is_deterministic(self) -> None:
        first, second = dataset(), dataset()
        assert torch.equal(first[5][0], second[5][0])
        assert torch.equal(first[5][2], second[5][2])


class TestBatch:
    def test_a_batch_is_built_in_one_call(self) -> None:
        data = dataset()
        generator = torch.Generator().manual_seed(1)
        ids, attention, labels = data.batch(torch.tensor([0, 1, 2]), generator)
        width = data.config.seq_length
        assert ids.shape == attention.shape == labels.shape == (3, width)
        assert bool(attention.all())

    def test_windows_start_with_the_class_token(self) -> None:
        windows = dataset().windows(torch.tensor([0, 3, 7]))
        assert bool((windows[:, 0] == CLS_ID).all())

    def test_windows_read_the_stream_in_order(self) -> None:
        data = dataset()
        windows = data.windows(torch.tensor([2]))
        start = 2 * data.window
        expected = data.tokens[start : start + data.window].astype("int64")
        assert torch.equal(windows[0, 1:], torch.from_numpy(expected))

    def test_a_batch_matches_the_single_item_path(self) -> None:
        data = dataset()
        generator = torch.Generator().manual_seed(7)
        batched = data.batch(torch.tensor([4, 9]), generator)
        assert batched[0].shape[0] == 2
        assert torch.equal(batched[1], torch.ones_like(batched[1], dtype=torch.bool))


class TestIterBatches:
    def test_rejects_a_non_positive_batch_size(self) -> None:
        with pytest.raises(ModelError, match="batch_size"):
            list(iter_batches(dataset(), 0, seed=1, epoch=0))

    def test_drops_the_ragged_tail_when_asked(self) -> None:
        data = dataset()
        assert len(list(iter_batches(data, 7, seed=1, epoch=0))) == len(data) // 7

    def test_keeps_the_tail_when_not_dropping(self) -> None:
        """A validation set smaller than one batch must still yield it. Dropping the tail
        there gives no batches at all, which reads downstream as an empty split."""
        data = dataset()
        kept = list(iter_batches(data, 7, seed=1, epoch=0, drop_last=False))
        assert len(kept) == -(-len(data) // 7)

    def test_a_set_smaller_than_one_batch_still_yields_a_batch(self) -> None:
        data = dataset(1_000)
        huge = len(data) * 10
        assert list(iter_batches(data, huge, seed=1, epoch=0)) == []
        assert len(list(iter_batches(data, huge, seed=1, epoch=0, drop_last=False))) == 1

    def test_resuming_skips_the_batches_already_seen(self) -> None:
        data = dataset()
        whole = list(iter_batches(data, 4, seed=1, epoch=0))
        resumed = list(iter_batches(data, 4, seed=1, epoch=0, start_batch=2))
        assert len(resumed) == len(whole) - 2
        assert torch.equal(resumed[0][0], whole[2][0])

    def test_each_epoch_shuffles_differently(self) -> None:
        data = dataset()
        first = next(iter(iter_batches(data, 4, seed=1, epoch=0)))
        second = next(iter(iter_batches(data, 4, seed=1, epoch=1)))
        assert not torch.equal(first[0], second[0])
