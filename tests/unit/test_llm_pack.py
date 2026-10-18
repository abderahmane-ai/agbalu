"""The training stream: eos framing, the pass schedule, block writing and its conservation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agbalu.llm import pack

EOS = 7


def encoded(sizes: list[int], kinds: list[int]) -> pack.Encoded:
    parts = []
    token = 100
    for size, kind in zip(sizes, kinds, strict=True):
        flat = np.arange(token, token + size, dtype=np.uint32)
        token += size
        parts.append((flat, np.array([size], dtype=np.int64), kind))
    return pack.assemble(parts)


class TestPrefixEos:
    def test_every_document_gains_one_leading_eos(self) -> None:
        flat = np.array([1, 2, 3, 4], dtype=np.uint32)
        out, sizes = pack.prefix_eos(flat, np.array([1, 3]), EOS)
        assert out.tolist() == [EOS, 1, EOS, 2, 3, 4]
        assert sizes.tolist() == [2, 4]

    def test_an_eos_inside_a_document_is_refused(self) -> None:
        with pytest.raises(pack.PackError, match="eos tokens inside"):
            pack.prefix_eos(np.array([1, EOS], dtype=np.uint32), np.array([2]), EOS)

    def test_lengths_that_do_not_cover_the_tokens_are_refused(self) -> None:
        with pytest.raises(pack.PackError):
            pack.prefix_eos(np.array([1, 2], dtype=np.uint32), np.array([1]), EOS)


class TestSchedule:
    def test_each_kind_appears_its_passes_times(self) -> None:
        kind_of = np.array([0, 0, 1, 2], dtype=np.uint8)
        order = pack.schedule(kind_of, [4, 2, 1], seed=1)
        counts = np.bincount(order, minlength=4).tolist()
        assert counts == [4, 4, 2, 1]

    def test_the_same_seed_gives_the_same_order(self) -> None:
        kind_of = np.array([0, 1, 1, 2, 0], dtype=np.uint8)
        assert (
            pack.schedule(kind_of, [2, 2, 1], 5).tolist()
            == pack.schedule(kind_of, [2, 2, 1], 5).tolist()
        )


class TestGather:
    def test_documents_come_out_in_the_requested_order(self) -> None:
        stream = encoded([2, 3], [0, 0])
        out = pack.gather(stream.tokens, stream.offsets, stream.lengths, np.array([1, 0, 1]))
        assert out.tolist() == [102, 103, 104, 100, 101, 102, 103, 104]


class TestWriteBlocks:
    def test_blocks_hold_the_stream_and_the_tail_is_counted(self, tmp_path: Path) -> None:
        stream = encoded([5, 4, 3], [0, 1, 2])
        order = np.array([2, 0, 1], dtype=np.int64)
        written = pack.write_blocks(stream, order, tmp_path / "b.npy", length=4)
        blocks = np.load(tmp_path / "b.npy")
        assert blocks.shape == (3, 4)
        assert blocks.reshape(-1).tolist() == [
            109,
            110,
            111,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
        ]
        assert written.tail == 0
        assert not (tmp_path / "b.npy.partial").exists()

    def test_a_stream_shorter_than_a_block_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(pack.PackError, match="short of one"):
            pack.write_blocks(encoded([2], [0]), np.array([0]), tmp_path / "b.npy", length=4)

    def test_the_summary_accounts_for_every_pass(self, tmp_path: Path) -> None:
        stream = encoded([4, 4, 4], [0, 1, 2])
        order = pack.schedule(stream.kind_of, [2, 1, 1], seed=3)
        written = pack.write_blocks(stream, order, tmp_path / "b.npy", length=4)
        payload = pack.summary(stream, order, written, ["k", "a", "r"], [2, 1, 1], ["s", "t", "u"])
        assert payload["stream_tokens"] == 16
        written_tokens, tail = payload["written_tokens"], payload["dropped_tail_tokens"]
        assert isinstance(written_tokens, int)
        assert isinstance(tail, int)
        assert written_tokens + tail == 16
        assert json.dumps(payload)

    def test_the_summary_fires_when_passes_disagree_with_the_order(self, tmp_path: Path) -> None:
        stream = encoded([4, 4], [0, 1])
        order = pack.schedule(stream.kind_of, [1, 1], seed=3)
        written = pack.write_blocks(stream, order, tmp_path / "b.npy", length=4)
        with pytest.raises(pack.PackError, match="passes predict"):
            pack.summary(stream, order, written, ["k", "a"], [2, 1], ["s", "t"])


class TestReadTexts:
    def test_rows_without_a_source_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        path.write_text('{"text": "azul"}\n', encoding="utf-8")
        with pytest.raises(pack.PackError, match=":1"):
            pack.read_texts(path)

    def test_kinds_carry_the_recipe_passes(self, tmp_path: Path) -> None:
        names = [k.name for k in pack.kinds(tmp_path, tmp_path, tmp_path)]
        assert names == ["kabyle", "aligned", "replay"]
