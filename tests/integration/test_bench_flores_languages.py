"""Every language the MT harness can name has bytes on disk to name it with.

A direction is only scoreable if FLORES+ shipped its source side and `make acquire-flores`
fetched it: `FETCH_PATTERNS` filters that repo down from 485 files, so adding a language to
`LANGUAGE_CODE` without adding its pattern leaves a direction that type-checks, passes every
unit test, and dies on the GPU reading a path that was never downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.acquire.fetch import FETCH_PATTERNS
from agbalu.bench.flores import DEFAULT_ROOT, SPLITS, Split, read_split
from agbalu.bench.mt import LANGUAGE_CODE

pytestmark = pytest.mark.integration

SOURCE_ID = "hf.flores-plus-kab"


def root_or_skip() -> Path:
    if not DEFAULT_ROOT.is_dir():
        pytest.skip("FLORES+ not acquired; run `make acquire-flores`")
    return DEFAULT_ROOT


@pytest.mark.parametrize("language", sorted(set(LANGUAGE_CODE.values())))
@pytest.mark.parametrize("split", SPLITS)
def test_every_named_language_has_both_splits(language: str, split: Split) -> None:
    sentences = read_split(root_or_skip(), split, language)
    assert sentences


def test_every_named_language_is_in_the_fetch_filter() -> None:
    """Checked against the pattern and not only against disk: a file left over from an
    earlier fetch passes the read above while a fresh clone would not fetch it at all."""
    patterns = set(FETCH_PATTERNS[SOURCE_ID])
    missing = [f"*/{code}.jsonl" for code in set(LANGUAGE_CODE.values())]
    assert not [pattern for pattern in missing if pattern not in patterns]


def test_the_splits_are_parallel_across_every_language() -> None:
    """Sources and references are paired by position, so a language whose split is short
    would silently misalign every pair after the gap."""
    root = root_or_skip()
    for split in SPLITS:
        counts = {code: len(read_split(root, split, code)) for code in set(LANGUAGE_CODE.values())}
        assert len(set(counts.values())) == 1, counts
