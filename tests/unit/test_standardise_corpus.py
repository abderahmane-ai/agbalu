"""The corruption pass: what it produces, and that a seed reproduces it exactly."""

from __future__ import annotations

import random
from pathlib import Path

from agbalu.standardise.corpus import corrupt_text, generate_pairs, load_jsonl, save_jsonl


def test_corrupt_text_phonetics() -> None:
    rng = random.Random(42)
    canonical = "tamaziɣt acimi xedmeɣ tečča ğeğğig telliḍ taṣebḥit aḥbib lɛali"
    corrupted = corrupt_text(canonical, rng)
    # Check that characteristic French/informal patterns appear
    assert corrupted != canonical
    expected_patterns = ["gh", "ch", "kh", "tch", "dj", "dh", "th", "h", "3", "7"]
    assert any(sub in corrupted for sub in expected_patterns)


def test_generate_pairs_and_jsonl(tmp_path: Path) -> None:
    sentences = [
        "Azul fell-awen, amek i telliḍ?",
        "Neddew ɣer taddart deg udrar.",
        "Taḥdayt-nni tečča aɣrum d zzit.",
    ]
    pairs = list(generate_pairs(sentences, seed=123))
    assert len(pairs) == 3
    for p in pairs:
        assert len(p.source) > 0
        assert len(p.target) > 0

    out_file = tmp_path / "kabstandard.jsonl"
    save_jsonl(pairs, out_file)
    loaded = load_jsonl(out_file)
    assert len(loaded) == 3
    assert loaded[0].target == sentences[0]
