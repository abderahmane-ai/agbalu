from __future__ import annotations

from pathlib import Path

import pytest

from agbalu.normalise.rules import ALPHABET
from agbalu.tokenizer.spec import (
    CLS_ID,
    MASK_PIECE,
    MAX_VOCAB_SIZE,
    MIN_VOCAB_SIZE,
    PAD_ID,
    SEP_ID,
    UNK_ID,
    TokenizerError,
    TokenizerSpec,
    required_chars,
)


class TestRequiredChars:
    def test_covers_the_normaliser_alphabet(self) -> None:
        chars = set(required_chars())
        assert set(ALPHABET) <= chars

    def test_includes_legacy_t_cedilla_in_both_cases(self) -> None:
        chars = set(required_chars())
        assert "ţ" in chars
        assert "Ţ" in chars

    def test_includes_the_hyphen_and_apostrophe(self) -> None:
        assert "-" in required_chars()
        assert "'" in required_chars()

    def test_excludes_whitespace_and_the_metaspace_marker(self) -> None:
        chars = set(required_chars())
        assert not any(c.isspace() for c in chars)
        assert "▁" not in chars

    def test_is_sorted_and_free_of_duplicates(self) -> None:
        chars = required_chars()
        assert list(chars) == sorted(chars)
        assert len(set(chars)) == len(chars)


class TestSpecValidation:
    @pytest.mark.parametrize("size", [0, -1, MIN_VOCAB_SIZE - 1, MAX_VOCAB_SIZE + 1, 10**9])
    def test_rejects_vocabulary_sizes_outside_the_band(self, size: int) -> None:
        with pytest.raises(TokenizerError, match="vocab_size"):
            TokenizerSpec(vocab_size=size)

    @pytest.mark.parametrize("size", [MIN_VOCAB_SIZE, 16_000, MAX_VOCAB_SIZE])
    def test_accepts_the_boundaries(self, size: int) -> None:
        assert TokenizerSpec(vocab_size=size).vocab_size == size

    @pytest.mark.parametrize("coverage", [0.0, -0.1, 1.01])
    def test_rejects_impossible_character_coverage(self, coverage: float) -> None:
        with pytest.raises(TokenizerError, match="character_coverage"):
            TokenizerSpec(character_coverage=coverage)

    def test_accepts_full_character_coverage(self) -> None:
        assert TokenizerSpec(character_coverage=1.0).character_coverage == 1.0

    @pytest.mark.parametrize("threads", [0, -4])
    def test_rejects_non_positive_thread_counts(self, threads: int) -> None:
        with pytest.raises(TokenizerError, match="num_threads"):
            TokenizerSpec(num_threads=threads)

    def test_is_frozen(self) -> None:
        spec = TokenizerSpec()
        with pytest.raises(AttributeError):
            spec.vocab_size = 8_000  # type: ignore[misc]


class TestSpecName:
    def test_base_and_seeded_arms_never_collide(self) -> None:
        base = TokenizerSpec(vocab_size=16_000)
        seeded = TokenizerSpec(vocab_size=16_000, seed_file=Path("seed.tsv"))
        assert base.name != seeded.name
        assert not base.seeded
        assert seeded.seeded

    @pytest.mark.parametrize(
        ("size", "expected"),
        [(8_000, "8k"), (16_000, "16k"), (12_500, "12.5k"), (1_000, "1k")],
    )
    def test_renders_sizes_without_trailing_zeroes(self, size: int, expected: str) -> None:
        assert TokenizerSpec(vocab_size=size).name == f"agbalu-tok-base-{expected}"


class TestTrainerKwargs:
    def test_pins_the_special_token_ids(self) -> None:
        kwargs = TokenizerSpec().trainer_kwargs(Path("c.txt"), Path("out/m"))
        assert kwargs["pad_id"] == PAD_ID
        assert kwargs["unk_id"] == UNK_ID
        assert kwargs["bos_id"] == CLS_ID
        assert kwargs["eos_id"] == SEP_ID
        assert kwargs["user_defined_symbols"] == [MASK_PIECE]

    def test_carries_the_settled_build_parameters(self) -> None:
        kwargs = TokenizerSpec().trainer_kwargs(Path("c.txt"), Path("out/m"))
        assert kwargs["model_type"] == "unigram"
        assert kwargs["byte_fallback"] is True
        assert kwargs["split_digits"] is True
        assert kwargs["normalization_rule_name"] == "identity"
        assert kwargs["required_chars"] == required_chars()

    def test_omits_the_seed_file_unless_seeded(self) -> None:
        base = TokenizerSpec().trainer_kwargs(Path("c.txt"), Path("out/m"))
        assert "seed_sentencepieces_file" not in base

    def test_passes_the_seed_file_when_seeded(self) -> None:
        spec = TokenizerSpec(seed_file=Path("pool.tsv"))
        kwargs = spec.trainer_kwargs(Path("c.txt"), Path("out/m"))
        assert kwargs["seed_sentencepieces_file"] == "pool.tsv"
