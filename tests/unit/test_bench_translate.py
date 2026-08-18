"""Target-language selection for public MT systems (task 7.7).

The failure this guards is silent: `convert_tokens_to_ids` returns `unk_token_id` for an
unknown language code, and a missing `>>tgt<<` marker makes a multi-target model pick a
language for itself. Either way the run completes and scores a fluent wrong language.
"""

from __future__ import annotations

import pytest

from agbalu.bench.mt import DIRECTIONS, Direction
from agbalu.bench.translate import (
    BASELINES,
    NLLB_CODE,
    ModelSpec,
    TranslationError,
    batches,
    generate,
    source_text,
    sources_for,
    target_token_id,
    tokenizer_options,
)

NLLB = ModelSpec("fake/nllb", "nllb", DIRECTIONS)
OPUS = ModelSpec("fake/opus", "opus-prefix", ("kab-eng", "eng-kab"))

KAB_LATN_ID = 256_080
"""Read from `facebook/nllb-200-distilled-600M` on 2026-08-08."""


class FakeTokenizer:
    """Resolves only the codes it is given; everything else is `unk`, as NLLB's does."""

    def __init__(self, known: dict[str, int], unk: int = 3) -> None:
        self.known = known
        self._unk = unk
        self.unk_token_id: int | None = unk

    def convert_tokens_to_ids(self, tokens: str) -> int:
        return self.known.get(tokens, self._unk)


class TestSourceMarker:
    @pytest.mark.parametrize(
        ("direction", "marker"),
        [("kab-eng", ">>eng<<"), ("eng-kab", ">>kab<<")],
    )
    def test_opus_models_get_a_target_marker(self, direction: Direction, marker: str) -> None:
        assert source_text(OPUS, direction, "Azul").startswith(f"{marker} ")

    def test_the_marker_names_the_target_not_the_source(self) -> None:
        """Prefixing the source language is the natural mistake and it silently produces
        a copy of the input rather than a translation."""
        assert source_text(OPUS, "eng-kab", "Hello") == ">>kab<< Hello"

    def test_nllb_source_is_left_alone(self) -> None:
        assert source_text(NLLB, "eng-kab", "Hello") == "Hello"

    def test_the_sentence_itself_survives_prefixing(self) -> None:
        text = "Iselman d aɣbalu axatar i wučči n yemdanen."
        assert source_text(OPUS, "kab-eng", text).endswith(text)


class TestTokenizerOptions:
    def test_nllb_pins_the_source_language(self) -> None:
        assert tokenizer_options(NLLB, "eng-kab") == {"src_lang": "eng_Latn"}
        assert tokenizer_options(NLLB, "kab-fra") == {"src_lang": "kab_Latn"}

    def test_opus_models_take_no_src_lang(self) -> None:
        """`MarianTokenizer` has no such attribute; passing one would raise."""
        assert tokenizer_options(OPUS, "eng-kab") == {}

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_every_direction_resolves_for_nllb(self, direction: Direction) -> None:
        assert tokenizer_options(NLLB, direction)["src_lang"] in NLLB_CODE.values()


class TestTargetTokenId:
    def test_a_known_code_resolves(self) -> None:
        tokenizer = FakeTokenizer({"kab_Latn": KAB_LATN_ID})
        assert target_token_id(NLLB, "eng-kab", tokenizer) == KAB_LATN_ID

    def test_an_unresolvable_code_raises_rather_than_returning_unk(self) -> None:
        """`unk` would generate fluent text in whatever language the decoder defaults to."""
        with pytest.raises(TranslationError, match="kab_Latn"):
            target_token_id(NLLB, "eng-kab", FakeTokenizer({}))

    def test_a_code_that_splits_into_several_tokens_raises(self) -> None:
        """The backends return a list when a string is not a single vocabulary entry, and a
        list is not usable as `forced_bos_token_id`."""

        class Splitting:
            unk_token_id: int | None = 3

            def convert_tokens_to_ids(self, tokens: str) -> int | list[int]:
                return [10, 11]

        with pytest.raises(TranslationError, match="no single token"):
            target_token_id(NLLB, "eng-kab", Splitting())

    def test_opus_models_force_nothing(self) -> None:
        assert target_token_id(OPUS, "eng-kab", FakeTokenizer({})) is None


class TestDirectionGuard:
    def test_an_undeclared_direction_raises(self) -> None:
        with pytest.raises(TranslationError, match="does not declare"):
            sources_for(OPUS, "kab-fra", ["Azul"])

    def test_a_declared_direction_prepares_every_source(self) -> None:
        assert len(sources_for(OPUS, "kab-eng", ["a", "b", "c"])) == 3


class TestBaselineRegistry:
    def test_every_direction_is_covered_by_some_baseline(self) -> None:
        covered = {d for spec in BASELINES for d in spec.directions}
        assert covered == set(DIRECTIONS)

    def test_no_baseline_claims_a_direction_twice_in_its_own_list(self) -> None:
        for spec in BASELINES:
            assert len(set(spec.directions)) == len(spec.directions), spec.repo

    def test_each_baseline_declares_at_least_one_direction(self) -> None:
        assert all(spec.directions for spec in BASELINES)


class TestBatching:
    def test_batches_cover_the_input_exactly_once(self) -> None:
        items = [str(i) for i in range(37)]
        flat = [x for batch in batches(items, 16) for x in batch]
        assert flat == items

    @pytest.mark.parametrize("size", [1, 2, 16, 100])
    def test_no_batch_exceeds_the_size(self, size: int) -> None:
        assert all(len(b) <= size for b in batches([str(i) for i in range(37)], size))

    def test_an_empty_input_yields_no_batches(self) -> None:
        assert list(batches([], 8)) == []

    def test_a_non_positive_size_raises(self) -> None:
        with pytest.raises(TranslationError, match="must be positive"):
            list(batches(["a"], 0))


class FakeEncoding:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def to(self, device: str) -> dict[str, list[str]]:
        return {"input_ids": self.texts}


class FakeModel:
    """Echoes its inputs so ordering is checkable without a real decoder."""

    def __init__(self) -> None:
        self.forced: list[int | None] = []

    def generate(
        self,
        *,
        forced_bos_token_id: int | None,
        num_beams: int,
        max_length: int,
        **inputs: object,
    ) -> list[str]:
        self.forced.append(forced_bos_token_id)
        ids = inputs["input_ids"]
        assert isinstance(ids, list)
        return [str(x) for x in ids]


class FakeCodec:
    def __init__(self) -> None:
        self.options: list[dict[str, object]] = []

    def __call__(
        self, text: list[str], *, return_tensors: str, padding: bool, truncation: bool
    ) -> FakeEncoding:
        self.options.append(
            {"return_tensors": return_tensors, "padding": padding, "truncation": truncation}
        )
        return FakeEncoding(text)

    def batch_decode(self, sequences: list[str], *, skip_special_tokens: bool) -> list[str]:
        return list(sequences)


class TestGenerationOrder:
    def test_hypotheses_come_back_in_input_order_across_batches(self) -> None:
        """Scoring pairs hypothesis i with reference i, so any reordering silently
        misaligns the whole test set and still produces a plausible number."""
        prepared = [f"s{i}" for i in range(37)]
        model, codec = FakeModel(), FakeCodec()
        assert generate(prepared, model, codec, KAB_LATN_ID, batch_size=8) == prepared

    def test_the_forced_token_reaches_every_batch(self) -> None:
        model, codec = FakeModel(), FakeCodec()
        generate([f"s{i}" for i in range(20)], model, codec, KAB_LATN_ID, batch_size=8)
        assert model.forced == [KAB_LATN_ID] * 3

    def test_every_batch_is_encoded_as_padded_tensors(self) -> None:
        """Without `return_tensors` the tokenizer returns lists and `generate` raises on
        `.shape`. The fake accepted anything, so the harness reached a GPU before failing."""
        model, codec = FakeModel(), FakeCodec()
        generate([f"s{i}" for i in range(20)], model, codec, KAB_LATN_ID, batch_size=8)
        assert codec.options == [{"return_tensors": "pt", "padding": True, "truncation": True}] * 3
