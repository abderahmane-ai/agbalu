"""Public MT systems under the two-condition protocol (task 7.7).

Two target-language conventions, both read off the model cards on 2026-08-08 rather than
recalled. NLLB tags the source through the tokenizer's `src_lang` and forces the target's
language id as the first generated token. The OPUS-MT Tatoeba-Bible multi-target models
take a sentence-initial `>>tgt<<` marker on the source text and have no `src_lang` at all —
`MarianTokenizer` does not carry the attribute, so the source language is chosen by
constructing the tokenizer, never by mutating it.

`transformers` removed `NllbTokenizer.lang_code_to_id` before 5.12. The id now comes from
`convert_tokens_to_ids`, which returns `unk_token_id` for an unknown code rather than
raising — an unchecked lookup yields fluent output in the wrong language.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from agbalu.bench.mt import Direction, source_language, target_language

Family = Literal["nllb", "opus-prefix"]

NLLB_CODE: Final[dict[str, str]] = {
    "kab": "kab_Latn",
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    "arb": "arb_Arab",
    "spa": "spa_Latn",
    "deu": "deu_Latn",
}
OPUS_CODE: Final[dict[str, str]] = {"kab": "kab", "eng": "eng", "fra": "fra"}

BATCH_SIZE: Final = 16
NUM_BEAMS: Final = 4
MAX_TARGET_LENGTH: Final = 256
"""FLORES+ rows are single sentences; 256 is slack, not a target length.

Passed as `max_length` rather than `max_new_tokens`: NLLB's shipped generation config sets
`max_length=200`, and transformers warns on every batch when both are present. For an
encoder-decoder the two bound the same thing, so this is the quieter spelling of one cap."""


class TranslationError(Exception):
    """A model cannot be driven for the requested direction."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    repo: str
    family: Family
    directions: tuple[Direction, ...]
    """Only the directions the card declares. A multi-target model emits some other
    language for a code it does not hold, so the set is enumerated, never inferred."""

    def supports(self, direction: Direction) -> bool:
        return direction in self.directions


NLLB_DIRECTIONS: Final[tuple[Direction, ...]] = (
    "kab-eng",
    "eng-kab",
    "kab-fra",
    "fra-kab",
    "arb-kab",
    "spa-kab",
    "deu-kab",
)
"""Enumerated rather than taken from `DIRECTIONS`: a spec states what its card declares,
and inheriting the harness's list would make a future direction a silent claim."""

BASELINES: Final[tuple[ModelSpec, ...]] = (
    ModelSpec("facebook/nllb-200-distilled-600M", "nllb", NLLB_DIRECTIONS),
    ModelSpec("facebook/nllb-200-distilled-1.3B", "nllb", NLLB_DIRECTIONS),
    ModelSpec(
        "Helsinki-NLP/opus-mt-tc-bible-big-afa-deu_eng_fra_por_spa",
        "opus-prefix",
        ("kab-eng", "kab-fra"),
    ),
    ModelSpec(
        "Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-afa",
        "opus-prefix",
        ("eng-kab", "fra-kab"),
    ),
)


def _code(table: Mapping[str, str], language: str) -> str:
    if language not in table:
        msg = f"no language code for {language!r}; have {sorted(table)}"
        raise TranslationError(msg)
    return table[language]


def source_text(spec: ModelSpec, direction: Direction, text: str) -> str:
    """The source as the model expects it, target marker included where one is required."""
    if spec.family == "opus-prefix":
        return f">>{_code(OPUS_CODE, target_language(direction))}<< {text}"
    return text


def tokenizer_options(spec: ModelSpec, direction: Direction) -> dict[str, str]:
    """`from_pretrained` keywords that fix the source language, empty where the marker
    carries it. Applied at construction because `MarianTokenizer` has no `src_lang`."""
    if spec.family == "nllb":
        return {"src_lang": _code(NLLB_CODE, source_language(direction))}
    return {}


class Tokenizer(Protocol):
    """The slice of the `transformers` tokenizer API this module drives.

    `convert_tokens_to_ids` is declared `int | list[int]` because that is what the backends
    return — one id per token, and a list when handed a list.
    """

    unk_token_id: int | None

    def convert_tokens_to_ids(self, tokens: str) -> int | list[int]: ...


def target_token_id(spec: ModelSpec, direction: Direction, tokenizer: Tokenizer) -> int | None:
    """The forced first generated token for NLLB, or None where the marker carries it."""
    if spec.family != "nllb":
        return None
    code = _code(NLLB_CODE, target_language(direction))
    token_id = tokenizer.convert_tokens_to_ids(code)
    if not isinstance(token_id, int) or token_id == tokenizer.unk_token_id:
        msg = f"{spec.repo} has no single token for {code!r}; it cannot target {direction}"
        raise TranslationError(msg)
    return token_id


def batches(items: Sequence[str], size: int = BATCH_SIZE) -> Iterator[Sequence[str]]:
    if size < 1:
        msg = f"batch size must be positive, got {size}"
        raise TranslationError(msg)
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Encoding[T](Protocol):
    def to(self, device: str) -> Mapping[str, T]: ...


class Generator[T](Protocol):
    """Generic in the batch representation so the loop is exercisable without torch."""

    def generate(
        self,
        *,
        forced_bos_token_id: int | None,
        num_beams: int,
        max_length: int,
        no_repeat_ngram_size: int | None,
        repetition_penalty: float | None,
        **inputs: T,
    ) -> T: ...


class Decoder[T](Protocol):
    def __call__(
        self, text: list[str], *, return_tensors: str, padding: bool, truncation: bool
    ) -> Encoding[T]: ...

    def batch_decode(self, sequences: T, *, skip_special_tokens: bool) -> list[str]: ...


class Remapper[T](Protocol):
    """`agbalu.mt.vocab.VocabTables`, structurally, so the loop stays torch-free to test."""

    def to_new(self, ids: T) -> T: ...

    def to_old(self, ids: T) -> T: ...

    def new_id(self, old: int) -> int: ...


def sources_for(spec: ModelSpec, direction: Direction, sources: Sequence[str]) -> list[str]:
    """Every source prepared for this model and direction, in input order."""
    if not spec.supports(direction):
        msg = f"{spec.repo} does not declare {direction}; it has {list(spec.directions)}"
        raise TranslationError(msg)
    return [source_text(spec, direction, text) for text in sources]


def generate[T](
    prepared: Sequence[str],
    model: Generator[T],
    encode: Decoder[T],
    forced: int | None,
    *,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
    tables: Remapper[T] | None = None,
    max_length: int = MAX_TARGET_LENGTH,
    no_repeat_ngram_size: int | None = None,
    repetition_penalty: float | None = None,
) -> list[str]:
    """Hypotheses in input order.

    Order is load-bearing: scoring pairs hypothesis *i* with reference *i*, so batches are
    never sorted by length however much that would help throughput.

    `tables` is required for a vocabulary-trimmed model and must be absent otherwise. The
    tokenizer speaks the full vocabulary and the model speaks the trimmed one, so the
    input ids, the forced language token and the generated ids each cross the boundary.

    The three decoding arguments default to what every published number in this project was
    measured with, and `None` is what transformers checks for before adding either penalty
    to the processor list. A caller that changes them is measuring something else, so only
    the document path does.
    """
    import torch

    hypotheses: list[str] = []
    for batch in batches(prepared, batch_size):
        # `return_tensors` is not optional: without it the tokenizer hands back lists and
        # `generate` reads `.shape` off one. Padding is what makes a ragged batch a tensor.
        encoded = encode(list(batch), return_tensors="pt", padding=True, truncation=True).to(device)
        inputs = dict(encoded)
        target = forced
        if tables is not None:
            inputs["input_ids"] = tables.to_new(inputs["input_ids"])
            target = None if forced is None else tables.new_id(forced)
        with torch.inference_mode():
            produced = model.generate(
                **inputs,
                forced_bos_token_id=target,
                num_beams=NUM_BEAMS,
                max_length=max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
            )
        if tables is not None:
            produced = tables.to_old(produced)
        hypotheses.extend(encode.batch_decode(produced, skip_special_tokens=True))

    if len(hypotheses) != len(prepared):
        msg = f"{len(hypotheses)} hypotheses from {len(prepared)} sources"
        raise TranslationError(msg)
    return hypotheses
