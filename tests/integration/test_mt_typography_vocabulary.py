"""The fold table, checked against the vocabulary it exists for.

A key the tokenizer can already represent would make the fold a silent rewrite of the
document; a value it cannot represent would move the `<unk>` rather than remove it. Neither
is visible in the table itself, and the table was written from a measurement that only this
test repeats.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agbalu.mt.finetune import SMALL_MODEL
from agbalu.mt.typography import FOLD, prepare

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(SMALL_MODEL)


def unknowns(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return sum(1 for identifier in ids if identifier == tokenizer.unk_token_id)


@pytest.mark.parametrize("mark", sorted(FOLD))
def test_every_key_is_unrepresentable(tokenizer: PreTrainedTokenizerBase, mark: str) -> None:
    assert unknowns(tokenizer, mark) > 0


@pytest.mark.parametrize("ascii_form", sorted(set(FOLD.values())))
def test_every_value_is_representable(tokenizer: PreTrainedTokenizerBase, ascii_form: str) -> None:
    assert unknowns(tokenizer, ascii_form) == 0


def test_a_line_of_typeset_dialogue_stops_being_unknown(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    line = "“Tut, tut, child!” said the Duchess—“everything’s got a moral.”"
    assert unknowns(tokenizer, line) > 0
    assert unknowns(tokenizer, prepare(line)) == 0
    assert prepare(line).count('"') == 4
