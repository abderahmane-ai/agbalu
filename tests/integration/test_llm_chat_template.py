"""SFT rendering against Qwen3.5-4B's real tokenizer and template at the pinned revision."""

from __future__ import annotations

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from agbalu.llm import chat
from agbalu.llm.bases import INSTRUCT, REVISIONS
from agbalu.llm.chat import IGNORE, Turn

pytestmark = pytest.mark.integration

THINK_ID, THINK_END_ID, IM_END_ID = 248068, 248069, 248046


def text(value: str | list[str]) -> str:
    assert isinstance(value, str)
    return value


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    path = snapshot_download(
        INSTRUCT,
        revision=REVISIONS[INSTRUCT],
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
    )
    loaded: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(path)
    return loaded


TURNS = [
    Turn("system", chat.SYSTEM_POLICY),
    Turn("user", "Acu i d tamurt n Leqbayel?"),
    Turn(
        "assistant",
        "D tamurt n yidurar.",
        reasoning="Ad d-iniɣ ɣef wakal d yidurar.\nSyin ad d-fkeɣ tiririt.",
    ),
    Turn("user", "What about the language?"),
    Turn("assistant", "Taqbaylit d tutlayt n Leqbayel."),
]


def test_the_markers_are_single_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    ids = tokenizer.convert_tokens_to_ids(["<think>", "</think>", "<|im_end|>"])
    assert ids == [THINK_ID, THINK_END_ID, IM_END_ID]


def test_each_sample_is_the_template_rendering_cut_at_its_end(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    for sample in chat.samples("c", TURNS, tokenizer):
        upto = [chat.message(turn) for turn in TURNS[: sample.turn + 1]]
        text = tokenizer.apply_chat_template(upto, tokenize=False, enable_thinking=sample.think)
        assert isinstance(text, str)
        reference = tokenizer.encode(text, add_special_tokens=False)
        assert list(sample.input_ids) == reference[: len(sample.input_ids)]
        assert sample.input_ids[-1] == IM_END_ID


def test_the_think_sample_supervises_from_inside_the_think_block(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    first = chat.samples("c", TURNS, tokenizer)[0]
    prompt = [i for i, label in zip(first.input_ids, first.labels, strict=True) if label == IGNORE]
    assert prompt[-2:] == [THINK_ID, *tokenizer.encode("\n", add_special_tokens=False)]
    supervised = [label for label in first.labels if label != IGNORE]
    assert THINK_END_ID in supervised
    assert THINK_ID not in supervised
    assert text(tokenizer.decode(supervised)).startswith("Ad d-iniɣ ɣef wakal")


def test_the_no_think_sample_supervises_only_the_answer(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    second = chat.samples("c", TURNS, tokenizer)[1]
    supervised = [label for label in second.labels if label != IGNORE]
    assert THINK_ID not in supervised
    assert THINK_END_ID not in supervised
    assert text(tokenizer.decode(supervised)) == "Taqbaylit d tutlayt n Leqbayel.<|im_end|>"
