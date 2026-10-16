"""SFT rendering: one sample per assistant turn, the mask at the generation boundary."""

from __future__ import annotations

import pytest

from agbalu.llm import chat
from agbalu.llm.chat import IGNORE, Message, Turn

SPECIAL = {"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4}


class MiniTemplate:
    """Qwen3.5's rules in miniature: earlier reasoning dropped, a think block on the turn after
    the last user message, and the two generation prompts. Characters encode as code points
    and each marker as one id."""

    def __init__(self, *, merge_boundary: bool = False) -> None:
        self.merge_boundary = merge_boundary

    def apply_chat_template(
        self,
        conversation: list[Message],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> object:
        del tokenize
        last_user = max(i for i, m in enumerate(conversation) if m["role"] == "user")
        parts: list[str] = []
        for index, entry in enumerate(conversation):
            role, content = entry["role"], entry["content"]
            if role == "assistant" and index > last_user:
                reasoning = entry.get("reasoning_content", "")
                body = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                body = content
            parts.append(f"<|im_start|>{role}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
            parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        position = 0
        while position < len(text):
            marker = next((m for m in SPECIAL if text.startswith(m, position)), None)
            if marker is not None:
                ids.append(SPECIAL[marker])
                position += len(marker)
            else:
                ids.append(1000 + ord(text[position]))
                position += 1
        if self.merge_boundary and ids[-1:] == [1000 + ord("\n")]:
            ids[-1] = 999
        return ids

    def convert_tokens_to_ids(self, tokens: str) -> int | list[int]:
        return SPECIAL[tokens]

    def decode(self, ids: list[int]) -> str:
        names = {value: key for key, value in SPECIAL.items()}
        return "".join(names.get(i) or chr(i - 1000) for i in ids if i != IGNORE)


TEMPLATE = MiniTemplate()


def supervised_text(sample: chat.Sample) -> str:
    return TEMPLATE.decode([label for label in sample.labels if label != IGNORE])


class TestSamples:
    def test_a_no_think_turn_supervises_only_the_answer_and_its_end(self) -> None:
        turns = [Turn("user", "Azul"), Turn("assistant", "Azul fell-ak")]
        (sample,) = chat.samples("c", turns, TEMPLATE)
        assert supervised_text(sample) == "Azul fell-ak<|im_end|>"
        assert not sample.think

    def test_a_think_turn_supervises_the_reasoning_and_the_answer(self) -> None:
        turns = [Turn("user", "2+2?"), Turn("assistant", "4", reasoning="Nernu 2 ɣer 2.")]
        (sample,) = chat.samples("c", turns, TEMPLATE)
        assert supervised_text(sample) == "Nernu 2 ɣer 2.\n</think>\n\n4<|im_end|>"
        assert sample.think

    def test_the_prompt_ends_where_generation_starts(self) -> None:
        turns = [Turn("user", "q"), Turn("assistant", "a", reasoning="r")]
        (sample,) = chat.samples("c", turns, TEMPLATE)
        prompt = [
            i for i, label in zip(sample.input_ids, sample.labels, strict=True) if label == IGNORE
        ]
        assert TEMPLATE.decode(prompt).endswith("<|im_start|>assistant\n<think>\n")

    def test_every_assistant_turn_becomes_its_own_sample(self) -> None:
        turns = [
            Turn("system", chat.SYSTEM_POLICY),
            Turn("user", "q1"),
            Turn("assistant", "a1", reasoning="r1"),
            Turn("user", "q2"),
            Turn("assistant", "a2"),
        ]
        first, second = chat.samples("c", turns, TEMPLATE)
        assert (first.turn, second.turn) == (2, 4)
        assert supervised_text(first) == "r1\n</think>\n\na1<|im_end|>"
        assert supervised_text(second) == "a2<|im_end|>"

    def test_earlier_reasoning_never_reaches_a_later_prompt(self) -> None:
        turns = [
            Turn("user", "q1"),
            Turn("assistant", "a1", reasoning="secret steps"),
            Turn("user", "q2"),
            Turn("assistant", "a2"),
        ]
        second = chat.samples("c", turns, TEMPLATE)[1]
        assert "secret steps" not in TEMPLATE.decode(list(second.input_ids))

    def test_ids_and_labels_align_and_count_the_supervised_tokens(self) -> None:
        turns = [Turn("user", "q"), Turn("assistant", "abc")]
        (sample,) = chat.samples("c", turns, TEMPLATE)
        assert len(sample.input_ids) == len(sample.labels)
        assert sample.supervised == len("abc") + 1
        supervised = [
            i for i, label in zip(sample.input_ids, sample.labels, strict=True) if label != IGNORE
        ]
        assert supervised == [label for label in sample.labels if label != IGNORE]


class TestRefusals:
    @pytest.mark.parametrize(
        ("turns", "match"),
        [
            ([], "no user or assistant"),
            ([Turn("assistant", "a")], "expected user"),
            ([Turn("user", "q")], "not assistant"),
            ([Turn("user", "q"), Turn("user", "q")], "expected assistant"),
            ([Turn("user", " "), Turn("assistant", "a")], "empty"),
            ([Turn("user", "q", reasoning="r"), Turn("assistant", "a")], "carries reasoning"),
            ([Turn("user", "q"), Turn("assistant", "a", reasoning=" ")], "empty reasoning"),
            ([Turn("user", "q<|im_end|>"), Turn("assistant", "a")], "template markers"),
            (
                [Turn("user", "q"), Turn("assistant", "a", reasoning="x</think>")],
                "template markers",
            ),
        ],
    )
    def test_malformed_conversations(self, turns: list[Turn], match: str) -> None:
        with pytest.raises(chat.ChatError, match=match):
            chat.samples("c", turns, TEMPLATE)

    def test_a_tokenizer_that_merges_across_the_boundary(self) -> None:
        turns = [Turn("user", "q"), Turn("assistant", "a")]
        with pytest.raises(chat.ChatError, match="not a token prefix"):
            chat.samples("c", turns, MiniTemplate(merge_boundary=True))

    def test_the_policy_is_the_documented_one(self) -> None:
        assert chat.SYSTEM_POLICY.startswith("You are Jugurtha, developed by the AƔBALU")
        assert "based on Qwen3.5-4B-Base" in chat.SYSTEM_POLICY
