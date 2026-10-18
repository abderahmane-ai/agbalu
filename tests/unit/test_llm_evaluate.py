"""The evaluation suite: likelihood accounting, prompts, topic scoring and the exit rule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from agbalu.llm import evaluate
from agbalu.llm.prompting import Shot
from tests.unit import tiny_lm

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def scorer(tmp_path_factory: pytest.TempPathFactory) -> evaluate.Scorer:
    model = tiny_lm.load(tiny_lm.save(tmp_path_factory.mktemp("tiny")), torch.float32)
    model.eval()
    return evaluate.Scorer.of(model, tiny_lm.EOS, CPU)


def alone(scorer: evaluate.Scorer, ids: list[int]) -> float:
    """One document's summed nll through a plain unpacked forward."""
    tokens = torch.tensor([[tiny_lm.EOS, *ids]])
    positions = torch.arange(tokens.shape[1])[None]
    bounds = torch.tensor([0, tokens.shape[1]], dtype=torch.int32)
    with torch.inference_mode():
        hidden = scorer.backbone(
            input_ids=tokens,
            position_ids=positions,
            cu_seq_lens_q=bounds,
            cu_seq_lens_k=bounds,
            max_length_q=tokens.shape[1],
            max_length_k=tokens.shape[1],
            use_cache=False,
        ).last_hidden_state[0]
        logits = (hidden[:-1] @ scorer.classifier.T).float()
        return float(torch.nn.functional.cross_entropy(logits, tokens[0, 1:], reduction="sum"))


class TestHeldOut:
    def test_the_sum_is_each_document_scored_alone(self, scorer: evaluate.Scorer) -> None:
        texts = ["azul fell-awen", "tanemmirt", "ass n tafat"]
        tokenizer = tiny_lm.CharTokenizer()
        scored = evaluate.held_out(scorer, tokenizer, texts, language="kab", capacity=16)
        expected = sum(alone(scorer, ids) for ids in tokenizer.encode_batch(texts))
        assert scored.nll == pytest.approx(expected, rel=1e-5)
        assert scored.tokens == sum(len(t) for t in texts)
        assert scored.characters == sum(len(t) for t in texts)
        assert scored.bits_per_character == pytest.approx(
            expected / (math.log(2) * scored.characters), rel=1e-5
        )
        assert scored.truncated == 0

    def test_a_document_past_capacity_is_counted_as_truncated(
        self, scorer: evaluate.Scorer
    ) -> None:
        scored = evaluate.held_out(
            scorer, tiny_lm.CharTokenizer(), ["x" * 20], language="kab", capacity=8
        )
        assert scored.truncated == 1


class TestContinuations:
    def test_matches_the_log_softmax_of_a_plain_forward(self, scorer: evaluate.Scorer) -> None:
        choice = evaluate.Choice((5, 6, 7), (8, 9))
        got = evaluate.continuation_logprobs(scorer, [choice], capacity=8)
        whole = alone(scorer, [5, 6, 7, 8, 9])
        context = alone(scorer, [5, 6, 7])
        assert got[0] == pytest.approx(-(whole - context), abs=1e-4)

    def test_results_come_back_in_input_order(self, scorer: evaluate.Scorer) -> None:
        choices = [evaluate.Choice((5,), (8,)), evaluate.Choice((6, 7), (9, 10))]
        together = evaluate.continuation_logprobs(scorer, choices, capacity=5)
        apart = [evaluate.continuation_logprobs(scorer, [c], capacity=5)[0] for c in choices]
        assert together == pytest.approx(apart, abs=1e-5)

    def test_an_empty_continuation_is_refused(self, scorer: evaluate.Scorer) -> None:
        with pytest.raises(evaluate.EvaluationError):
            evaluate.continuation_logprobs(scorer, [evaluate.Choice((5,), ())])


class FakeGenerator:
    """Returns the prompt followed by one token, recording what it was given."""

    def __init__(self) -> None:
        self.batches: list[torch.Tensor] = []

    def generate(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object
    ) -> torch.Tensor:
        self.batches.append(input_ids)
        width = input_ids.shape[1]
        tail = attention_mask.sum(dim=1, keepdim=True) % 26 + 1
        return torch.cat([input_ids, tail], dim=1)[:, : width + 1]


class TestGreedy:
    def test_prompts_are_left_padded_and_answers_return_in_order(self) -> None:
        fake = FakeGenerator()
        tokenizer = tiny_lm.CharTokenizer()
        prompts = ["ab", "abcd", "a"]
        out = evaluate.greedy(fake, tokenizer, prompts, [4, 4, 4], device=CPU, batch_size=3)
        batch = fake.batches[0]
        assert batch[:, 0].tolist() == [tiny_lm.EOS] * 3
        assert batch.shape[1] == 5
        assert out == [
            tokenizer.decode([3 % 26 + 1]),
            tokenizer.decode([5 % 26 + 1]),
            tokenizer.decode([2 % 26 + 1]),
        ]

    def test_budgets_must_match_prompts(self) -> None:
        with pytest.raises(evaluate.EvaluationError):
            evaluate.greedy(FakeGenerator(), tiny_lm.CharTokenizer(), ["a"], [], device=CPU)


class TestPrompts:
    def test_translation_uses_the_training_tags(self) -> None:
        prompt = evaluate.translation_prompt([Shot("hi", "azul")], "eng_Latn", "kab_Latn", "yes")
        assert prompt == "eng_Latn: hi\nkab_Latn: azul\n\neng_Latn: yes\nkab_Latn:"

    def test_only_the_first_line_is_the_answer(self) -> None:
        assert evaluate.first_line(" Azul.\neng_Latn: next") == "Azul."


def topics(root: Path, split: str, labels: list[int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    names = ["entertainment", "geography", "health"]
    metadata = {
        b"huggingface": json.dumps(
            {"info": {"features": {"label": {"names": names, "_type": "ClassLabel"}}}}
        ).encode()
    }
    table = pa.table({"label": labels, "text": [f"t{i}" for i in range(len(labels))]})
    pq.write_table(
        table.replace_schema_metadata(metadata), root / f"{split}-00000-of-00001.parquet"
    )


class TestTopics:
    def test_label_names_come_from_the_file(self, tmp_path: Path) -> None:
        topics(tmp_path, "test", [0, 2, 1])
        found = evaluate.read_topics("test", tmp_path)
        assert found.names == ("entertainment", "geography", "health")
        assert found.labels == (0, 2, 1)

    def test_a_label_outside_the_names_is_refused(self, tmp_path: Path) -> None:
        topics(tmp_path, "test", [0, 3])
        with pytest.raises(evaluate.EvaluationError):
            evaluate.read_topics("test", tmp_path)

    def test_one_shot_per_label_reproducibly(self, tmp_path: Path) -> None:
        topics(tmp_path, "train", [0, 0, 1, 2, 2, 1])
        found = evaluate.read_topics("train", tmp_path)
        first = evaluate.topic_shots(found, seed=3)
        assert sorted(name for _, name in first) == sorted(found.names)
        assert first == evaluate.topic_shots(found, seed=3)


def report(
    bpc: float, chrf: float, scores: dict[str, float], limit: int | None = None
) -> dict[str, object]:
    rows = {name: {"score": value} for name, value in scores.items()}
    return {
        "limit": limit,
        "heldout": {"kab": {"bits_per_character": bpc}},
        "mt": {"eng-kab": {"chrf++": chrf}},
        "retention": {"tasks": rows, "mean_points": evaluate.retention_mean(rows)},
    }


class TestExitCriterion:
    def test_all_three_rules_must_pass(self) -> None:
        base = report(1.5, 20.0, {"a": 0.60, "b": 0.50})
        better = report(1.2, 30.0, {"a": 0.59, "b": 0.49})
        assert evaluate.exit_criterion(better, base)["adopted"] is True

    def test_a_retention_drop_over_two_points_rejects(self) -> None:
        base = report(1.5, 20.0, {"a": 0.60, "b": 0.50})
        worse = report(1.2, 30.0, {"a": 0.57, "b": 0.48})
        verdict = evaluate.exit_criterion(worse, base)
        assert verdict["adopted"] is False

    def test_a_limited_report_decides_nothing(self) -> None:
        base = report(1.5, 20.0, {"a": 0.6})
        with pytest.raises(evaluate.EvaluationError, match="limited"):
            evaluate.exit_criterion(report(1.2, 30.0, {"a": 0.6}, limit=16), base)

    def test_different_task_sets_are_refused(self) -> None:
        with pytest.raises(evaluate.EvaluationError, match="differ"):
            evaluate.exit_criterion(report(1, 1, {"a": 0.5}), report(1, 1, {"b": 0.5}))


class TestRetention:
    def test_a_missing_metric_names_what_is_there(self) -> None:
        task = evaluate.RetentionTask("mmlu", 5, "acc,none", "eng")
        with pytest.raises(evaluate.EvaluationError, match="acc_norm"):
            evaluate.read_metric({"results": {"mmlu": {"acc_norm,none": 0.4}}}, task)

    def test_the_metric_is_read_as_a_float(self) -> None:
        task = evaluate.RetentionTask("mmlu", 5, "acc,none", "eng")
        assert evaluate.read_metric({"results": {"mmlu": {"acc,none": 0.4}}}, task) == 0.4

    def test_the_mean_is_in_points(self) -> None:
        assert evaluate.retention_mean({"a": {"score": 0.5}, "b": {"score": 0.7}}) == 60.0


class TestFlores:
    def test_the_corrected_kabyle_side_is_in_id_order(self, tmp_path: Path) -> None:
        path = tmp_path / "c.jsonl"
        rows = [
            {"id": 1, "split": "dev", "text": "b"},
            {"id": 0, "split": "dev", "text": "a"},
            {"id": 0, "split": "devtest", "text": "z"},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        assert evaluate.kabyle_corrected("dev", path) == ["a", "b"]
