"""Chat-mode measurements through lm-eval's vLLM backend, in both thinking modes (task 11.21).

The completion-mode suite (`agbalu.llm.evaluate`) asks a model in the shape CPT trained.
A merged or fine-tuned assistant is also asked as an assistant: through the official chat
template, with `enable_thinking` on and off. Those runs generate thousands of tokens per
question in thinking mode, so they go through vLLM in its own environment
(`/home/jovyan/envs/vllm`), never through `transformers.generate`.

Decoding is fixed for every model compared: greedy with thinking off; with thinking on,
Qwen's own recommendation for precise work (temperature 0.6, top_p 0.95, top_k 20) under one
seed, because greedy decoding in thinking mode loops (Qwen3.5 README, read 2026-10-16).
Large sets are cut by a seeded draw over every example, never by a prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

from agbalu.llm.progress import progress, stage

Mode = Literal["think", "no_think"]
Suite = Literal["select", "release"]
MODES: Final[tuple[Mode, ...]] = ("no_think", "think")

THINK_END: Final = "</think>"
SEED: Final = 20261016
GPU_MEMORY: Final = 0.85
"""vLLM's share of the 47 GB slice; the rest is the CUDA context and the sampler."""

NO_THINK_TOKENS: Final = 2_048
THINK_TOKENS: Final = 12_288
"""A Kabyle reasoning trace costs about three tokens per word; the cap leaves the trace room
to finish on MMLU-Pro with a 5-shot prompt."""

LONG_CONTEXT: Final = 32_768
"""The `niah` length, past every training row (16,384) so it measures what training never saw."""

MAX_MODEL_LENGTH: Final = LONG_CONTEXT + NO_THINK_TOKENS
"""One engine length serves every task: vLLM sizes its cache by `GPU_MEMORY`, not by this."""


class ChatEvaluationError(Exception):
    """A task returned no usable score, or a report cannot be read."""


@dataclass(frozen=True, slots=True)
class HarnessTask:
    name: str
    metric: str
    shots: int
    modes: tuple[Mode, ...]
    suites: tuple[Suite, ...]
    sample: int | None = None
    """Examples drawn per subtask under `SEED`; None scores every example."""

    long_context: bool = False
    code: bool = False


HARNESS: Final = (
    HarnessTask("ifeval", "prompt_level_strict_acc,none", 0, ("no_think",), ("select", "release")),
    HarnessTask("gsm8k", "exact_match,flexible-extract", 5, MODES, ("select", "release")),
    HarnessTask("mmlu_pro", "exact_match,custom-extract", 5, MODES, ("select", "release"), 50),
    HarnessTask(
        "niah_single_1", "score,none", 0, ("no_think",), ("select", "release"), long_context=True
    ),
    HarnessTask(
        "niah_single_2", "score,none", 0, ("no_think",), ("select", "release"), long_context=True
    ),
    HarnessTask(
        "niah_single_3", "score,none", 0, ("no_think",), ("select", "release"), long_context=True
    ),
    HarnessTask("humaneval_plus", "pass@1,create_test", 0, MODES, ("release",), code=True),
    HarnessTask("mbpp_plus", "pass_at_1,none", 3, MODES, ("release",), code=True),
)
"""Task names verified in lm-eval 0.4.13 on the pod, 2026-10-16. Metric keys are confirmed by
the smoke at `limit` 8 before any full run; a missing key fails by name, never as zero."""


def tasks_for(suite: Suite) -> list[HarnessTask]:
    return [task for task in HARNESS if suite in task.suites]


def drawn(size: int, count: int | None, seed: int = SEED) -> list[int] | None:
    """`count` indices drawn from `range(size)`, sorted; every index when `count` is None."""
    if count is None or count >= size:
        return None
    return sorted(random.Random(seed).sample(range(size), count))


def generation(mode: Mode) -> dict[str, object]:
    if mode == "no_think":
        return {"max_gen_toks": NO_THINK_TOKENS, "do_sample": False, "temperature": 0.0}
    return {
        "max_gen_toks": THINK_TOKENS,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }


def read_score(output: object, task: HarnessTask) -> tuple[float, int | None]:
    """The task's metric and its example count from an lm-eval result, or a named error."""
    results = output.get("results") if isinstance(output, dict) else None
    row = results.get(task.name) if isinstance(results, dict) else None
    if not isinstance(row, dict):
        message = f"lm-eval returned no result row for {task.name!r}"
        raise ChatEvaluationError(message)
    value = row.get(task.metric)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        message = f"{task.name}: no finite {task.metric!r}; the row has {sorted(row)}"
        raise ChatEvaluationError(message)
    counts = output.get("n-samples") if isinstance(output, dict) else None
    effective = counts.get(task.name) if isinstance(counts, dict) else None
    size = effective.get("effective") if isinstance(effective, dict) else None
    return float(value), size if isinstance(size, int) else None


class Harness(Protocol):
    """What scoring one task needs; the vLLM-backed one lives in `run`."""

    def score(self, task: HarnessTask, limit: int | None) -> object: ...


def evaluate_tasks(
    harness: Harness, tasks: Sequence[HarnessTask], *, limit: int | None
) -> dict[str, dict[str, object]]:
    """Every task: score in points, metric, examples scored, sample size."""
    rows: dict[str, dict[str, object]] = {}
    for task in tasks:
        value, examples = read_score(harness.score(task, limit), task)
        rows[task.name] = {
            "score": round(100 * value, 4),
            "metric": task.metric,
            "examples": examples,
            "sample": task.sample,
        }
    return rows


def task_metadata(task: HarnessTask, model: Path) -> dict[str, object] | None:
    """What a long-context task builds its inputs from: the length and the tokenizer that
    sizes them. lm-eval passes a task only this metadata when it is given an engine object,
    not a model path, so the tokenizer must be named here."""
    if not task.long_context:
        return None
    return {"max_seq_lengths": [LONG_CONTEXT], "tokenizer": str(model)}


class VllmHarness:
    """lm-eval over one vLLM engine in one thinking mode.

    One engine per process: a second engine in the same process is not guaranteed its GPU
    memory back, so each mode is its own `chat-eval` invocation with its own report.
    """

    def __init__(self, model: Path, mode: Mode, *, gpu_memory: float = GPU_MEMORY) -> None:
        from lm_eval.models.vllm_causallms import VLLM

        self.engine = VLLM(
            pretrained=str(model),
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory,
            max_model_len=MAX_MODEL_LENGTH,
            seed=SEED,
            enable_thinking=mode == "think",
            think_end_token=THINK_END if mode == "think" else None,
        )
        self.mode = mode
        self.model = model

    def samples(self, task: HarnessTask) -> dict[str, list[int]] | None:
        """Seeded indices per subtask for a task scored on a draw."""
        if task.sample is None:
            return None
        from lm_eval.tasks import TaskManager, get_task_dict

        chosen: dict[str, list[int]] = {}
        for name, entry in flatten(get_task_dict([task.name], TaskManager())).items():
            docs = getattr(entry, "eval_docs", None)
            if not isinstance(docs, Sized):
                message = f"{task.name}: subtask {name} exposes no evaluation documents"
                raise ChatEvaluationError(message)
            size = len(docs)
            indices = drawn(size, task.sample, seed=SEED)
            chosen[name] = indices if indices is not None else list(range(size))
        return chosen

    def score(self, task: HarnessTask, limit: int | None) -> object:
        from lm_eval import simple_evaluate

        return simple_evaluate(
            model=self.engine,
            tasks=[task.name],
            num_fewshot=task.shots,
            limit=limit,
            samples=None if limit is not None else self.samples(task),
            apply_chat_template=True,
            fewshot_as_multiturn=task.shots > 0,
            gen_kwargs=generation(self.mode),
            bootstrap_iters=0,
            confirm_run_unsafe_code=task.code,
            metadata=task_metadata(task, self.model),
            random_seed=SEED,
            numpy_random_seed=SEED,
            torch_random_seed=SEED,
            fewshot_random_seed=SEED,
        )


def flatten(tasks: object) -> dict[str, object]:
    """Leaf tasks of lm-eval's nested task dictionary, by name."""
    leaves: dict[str, object] = {}
    if not isinstance(tasks, Mapping):
        return leaves
    for key, value in tasks.items():
        if isinstance(value, Mapping):
            leaves.update(flatten(value))
        else:
            leaves[getattr(key, "group_name", None) or str(key)] = value
    return leaves


def tasks_in(suite: Suite, mode: Mode) -> list[HarnessTask]:
    return [task for task in tasks_for(suite) if mode in task.modes]


def run(model: Path, suite: Suite, mode: Mode, *, limit: int | None = None) -> dict[str, object]:
    """One mode's report; `chat-<mode>.json` beside the completion suite's report."""
    harness = VllmHarness(model, mode)
    return {
        "model": str(model),
        "suite": suite,
        "mode": mode,
        "limit": limit,
        "seed": SEED,
        "decoding": generation(mode),
        "tasks": evaluate_tasks(harness, tasks_in(suite, mode), limit=limit),
    }


def report_name(mode: Mode) -> str:
    return f"chat-{mode}.json"


def write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chat_eval", description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suite", choices=("select", "release"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int, help="examples per task; a smoke, never a result")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run in the vLLM environment by `make evaluate SUITE=select|release`."""
    args = build_parser().parse_args(argv)
    try:
        with stage("chat evaluation", suite=args.suite, mode=args.mode):
            report = run(args.model, args.suite, args.mode, limit=args.limit)
    except ChatEvaluationError as error:
        sys.stderr.write(f"error: {error}\n")
        return 1
    write_report(args.out, report)
    progress("chat_report", out=str(args.out), tasks=sorted(report_tasks(report)))
    return 0


def report_tasks(report: Mapping[str, object]) -> list[str]:
    tasks = report.get("tasks")
    return [str(name) for name in tasks] if isinstance(tasks, Mapping) else []


if __name__ == "__main__":
    raise SystemExit(main())
