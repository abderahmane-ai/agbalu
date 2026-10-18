"""The merge coefficient chosen by the rule fixed before any merge was scored (task 11.21).

A candidate passes when, against α = 0 (the post-trained model unchanged), it loses at most
2 points of the completion-mode retention mean, 3 of IFEval, 3 of GSM8K and of MMLU-Pro in
each thinking mode, and 5 of the long-context needle score. Among passing candidates with
α > 0 the best eng→kab chrF++ wins; candidates within 0.5 chrF++ of it are tied and the tie
goes to the lowest α. `docs/llm_post_training_design.md` §3 states the rule; nothing here may
change after a result is seen.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agbalu.llm.chat_eval import MODES, report_name

ALLOWED_DROP: Final[dict[str, float]] = {
    "retention": 2.0,
    "ifeval": 3.0,
    "gsm8k_no_think": 3.0,
    "gsm8k_think": 3.0,
    "mmlu_pro_no_think": 3.0,
    "mmlu_pro_think": 3.0,
    "niah": 5.0,
}
TIE: Final = 0.5
CHOSEN_BY: Final = "eng_kab_chrf"
NEEDLES: Final = ("niah_single_1", "niah_single_2", "niah_single_3")

MERGE_REPORT: Final = "merge.json"
COMPLETION_REPORT: Final = "evaluation.json"


class SelectionError(Exception):
    """The candidates cannot be compared under the fixed rule."""


@dataclass(frozen=True, slots=True)
class Candidate:
    alpha: float
    run: Path
    completion: Mapping[str, object]
    chat: Mapping[str, Mapping[str, object]]


def load(path: Path) -> dict[str, object]:
    if not path.is_file():
        message = f"{path} not found"
        raise SelectionError(message)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"{path} is not a JSON object"
        raise SelectionError(message)
    return payload


def read_candidate(run: Path) -> Candidate:
    """A merge run directory holding its merge report and every evaluation report."""
    alpha = load(run / MERGE_REPORT).get("alpha")
    if not isinstance(alpha, int | float) or isinstance(alpha, bool):
        message = f"{run / MERGE_REPORT}: alpha is {alpha!r}"
        raise SelectionError(message)
    chat: dict[str, Mapping[str, object]] = {mode: load(run / report_name(mode)) for mode in MODES}
    return Candidate(float(alpha), run, load(run / COMPLETION_REPORT), chat)


def value(report: Mapping[str, object], *path: str) -> float:
    node: object = report
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            message = f"no {'.'.join(path)} in the report"
            raise SelectionError(message)
        node = node[key]
    if not isinstance(node, int | float) or isinstance(node, bool) or not math.isfinite(node):
        message = f"{'.'.join(path)} is {node!r}, not a finite number"
        raise SelectionError(message)
    return float(node)


def measures(candidate: Candidate) -> dict[str, float]:
    """Every number the rule reads, in points."""
    think, plain = candidate.chat["think"], candidate.chat["no_think"]
    needles = [value(plain, "tasks", name, "score") for name in NEEDLES]
    return {
        CHOSEN_BY: value(candidate.completion, "mt", "eng-kab", "chrf++"),
        "retention": value(candidate.completion, "retention", "mean_points"),
        "ifeval": value(plain, "tasks", "ifeval", "score"),
        "gsm8k_no_think": value(plain, "tasks", "gsm8k", "score"),
        "gsm8k_think": value(think, "tasks", "gsm8k", "score"),
        "mmlu_pro_no_think": value(plain, "tasks", "mmlu_pro", "score"),
        "mmlu_pro_think": value(think, "tasks", "mmlu_pro", "score"),
        "niah": math.fsum(needles) / len(needles),
    }


def check_comparable(candidates: Sequence[Candidate]) -> Candidate:
    """The α = 0 control, after refusing limited reports and duplicate coefficients."""
    alphas = [candidate.alpha for candidate in candidates]
    if len(set(alphas)) != len(alphas):
        message = f"duplicate coefficients among {sorted(alphas)}"
        raise SelectionError(message)
    for candidate in candidates:
        reports = [("completion", candidate.completion)]
        reports += [(f"chat {mode}", report) for mode, report in candidate.chat.items()]
        for label, report in reports:
            if report.get("limit") is not None:
                message = f"{candidate.run} {label} was limited to {report['limit']}"
                raise SelectionError(message)
    controls = [candidate for candidate in candidates if candidate.alpha == 0.0]
    if len(controls) != 1:
        message = f"exactly one α = 0 control is required, found {len(controls)}"
        raise SelectionError(message)
    return controls[0]


def select_alpha(candidates: Sequence[Candidate]) -> dict[str, object]:
    control = check_comparable(candidates)
    reference = measures(control)
    rows: dict[str, dict[str, object]] = {}
    passing: list[tuple[float, float]] = []
    for candidate in sorted(candidates, key=lambda c: c.alpha):
        numbers = measures(candidate)
        drops = {name: round(reference[name] - numbers[name], 4) for name in ALLOWED_DROP}
        failed = sorted(name for name, drop in drops.items() if drop > ALLOWED_DROP[name])
        rows[str(candidate.alpha)] = {
            "run": str(candidate.run),
            "measures": numbers,
            "drops": drops,
            "failed": failed,
            "passed": not failed,
        }
        if candidate.alpha > 0 and not failed:
            passing.append((candidate.alpha, numbers[CHOSEN_BY]))
    selected: float | None = None
    if passing:
        best = max(chrf for _, chrf in passing)
        selected = min(alpha for alpha, chrf in passing if best - chrf <= TIE)
    return {
        "rule": {"allowed_drop": ALLOWED_DROP, "chosen_by": CHOSEN_BY, "tie": TIE},
        "candidates": rows,
        "selected_alpha": selected,
        "fallback": None if selected is not None else "start SFT from the CPT checkpoint",
    }
