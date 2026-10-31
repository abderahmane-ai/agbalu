"""Select downstream checkpoints under explicit regression guards."""

import math
from dataclasses import dataclass

from agbalu.encoder.checkpoint import State
from agbalu.encoder.configuration import Recipe


@dataclass(frozen=True)
class Decision:
    eligible: bool
    improved: bool
    reason: str


def choose(clean: float, corrupted: float, state: State, recipe: Recipe) -> Decision:
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (clean, corrupted)):
        return Decision(False, False, "invalid development scores")
    if (
        state.reference_clean is not None
        and clean < state.reference_clean - recipe.regression_tolerance
    ):
        return Decision(False, False, "clean POS regression")
    if (
        state.reference_corrupted is not None
        and corrupted < state.reference_corrupted - recipe.regression_tolerance
    ):
        return Decision(False, False, "corrupted POS regression")
    if state.best_score is None or clean > state.best_score + recipe.selection_delta:
        return Decision(True, True, "canonical POS improvement")
    return Decision(True, False, "improvement below the frozen selection delta")
