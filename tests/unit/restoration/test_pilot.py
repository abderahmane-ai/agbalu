"""Reject full GPU schedules without matching measured selection evidence."""

from pathlib import Path

import pytest
import torch

from agbalu.restoration.configuration import Locations, Settings
from agbalu.restoration.pilot import candidate_metrics
from agbalu.restoration.training import Trainer


@pytest.mark.integration
def test_full_gpu_training_requires_a_smoke_proof(tmp_path: Path) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.settings = Settings()
    trainer.locations = Locations(data=tmp_path)
    trainer.device = torch.device("cuda")
    with pytest.raises(ValueError, match="missing smoke proof"):
        trainer.run_locked(stop_after=20000, smoke=False)


@pytest.mark.integration
def test_full_gpu_training_requires_a_selected_pilot_when_enabled(tmp_path: Path) -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.settings = Settings(require_pilot=True)
    trainer.locations = Locations(data=tmp_path)
    trainer.device = torch.device("cuda")
    with pytest.raises(ValueError, match="selected, measured pilot"):
        trainer.run_locked(stop_after=20000, smoke=False)


@pytest.mark.parametrize(
    ("punctuation", "streaming", "false", "exact", "expected"),
    [
        (0.5, 0.45, 30.0, 0.91, True),
        (0.5, 0.45, 38.0, 0.91, False),
        (0.5, 0.45, 30.0, 0.84, False),
        (0.29, 0.45, 30.0, 0.91, False),
        (0.5, 0.29, 30.0, 0.91, False),
    ],
)
def test_selection_rejects_false_insertions_and_casing_regressions(
    punctuation: float, streaming: float, false: float, exact: float, *, expected: bool
) -> None:
    offline: dict[str, object] = {
        "punctuation_macro_f1": punctuation,
        "casing_macro_f1": 0.7,
        "false_insertions_per_1000_words": false,
        "exact_word_casing_accuracy": exact,
    }
    score, is_eligible = candidate_metrics(
        offline, {"punctuation_macro_f1": streaming}, {"exact_word_casing_accuracy": 0.85}
    )
    assert score == pytest.approx(0.4 * punctuation + 0.3 * streaming + 0.21)
    assert is_eligible is expected
