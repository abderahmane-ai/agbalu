"""Check rendering output contracts for empty predictions."""

from agbalu.restoration.inference import Prediction


def test_empty_prediction_keeps_a_terminal_gap_without_emitting_text() -> None:
    assert Prediction((), (0,), (), ("@",)).text == ""
