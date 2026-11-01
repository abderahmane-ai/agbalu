"""Validate restoration recovery state without legacy MLM counters."""

from agbalu.restoration.checkpoint import State


def test_recovery_state_retains_exact_epoch_and_window_cursor() -> None:
    state = State(step=7, epoch=2, cursor=41, tokens=9000, words=2100)
    restored = State.model_validate(state.model_dump())
    assert (restored.step, restored.epoch, restored.cursor) == (7, 2, 41)
    assert restored.words == 2100
