"""The parts of the TTS baseline that run without a GPU (task 12.1).

Two of them decide whether the published number means what it says. The front end
deletes characters it cannot represent and says nothing, so what it voiced is read back
and tallied; and the second, restricted score is computed from the *same* decode, keyed
on the reference text rather than on position — the decode comes back in duration order.

The path assertions are here because a writer and a reader that disagree about where the
prompt set lives fail inside a paid container, and this project has already shipped a
default no command could reach.
"""

from __future__ import annotations

import numpy as np
from modal_app.asr import Evaluation
from modal_app.tts import (
    LOCAL_PROMPTS,
    LOCAL_RESULT,
    PROMPTS_FILE,
    REMOTE_TTS,
    Synthesis,
    _condition,
    deletions,
)

from agbalu.speech.vocabulary import ctc_target
from agbalu.tts import cli as tts_cli
from agbalu.tts import cycle
from agbalu.tts.prompts import Prompt


def prompt(clip: str, text: str) -> Prompt:
    """A prompt whose target is reduced exactly as the corpus builder reduces it."""
    return Prompt(clip=clip, speaker="s1", duration_ms=4000, text=text, target=ctc_target(text))


def synthesis(voiced: dict[str, str]) -> Synthesis:
    waves = {name: np.zeros(16_000, dtype=np.float32) for name in voiced}
    return Synthesis(waves=waves, voiced=voiced, seconds=float(len(voiced)), elapsed=1.0)


def test_a_deleted_character_is_counted_against_its_prompt() -> None:
    """`povo` reaches `mms-tts-kab` as nothing at all: `o`, `p` and `v` are outside its
    38-symbol vocabulary and `VitsTokenizer` drops them without raising."""
    prompts = [prompt("a.mp3", "bonjour d povo"), prompt("b.mp3", "azul fell-awen")]
    voiced = {"a.mp3": "bnjur d ", "b.mp3": "azul fell-awen"}

    summary, intact = deletions(prompts, synthesis(voiced))

    assert summary["prompts_with_deleted_characters"] == 1
    assert summary["deleted_characters"] == {"o": 4, "p": 1, "v": 1}
    assert intact == {"azul fell-awen"}


def test_punctuation_the_front_end_drops_is_not_a_deletion() -> None:
    """`?` is outside the baseline's vocabulary and outside the CTC target both error
    rates are computed under, so counting it would swamp the letters that matter."""
    prompts = [prompt("a.mp3", "d acu ara d-tiniḍ deg wa?")]
    summary, intact = deletions(prompts, synthesis({"a.mp3": "d acu ara d-tiniḍ deg wa"}))

    assert summary["deleted_characters"] == {}
    assert intact == {"d acu ara d-tiniḍ deg wa"}


def test_a_prompt_voiced_in_full_leaves_no_tally() -> None:
    prompts = [prompt("a.mp3", "Azul fell-awen")]
    summary, intact = deletions(prompts, synthesis({"a.mp3": "azul fell-awen"}))

    assert summary["prompts_with_deleted_characters"] == 0
    assert summary["deleted_characters"] == {}
    assert intact == {"azul fell-awen"}


def test_case_alone_is_not_a_deletion() -> None:
    """The front end lower-cases before it filters, so an upper-case letter that survives
    as its lower-case form was represented, not dropped."""
    prompts = [prompt("a.mp3", "AZUL FELL-AWEN")]
    summary, _ = deletions(prompts, synthesis({"a.mp3": "azul fell-awen"}))

    assert summary["deleted_characters"] == {}


def test_the_waveform_is_addressed_by_clip_name() -> None:
    built = synthesis({"a.mp3": "azul"})

    assert built.waveform("a.mp3").shape == (16_000,)


def test_a_condition_carries_the_evaluation_s_own_rates() -> None:
    """The harness is handed what `evaluate` already computed, not a second scoring of
    the same pairs: two rates for one decode is the stale-copy defect one layer down."""
    scored = Evaluation(
        loss=0.5696,
        word_error=0.343551,
        character_error=0.118902,
        utterances=1000,
        previews=(("azul", "azuk"),),
    )

    condition = _condition(cycle.BASELINE, scored, audio_seconds=2656.24)

    assert condition.cer_percent == 11.8902
    assert condition.wer_percent == 34.3551
    assert condition.loss == 0.5696
    assert condition.audio_seconds == 2656.2


def test_the_reader_and_the_writer_agree_on_where_the_prompt_set_lives() -> None:
    assert LOCAL_PROMPTS == tts_cli.DEFAULT_PROMPTS
    assert REMOTE_TTS.name == "tts"
    assert LOCAL_PROMPTS.name == PROMPTS_FILE


def test_the_result_lands_where_the_roadmap_declares_the_deliverable() -> None:
    assert LOCAL_RESULT.as_posix() == "data/processed/bench/tts-baseline.json"
