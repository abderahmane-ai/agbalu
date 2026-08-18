"""The prompt set against the real corpus, not against a fixture.

A fixture satisfies the filter it was written for. These assertions run against
`data/processed/speech/` and `data/raw/opus.bible-uedin-kab/`, which is where the
properties task 12.1's number rests on can actually fail: 220 of the 9,494 admissible
test clips carry a sentence Fadhma trained on, and neither the split file nor the speaker
disjointness says anything about that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from agbalu.speech.corpus import read as read_clips
from agbalu.tts import prompts as prompt_set
from agbalu.tts import scripture

pytestmark = pytest.mark.integration

SPEECH: Final = Path("data/processed/speech")
SIZE: Final = 200
"""Smaller than the built set: these assertions are about the filters, and reading the
144-hour train split's manifest is already the slowest part of the test."""


@pytest.fixture(scope="module")
def heard() -> set[str]:
    """Every sentence Fadhma trained or validated on, as the decoder's targets."""
    if not (SPEECH / "train.jsonl").is_file():
        pytest.skip("speech corpus not built; run `make speech TASK=corpus`")
    return {
        clip.target for split in ("train", "dev") for clip in read_clips(SPEECH / f"{split}.jsonl")
    }


@pytest.fixture(scope="module")
def verses() -> scripture.Scripture:
    if not scripture.BIBLE.is_file():
        pytest.skip(f"{scripture.BIBLE} not fetched")
    return scripture.load()


@pytest.fixture(scope="module")
def selection(heard: set[str], verses: scripture.Scripture) -> prompt_set.Selection:
    return prompt_set.select(
        read_clips(SPEECH / "test.jsonl"), exclude=verses.match, seen=heard, size=SIZE
    )


def test_the_bible_index_matches_its_own_source(verses: scripture.Scripture) -> None:
    """`load` proves this before returning, and it is asserted again here because a
    zero-hit filter and a broken filter look identical from the outside."""
    assert verses.verses > 15_000
    assert verses.match(next(iter(scripture.verses()))) == scripture.EXACT


def test_the_decoder_has_not_heard_a_single_prompt(
    selection: prompt_set.Selection, heard: set[str]
) -> None:
    assert not [prompt for prompt in selection.prompts if prompt.target in heard]


def test_no_prompt_is_biblical(
    selection: prompt_set.Selection, verses: scripture.Scripture
) -> None:
    assert not [prompt for prompt in selection.prompts if verses.match(prompt.text) is not None]


def test_every_prompt_clears_the_length_floor(selection: prompt_set.Selection) -> None:
    assert min(prompt.words for prompt in selection.prompts) >= prompt_set.MIN_WORDS


def test_no_voice_dominates_the_floor_condition(selection: prompt_set.Selection) -> None:
    counts = {
        speaker: sum(1 for prompt in selection.prompts if prompt.speaker == speaker)
        for speaker in {prompt.speaker for prompt in selection.prompts}
    }
    assert max(counts.values()) <= prompt_set.SPEAKER_CAP
    assert len(counts) >= SIZE // prompt_set.SPEAKER_CAP


def test_the_prompts_are_distinct_sentences(selection: prompt_set.Selection) -> None:
    """The restricted score is keyed on the reference text, so two identical targets
    would silently merge two measurements into one."""
    targets = [prompt.target for prompt in selection.prompts]
    assert len(set(targets)) == len(targets)


def test_the_selection_is_reproducible(heard: set[str], verses: scripture.Scripture) -> None:
    again = prompt_set.select(
        read_clips(SPEECH / "test.jsonl"), exclude=verses.match, seen=heard, size=SIZE
    )
    first = prompt_set.select(
        read_clips(SPEECH / "test.jsonl"), exclude=verses.match, seen=heard, size=SIZE
    )
    assert again.prompts == first.prompts
