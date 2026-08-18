"""The prompt set's filters, and the biblical index that cannot silently pass everything.

Every property here is one a downstream number depends on. A prompt whose sentence the
decoder trained on measures its memory; a biblical prompt measures how much of its own
training text `mms-tts-kab` remembers; an uncapped speaker turns the floor condition into
one voice's number. The index gets its own tests because a join that cannot match returns
zero and reads as a clean result — which has already happened once in this project.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from agbalu.speech.corpus import Clip
from agbalu.tts import prompts as prompt_set
from agbalu.tts import scripture
from agbalu.tts.prompts import CAPPED, DUPLICATE, SEEN, TOO_SHORT, Prompt, PromptError

VERSES: Final = (
    "Yal wa amek yebɣa ad yefhem awal-agi n Sidi Ṛebbi di temdint n Yerusalem",
    "Imiren kan yekker-ed wergaz-nni yerra-yas awal i wid akk yellan dinna",
    "Ur ilaq ara ad tettuɣalem ɣer deffir m'ara tebdum abrid",
)


def clip(name: str, text: str, speaker: str = "s1", duration_ms: int = 4000) -> Clip:
    """A corpus record whose target is its text lower-cased, as `ctc_target` leaves it."""
    return Clip(
        clip=name,
        speaker=speaker,
        split="test",
        duration_ms=duration_ms,
        text=text,
        target=text.casefold(),
        repaired=False,
    )


def nothing_excluded(_text: str) -> str | None:
    return None


def test_a_clip_shorter_than_the_floor_is_rejected() -> None:
    clips = [clip("a.mp3", "yiwen sin kraḍ"), clip("b.mp3", "yiwen sin kraḍ kuẓ")]
    selection = prompt_set.select(clips, exclude=nothing_excluded)

    assert [prompt.clip for prompt in selection.prompts] == ["b.mp3"]
    assert selection.rejected[TOO_SHORT] == 1


def test_the_boundary_word_count_is_kept() -> None:
    exact = clip("a.mp3", " ".join(["awal"] * prompt_set.MIN_WORDS))
    below = clip("b.mp3", " ".join(["awal"] * (prompt_set.MIN_WORDS - 1)))
    selection = prompt_set.select([exact, below], exclude=nothing_excluded)

    assert [prompt.clip for prompt in selection.prompts] == ["a.mp3"]


def test_a_repeated_sentence_enters_once() -> None:
    """Common Voice records the same sentence from several speakers, and two identical
    prompts are one measurement charged twice."""
    clips = [
        clip("a.mp3", "d acu i txedmeḍ ass-a", speaker="s1"),
        clip("b.mp3", "d acu i txedmeḍ ass-a", speaker="s2"),
    ]
    selection = prompt_set.select(clips, exclude=nothing_excluded)

    assert len(selection.prompts) == 1
    assert selection.rejected[DUPLICATE] == 1


def test_a_sentence_the_decoder_trained_on_is_rejected() -> None:
    heard = clip("a.mp3", "ur ẓriɣ ara d acu i yedran")
    fresh = clip("b.mp3", "azekka ad nruḥ ɣer temdint")
    selection = prompt_set.select([heard, fresh], exclude=nothing_excluded, seen={heard.target})

    assert [prompt.clip for prompt in selection.prompts] == ["b.mp3"]
    assert selection.rejected[SEEN] == 1


def test_the_speaker_cap_bounds_one_voice() -> None:
    clips = [clip(f"{i}.mp3", f"awal wis {i} n tefyirt", speaker="s1") for i in range(6)]
    selection = prompt_set.select(clips, exclude=nothing_excluded, speaker_cap=2)

    assert len(selection.prompts) == 2
    assert selection.rejected[CAPPED] == 4


def test_the_selection_does_not_depend_on_the_order_the_file_was_written_in() -> None:
    clips = [clip(f"{i}.mp3", f"tafyirt wis {i} deg umuqel", speaker=f"s{i}") for i in range(20)]
    forward = prompt_set.select(clips, exclude=nothing_excluded, size=5)
    backward = prompt_set.select(list(reversed(clips)), exclude=nothing_excluded, size=5)

    assert forward.prompts == backward.prompts


def test_two_seeds_choose_differently() -> None:
    clips = [clip(f"{i}.mp3", f"tafyirt wis {i} deg umuqel", speaker=f"s{i}") for i in range(40)]
    first = prompt_set.select(clips, exclude=nothing_excluded, size=5, seed=1)
    second = prompt_set.select(clips, exclude=nothing_excluded, size=5, seed=2)

    assert first.prompts != second.prompts


def test_a_pool_smaller_than_the_request_yields_the_pool() -> None:
    clips = [clip(f"{i}.mp3", f"tafyirt wis {i} deg umuqel", speaker=f"s{i}") for i in range(3)]
    selection = prompt_set.select(clips, exclude=nothing_excluded, size=100)

    assert len(selection.prompts) == 3
    assert selection.candidates == 3


def test_an_empty_pool_raises_rather_than_writing_an_unscoreable_set() -> None:
    with pytest.raises(PromptError, match=TOO_SHORT):
        prompt_set.select([clip("a.mp3", "yiwen sin")], exclude=nothing_excluded)


def test_no_clips_at_all_raises() -> None:
    with pytest.raises(PromptError):
        prompt_set.select([], exclude=nothing_excluded)


def test_a_size_below_one_is_refused() -> None:
    clips = [clip("a.mp3", "d acu i txedmeḍ ass-a")]
    with pytest.raises(PromptError, match="cannot be scored"):
        prompt_set.select(clips, exclude=nothing_excluded, size=0)


def test_the_synthetic_clip_carries_the_synthesis_duration() -> None:
    prompt = Prompt(
        clip="a.mp3", speaker="s1", duration_ms=4000, text="Azul fell-awen", target="azul fell-awen"
    )

    assert prompt.as_clip().duration_ms == 4000
    assert prompt.as_clip(duration_ms=2500).duration_ms == 2500
    assert prompt.as_clip().target == prompt.target


def test_a_prompt_set_round_trips(tmp_path: Path) -> None:
    written = [
        Prompt(clip="a.mp3", speaker="s1", duration_ms=4000, text="Azul ɣer tmurt", target="azul"),
        Prompt(clip="b.mp3", speaker="s2", duration_ms=3000, text="Ṛuḥ ɣer wexxam", target="ṛuḥ"),
    ]
    path = prompt_set.write(tmp_path / "prompts.jsonl", written)

    assert prompt_set.read(path) == written


def test_a_malformed_row_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"clip": "a.mp3"}) + "\n", encoding="utf-8")

    with pytest.raises(PromptError, match=r"prompts\.jsonl:1"):
        prompt_set.read(path)


def test_an_empty_file_is_not_a_prompt_set(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(PromptError, match="empty"):
        prompt_set.read(path)


def test_a_missing_file_says_how_to_build_it(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match="make tts TASK=prompts"):
        prompt_set.read(tmp_path / "absent.jsonl")


def test_a_verse_is_caught_exactly() -> None:
    index = scripture.index(VERSES)

    assert index.match(VERSES[0]) == scripture.EXACT
    assert index.verses == len(VERSES)


def test_a_clipped_verse_is_caught_by_the_ngram_and_not_by_the_fingerprint() -> None:
    index = scripture.index(VERSES)
    clipped = " ".join(VERSES[0].split()[1:])

    assert index.match(clipped) == scripture.NGRAM_HIT


def test_ordinary_kabyle_is_not_biblical() -> None:
    index = scripture.index(VERSES)

    assert index.match("Azekka ad nruḥ ɣer ssuq n taddart") is None


def test_punctuation_and_case_do_not_hide_a_verse() -> None:
    """The fingerprint is the corpus's own dedup key, so a re-punctuated verse still hits."""
    index = scripture.index(VERSES)

    assert index.match(VERSES[1].upper() + " !") == scripture.EXACT


def test_a_biblical_prompt_is_rejected_under_its_own_reason() -> None:
    index = scripture.index(VERSES)
    clips = [clip("a.mp3", VERSES[0]), clip("b.mp3", "azekka ad nruḥ ɣer ssuq n taddart")]
    selection = prompt_set.select(clips, exclude=index.match)

    assert [prompt.clip for prompt in selection.prompts] == ["b.mp3"]
    assert selection.rejected[scripture.EXACT] == 1


def test_the_control_needs_a_verse_long_enough_to_prove_both_tests() -> None:
    with pytest.raises(scripture.ScriptureError, match="control"):
        scripture.control(["awal", "sin wawalen"])


def test_an_index_that_cannot_match_is_refused(tmp_path: Path) -> None:
    """The whole point of the control: a build that indexes nothing returns zero hits on
    every prompt, which is indistinguishable from a clean corpus."""
    empty = tmp_path / "empty.zip"
    empty.write_bytes(b"not a zip")

    with pytest.raises(scripture.ScriptureError):
        scripture.load(empty)


def test_a_missing_bundle_is_named() -> None:
    with pytest.raises(scripture.ScriptureError, match="not found"):
        scripture.load(Path("data/raw/absent/en-kab.txt.zip"))
