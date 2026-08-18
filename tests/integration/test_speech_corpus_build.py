"""The built speech corpus, checked against the artifact rather than a fixture.

A fixture satisfies the invariant it was written for; the corpus is where an invariant can
fail. These assertions run against `data/processed/speech/`, so they fail when what is on
disk stops matching what the phase record claims about it.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Final

import pytest

from agbalu.normalise.rules import ALPHABET, HYPHEN
from agbalu.speech import corpus
from agbalu.speech.vocabulary import PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER, ctc_target, encoder

pytestmark = pytest.mark.integration

PROCESSED: Final = Path("data/processed/speech")
VOCAB: Final = Path("artifacts/asr/vocab.json")
SPLITS: Final = ("train", "dev", "test")


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} is not built; run `make speech TASK=corpus`")
    return path


@pytest.fixture(scope="module")
def clips() -> dict[str, list[corpus.Clip]]:
    return {split: corpus.read(_require(PROCESSED / f"{split}.jsonl")) for split in SPLITS}


@pytest.fixture(scope="module")
def vocabulary() -> dict[str, int]:
    payload = json.loads(_require(VOCAB).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return {str(token): int(index) for token, index in payload.items()}


class TestSplits:
    def test_every_split_has_clips(self, clips: dict[str, list[corpus.Clip]]) -> None:
        assert all(clips[split] for split in SPLITS)

    def test_speakers_are_disjoint_on_the_artifact(
        self, clips: dict[str, list[corpus.Clip]]
    ) -> None:
        """`build` refuses an overlap, but the file on disk is what training reads."""
        speakers = {split: {clip.speaker for clip in clips[split]} for split in SPLITS}
        assert not speakers["train"] & speakers["test"]
        assert not speakers["train"] & speakers["dev"]
        assert not speakers["dev"] & speakers["test"]

    def test_no_clip_appears_in_two_splits(self, clips: dict[str, list[corpus.Clip]]) -> None:
        names = [clip.clip for split in SPLITS for clip in clips[split]]
        assert len(names) == len(set(names))

    def test_each_record_carries_its_own_split(self, clips: dict[str, list[corpus.Clip]]) -> None:
        assert all(clip.split == split for split in SPLITS for clip in clips[split])


class TestTargets:
    def test_no_target_is_empty(self, clips: dict[str, list[corpus.Clip]]) -> None:
        assert all(clip.target for split in SPLITS for clip in clips[split])

    def test_every_target_fits_its_frame_budget(self, clips: dict[str, list[corpus.Clip]]) -> None:
        """CTC returns infinite loss when the target is longer than the input."""
        for split in SPLITS:
            for clip in clips[split]:
                assert len(clip.target) <= corpus.frames(clip.duration_ms)

    def test_targets_are_already_reduced(self, clips: dict[str, list[corpus.Clip]]) -> None:
        """`ctc_target` is idempotent on what was written, so scoring cannot shift it."""
        for clip in clips["dev"]:
            assert ctc_target(clip.target) == clip.target

    def test_every_target_character_is_a_class(
        self, clips: dict[str, list[corpus.Clip]], vocabulary: dict[str, int]
    ) -> None:
        encode = encoder(vocabulary)
        for split in SPLITS:
            for clip in clips[split]:
                assert all(character in encode for character in clip.target)

    def test_text_is_composed(self, clips: dict[str, list[corpus.Clip]]) -> None:
        for clip in clips["dev"]:
            assert clip.text == unicodedata.normalize("NFC", clip.text)


class TestVocabulary:
    def test_pad_is_the_blank_at_zero(self, vocabulary: dict[str, int]) -> None:
        assert vocabulary[PAD_TOKEN] == 0

    def test_ids_are_contiguous(self, vocabulary: dict[str, int]) -> None:
        assert sorted(vocabulary.values()) == list(range(len(vocabulary)))

    def test_it_is_small_enough_to_read(self, vocabulary: dict[str, int]) -> None:
        """The reference Mongolian CTC vocabulary is 37 classes; a Kabyle one that came
        out at hundreds would mean the transcripts were never normalised."""
        assert len(vocabulary) < 60

    def test_every_class_is_kabyle_or_structural(self, vocabulary: dict[str, int]) -> None:
        structural = {PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER, HYPHEN}
        for token in vocabulary:
            assert token in structural or token in ALPHABET

    def test_the_homoglyphs_are_not_classes(self, vocabulary: dict[str, int]) -> None:
        """Greek epsilon and gamma are 2.6-3.2% of raw Kabyle text. A vocabulary taken
        before normalisation contains them beside their Latin counterparts."""
        assert "ε" not in vocabulary
        assert "γ" not in vocabulary
        assert "ɛ" in vocabulary
        assert "ɣ" in vocabulary

    def test_the_hyphen_is_a_class(self, vocabulary: dict[str, int]) -> None:
        """`docs/orthography.md` §2: a letter of the writing system, never stripped."""
        assert HYPHEN in vocabulary


class TestStats:
    def test_the_stats_describe_the_files(self, clips: dict[str, list[corpus.Clip]]) -> None:
        payload = json.loads(_require(PROCESSED / "speech.stats.json").read_text("utf-8"))
        assert payload["clips"] == sum(len(clips[split]) for split in SPLITS)
        assert payload["normaliser_version"]
        for row in payload["splits"]:
            assert row["clips"] == len(clips[row["split"]])

    def test_recorded_overlap_is_zero(self) -> None:
        payload = json.loads(_require(PROCESSED / "speech.stats.json").read_text("utf-8"))
        assert set(payload["speaker_overlap"].values()) == {0}
