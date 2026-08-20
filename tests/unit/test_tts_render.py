"""The corpus renderer, in isolation from the volume and the decoder.

What is under test is the resilience and the arithmetic the full run depends on, both of
which are invisible from a clip count: a source clip is decoded **once** and feeds both
arms, a clip already on the volume is neither re-read nor re-rendered, and its profile
still reaches the summary so a resumed voice reports the whole arm rather than this call's
share of it. A killed writer must leave no file behind that resume would trust.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import soundfile as sf

from agbalu.tts.corpus import OUTPUT_RATE, Utterance
from agbalu.tts.restore import OUTPUT_RATE as RESTORED_RATE
from agbalu.tts.restore import Restorer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

from modal_app import tts
from modal_app.common import data_volume

RATE: int = 16_000
CLIPS: int = 3
WORKERS: int = 3
DEVICE_SECONDS: float = 0.05
ROUNDING_SLOP: float = 0.3
"""Every clock is rounded to a tenth of a second before it is reported, and there are
three of them plus the wall."""


def _write_wav(path: Path, *, seconds: float = 3.0, rate: int = RATE) -> None:
    steps = int(seconds * rate)
    tone = np.sin(2 * np.pi * 220 * np.arange(steps) / rate).astype(np.float32)
    sf.write(path, tone, rate)


def _utterance(name: str, *, seconds: float = 2.5) -> Utterance:
    target = "azul " * 6
    return Utterance(
        clip=name,
        text=target,
        target=target,
        ipa="æzul",
        start_s=0.0,
        stop_s=seconds,
        speech_dbfs=-20.0,
        peak=0.5,
        cer=0.05,
    )


class _FakeRestorer:
    """Shape-preserving stand-in. `prepare` hands the waveform through as its own
    features, so a test can read back exactly which clips reached the device."""

    def __init__(self) -> None:
        self.prepared: list[int] = []

    def prepare(self, audio: NDArray[np.float32], rate: int) -> tuple[NDArray[np.float32], int]:
        self.prepared.append(audio.size)
        return audio, round(audio.size / rate * RESTORED_RATE)

    def restore_prepared(
        self, features: Sequence[NDArray[np.float32]], lengths: Sequence[int]
    ) -> list[NDArray[np.float32]]:
        assert len(features) == len(lengths)
        return list(features)


@pytest.fixture
def audio_map(tmp_path: Path) -> Mapping[str, Path]:
    clips: dict[str, Path] = {}
    for index in range(CLIPS):
        path = tmp_path / f"clip{index}.mp3"
        _write_wav(path)
        clips[f"clip{index}"] = path
    return clips


@pytest.fixture
def _no_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_volume, "commit", lambda: None)


def _render(
    utterances: Sequence[Utterance],
    audio: Mapping[str, Path],
    restorer: _FakeRestorer,
    voice_root: Path,
    *,
    resume: bool,
) -> dict[str, object]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return tts.render_arms(
            utterances,
            audio,
            cast("Restorer", restorer),
            voice_root,
            pool,
            workers=WORKERS,
            resume=resume,
        )


def _arms(rendered: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    return cast("Mapping[str, Mapping[str, object]]", rendered["arms"])


class TestTodo:
    def test_an_arm_holding_the_clip_is_skipped_and_the_other_is_not(self) -> None:
        already = {"restored": {"clip0"}, "raw": set[str]()}
        todo, present = tts._arms_todo(already, _utterance("clip0"), resume=True)
        assert (todo, present) == (["raw"], ["restored"])

    def test_a_clip_neither_arm_holds_owes_both(self) -> None:
        already = {"restored": {"clip9"}, "raw": {"clip9"}}
        todo, present = tts._arms_todo(already, _utterance("clip0"), resume=True)
        assert (sorted(todo), present) == (["raw", "restored"], [])

    def test_resume_off_renders_everything_however_full_the_arms_are(self) -> None:
        already = {"restored": {"clip0"}, "raw": {"clip0"}}
        todo, present = tts._arms_todo(already, _utterance("clip0"), resume=False)
        assert (sorted(todo), present) == (["raw", "restored"], [])

    def test_an_arm_is_listed_once_rather_than_stat_ed_per_clip(self, tmp_path: Path) -> None:
        for index in range(CLIPS):
            _write_wav(tmp_path / f"clip{index}.wav")
        (tmp_path / "clip9.wav.partial").write_bytes(b"x")
        assert tts._written(tmp_path) == {"clip0", "clip1", "clip2"}


class TestRenderArms:
    @pytest.mark.usefixtures("_no_commit")
    def test_one_source_read_feeds_both_arms(
        self, audio_map: Mapping[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One decode per clip, not one per arm: the restored pass and the raw resample are
        two derivations of the same decode, and the volume charges per read."""
        reads: Counter[str] = Counter()
        original = tts._native

        def _counted(path: Path) -> tuple[NDArray[np.float32], int]:
            reads[path.name] += 1
            return original(path)

        monkeypatch.setattr(tts, "_native", _counted)
        utterances = [_utterance(f"clip{index}") for index in range(CLIPS)]
        _render(utterances, audio_map, _FakeRestorer(), tmp_path / "voice", resume=False)

        assert reads == Counter({f"clip{index}.mp3": 1 for index in range(CLIPS)})

    @pytest.mark.usefixtures("_no_commit")
    def test_both_arms_are_written_for_every_clip_and_profiled(
        self, audio_map: Mapping[str, Path], tmp_path: Path
    ) -> None:
        voice_root = tmp_path / "voice"
        utterances = [_utterance(f"clip{index}") for index in range(CLIPS)]
        rendered = _render(utterances, audio_map, _FakeRestorer(), voice_root, resume=False)

        for arm in ("raw", "restored"):
            written = sorted(path.name for path in (voice_root / arm).glob("*.wav"))
            assert written == [f"clip{index}.wav" for index in range(CLIPS)]
            assert _arms(rendered)[arm]["clips"] == CLIPS
        assert rendered["clips_rendered"] == 2 * CLIPS

    @pytest.mark.usefixtures("_no_commit")
    def test_the_written_audio_carries_the_utterance_cut_and_rate(
        self, audio_map: Mapping[str, Path], tmp_path: Path
    ) -> None:
        voice_root = tmp_path / "voice"
        utterance = _utterance("clip0", seconds=1.5)
        _render([utterance], audio_map, _FakeRestorer(), voice_root, resume=False)

        audio, rate = sf.read(voice_root / "raw" / "clip0.wav", dtype="float32")
        assert rate == OUTPUT_RATE
        assert audio.size == pytest.approx(1.5 * OUTPUT_RATE, abs=2)

    @pytest.mark.usefixtures("_no_commit")
    def test_a_resumed_clip_is_neither_read_nor_rendered_but_is_still_profiled(
        self, audio_map: Mapping[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resume's whole claim: a committed wav costs no device time and no volume
        read, and the arm summary still counts it — a summary over this call's share
        alone would report a resumed voice as a fraction of itself."""
        voice_root = tmp_path / "voice"
        restorer = _FakeRestorer()
        _render(
            [_utterance(f"clip{index}") for index in range(CLIPS)],
            audio_map,
            restorer,
            voice_root,
            resume=False,
        )

        sources: list[str] = []
        original = tts._native

        def _counted(path: Path) -> tuple[NDArray[np.float32], int]:
            sources.append(path.name)
            return original(path)

        monkeypatch.setattr(tts, "_native", _counted)
        second = _FakeRestorer()
        rendered = _render(
            [_utterance(f"clip{index}") for index in range(CLIPS)],
            audio_map,
            second,
            voice_root,
            resume=True,
        )

        assert second.prepared == []
        assert not [name for name in sources if name.endswith(".mp3")]
        assert rendered["clips_rendered"] == 0
        for arm in ("raw", "restored"):
            assert _arms(rendered)[arm]["clips"] == CLIPS

    @pytest.mark.usefixtures("_no_commit")
    def test_a_half_written_voice_renders_only_what_is_missing(
        self, audio_map: Mapping[str, Path], tmp_path: Path
    ) -> None:
        voice_root = tmp_path / "voice"
        _render([_utterance("clip0")], audio_map, _FakeRestorer(), voice_root, resume=True)

        restorer = _FakeRestorer()
        rendered = _render(
            [_utterance(f"clip{index}") for index in range(CLIPS)],
            audio_map,
            restorer,
            voice_root,
            resume=True,
        )
        assert len(restorer.prepared) == CLIPS - 1
        assert rendered["clips_rendered"] == 2 * (CLIPS - 1)
        for arm in ("raw", "restored"):
            assert _arms(rendered)[arm]["clips"] == CLIPS

    @pytest.mark.usefixtures("_no_commit")
    def test_the_stage_clocks_attribute_device_time_and_do_not_double_count(
        self, audio_map: Mapping[str, Path], tmp_path: Path
    ) -> None:
        """They are the instrument that says whether the device, the volume or the writers
        is the wall. Nesting one clock inside another — draining the writers with the
        device clock running, say — would report a stage that dominates because it
        contains the others."""

        class _SlowRestorer(_FakeRestorer):
            def restore_prepared(
                self, features: Sequence[NDArray[np.float32]], lengths: Sequence[int]
            ) -> list[NDArray[np.float32]]:
                time.sleep(DEVICE_SECONDS)
                return super().restore_prepared(features, lengths)

        rendered = _render(
            [_utterance(f"clip{index}") for index in range(CLIPS)],
            audio_map,
            _SlowRestorer(),
            tmp_path / "voice",
            resume=False,
        )
        stages = cast("Mapping[str, float]", rendered["stage_seconds"])
        assert set(stages) == {"host", "device", "write"}
        assert stages["device"] >= DEVICE_SECONDS
        assert sum(stages.values()) <= cast("float", rendered["seconds"]) + ROUNDING_SLOP


class TestPartialWrites:
    def test_a_wav_is_staged_before_it_takes_its_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`resume` reads a wav's presence as proof it was rendered, so a container killed
        mid-write would leave a truncated clip that nothing downstream can distinguish
        from a short utterance."""
        seen: list[str] = []
        original = sf.write

        def _spy(path: object, *args: object, **kwargs: object) -> None:
            seen.append(Path(str(path)).name)
            original(path, *args, **kwargs)

        monkeypatch.setattr(sf, "write", _spy)
        target = tmp_path / "clip0.wav"
        tts._write_wav(target, np.zeros(100, dtype=np.float32), OUTPUT_RATE)

        assert seen == ["clip0.wav.partial"]
        assert target.is_file()
        assert not list(tmp_path.glob("*.partial"))

    def test_a_partial_left_by_a_kill_is_not_mistaken_for_a_rendered_clip(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "clip0.wav.partial").write_bytes(b"truncated")
        assert tts._written(tmp_path) == set()


class TestReportPath:
    def test_a_voice_is_named_under_its_own_root(self, tmp_path: Path) -> None:
        assert tts.report_path(tmp_path, "kab_male") == (
            tmp_path / "kab_male" / "kab_male.report.json"
        )
