"""Restore complete documents and stable streamed words without dropping tails."""

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from numpy.typing import NDArray

from agbalu.encoder.checkpoint import verify_snapshot
from agbalu.encoder.corpus import read_json
from agbalu.encoder.corpus import tokenizer as load_tokenizer
from agbalu.restoration.alignment import EncodedWord, align_cases, encode_word
from agbalu.restoration.annotations import IGNORE, MARKS, PRIMARY, CasePrediction, annotate
from agbalu.restoration.calibration import Calibration
from agbalu.restoration.configuration import Locations, Settings
from agbalu.restoration.dataset import Example, collate
from agbalu.restoration.decoding import StructureState, joint_logits, render
from agbalu.restoration.model import Restorer
from agbalu.restoration.windows import Window, tile


@dataclass(frozen=True)
class Prediction:
    words: tuple[str, ...]
    marks: tuple[int, ...]
    casing: tuple[CasePrediction, ...]
    structures: tuple[str, ...]

    @property
    def text(self) -> str:
        return render(self.words, self.marks, self.casing, self.structures)


@dataclass(frozen=True)
class EncodedText:
    ids: list[int]
    offsets: list[int]
    case_offsets: list[int]
    case_rows: NDArray[np.int64]

    def predictions(
        self, labels: NDArray[np.int64], first: int, last: int, origin_word: int = 0
    ) -> tuple[CasePrediction, ...]:
        origin = self.case_offsets[origin_word]
        result: list[CasePrediction] = []
        for word in range(first, last):
            values = tuple(
                int(value)
                for value in labels[
                    self.case_offsets[word] - origin : self.case_offsets[word + 1] - origin
                ]
            )
            result.append(values[0] if len(values) == 1 else values)
        return tuple(result)


def encode(words: tuple[str, ...], tokenizer: spm.SentencePieceProcessor) -> EncodedText:
    ids: list[int] = []
    offsets, case_offsets = [0], [0]
    rows: list[tuple[int, ...]] = []
    cache: dict[str, EncodedWord] = {}
    for index, word in enumerate(words):
        if word not in cache:
            cache[word] = encode_word(word, tokenizer)
        encoded = cache[word]
        for first, last, start, end, length, _ in align_cases(word, encoded):
            rows.append((len(ids) + first, len(ids) + last, index, start, end, length))
        ids.extend(encoded.ids)
        offsets.append(len(ids))
        case_offsets.append(len(rows))
    return EncodedText(ids, offsets, case_offsets, np.asarray(rows, np.int64).reshape(-1, 6))


def example(
    words: tuple[str, ...],
    encoded: EncodedText,
    window: Window,
    *,
    actual_start: bool,
    actual_end: bool,
) -> Example:
    ids, offsets = encoded.ids, encoded.offsets
    begin, end = offsets[window.start], offsets[window.stop]
    spans = np.asarray(
        list(
            zip(
                offsets[window.start : window.stop],
                offsets[window.start + 1 : window.stop + 1],
                strict=True,
            )
        ),
        np.int64,
    )
    count = window.stop - window.start
    case_first, case_last = encoded.case_offsets[window.start], encoded.case_offsets[window.stop]
    case_rows = encoded.case_rows[case_first:case_last]
    case_features = np.stack(
        (case_rows[:, 3], case_rows[:, 4], case_rows[:, 4] - case_rows[:, 3]), 1
    )
    return Example(
        np.asarray([2, *ids[begin:end], 3], np.int64),
        spans - begin + 1,
        np.full(count + 1, IGNORE, np.int64),
        np.full(case_last - case_first, IGNORE, np.int64),
        np.full(count + 1, IGNORE, np.int64),
        (actual_start and window.start == 0, actual_end and window.stop == len(words)),
        owned_words=window.core_stop - window.core_start,
        case_spans=case_rows[:, :2] - begin + 1,
        case_words=case_rows[:, 2] - window.start,
        case_features=np.asarray(case_features / case_rows[:, 5:6], np.float32),
    )


class Inference:
    def __init__(
        self,
        model: Restorer,
        tokenizer: spm.SentencePieceProcessor,
        settings: Settings,
        calibration: Calibration | None = None,
    ) -> None:
        self.model, self.tokenizer, self.settings = model, tokenizer, settings
        self.model.eval()
        self.device = next(model.parameters()).device
        self.calibration = calibration or Calibration()

    @classmethod
    def load(cls, run: Path, device: torch.device) -> "Inference":
        config = read_json(run / "config.json")
        settings = Settings.model_validate(config["settings"])
        locations = Locations.model_validate(config["locations"])
        templates = config["templates"]
        if not isinstance(templates, list):
            msg = f"{run}: templates are missing"
            raise TypeError(msg)
        pointer = read_json(run / "best_deployment.json")
        snapshot = run / str(pointer["snapshot"])
        verify_snapshot(snapshot)
        model = Restorer.from_settings(
            settings.model_copy(
                update={
                    "geometry": settings.geometry.model_copy(
                        update={"backend": "flex" if device.type == "cuda" else "reference"}
                    )
                }
            ),
            tuple(str(value) for value in templates),
        ).to(device)
        payload = torch.load(snapshot / "training.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"], strict=True)
        tokenizer = load_tokenizer(locations.tokenizer)
        metrics = pointer.get("metrics", {})
        calibration = Calibration.model_validate(
            metrics.get("calibration", {}) if isinstance(metrics, dict) else {}
        )
        return cls(model, tokenizer, settings, calibration)

    @torch.inference_mode()
    def window(self, row: Example) -> tuple[NDArray[np.float32], ...]:
        batch = collate([row], self.settings.geometry.context_tokens).to(self.device)
        with torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            scores = self.calibration.apply(self.model.forward(batch))
        count = len(row.spans)
        return (
            scores.presence[0, : count + 1].float().cpu().numpy(),
            scores.punctuation[0, : count + 1].float().cpu().numpy(),
            scores.casing[0, : len(row.casing)].float().cpu().numpy(),
            scores.structure[0, : count + 1].float().cpu().numpy(),
        )

    def predict(self, text: str) -> Prediction:
        words = annotate(text).words
        if not words:
            return Prediction((), (0,), (), (PRIMARY,))
        encoded = encode(words, self.tokenizer)
        offsets = encoded.offsets
        windows = tile(
            offsets,
            self.settings.geometry.context_tokens - 2,
            max(1, self.settings.core_tokens - 128),
            self.settings.left_tokens,
        )
        presence = np.zeros(len(words) + 1, np.float32)
        types = np.zeros((len(words) + 1, 8), np.float32)
        cases = np.zeros((encoded.case_offsets[-1], 3), np.float32)
        case_weights = np.zeros(encoded.case_offsets[-1], np.float32)
        structure = np.zeros((len(words) + 1, len(self.model.templates)), np.float32)
        gap_weights = np.zeros(len(words) + 1, np.float32)
        word_weights = np.zeros(len(words), np.float32)
        for window in windows:
            row = example(words, encoded, window, actual_start=True, actual_end=True)
            p, t, c, s = self.window(row)
            extended = bisect.bisect_right(offsets, offsets[window.core_stop] + 128) - 1
            stop = min(len(words), window.stop - int(window.stop < len(words)), extended)
            for index in range(window.core_start, stop):
                local = index - window.start
                weight = float(
                    1
                    + min(
                        offsets[index] - offsets[window.start],
                        offsets[window.stop] - offsets[index + 1],
                    )
                )
                case_first, case_last = encoded.case_offsets[index], encoded.case_offsets[index + 1]
                origin = encoded.case_offsets[window.start]
                cases[case_first:case_last] += c[case_first - origin : case_last - origin] * weight
                case_weights[case_first:case_last] += weight
                word_weights[index] += weight
                gap = index + 1
                presence[gap] += p[local + 1] * weight
                types[gap] += t[local + 1] * weight
                structure[gap] += s[local + 1] * weight
                gap_weights[gap] += weight
            if window.start == 0:
                presence[0], types[0], structure[0], gap_weights[0] = p[0], t[0], s[0], 1
        if np.any(word_weights == 0) or np.any(gap_weights == 0):
            msg = "document restoration left an uncovered word or gap"
            raise ValueError(msg)
        presence /= gap_weights
        types /= gap_weights[:, None]
        structure /= gap_weights[:, None]
        cases /= case_weights[:, None]
        return Prediction(
            words,
            tuple(int(value) for value in joint_logits(presence, types).argmax(-1)),
            encoded.predictions(cases.argmax(-1), 0, len(words)),
            tuple(self.model.templates[int(value)] for value in structure.argmax(-1)),
        )


class Stream:
    """Commit only stable words after a complete target chunk and its lookahead arrive."""

    def __init__(self, inference: Inference) -> None:
        self.inference = inference
        self.words: tuple[str, ...] = ()
        self.committed = 0
        self.structure = StructureState()
        self.closed = False

    def push(self, stable_text: str, *, final: bool = False) -> str:
        if self.closed:
            msg = "stream is already closed"
            raise ValueError(msg)
        words = annotate(stable_text).words
        if words[: self.committed] != self.words[: self.committed]:
            msg = "ASR revised words that were already committed"
            raise ValueError(msg)
        self.words = words
        output: list[str] = []
        encoded = encode(words, self.inference.tokenizer)
        offsets = encoded.offsets
        settings = self.inference.settings
        while self.committed < len(words):
            target = offsets[self.committed] + settings.streaming_core_tokens
            core_stop = max(self.committed + 1, bisect.bisect_right(offsets, target) - 1)
            core_stop = min(core_stop, len(words))
            future = offsets[core_stop] + settings.lookahead_tokens
            if not final and offsets[-1] < future:
                break
            stop = min(len(words), bisect.bisect_right(offsets, future) - 1)
            stop = max(core_stop, stop)
            start = bisect.bisect_left(
                offsets, max(0, offsets[stop] - settings.geometry.context_tokens + 2)
            )
            if start > self.committed:
                msg = "stream word exceeds context capacity"
                raise ValueError(msg)
            window = Window(start, stop, self.committed, core_stop)
            row = example(words, encoded, window, actual_start=True, actual_end=final)
            presence, types, cases, structures = self.inference.window(row)
            first, last = self.committed - start, core_stop - start
            marks = joint_logits(presence, types).argmax(-1)
            templates = tuple(
                self.inference.model.templates[int(value)]
                for value in structures[first : last + 1].argmax(-1)
            )
            # A committed gap is emitted once, immediately before its following word.
            emitted_marks = (*tuple(int(value) for value in marks[first:last]), 0)
            emitted_structures = (*templates[:-1], PRIMARY)
            output.append(
                render(
                    words[self.committed : core_stop],
                    emitted_marks,
                    encoded.predictions(cases.argmax(-1), self.committed, core_stop, start),
                    emitted_structures,
                    self.structure,
                    continuing=self.committed > 0,
                )
            )
            self.committed = core_stop
            if final and core_stop == len(words):
                terminal = templates[-1].replace(PRIMARY, MARKS[int(marks[last])])
                output.append(self.structure.constrain(terminal))
        self.closed = final
        return "".join(output)
