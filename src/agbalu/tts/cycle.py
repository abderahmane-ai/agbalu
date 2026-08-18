"""Task 12.5: the instrument every Phase 12 claim is read through.

Cycle-CER synthesises held-out text, transcribes it with `Fadhma-300M` and measures the
character error against the text that was synthesised. A bare Cycle-CER is not a result:
Fadhma's own floor is 8.01% CER, so 9% is not "9% error" — most of it is the decoder's.
The harness therefore scores three conditions on one prompt set, one decoder and one
normalisation policy, and publishes a difference between two of them:

    floor     Fadhma on the prompt set's real human audio   this text's floor
    baseline  Fadhma on `mms-tts-kab`'s synthesis           what Matoub must beat
    cycle     Fadhma on Matoub's synthesis                  the measurement

`Report.delta` is a condition minus the floor. A system wins by holding the smaller
delta, never by holding a small CER.

The floor is also the positive control. It is the same decoder scoring the same kind of
audio the published 8.0136 was measured over, so a floor far from that number means the
decoder is not the published one and every delta beside it is uninterpretable.

Nothing here reads audio. It scores strings and does arithmetic over scores, which is
what lets one definition serve the container that spends the GPU and the laptop that
reads a result back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from agbalu.speech.metrics import cer, wer

if TYPE_CHECKING:
    from collections.abc import Container, Iterable, Mapping, Sequence


class CycleError(Exception):
    """A condition that cannot be scored, or a report that cannot be interpreted."""


FLOOR: Final = "floor_real_audio"
BASELINE: Final = "baseline_mms_tts_kab"
CYCLE: Final = "cycle_matoub"

PUBLISHED_CER: Final = 8.0136
"""Fadhma-300M over the whole 15,003-clip speaker-disjoint test split under the same
5-gram fusion, read from `data/processed/bench/fadhma-v1-test.json` — the artifact, not
a card."""

TOLERANCE: Final = 1.00
"""How far the floor may sit from `PUBLISHED_CER` before the run is uninterpretable.

Sized to separate a decoder defect from sampling rather than to be tight. The failures
this exists to catch — no n-gram loaded, the wrong checkpoint, a vocabulary whose blank
moved — cost whole points; task 12.1's 1,000-prompt subset of that same split measured
+0.3195, and that is the only evidence there is for what a correct floor looks like on a
subset.
"""


@dataclass(frozen=True, slots=True)
class Condition:
    """One scored decode, and the audio it was measured over.

    `cer_percent` and `wer_percent` are `None` exactly when nothing was scored. A subset
    that matched no reference has no error rate, and writing 0.0 there would turn "not
    measured" into a system that made no mistakes.
    """

    name: str
    utterances: int
    cer_percent: float | None
    wer_percent: float | None
    loss: float | None = None
    audio_seconds: float | None = None
    previews: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.utterances < 0:
            message = f"{self.name}: negative utterance count {self.utterances}"
            raise CycleError(message)
        measured = self.cer_percent is not None and self.wer_percent is not None
        if measured != (self.utterances > 0):
            message = (
                f"{self.name}: {self.utterances} utterances against "
                f"cer={self.cer_percent}, wer={self.wer_percent} — a rate exists if and "
                f"only if something was scored"
            )
            raise CycleError(message)
        for rate in (self.cer_percent, self.wer_percent):
            if rate is not None and rate < 0:
                message = f"{self.name}: negative error rate {rate}"
                raise CycleError(message)

    @classmethod
    def score(
        cls,
        name: str,
        pairs: Iterable[tuple[str, str]],
        *,
        loss: float | None = None,
        audio_seconds: float | None = None,
        previews: Sequence[tuple[str, str]] = (),
    ) -> Condition:
        """Score reference/hypothesis pairs through the project's one metric module.

        Never its own edit distance: two conditions reduced differently are not
        comparable, and `agbalu.speech.metrics` is where that policy is fixed.
        """
        kept = list(pairs)
        if not kept:
            return cls(name=name, utterances=0, cer_percent=None, wer_percent=None)
        references = [reference for reference, _ in kept]
        hypotheses = [hypothesis for _, hypothesis in kept]
        return cls(
            name=name,
            utterances=len(kept),
            cer_percent=round(100 * cer(references, hypotheses).rate, 4),
            wer_percent=round(100 * wer(references, hypotheses).rate, 4),
            loss=loss,
            audio_seconds=audio_seconds,
            previews=tuple(previews),
        )

    def as_dict(self) -> dict[str, object]:
        """The condition's own fields, without its name: a payload keys conditions by it."""
        payload: dict[str, object] = {}
        if self.loss is not None:
            payload["loss"] = self.loss
        if self.wer_percent is not None:
            payload["wer_percent"] = self.wer_percent
        if self.cer_percent is not None:
            payload["cer_percent"] = self.cer_percent
        payload["utterances"] = self.utterances
        if self.previews:
            payload["previews"] = [
                {"reference": reference, "hypothesis": hypothesis}
                for reference, hypothesis in self.previews
            ]
        if self.audio_seconds is not None:
            payload["audio_seconds"] = self.audio_seconds
        return payload


def restricted(name: str, pairs: Iterable[tuple[str, str]], targets: Container[str]) -> Condition:
    """The same decode, scored again over the subset of its references in `targets`.

    Keyed on the reference text because the prompt set holds no two identical targets — a
    repeat is rejected when it is built — and a decode comes back in duration order rather
    than in the order the prompts were written.
    """
    return Condition.score(name, [pair for pair in pairs if pair[0] in targets])


@dataclass(frozen=True, slots=True)
class Control:
    """Whether the floor reproduces Fadhma's published rate closely enough for the deltas
    measured beside it to mean anything."""

    measured: float
    published: float = PUBLISHED_CER
    tolerance: float = TOLERANCE

    @property
    def gap(self) -> float:
        """Signed, because which side the floor falls on says different things: above is
        harder text, below is text the decoder finds easier than its own test split."""
        return round(self.measured - self.published, 4)

    @property
    def holds(self) -> bool:
        return abs(self.measured - self.published) <= self.tolerance

    def as_dict(self) -> dict[str, object]:
        return {
            "floor_cer_percent": round(self.measured, 4),
            "published_cer_percent": self.published,
            "gap": self.gap,
            "tolerance": self.tolerance,
            "holds": self.holds,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """The conditions of one Cycle-CER run, and what they are allowed to claim."""

    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        names = [condition.name for condition in self.conditions]
        if len(set(names)) != len(names):
            message = f"two conditions share a name: {sorted(names)}"
            raise CycleError(message)
        if FLOOR not in names:
            message = (
                f"no {FLOOR!r} condition, so there is no floor to subtract and no control; "
                f"got {sorted(names)}"
            )
            raise CycleError(message)

    def condition(self, name: str) -> Condition:
        for condition in self.conditions:
            if condition.name == name:
                return condition
        message = f"no condition named {name!r}; got {sorted(c.name for c in self.conditions)}"
        raise CycleError(message)

    @property
    def floor(self) -> Condition:
        return self.condition(FLOOR)

    def delta(self, name: str) -> float:
        """This condition's CER minus the floor's — the only form Cycle-CER is quoted in."""
        if name == FLOOR:
            message = f"{FLOOR!r} is the floor; its delta against itself is not a result"
            raise CycleError(message)
        measured = self.condition(name).cer_percent
        floor = self.floor.cer_percent
        if measured is None or floor is None:
            message = f"{name!r} or the floor scored no utterances, so no delta exists"
            raise CycleError(message)
        return round(measured - floor, 4)

    @property
    def deltas(self) -> dict[str, float]:
        return {
            condition.name: self.delta(condition.name)
            for condition in self.conditions
            if condition.name != FLOOR
        }

    @property
    def control(self) -> Control:
        floor = self.floor.cer_percent
        if floor is None:
            message = "the floor scored no utterances, so the control cannot be evaluated"
            raise CycleError(message)
        return Control(measured=floor)

    def beats(self, name: str, other: str) -> bool:
        """Whether `name` is the better system: the smaller delta wins.

        Against the delta and not the CER, because two systems whose audio the decoder
        finds equally hard differ only in what they added to its floor.
        """
        return self.delta(name) < self.delta(other)

    def as_dict(self) -> dict[str, object]:
        return {
            "conditions": {condition.name: condition.as_dict() for condition in self.conditions},
            "deltas": self.deltas,
            "control": self.control.as_dict(),
        }


def _previews(raw: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(
        (str(item["reference"]), str(item["hypothesis"]))
        for item in raw
        if isinstance(item, dict) and "reference" in item and "hypothesis" in item
    )


def _number(raw: object) -> float | None:
    return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None


def read_result(payload: Mapping[str, object]) -> Report:
    """A written Cycle-CER payload, read back into the conditions it recorded.

    Every delta is recomputed from the conditions rather than trusted, and a stored one
    that disagrees raises: a payload carrying both a rate and a difference of rates is
    two copies of the same fact, and this is the check that keeps them one.
    """
    raw = payload.get("conditions")
    if not isinstance(raw, dict) or not raw:
        message = "payload carries no `conditions` block, so it is not a Cycle-CER result"
        raise CycleError(message)

    conditions: list[Condition] = []
    for name, fields in raw.items():
        if not isinstance(fields, dict):
            message = f"condition {name!r} is {type(fields).__name__}, not an object"
            raise CycleError(message)
        utterances = fields.get("utterances")
        if not isinstance(utterances, int) or isinstance(utterances, bool):
            message = f"condition {name!r} carries no integer `utterances`"
            raise CycleError(message)
        conditions.append(
            Condition(
                name=str(name),
                utterances=utterances,
                cer_percent=_number(fields.get("cer_percent")),
                wer_percent=_number(fields.get("wer_percent")),
                loss=_number(fields.get("loss")),
                audio_seconds=_number(fields.get("audio_seconds")),
                previews=_previews(fields.get("previews")),
            )
        )

    report = Report(tuple(conditions))
    stored = payload.get("deltas")
    if isinstance(stored, dict):
        for name, value in stored.items():
            recomputed = report.delta(str(name))
            if _number(value) != recomputed:
                message = (
                    f"payload records delta {value} for {name!r} where its own conditions "
                    f"give {recomputed}"
                )
                raise CycleError(message)
    return report
