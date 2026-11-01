"""Measure a bounded restoration candidate before authorizing a full schedule."""

import time
from collections.abc import Mapping

import numpy as np

from agbalu.encoder.corpus import read_json, write_json
from agbalu.encoder.corpus import tokenizer as load_tokenizer
from agbalu.restoration import calibration, checkpoint
from agbalu.restoration.configuration import Settings
from agbalu.restoration.evaluation import evaluate
from agbalu.restoration.inference import Inference
from agbalu.restoration.training import Trainer, smoke_identity

MINIMUM_PILOT_F1 = 0.30


def previews(
    trainer: Trainer, correction: calibration.Calibration | None = None
) -> list[dict[str, object]]:
    tokenizer = load_tokenizer(trainer.locations.tokenizer)
    inference = Inference(trainer.model, tokenizer, trainer.settings, correction)
    order = np.random.default_rng(1618).permutation(len(trainer.dev))[:4]
    rows: list[dict[str, object]] = []
    for index in order:
        example = trainer.dev[int(index)]
        text = str(tokenizer.decode(example.ids.tolist()))
        prediction = inference.predict(text)
        rows.append(
            {
                "window_index": int(index),
                "input": text,
                "restored": prediction.text,
                "gold_marks": example.marks.tolist(),
                "gold_components": example.casing.tolist(),
            }
        )
    return rows


def run(trainer: Trainer) -> dict[str, object]:
    started = time.monotonic()
    baseline = evaluate(
        trainer.model,
        trainer.dev,
        trainer.device,
        trainer.settings.validation_windows,
        baseline=True,
    )
    write_json(trainer.directory / "baseline.stats.json", baseline)
    training = trainer.run(stop_after=trainer.settings.pilot_updates, smoke=True)
    diagnostic = trainer.validate()
    offline = evaluate(trainer.model, trainer.dev, trainer.device, 512)
    streaming = evaluate(trainer.model, trainer.dev_stream, trainer.device, 512)
    write_json(trainer.directory / "predictions.stats.json", {"examples": previews(trainer)})
    result: dict[str, object] = {
        "kind": "bounded_candidate_measurement",
        "identity": smoke_identity(trainer.settings, trainer.locations),
        "settings": trainer.settings.model_dump(),
        "step": trainer.state.step,
        "tokens_total": trainer.state.tokens,
        "supervised_case_components_total": trainer.state.words,
        "training": training,
        "diagnostic": diagnostic,
        "offline": offline,
        "streaming": streaming,
        "baseline": baseline,
        "passed": False,
        "selection_required": True,
        "seconds": time.monotonic() - started,
    }
    write_json(trainer.directory / "pilot.stats.json", result)
    return result


def assess(trainer: Trainer) -> dict[str, object]:
    saved = checkpoint.latest(trainer.directory)
    if saved is None:
        message = f"{trainer.directory}: assessment requires a completed candidate"
        raise ValueError(message)
    trainer.state = checkpoint.load(saved, trainer.model, trainer.optimizer, trainer.identity)
    fitted, calibration_report = calibration.fit(trainer.model, trainer.dev, trainer.device)
    variants: dict[str, object] = {}
    for name, transform in (("raw", None), ("calibrated", fitted.apply)):
        variants[name] = {
            "offline": evaluate(
                trainer.model,
                trainer.dev,
                trainer.device,
                512,
                fold="selection",
                calibration=transform,
            ),
            "streaming": evaluate(
                trainer.model,
                trainer.dev_stream,
                trainer.device,
                512,
                fold="selection",
                calibration=transform,
            ),
        }
    result: dict[str, object] = {
        "identity": smoke_identity(trainer.settings, trainer.locations),
        "config_hash": trainer.identity,
        "settings": trainer.settings.model_dump(),
        "step": trainer.state.step,
        "tokens": trainer.state.tokens,
        "snapshot": saved.name,
        "variants": variants,
        "calibration_fit": calibration_report,
        "baseline": evaluate(
            trainer.model, trainer.dev, trainer.device, 512, fold="selection", baseline=True
        ),
    }
    write_json(trainer.directory / "assessment.stats.json", result)
    write_json(trainer.directory / "assessments" / f"{time.time_ns()}.stats.json", result)
    write_json(
        trainer.directory / "calibrated-predictions.stats.json",
        {"examples": previews(trainer, fitted)},
    )
    return result


def candidate_metrics(
    offline: Mapping[str, object], streaming: Mapping[str, object], baseline: Mapping[str, object]
) -> tuple[float, bool]:
    p, s, c, false, exact, floor = (
        float(str(value))
        for value in (
            offline["punctuation_macro_f1"],
            streaming["punctuation_macro_f1"],
            offline["casing_macro_f1"],
            offline["false_insertions_per_1000_words"],
            offline["exact_word_casing_accuracy"],
            baseline["exact_word_casing_accuracy"],
        )
    )
    return 0.4 * p + 0.3 * s + 0.3 * c, (
        p >= MINIMUM_PILOT_F1
        and s >= MINIMUM_PILOT_F1
        and false <= calibration.FALSE_INSERTION_BUDGET
        and exact > floor
    )


def select(trainer: Trainer) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    exposures: set[tuple[object, object]] = set()
    for name in ("restoration-linear-joint", "restoration-linear-balanced"):
        report = read_json(trainer.locations.runs / name / "assessment.stats.json")
        settings = Settings.model_validate(report["settings"])
        if report["identity"] != smoke_identity(settings, trainer.locations):
            message = f"{name}: assessment does not match current code/data/environment"
            raise ValueError(message)
        exposures.add((report["step"], report["tokens"]))
        variants, baseline = report["variants"], report["baseline"]
        if not isinstance(variants, dict) or not isinstance(baseline, dict):
            message = f"{name}: malformed assessment metrics"
            raise TypeError(message)
        for variant, metrics in variants.items():
            if not isinstance(metrics, dict):
                message = f"{name}/{variant}: malformed candidate metrics"
                raise TypeError(message)
            offline, streaming = metrics["offline"], metrics["streaming"]
            if not isinstance(offline, dict) or not isinstance(streaming, dict):
                message = f"{name}/{variant}: missing mode metrics"
                raise TypeError(message)
            score, is_eligible = candidate_metrics(offline, streaming, baseline)
            candidates.append(
                {
                    "run": name,
                    "variant": variant,
                    "settings": settings.model_dump(),
                    "score": score,
                    "eligible": is_eligible,
                    "offline": offline,
                    "streaming": streaming,
                    "calibration_fit": report["calibration_fit"],
                }
            )
    if len(exposures) != 1:
        message = f"candidate exposure differs: {exposures}"
        raise ValueError(message)
    eligible = [candidate for candidate in candidates if candidate["eligible"] is True]
    if not eligible:
        write_json(
            trainer.locations.data / "selection.stats.json",
            {"passed": False, "candidates": candidates},
        )
        message = "no candidate passed punctuation, insertion and exact-casing guards"
        raise ValueError(message)
    chosen = max(eligible, key=lambda row: float(str(row["score"])))
    selected = Settings.model_validate(chosen["settings"])
    result = {
        "passed": True,
        "identity": smoke_identity(selected, trainer.locations),
        "chosen": chosen,
        "candidates": candidates,
        "training_trials": 2,
        "selection_trials": len(candidates),
    }
    directory = trainer.locations.runs / str(chosen["run"])
    assessment = read_json(directory / "assessment.stats.json")
    fit_report = chosen["calibration_fit"]
    correction = (
        fit_report["calibration"]
        if chosen["variant"] == "calibrated" and isinstance(fit_report, dict)
        else {}
    )
    snapshot = checkpoint.latest(directory)
    if snapshot is None or snapshot.name != assessment["snapshot"]:
        message = f"{directory}: weights changed after the assessment"
        raise ValueError(message)
    checkpoint.promote(
        directory,
        snapshot,
        {"score": chosen["score"], "calibration": correction, "selection": result},
    )
    write_json(trainer.locations.data / "selection.stats.json", result)
    return result
