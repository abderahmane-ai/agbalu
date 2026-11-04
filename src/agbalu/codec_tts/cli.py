"""Run codec speech preparation, adaptation and reference synthesis."""

from __future__ import annotations

import argparse
import gc
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from agbalu.codec_tts.adaptation import inject_adapter, restore_adapter
from agbalu.codec_tts.artifacts import atomic_binary, digest_file, provenance, write_json
from agbalu.codec_tts.audio import decode_audio, prepare_audio, prepare_pairs, source_files
from agbalu.codec_tts.backbone import Generator, load_generator
from agbalu.codec_tts.checkpoint import (
    Pointer,
    Snapshot,
    snapshot_path,
    tensor_mapping,
    verify_snapshot,
)
from agbalu.codec_tts.codec import load_codec
from agbalu.codec_tts.configuration import (
    DEFAULT_RECIPE,
    Geometry,
    Recipe,
    is_run_name,
    read_recipe,
)
from agbalu.codec_tts.dataset import SpeechDataset, TextTokenizer, cache_codes
from agbalu.codec_tts.evaluation import (
    ListenerReview,
    SpeechObservation,
    measure_speech,
    selection_report,
)
from agbalu.codec_tts.inference import Sampling, generate_codes
from agbalu.codec_tts.language import observe_languages
from agbalu.codec_tts.layout import prompt_layout
from agbalu.codec_tts.smoke import PARTS, SMOKE_UPDATES, admit_training, run_smoke
from agbalu.codec_tts.training import TrainingRun
from agbalu.codec_tts.waveforms import resample_waveform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="inventory",
        choices=(
            "environment",
            "fetch",
            "inventory",
            "prepare",
            "language",
            "partition",
            "encode",
            "smoke",
            "train",
            "evaluate",
            "select",
            "synthesize",
            "export",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--run")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--part", choices=PARTS, default="throughput")
    parser.add_argument(
        "--suite", choices=("validation", "release", "baseline"), default="validation"
    )
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--role", choices=("latest", "best_ce", "best_speech"), default="best_ce")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-text-file", type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--review-file", type=Path)
    return parser


def run_directory(value: str | None) -> Path:
    if value is None or not is_run_name(value):
        message = f"a simple --run name is required, got {value!r}"
        raise ValueError(message)
    return Path("artifacts/runs") / value


def native_model(recipe: Recipe, device: torch.device) -> Generator:
    path = recipe.base_directory / "geometry.json"
    geometry = Geometry.model_validate_json(path.read_bytes()) if path.is_file() else Geometry()
    model, _ = load_generator(recipe.base_directory / "model.safetensors", geometry, device)
    inject_adapter(model)
    return model


def restore_selected(model: Generator, recipe: Recipe, root: Path, role: str) -> Path:
    pointer = Pointer.model_validate_json((root / f"{role}.json").read_bytes())
    selected = snapshot_path(root, pointer.snapshot)
    manifest = Snapshot.model_validate_json((selected / "snapshot.stats.json").read_bytes())
    verify_snapshot(selected, manifest.identity)
    identity = json.loads((root / "identity.json").read_bytes())
    stored = Recipe.model_validate(identity["recipe"])
    if identity.get("identity") != manifest.identity or any(
        getattr(stored, name) != getattr(recipe, name)
        for name in ("base_id", "base_revision", "codec_id", "codec_revision", "rank", "alpha")
    ):
        message = f"selected adapter/base identity mismatch at {selected}"
        raise ValueError(message)
    payload: object = torch.load(selected / "training.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        message = f"invalid adapter payload at {selected}"
        raise TypeError(message)
    restore_adapter(model, tensor_mapping(payload.get("adapter")))
    return selected


def speech_evaluation(
    model: Generator, recipe: Recipe, dataset: SpeechDataset, output: Path
) -> dict[str, object]:
    codec, _ = load_codec(
        recipe.base_directory / "model.safetensors",
        recipe.codec_configuration,
        model.audio_embedding.weight.device,
    )
    try:
        return measure_speech(model, dataset, codec, output)
    finally:
        del codec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def environment_report() -> dict[str, object]:
    report: dict[str, object] = {
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "torchaudio",
                "transformers",
                "numpy",
                "scipy",
                "soundfile",
                "pyarrow",
                "pydantic",
            )
        }
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report.update(gpu=torch.cuda.get_device_name(0), free_bytes=free, total_bytes=total)
    return report


def select_review(
    root: Path, selected: Path, evaluation: Path, review_file: Path
) -> dict[str, object]:
    report = json.loads(evaluation.read_bytes())
    review = ListenerReview.model_validate_json(review_file.read_bytes())
    if review.evaluation_sha256 != digest_file(evaluation) or report.get(
        "checkpoint_sha256"
    ) != digest_file(selected / "training.pt"):
        message = f"review/checkpoint evidence mismatch at {evaluation}"
        raise ValueError(message)
    observations = tuple(SpeechObservation.model_validate(row) for row in report["observations"])
    selection = selection_report(observations, review)
    if selection["eligible"]:
        write_json(
            root / "best_speech.json",
            Pointer(
                snapshot=selected.name,
                score=float(str(selection["cer_excess"])),
                evidence_sha256=digest_file(evaluation),
            ).model_dump(),
        )
    write_json(
        root / "selection.stats.json",
        {
            **provenance(str(report["config_hash"])),
            **selection,
            "evaluation_sha256": digest_file(evaluation),
            "review_sha256": digest_file(review_file),
            "snapshot": selected.name,
        },
    )
    return selection


def synthesize(
    args: argparse.Namespace, recipe: Recipe, model: Generator, tokenizer: TextTokenizer
) -> dict[str, object]:
    if args.text_file is None:
        message = "synthesis requires --text-file"
        raise ValueError(message)
    text = Path(args.text_file).read_text(encoding="utf-8")
    if not text.strip():
        message = f"empty synthesis text at {args.text_file}"
        raise ValueError(message)
    codec, _ = load_codec(
        recipe.base_directory / "model.safetensors",
        recipe.codec_configuration,
        model.audio_embedding.weight.device,
    )
    reference = None
    reference_text: list[int] = []
    if args.reference is not None:
        samples, rate_hz = decode_audio(Path(args.reference).read_bytes())
        reference = codec.encode(resample_waveform(samples, rate_hz))
        if args.reference_text_file is not None:
            reference_text = tokenizer.encode(
                Path(args.reference_text_file).read_text(encoding="utf-8"), add_special_tokens=False
            )
    prompt = prompt_layout(
        tokenizer.get_added_vocab(),
        tokenizer.encode(text, add_special_tokens=False),
        reference,
        reference_text,
    )
    codes, trace = generate_codes(model, prompt, Sampling())
    waveform = np.asarray(codec.decode(codes).numpy(), dtype=np.float32)
    output = args.output or Path("artifacts/codec_tts/synthesis.wav")
    with atomic_binary(output) as stream:
        sf.write(stream, waveform, 24000, format="WAV", subtype="FLOAT")
    report = {
        **provenance(recipe.digest),
        **trace,
        "waveform_sha256": digest_file(output),
        "output": str(output),
        "reference_sha256": digest_file(args.reference) if args.reference else None,
        "reference_transcript_provided": bool(reference_text),
    }
    write_json(output.with_suffix(".stats.json"), report)
    return report


def execute_model_action(
    args: argparse.Namespace, recipe: Recipe, device: torch.device
) -> dict[str, object]:
    tokenizer: TextTokenizer = AutoTokenizer.from_pretrained(
        recipe.base_directory, local_files_only=True
    )
    if args.action == "encode":
        codec, _ = load_codec(
            recipe.base_directory / "model.safetensors", recipe.codec_configuration, device
        )
        return cache_codes(recipe, codec, tokenizer)
    model = native_model(recipe, device)
    if args.action == "synthesize":
        if args.run:
            restore_selected(model, recipe, run_directory(args.run), args.role)
        return synthesize(args, recipe, model, tokenizer)
    root = run_directory(args.run)
    if args.action in {"select", "export"}:
        selected = restore_selected(model, recipe, root, args.role)
        if args.action == "select":
            if args.evaluation is None or args.review_file is None:
                message = "selection requires --evaluation and --review-file"
                raise ValueError(message)
            return select_review(root, selected, args.evaluation, args.review_file)
        from tools.export_checkpoint import export_codec_tts

        if args.role != "best_speech":
            message = "export requires --role best_speech and qualified speech review"
            raise ValueError(message)
        return export_codec_tts(model, recipe, args.output or root / "export")
    validation = SpeechDataset(
        recipe, tokenizer.get_added_vocab(), "test" if args.suite == "release" else "validation"
    )
    if args.action == "evaluate":
        evaluated_snapshot = (
            restore_selected(model, recipe, root, args.role) if args.suite != "baseline" else None
        )
        report = speech_evaluation(model, recipe, validation, root / "evaluation" / args.suite)
        report["checkpoint_sha256"] = (
            digest_file(evaluated_snapshot / "training.pt") if evaluated_snapshot else None
        )
        write_json(root / "evaluation" / args.suite / "evaluation.stats.json", report)
        return report
    training = SpeechDataset(recipe, tokenizer.get_added_vocab(), "train")
    run = TrainingRun(
        model,
        recipe,
        training,
        validation,
        root,
        lambda generator, output: speech_evaluation(generator, recipe, validation, output),
    )
    if args.action == "smoke":
        if args.steps not in (None, SMOKE_UPDATES):
            message = "the codec qualification smoke requires exactly twenty updates"
            raise ValueError(message)
        return run_smoke(run, args.part)
    admit_training(run)
    state = run.train(stop_after=args.stop_after)
    return {"step": state.step, "signal": run.signal_number}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps is not None and args.action != "smoke":
        message = "--steps is only valid for smoke; train uses the prepared exposure schedule"
        raise ValueError(message)
    if args.stop_after is not None and (args.action != "train" or args.stop_after < 0):
        message = "--stop-after must be nonnegative and is only valid for train"
        raise ValueError(message)
    recipe = read_recipe(args.config)
    torch.manual_seed(recipe.seed)
    device = torch.device(args.device)
    if args.action == "environment":
        report: dict[str, object] = environment_report()
    elif args.action == "fetch":
        path = snapshot_download(
            recipe.base_id,
            revision=recipe.base_revision,
            local_dir=recipe.base_directory,
            max_workers=2,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE", "README.md"],
        )
        report = {"base_directory": str(path), "revision": recipe.base_revision}
    elif args.action == "inventory":
        report = {"shards": [str(path) for path in source_files(recipe)]}
    elif args.action == "prepare":
        report = prepare_audio(recipe)
    elif args.action == "partition":
        report = prepare_pairs(recipe)
    elif args.action == "language":
        report = observe_languages(recipe, device)
    else:
        report = execute_model_action(args, recipe, device)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    number = report.get("signal")
    return 128 + number if isinstance(number, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
