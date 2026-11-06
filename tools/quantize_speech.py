"""Qualify speech quantization with matched inputs and protected audio weights."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from pydantic import ConfigDict, Field
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForTextToWaveform, AutoTokenizer, BitsAndBytesConfig

from agbalu.codec_tts.artifacts import digest_file, provenance, write_json
from agbalu.codec_tts.backbone import load_generator, map_weight
from agbalu.codec_tts.codec import Codec, load_codec
from agbalu.codec_tts.configuration import Geometry, Specification
from agbalu.codec_tts.inference import Decoding, GenerationError, Sampling, generate_codes
from agbalu.codec_tts.layout import prompt_layout
from agbalu.codec_tts.release import remote_sources
from agbalu.codec_tts.synthesis import SpeechRequest, read_waveform, save_waveform
from agbalu.codec_tts.verification import INSTRUMENTS, Quality, SpeechVerifier

ROOT = Path("artifacts/runs/matoub-quantization")
STAGE = Path("artifacts/runs/codec-tts-r32-stage/model")
BUILD = ROOT / "audiocpp-build/bin"
SEEDS = (20261102, 20261103)


class SpeechCase(Specification):
    name: str
    text: str
    reference: Path
    reference_text: str


class Trial(Specification):
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    case: str
    seed: int
    output: Path
    seconds: float | None = Field(default=None, gt=0)
    duration_s: float | None = Field(default=None, gt=0)
    quality: Quality | None = None
    error: str | None = None
    sha256: str | None = None


def summarize(trials: Sequence[Trial]) -> dict[str, object]:
    good = [t for t in trials if t.error is None and t.quality is not None]
    qualities = [t.quality for t in good if t.quality is not None]
    seconds = sum(t.seconds for t in good if t.seconds is not None)
    duration = sum(t.duration_s for t in good if t.duration_s is not None)
    characters = sum(q.characters for q in qualities)
    return {
        "requests": len(trials),
        "completed": len(good),
        "failed": len(trials) - len(good),
        "characters": characters,
        "character_errors": sum(q.character_errors for q in qualities),
        "cer": sum(q.character_errors for q in qualities) / characters if characters else None,
        "voice_cosine_mean": sum(q.voice_cosine for q in qualities) / len(qualities)
        if qualities
        else None,
        "predicted_mos_mean": sum(q.predicted_mos for q in qualities) / len(qualities)
        if qualities
        else None,
        "rtf_audio_weighted": seconds / duration if duration else None,
        "synthesis_s": seconds if good else None,
        "audio_s": duration if good else None,
    }


def validate_quantized_modules(names: Sequence[str], expected: int) -> None:
    invalid = [n for n in names if not n.startswith("generator.")]
    if invalid or len(names) != expected:
        message = (
            f"quantization preservation failed: layers={len(names)}/{expected}, invalid={invalid}"
        )
        raise ValueError(message)


def validate_codec_types(types: Mapping[str, str]) -> int:
    codec = {
        name: kind
        for name, kind in types.items()
        if "tied.embedding.modality_embeddings.0.model." in name
    }
    if not codec:
        message = "GGUF codec namespace is missing"
        raise ValueError(message)
    for name, kind in codec.items():
        if kind != "F32":
            message = f"codec tensor was quantized: {name}, type={kind}"
            raise ValueError(message)
    return len(codec)


def metric_seconds(log: str, request_id: str) -> float | None:
    pattern = rf"^metrics\[{re.escape(request_id)}\]\.wall_ms=([^\s]+)$"
    match = re.search(pattern, log, re.MULTILINE)
    if match is None:
        return None
    try:
        value = float(match[1]) / 1000
    except ValueError as error:
        message = f"invalid native timing for request={request_id}: {match[1]!r}"
        raise ValueError(message) from error
    if not np.isfinite(value) or value <= 0:
        message = f"invalid native timing for request={request_id}: {match[1]!r}"
        raise ValueError(message)
    return value


def read_cases() -> tuple[SpeechCase, ...]:
    panel = Path("artifacts/runs/teacher-inference-audit/final-qualification")
    first = next(iter(sorted(panel.glob("9175*.stats.json"))))
    request = SpeechRequest.model_validate(json.loads(first.read_bytes())["request"])
    hard = SpeechRequest.model_validate(
        json.loads((panel / "english-reference-hard.stats.json").read_bytes())["request"]
    )
    cases = [
        SpeechCase(
            name="native-short",
            text=request.text,
            reference=request.reference,
            reference_text=request.reference_text,
        ),
        SpeechCase(
            name="native-hard",
            text=hard.text,
            reference=request.reference,
            reference_text=request.reference_text,
        ),
    ]
    for name, directory in [("english-male", "trump"), ("english-female", "sabrina")]:
        root = Path(f"artifacts/runs/{directory}-kabyle-listening")
        cases.append(
            SpeechCase(
                name=name,
                text=(root / "target.txt").read_text().strip(),
                reference=root / "reference.wav",
                reference_text=(root / "reference.txt").read_text().strip(),
            )
        )
    for case in cases:
        if not case.reference.is_file():
            message = f"missing benchmark reference: {case.reference}, case={case.name}"
            raise FileNotFoundError(message)
    return tuple(cases)


class DeviceMemory:
    def __init__(self) -> None:
        torch.cuda.init()
        samples = [torch.cuda.mem_get_info()[0] for _ in range(5)]
        self.idle_free = max(samples)
        self.noise_bytes = max(samples) - min(samples)
        self.minimum_free = min(samples)
        self.stop = threading.Event()
        self.failure: RuntimeError | None = None
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        try:
            while not self.stop.wait(0.02):
                self.minimum_free = min(self.minimum_free, torch.cuda.mem_get_info()[0])
        except RuntimeError as error:
            self.failure = error
            self.stop.set()

    def __enter__(self) -> DeviceMemory:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            message = "device memory sampler did not stop"
            raise RuntimeError(message)
        if self.failure is not None:
            raise self.failure

    def report(self) -> dict[str, object]:
        return {
            "peak_device_delta_bytes": self.idle_free - self.minimum_free,
            "idle_noise_bytes": self.noise_bytes,
            "sampling_interval_s": 0.02,
        }


def merge_native(model: Path, output: Path) -> dict[str, object]:
    template = Path("artifacts/runs/codec-tts-base/native")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "model.safetensors"
    identity = digest_file(model / "model.safetensors")
    record = output / "merge.stats.json"
    if record.exists():
        previous = json.loads(record.read_bytes())
        if previous["source_sha256"] == identity and digest_file(destination) == previous["sha256"]:
            return dict(previous)
        message = f"incompatible native merge already exists at {output}"
        raise FileExistsError(message)
    values: dict[str, torch.Tensor] = {}
    merged = 0
    with (
        safe_open(template / "model.safetensors", framework="pt") as base,
        safe_open(model / "model.safetensors", framework="pt") as speech,
    ):
        names = set(speech.keys())
        base_names = base.keys()
        for name in base_names:
            mapped = map_weight(name)
            if mapped is None:
                values[name] = base.get_tensor(name)
                continue
            key = "generator." + mapped
            stem = key.removesuffix(".weight")
            if stem + ".factor_a" in names:
                weight = speech.get_tensor(stem + ".original.weight")
                delta = (
                    speech.get_tensor(stem + ".factor_b").float()
                    @ speech.get_tensor(stem + ".factor_a").float()
                )
                values[name] = (weight.float() + 2 * delta).to(weight.dtype)
                merged += 1
            else:
                values[name] = speech.get_tensor(key)
    config = json.loads((model / "config.json").read_bytes())
    expected = int(config["geometry"]["layers"]) * 7
    if merged != expected:
        message = f"native merge layer accounting failed: {merged}/{expected}"
        raise ValueError(message)
    temporary = output / ".model.safetensors.tmp"
    save_file(values, temporary)
    temporary.replace(destination)
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "LICENSE",
    ]:
        source = template / name
        if source.is_file():
            shutil.copy2(source, output / name)
    write_json(output / "codec.json", config["audio_configuration"])
    report = {
        **provenance(identity),
        "source_sha256": identity,
        "sha256": digest_file(destination),
        "merged_projections": merged,
        "tensor_count": len(values),
        "adapter_multiplier": 1.0,
        "merge_arithmetic": "FP32 base plus 2*B@A, rounded to original weight dtype",
        "codec_unchanged": True,
    }
    write_json(record, report)
    return report


def inspect_gguf(path: Path) -> dict[str, object]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    counts: dict[str, int] = {}
    field = reader.fields.get("audiocpp.tensor_names")
    if field is None:
        message = f"GGUF logical tensor-name metadata missing: {path}"
        raise ValueError(message)
    names: list[str] = []
    for name in field.contents():
        if not isinstance(name, str):
            message = f"invalid logical tensor name in {path}: {name!r}"
            raise TypeError(message)
        names.append(name)
    if len(names) != len(reader.tensors):
        message = f"GGUF logical/physical tensor conservation failed at {path}"
        raise ValueError(message)
    types: dict[str, str] = {}
    for name, tensor in zip(names, reader.tensors, strict=True):
        kind = tensor.tensor_type.name
        counts[kind] = counts.get(kind, 0) + 1
        types[name] = kind
    codec_count = validate_codec_types(types)
    source_path = ROOT / "native-merged/model.safetensors"
    with safe_open(source_path, framework="pt") as source:
        expected = set(source.keys())
        if set(names) != expected:
            message = f"GGUF tensor conservation failed at {path}: missing={expected - set(names)}"
            raise ValueError(message)
        for name, tensor in zip(names, reader.tensors, strict=True):
            if "tied.embedding.modality_embeddings.0.model." in name and not np.array_equal(
                tensor.data.reshape(-1), source.get_tensor(name).float().numpy().reshape(-1)
            ):
                message = f"codec values changed in GGUF: {path}:{name}"
                raise ValueError(message)
    if len(types) != len(names):
        message = f"duplicate logical tensor names in GGUF: {path}"
        raise ValueError(message)
    return {
        "tensor_types": counts,
        "codec_tensors_fp32": codec_count,
        "codec_values_equal_to_fp32_source": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest_file(path),
    }


def merge_transformers(model: Path, output: Path) -> dict[str, object]:
    native = ROOT / "native-merged/model.safetensors"
    identity = digest_file(native)
    if (output / "merge.stats.json").exists():
        previous = json.loads((output / "merge.stats.json").read_bytes())
        if previous["native_sha256"] == identity:
            return dict(previous)
        message = f"different merged export occupies {output}"
        raise FileExistsError(message)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((model / "config.json").read_bytes())
    config["has_adapter"] = False
    strategies = config["inference"].get("strategies")
    if isinstance(strategies, list):
        config["inference"]["strategies"] = [
            value for value in strategies if value["adapter_multiplier"] == 1.0
        ]
    values: dict[str, torch.Tensor] = {}
    prefix = "tied.embedding.modality_embeddings.0.model."
    with safe_open(native, framework="pt") as source:
        names = source.keys()
        for name in names:
            if name.startswith(prefix):
                if name.endswith("semantic_model.masked_spec_embed"):
                    continue
                values["audio_tokenizer." + name.removeprefix(prefix)] = source.get_tensor(name)
            else:
                mapped = map_weight(name)
                if mapped is not None:
                    values["generator." + mapped] = source.get_tensor(name)
    save_file(values, output / "model.safetensors", metadata={"format": "pt"})
    write_json(output / "config.json", config)
    for name, content in remote_sources(Path.cwd()).items():
        (output / name).write_text(content, encoding="utf-8")
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "LICENSE",
        "NOTICE",
        "README.md",
        "requirements.txt",
    ]:
        shutil.copy2(model / name, output / name)
    report = {
        **provenance(identity),
        "native_sha256": identity,
        "has_adapter": False,
        "tensor_count": len(values),
        "sha256": digest_file(output / "model.safetensors"),
    }
    write_json(output / "merge.stats.json", report)
    return report


def convert(format_name: str, output: Path) -> dict[str, object]:
    source = ROOT / "native-merged"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(BUILD / "audiocpp_gguf"),
        "--input",
        str(source / "model.safetensors"),
        "--root",
        str(source),
        "--family",
        "higgs_audio_tts",
        "--type",
        format_name,
        "--keep-type",
        "tied.embedding.modality_embeddings.0.model.*=f32",
        "--output",
        str(output),
    ]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=720, check=False)
    output.with_suffix(".log.txt").write_text(result.stdout + result.stderr)
    if result.returncode:
        message = f"GGUF conversion failed: type={format_name}, exit={result.returncode}"
        message += f"; see {output.with_suffix('.log.txt')}"
        raise RuntimeError(message)
    report = {
        **provenance(digest_file(source / "model.safetensors")),
        **inspect_gguf(output),
        "format": format_name,
        "conversion_s": time.monotonic() - started,
        "command": command,
    }
    write_json(output.with_suffix(".stats.json"), report)
    return report


def benchmark_transformers(
    format_name: str,
    model_path: Path,
    output: Path,
    cases: Sequence[SpeechCase],
    export: bool = False,
) -> tuple[list[Trial], dict[str, object]]:
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if format_name == "merged":
        config = json.loads((model_path / "config.json").read_bytes())
        native = ROOT / "native-merged"
        generator, _ = load_generator(
            native / "model.safetensors",
            Geometry.model_validate(config["geometry"]),
            torch.device("cuda"),
        )
        codec, _ = load_codec(
            native / "model.safetensors", native / "codec.json", torch.device("cuda")
        )
        audit: dict[str, object] = {"quantized_modules": 0}
    else:
        quantization = (
            None
            if format_name == "bf16"
            else BitsAndBytesConfig(
                load_in_4bit=format_name == "nf4",
                load_in_8bit=format_name == "int8",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        )
        model = AutoModelForTextToWaveform.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": "cuda:0"},
            quantization_config=quantization,
        ).eval()
        generator = model.generator
        codec = Codec(model.audio_tokenizer, torch.device("cuda"))
        quantized = [
            name
            for name, layer in model.named_modules()
            if type(layer).__module__.startswith("bitsandbytes.")
            and type(layer).__name__ in ("Linear4bit", "Linear8bitLt")
        ]
        validate_quantized_modules(
            quantized, 0 if format_name == "bf16" else generator.geometry.layers * 7
        )
        if "audio_tokenizer" not in model._keep_in_fp32_modules:
            message = "BitsAndBytes codec exclusion is missing from loaded model"
            raise ValueError(message)
        with safe_open(model_path / "model.safetensors", framework="pt") as source:
            for name, value in model.audio_tokenizer.state_dict().items():
                if value.dtype != torch.float32 or not torch.equal(
                    value.cpu(), source.get_tensor("audio_tokenizer." + name).float()
                ):
                    message = (
                        f"codec changed during {format_name} loading: {name}, dtype={value.dtype}"
                    )
                    raise ValueError(message)
            for name, value in generator.named_parameters():
                if name.endswith(("factor_a", "factor_b")) and (
                    value.dtype != torch.float32
                    or not torch.equal(value.cpu(), source.get_tensor("generator." + name))
                ):
                    message = f"adapter factor changed during {format_name} loading: {name}"
                    raise ValueError(message)
        audit = {
            "quantized_modules": len(quantized),
            "codec_equal_to_fp32_source": True,
            "model_footprint_bytes": model.get_memory_footprint(),
        }
        if export and format_name in {"nf4", "int8"}:
            model.save_pretrained(output / "model")
            audit["saved_model_bytes"] = sum(
                p.stat().st_size for p in (output / "model").rglob("*") if p.is_file()
            )
    torch.cuda.synchronize()
    audit["load_audit_and_export_s"] = time.monotonic() - started
    trials: list[Trial] = []
    work = [(cases[0], SEEDS[0] - 1, True)] + [
        (case, seed, False) for case in cases for seed in SEEDS
    ]
    for case, seed, warmup in work:
        path = output / ("warmup.wav" if warmup else f"{case.name}-s{seed}.wav")
        trial = Trial(case=case.name, seed=seed, output=path)
        try:
            reference = read_waveform(case.reference)
            torch.cuda.synchronize()
            start = time.monotonic()
            ref_codes = codec.encode(reference)
            prompt = prompt_layout(
                tokenizer.get_added_vocab(),
                tokenizer.encode(case.text, add_special_tokens=False),
                ref_codes,
                tokenizer.encode(case.reference_text, add_special_tokens=False),
            )
            codes, trace = generate_codes(
                generator,
                prompt,
                Sampling(seed=seed, max_new_positions=600),
                Decoding(),
                deadline_monotonic=start + 100,
            )
            waveform = codec.decode(codes).numpy()
            torch.cuda.synchronize()
            trial.seconds = time.monotonic() - start
            trial.duration_s = len(waveform) / 24000
            save_waveform(
                path, np.asarray(waveform * min(1.0, 0.99 / max(abs(waveform))), dtype=np.float32)
            )
            trial.sha256 = digest_file(path)
            write_json(path.with_suffix(".trace.json"), trace)
        except (RuntimeError, ValueError, TimeoutError) as error:
            trial.error = f"{type(error).__name__}: {error}"
            if isinstance(error, GenerationError):
                write_json(path.with_suffix(".trace.json"), error.trace)
        if warmup:
            audit["warmup_error"] = trial.error
        else:
            trials.append(trial)
            write_json(
                output / "progress.stats.json",
                {"trials": [t.model_dump(mode="json") for t in trials]},
            )
        print(
            json.dumps(
                {"case": case.name, "seed": seed, "error": trial.error, "seconds": trial.seconds}
            ),
            flush=True,
        )
    audit["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    audit["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return trials, audit


def benchmark_native(
    format_name: str, model_path: Path, output: Path, cases: Sequence[SpeechCase]
) -> tuple[list[Trial], dict[str, object]]:
    work = [(cases[0].model_copy(update={"name": "warmup"}), SEEDS[0] - 1)]
    work += [(case, seed) for case in cases for seed in SEEDS]
    trials: list[Trial] = []
    deadline = time.monotonic() + 700
    process_count = 0
    while work:
        requests = [
            {
                "id": f"{case.name}-s{seed}",
                "text": case.text,
                "voice_ref": str(case.reference.resolve()),
                "reference_text": case.reference_text,
                "seed": seed,
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 30,
                "max_tokens": 600,
                "allow_incomplete_generation": False,
            }
            for case, seed in work
        ]
        sequence = output / f"requests-{process_count}.json"
        write_json(sequence, {"requests": requests})
        command = [
            str(BUILD / "audiocpp_cli"),
            "--family",
            "higgs_audio_tts",
            "--task",
            "clon",
            "--model",
            str(model_path),
            "--backend",
            "cuda",
            "--request-sequence",
            str(sequence),
            "--out-dir",
            str(output),
            "--out-format",
            "float32",
            "--metrics",
            "--threads",
            "4",
            "--session-option",
            "higgs_audio_tts.codec_weight_type=f32",
            "--session-option",
            "higgs_audio_tts.reference_cache_slots=0",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(1, deadline - time.monotonic()),
        )
        log = result.stdout + result.stderr
        (output / f"native-{process_count}.log").write_text(log)
        process_count += 1
        completed = 0
        for case, seed in work:
            request_id = f"{case.name}-s{seed}"
            path = output / f"{request_id}.wav"
            seconds = metric_seconds(log, request_id)
            trial = Trial(case=case.name, seed=seed, output=path, seconds=seconds)
            if not path.is_file() or seconds is None:
                trial.error = f"native exit={result.returncode}: {log[-1800:]}"
                trials.append(trial)
                completed += 1
                break
            wave = read_waveform(path)
            trial.duration_s = wave.size / 24000
            save_waveform(
                path, np.asarray(wave * min(1.0, 0.99 / max(abs(wave))), dtype=np.float32)
            )
            trial.sha256 = digest_file(path)
            trials.append(trial)
            completed += 1
        work = work[completed:]
        write_json(
            output / "progress.stats.json", {"trials": [t.model_dump(mode="json") for t in trials]}
        )
    warmup = next(t for t in trials if t.case == "warmup")
    return [t for t in trials if t.case != "warmup"], {
        "warmup_error": warmup.error,
        "native_processes": process_count,
        "reference_cache_slots": 0,
        "gguf": inspect_gguf(model_path),
        "format": format_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["benchmark", "merge", "merge-hf", "convert", "inspect"],
        default="benchmark",
    )
    parser.add_argument(
        "--format",
        choices=["bf16", "merged", "nf4", "int8", "q8_0", "q4_k", "q4_0"],
        default="bf16",
    )
    parser.add_argument("--backend", choices=["transformers", "audiocpp"], default="transformers")
    parser.add_argument("--target", choices=["local", "hpc"], default="hpc")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.target == "hpc":
        from tools.launch import CODEC_PYTHON, start

        run = f"matoub-quant-{args.backend}-{args.format}-{args.task}"
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=30s",
            "840s",
            CODEC_PYTHON,
            "-m",
            "tools.quantize_speech",
            "--target",
            "local",
            "--task",
            args.task,
            "--backend",
            args.backend,
            "--format",
            args.format,
        ]
        if args.model is not None:
            command += ["--model", str(args.model)]
        if args.output is not None:
            command += ["--output", str(args.output)]
        if args.export:
            command += ["--export"]
        start(run, ["__raw__", shlex.join(command)], f"{run}/log.txt")
        return
    torch.set_num_threads(4)
    model = args.model or (
        STAGE
        if args.backend == "transformers" or args.task in {"merge", "merge-hf"}
        else ROOT / "gguf" / f"Matoub-4B-{args.format}.gguf"
    )
    if args.task == "merge":
        result = merge_native(model, args.output or ROOT / "native-merged")
    elif args.task == "merge-hf":
        result = merge_transformers(model, args.output or ROOT / "transformers-merged-model")
    elif args.task == "convert":
        result = convert(
            args.format, args.output or ROOT / "gguf" / f"Matoub-4B-{args.format}.gguf"
        )
    elif args.task == "inspect":
        result = inspect_gguf(model)
    else:
        output = args.output or ROOT / f"{args.backend}-{args.format}"
        output.mkdir(parents=True, exist_ok=True)
        existing = output / "benchmark.stats.json"
        if existing.is_file():
            stored = json.loads(existing.read_bytes())
            for row in stored["trials"]:
                if row["sha256"] and digest_file(Path(row["output"])) != row["sha256"]:
                    message = f"cached benchmark audio changed: {row['output']}"
                    raise ValueError(message)
            print(
                json.dumps(
                    {"cached": True, "generated_requests_this_run": 0, "report": str(existing)}
                ),
                flush=True,
            )
            return
        if any(output.glob("*.wav")):
            message = f"incomplete prior benchmark at {output}; use a fresh --output"
            raise FileExistsError(message)
        cases = read_cases()
        signature = digest_file(Path(__file__))
        with DeviceMemory() as memory:
            if args.backend == "transformers":
                trials, audit = benchmark_transformers(
                    args.format, model, output, cases, args.export
                )
            else:
                trials, audit = benchmark_native(args.format, model, output, cases)
        verifier = SpeechVerifier(torch.device("cpu"))
        for trial in trials:
            if trial.error is None:
                case = next(c for c in cases if c.name == trial.case)
                try:
                    trial.quality = verifier.assess(
                        read_waveform(trial.output), case.text, read_waveform(case.reference)
                    )
                except (ValueError, RuntimeError) as error:
                    trial.error = f"quality assessment failed at {trial.output}: {error}"
        result = {
            **provenance(signature),
            "complete": True,
            "backend": args.backend,
            "format": args.format,
            "model_source": str(model),
            "cases": [c.model_dump(mode="json") for c in cases],
            "seeds": SEEDS,
            "sampling": {"temperature": 0.8, "top_p": 0.8, "top_k": 30, "max_positions": 600},
            "timing_scope": (
                "reference processing, prefill, AR decode and waveform decode; "
                "excludes quality scoring"
            ),
            "instruments": INSTRUMENTS,
            "memory": memory.report(),
            "audit": audit,
            "trials": [t.model_dump(mode="json") for t in trials],
            "summary": summarize(trials),
        }
        write_json(output / "benchmark.stats.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
