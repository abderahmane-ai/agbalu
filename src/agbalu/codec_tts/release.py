"""Stage native merged speech weights with independently usable source code."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from agbalu.codec_tts.adaptation import PROJECTIONS, AdaptedLinear, adapter_state
from agbalu.codec_tts.artifacts import digest_file, provenance, sync_directory, write_json
from agbalu.codec_tts.backbone import Generator, map_weight
from agbalu.codec_tts.configuration import Recipe
from agbalu.codec_tts.synthesis import InferenceSettings
from agbalu.codec_tts.transformers_model import SpeechConfig, SpeechModel


def native_tensors(model: Generator, source: Path) -> dict[str, Tensor]:
    active = model.state_dict()
    tensors: dict[str, Tensor] = {}
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        names = checkpoint.keys()
        for name in names:
            key = map_weight(name)
            if key is None:
                tensors[name] = checkpoint.get_tensor(name)
            elif key in active:
                tensors[name] = active[key].detach().cpu().contiguous()
            else:
                layer = model.get_submodule(key.removesuffix(".weight"))
                if not isinstance(layer, AdaptedLinear):
                    message = f"unmapped active export tensor {key}"
                    raise ValueError(message)
                tensors[name] = layer.merged_weight().detach().cpu().contiguous()
    return tensors


def export_model(
    model: Generator,
    recipe: Recipe,
    destination: Path,
    *,
    inference_config: Path | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    retain_adapter = inference_config is not None
    license_path = recipe.base_directory / "LICENSE"
    license_sha256 = digest_file(license_path) if retain_adapter else None
    root = Path(__file__).resolve().parents[3]
    sources = sorted((root / "src/agbalu/codec_tts").glob("*.py"))
    sources += [root / "requirements/codec_tts.lock", root / "tools/infer_speech.py"]
    if retain_adapter:
        sources += [root / "docs/cards/matoub4b.md"]
    source_files = {str(path.relative_to(root)): digest_file(path) for path in sources}
    adapter_digest = hashlib.sha256()
    for name, tensor in sorted(adapter_state(model).items()):
        adapter_digest.update(name.encode())
        adapter_digest.update(tensor.numpy().tobytes())
    fingerprint = adapter_digest.hexdigest()
    if destination.exists():
        report = json.loads((destination / "export.stats.json").read_bytes())
        if (
            report.get("config_hash") == recipe.digest
            and report.get("adapter_sha256") == fingerprint
            and report.get("weights_sha256") == digest_file(destination / "model.safetensors")
            and report.get("source_files_sha256") == source_files
            and report.get("retains_adapter") == retain_adapter
            and report.get("inference_config_sha256")
            == (digest_file(inference_config) if inference_config else None)
            and report.get("checkpoint_sha256") == checkpoint_sha256
            and report.get("upstream_license_sha256") == license_sha256
        ):
            verify_stage(destination)
            return dict(report)
        message = f"incompatible export already exists at {destination}; use a new artifact path"
        raise ValueError(message)
    source = recipe.base_directory / "model.safetensors"
    available = shutil.disk_usage(
        destination.parent if destination.parent.exists() else Path.cwd()
    ).free
    if available < 2 * source.stat().st_size:
        message = f"insufficient old/staged export space for {source}"
        raise OSError(message)
    tensors = {} if retain_adapter else native_tensors(model, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-export-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir(parents=True, exist_ok=True)
        if retain_adapter:
            shutil.copy2(source, stage / "model.safetensors")
            save_file(adapter_state(model), stage / "adapter.safetensors")
            write_json(
                stage / "adapter_config.json",
                {
                    "format": "agbalu-codec-lora",
                    "rank": recipe.rank,
                    "alpha": recipe.alpha,
                    "target_modules": sorted(PROJECTIONS),
                },
            )
        else:
            save_file(tensors, stage / "model.safetensors")
        with (stage / "model.safetensors").open("rb") as stream:
            os.fsync(stream.fileno())
        for path in recipe.base_directory.iterdir():
            if path.is_file() and (
                path.suffix in {".json", ".txt", ".jinja", ".md"} or path.name == "LICENSE"
            ):
                shutil.copy2(path, stage / path.name)
        shutil.copy2(recipe.codec_configuration, stage / "codec.json")
        package = stage / "src" / "agbalu"
        package.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            root / "src/agbalu/codec_tts",
            package / "codec_tts",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in ("__init__.py", "_about.py"):
            shutil.copy2(root / "src/agbalu" / name, package / name)
        (package / "llm").mkdir(parents=True, exist_ok=True)
        for name in ("__init__.py", "provenance.py", "progress.py"):
            shutil.copy2(root / "src/agbalu/llm" / name, package / "llm" / name)
        (stage / "tools").mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "tools/export_checkpoint.py", stage / "tools/export_checkpoint.py")
        shutil.copy2(root / "requirements/codec_tts.lock", stage / "requirements.txt")
        if inference_config is not None:
            settings = InferenceSettings.model_validate_json(inference_config.read_bytes())
            write_json(stage / "inference.json", settings.model_dump())
            shutil.copy2(root / "tools/infer_speech.py", stage / "inference.py")
            if (stage / "README.md").exists():
                (stage / "README.md").rename(stage / "UPSTREAM_README.md")
            shutil.copy2(root / "docs/cards/matoub4b.md", stage / "README.md")
        portable = Recipe.model_validate(
            {
                **recipe.model_dump(),
                "base_directory": Path(),
                "codec_configuration": Path("codec.json"),
                "output": Path("artifacts"),
                "source": Path("source-not-shipped"),
            }
        )
        write_json(stage / "recipe.json", portable.model_dump(mode="json"))
        write_json(stage / "geometry.json", model.geometry.model_dump())
        makefile = (
            "PY ?= python3\nexport PYTHONPATH := src\n.PHONY: check clean synthesize\n"
            "check: ## Verify standalone imports\n"
            "\t$(PY) -m agbalu.codec_tts.cli environment --config recipe.json\n"
            "clean: ## Remove interpreter caches\n"
            "\tfind src tools -name __pycache__ -type d -prune -exec rm -rf {} +\n"
            "synthesize: ## Synthesize TEXT_FILE with optional REFERENCE and REFERENCE_TEXT_FILE\n"
            "\t$(PY) -m agbalu.codec_tts.cli synthesize --config recipe.json "
            "$(if $(TEXT_FILE),--text-file $(TEXT_FILE),) "
            "$(if $(REFERENCE),--reference $(REFERENCE),) "
            "$(if $(REFERENCE_TEXT_FILE),--reference-text-file $(REFERENCE_TEXT_FILE),) "
            "$(if $(OUTPUT),--output $(OUTPUT),) $(if $(DEVICE),--device $(DEVICE),)\n"
        )
        if retain_adapter:
            makefile = (
                "PY ?= python3\nexport PYTHONPATH := src\n.PHONY: check clean synthesize\n"
                "check: ## Verify every staged file\n"
                "\t$(PY) -m agbalu.codec_tts.cli verify-stage --output .\n"
                "clean: ## Remove interpreter caches\n"
                "\tfind src -name __pycache__ -type d -prune -exec rm -rf {} +\n"
                "synthesize: ## Generate verified reference-conditioned speech\n"
                "\t$(PY) inference.py $(if $(TEXT_FILE),--text-file $(TEXT_FILE),) "
                "$(if $(REFERENCE),--reference $(REFERENCE),) "
                "$(if $(REFERENCE_TEXT_FILE),--reference-text-file $(REFERENCE_TEXT_FILE),) "
                "$(if $(OUTPUT),--output $(OUTPUT),) $(if $(DEVICE),--device $(DEVICE),) "
                "$(if $(INFERENCE_CONFIG),--inference-config $(INFERENCE_CONFIG),)\n"
            )
        (stage / "Makefile").write_text(makefile, encoding="utf-8")
        write_json(stage / "STAMP.json", provenance(recipe.digest))
        report = {
            **provenance(recipe.digest),
            "source_sha256": digest_file(source),
            "adapter_sha256": fingerprint,
            "weights_sha256": digest_file(stage / "model.safetensors"),
            "native_tensor_count": native_tensor_count(stage / "model.safetensors"),
            "retains_adapter": retain_adapter,
            "source_files_sha256": source_files,
            "inference_config_sha256": digest_file(inference_config) if inference_config else None,
            "checkpoint_sha256": checkpoint_sha256,
            "upstream_license_sha256": license_sha256,
            "publication_ready": False,
            "publication_pending": ["independent pronunciation, speaker and naturalness review"],
            "dropped_native_keys": [],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "merge_speech_parity_verified": False,
            "files": {
                str(path.relative_to(stage)): digest_file(path)
                for path in stage.rglob("*")
                if path.is_file()
            },
        }
        write_json(stage / "export.stats.json", report)
        for path in stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        stage.rename(destination)
        sync_directory(destination.parent)
    return report


def stage_model(
    model: Generator,
    recipe: Recipe,
    destination: Path,
    inference_config: Path,
    checkpoint_sha256: str,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    sources = remote_sources(root)
    source_files = {
        name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()
    }
    license_path = recipe.base_directory / "LICENSE"
    license_sha256 = digest_file(license_path)
    adapter_digest = hashlib.sha256()
    for name, value in sorted(adapter_state(model).items()):
        adapter_digest.update(name.encode())
        adapter_digest.update(value.numpy().tobytes())
    fingerprint = {
        "checkpoint_sha256": checkpoint_sha256,
        "adapter_sha256": adapter_digest.hexdigest(),
        "source_files_sha256": source_files,
        "card_sha256": digest_file(root / "docs/cards/matoub4b.md"),
        "inference_config_sha256": digest_file(inference_config),
        "upstream_license_sha256": license_sha256,
        "notice_sha256": digest_file(root / "resources/codec_tts/NOTICE"),
        "requirements_sha256": digest_file(root / "requirements/codec_tts.lock"),
    }
    if destination.exists():
        report = json.loads((destination / "export.stats.json").read_bytes())
        if all(report.get(key) == value for key, value in fingerprint.items()):
            verify_stage(destination)
            return dict(report)
        message = (
            f"incompatible export already exists at {destination}; preserve it before restaging"
        )
        raise FileExistsError(message)
    settings = InferenceSettings.model_validate_json(inference_config.read_bytes())
    config = SpeechConfig(
        geometry=model.geometry.model_dump(),
        audio_configuration=json.loads(recipe.codec_configuration.read_bytes()),
        inference=settings.model_dump(),
        checkpoint_sha256=checkpoint_sha256,
        source_stamp=provenance(recipe.digest),
        dtype="bfloat16",
        architectures=["SpeechModel"],
        auto_map={
            "AutoConfig": "modeling_speech.SpeechConfig",
            "AutoModel": "modeling_speech.SpeechModel",
            "AutoModelForTextToWaveform": "modeling_speech.SpeechModel",
        },
    )
    with torch.device("meta"):
        expected = SpeechModel(config).state_dict()
    tensors, dropped = transformers_tensors(
        model, recipe.base_directory / "model.safetensors", expected
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-export-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir()
        for name, content in sources.items():
            (stage / name).write_text(content, encoding="utf-8")
        save_file(tensors, stage / "model.safetensors", metadata={"format": "pt"})
        config.save_pretrained(stage)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            source = recipe.base_directory / name
            if name == "chat_template.jinja" and not source.is_file():
                continue
            if source.is_file():
                shutil.copy2(source, stage / name)
            else:
                message = f"required tokenizer file missing: {source}"
                raise FileNotFoundError(message)
        shutil.copy2(license_path, stage / "LICENSE")
        shutil.copy2(root / "resources/codec_tts/NOTICE", stage / "NOTICE")
        shutil.copy2(root / "docs/cards/matoub4b.md", stage / "README.md")
        shutil.copy2(root / "requirements/codec_tts.lock", stage / "requirements.txt")
        write_json(stage / "STAMP.json", provenance(recipe.digest))
        report = {
            **provenance(recipe.digest),
            **fingerprint,
            "format": "transformers-auto-class",
            "weights_sha256": digest_file(stage / "model.safetensors"),
            "native_tensor_count": native_tensor_count(recipe.base_directory / "model.safetensors"),
            "exported_tensor_count": len(tensors),
            "dropped_native_keys": dropped,
            "publication_ready": False,
            "publication_pending": ["model card review and required attribution"],
            "files": {
                str(path.relative_to(stage)): digest_file(path)
                for path in stage.rglob("*")
                if path.is_file()
            },
        }
        write_json(stage / "export.stats.json", report)
        stage.rename(destination)
        sync_directory(destination.parent)
    return report


def remote_sources(root: Path) -> dict[str, str]:
    pending = ["agbalu.codec_tts.transformers_model"]
    visited: set[str] = set()
    result: dict[str, str] = {}
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = root / "src" / (module.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("agbalu.")
            ):
                pending.append(node.module)
                node.module = node.module.rsplit(".", 1)[-1]
                node.level = 1
        name = "modeling_speech.py" if module.endswith("transformers_model") else path.name
        if name in result:
            message = f"remote-code filename collision: {module} -> {name}"
            raise ValueError(message)
        content = ast.unparse(tree) + "\n"
        if name == "modeling_speech.py":
            content += "SpeechConfig.register_for_auto_class()\n"
            content += "SpeechModel.register_for_auto_class('AutoModelForTextToWaveform')\n"
        result[name] = content
    return result


def transformers_tensors(
    model: Generator, source: Path, expected: dict[str, Tensor]
) -> tuple[dict[str, Tensor], list[str]]:
    tensors: dict[str, Tensor] = {}
    dropped: list[str] = []
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        names = checkpoint.keys()
        for name in names:
            if name.startswith("tied.head.") or name.endswith("semantic_model.masked_spec_embed"):
                dropped.append(name)
                continue
            prefix = "tied.embedding.modality_embeddings.0.model."
            if name.startswith(prefix):
                destination = "audio_tokenizer." + name.removeprefix(prefix)
            else:
                mapped = map_weight(name)
                destination = "generator." + str(mapped)
                if destination not in expected and destination.endswith(".weight"):
                    destination = destination.removesuffix(".weight") + ".original.weight"
            if destination in tensors or destination not in expected:
                message = f"unexpected/duplicate export tensor {source}:{name} -> {destination}"
                raise ValueError(message)
            value = checkpoint.get_tensor(name)
            if value.shape != expected[destination].shape:
                message = f"export shape mismatch {source}:{name}: {value.shape}"
                message += f" != {expected[destination].shape}"
                raise ValueError(message)
            tensors[destination] = value
    tensors.update({"generator." + name: value for name, value in adapter_state(model).items()})
    if set(tensors) != set(expected):
        message = (
            f"export tensor conservation failed: missing={sorted(set(expected) - set(tensors))}"
        )
        raise ValueError(message)
    return tensors, dropped


def native_tensor_count(path: Path) -> int:
    with safe_open(path, framework="pt", device="cpu") as weights:
        return len(weights.keys())


def verify_stage(directory: Path) -> dict[str, object]:
    report = json.loads((directory / "export.stats.json").read_bytes())
    files = report.get("files")
    if not isinstance(files, dict) or not files:
        message = f"empty or malformed export manifest at {directory}"
        raise ValueError(message)
    for name, expected in files.items():
        path = Path(str(name))
        if path.is_absolute() or ".." in path.parts:
            message = f"unsafe export manifest path={name!r}"
            raise ValueError(message)
        target = directory / path
        if not target.is_file() or digest_file(target) != expected:
            message = f"staged file missing or changed: {target}"
            raise ValueError(message)
    return {
        "verified": True,
        "files": len(files),
        "weights_sha256": report["weights_sha256"],
        "publication_ready": report.get("publication_ready", False),
    }
