"""Stage native merged speech weights with independently usable source code."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from agbalu.codec_tts.adaptation import AdaptedLinear, adapter_state
from agbalu.codec_tts.artifacts import digest_file, provenance, sync_directory, write_json
from agbalu.codec_tts.backbone import Generator, map_weight
from agbalu.codec_tts.configuration import Recipe


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


def export_model(model: Generator, recipe: Recipe, destination: Path) -> dict[str, object]:
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
        ):
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
    tensors = native_tensors(model, source)
    root = Path(__file__).resolve().parents[3]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".speech-export-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir(parents=True, exist_ok=True)
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
        (stage / "Makefile").write_text(makefile, encoding="utf-8")
        write_json(stage / "STAMP.json", provenance(recipe.digest))
        report = {
            **provenance(recipe.digest),
            "source_sha256": digest_file(source),
            "adapter_sha256": fingerprint,
            "weights_sha256": digest_file(stage / "model.safetensors"),
            "native_tensor_count": len(tensors),
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
        stage.rename(destination)
        sync_directory(destination.parent)
    return report
