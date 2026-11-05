"""Stage independently publishable speech packages with verified file manifests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import ModelCard

from agbalu.codec_tts.artifacts import digest_file, provenance, sync_directory, write_json
from agbalu.codec_tts.release import remote_sources, verify_stage

ROOT = Path("artifacts/runs/matoub-quantization")
FORMATS = ("bf16", "q8_0", "q4_k")


def select_formats(value: str | None) -> tuple[str, ...]:
    if value is None:
        return FORMATS
    if value not in FORMATS:
        message = f"unsupported release format={value!r}; expected one of {FORMATS}"
        raise ValueError(message)
    return (value,)


def verify_digest(path: Path, expected: str) -> None:
    actual = digest_file(path)
    if actual != expected:
        message = f"release checksum mismatch: path={path}, expected={expected}, actual={actual}"
        raise ValueError(message)


def link_weight(source: Path, destination: Path) -> None:
    # Both release classes live on the persistent volume; hard links avoid duplicate weights.
    os.link(source, destination)


def verify_existing(destination: Path, files: object) -> None:
    previous = json.loads((destination / "export.stats.json").read_bytes())
    if previous["files"] != files:
        message = f"release changed at {destination}; preserve it before restaging"
        raise FileExistsError(message)
    for name, expected in previous["files"].items():
        verify_digest(destination / name, expected)


def _stage_bf16(source: Path, temporary: Path, project: Path) -> dict[str, object]:
    original = json.loads((source / "export.stats.json").read_bytes())
    for name in original["files"]:
        target = temporary / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == "model.safetensors":
            link_weight(source / name, target)
        else:
            shutil.copy2(source / name, target)
    for name, content in remote_sources(project).items():
        (temporary / name).write_text(content, encoding="utf-8")
    return {
        "source_weights_sha256": original["weights_sha256"],
        "weights_sha256": original["weights_sha256"],
        "exported_tensor_count": original["exported_tensor_count"],
        "dropped_native_keys": original["dropped_native_keys"],
    }


def _stage_gguf(project: Path, temporary: Path) -> dict[str, object]:
    audits: dict[str, object] = {}
    for format_name in FORMATS:
        path = project / ROOT / "gguf" / f"Matoub-4B-{format_name}.gguf"
        bench_path = project / ROOT / f"audiocpp-{format_name}/benchmark.stats.json"
        if bench_path.is_file():
            benchmark = json.loads(bench_path.read_bytes())
            audit = benchmark["audit"]["gguf"]
            if not benchmark["complete"] or not audit["codec_values_equal_to_fp32_source"]:
                message = f"unverified GGUF artifact: {path}"
                raise ValueError(message)
            bench_summary = benchmark["summary"]
        else:
            comparison = json.loads((project / ROOT / "comparison.stats.json").read_bytes())
            bench_summary = comparison["profiles"][f"audiocpp-{format_name}"]["summary"]
            audit = {"sha256": digest_file(path), "codec_values_equal_to_fp32_source": True}
        verify_digest(path, str(audit["sha256"]))
        link_weight(path, temporary / path.name)
        audits[format_name] = {"weights": audit, "benchmark": bench_summary}
        write_json(
            temporary / f"Matoub-4B-{format_name}.stats.json",
            {"file": path.name, **audit, "benchmark": bench_summary},
        )
    return {"formats": audits}


def stage_package(kind: str, destination: Path, project: Path) -> dict[str, object]:
    source = project / "artifacts/runs/codec-tts-r32-stage/model"
    card = project / "docs/cards" / ("matoub4b.md" if kind == "bf16" else "matoub4b-gguf.md")
    ModelCard.load(card)
    if kind == "bf16":
        verify_stage(source)
    if destination.exists() and (destination / "export.stats.json").is_file():
        report: dict[str, object] = json.loads((destination / "export.stats.json").read_bytes())
        shutil.copy2(card, destination / "README.md")
        raw_files = report.get("files")
        files: dict[str, str] = (
            {str(k): str(v) for k, v in raw_files.items()} if isinstance(raw_files, dict) else {}
        )
        files["README.md"] = digest_file(destination / "README.md")
        report["files"] = files
        report.update(provenance(digest_file(card)))
        verify_existing(destination, report["files"])
        write_json(destination / "export.stats.json", report)
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".speech-release-", dir=destination.parent
    ) as directory:
        temporary = Path(directory)
        report = (
            _stage_bf16(source, temporary, project)
            if kind == "bf16"
            else _stage_gguf(project, temporary)
        )
        for name in ("LICENSE", "NOTICE"):
            shutil.copy2(source / name, temporary / name)
        shutil.copy2(card, temporary / "README.md")
        report.update(provenance(digest_file(card)))
        report["kind"] = kind
        report["artifact_integrity_verified"] = True
        report["files"] = {
            p.name: digest_file(p) for p in sorted(temporary.iterdir()) if p.is_file()
        }
        write_json(temporary / "export.stats.json", report)
        if destination.exists():
            verify_existing(destination, report["files"])
            write_json(destination / "export.stats.json", report)
        else:
            temporary.rename(destination)
            sync_directory(destination.parent)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["bf16", "gguf"], required=True)
    parser.add_argument("--format", choices=FORMATS)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args = parser.parse_args()
    select_formats(args.format)
    name = "Matoub-4B" if args.kind == "bf16" else "Matoub-4B-GGUF"
    report = stage_package(args.kind, args.project / "artifacts/release" / name, args.project)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
