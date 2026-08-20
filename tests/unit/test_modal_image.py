"""The Modal image must install every third party its container-side imports reach.

`make check` never imports `modal_app`, so a missing dependency stays green locally and
fails only after the image build has been paid for. This recomputes the import closure
from the entrypoint and compares it against `PIP_PACKAGES`.

The same gap exists one layer down, in apt: `kenlm_image` ran `git clone` as its first
build step and never installed `git`, which nothing caught because the 5-gram it produces
was compiled by hand in a container that predates this file. `run_commands` is checked
against `apt_install` for that reason.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import Final

import pytest
from modal_app.common import (
    ASR_PIP_PACKAGES,
    BENCH_PIP_PACKAGES,
    EMBED_PIP_PACKAGES,
    IMPORT_NAME,
    JUGURTHA_PIP_PACKAGES,
    LLM_PIP_PACKAGES,
    MATOUB_PIP_PACKAGES,
    MT_PIP_PACKAGES,
    OCR_PIP_PACKAGES,
    PIP_PACKAGES,
    RESOURCES,
    TIFINAGH_PIP_PACKAGES,
    TTS_PIP_PACKAGES,
)

ROOT: Final = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES: Final = ("agbalu", "modal_app")
ENTRYPOINT: Final = "modal_app.train"

IMAGES: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    ("train", "modal_app.train", PIP_PACKAGES),
    ("bench", "modal_app.bench", BENCH_PIP_PACKAGES),
    ("mt", "modal_app.mt", MT_PIP_PACKAGES),
    ("translate", "modal_app.translate", BENCH_PIP_PACKAGES),
    ("infer", "modal_app.infer", PIP_PACKAGES),
    ("synth", "modal_app.synth", BENCH_PIP_PACKAGES),
    ("llm", "modal_app.llm", LLM_PIP_PACKAGES),
    ("jugurtha", "modal_app.jugurtha", JUGURTHA_PIP_PACKAGES),
    ("asr", "modal_app.asr", ASR_PIP_PACKAGES),
    ("tts", "modal_app.tts", TTS_PIP_PACKAGES),
    ("matoub", "modal_app.matoub", (*TTS_PIP_PACKAGES, *MATOUB_PIP_PACKAGES)),
    ("tifinagh", "modal_app.tifinagh", TIFINAGH_PIP_PACKAGES),
    ("sentiment", "modal_app.sentiment", BENCH_PIP_PACKAGES),
    ("punctuation", "modal_app.punctuation", PIP_PACKAGES),
    ("boulifa", "modal_app.boulifa", PIP_PACKAGES),
    ("simohand", "modal_app.simohand", EMBED_PIP_PACKAGES),
    ("ocr", "modal_app.ocr", OCR_PIP_PACKAGES),
)
"""Each entrypoint with the package list of the image it runs under. Three images, because
the bench and MT stacks pull transformers and sacrebleu and adding them to the training
image would invalidate it."""

ALWAYS_PRESENT: Final = frozenset({"modal", "Modules", "Utils", "models", "utils"})
"""Installed in Modal's runtime layer or cloned StyleTTS2 repository root, not via pip."""

RUNTIME_ONLY: Final[dict[str, frozenset[str]]] = {
    "train": frozenset({"pyarrow"}),
    # Shares the training image, which carries pyarrow for `agbalu.model.data`. This
    # entrypoint reads JSONL splits and reaches no parquet.
    "punctuation": frozenset({"pyarrow"}),
    "bench": frozenset({"sentencepiece", "numpy"}),
    "mt": frozenset({"sentencepiece", "accelerate"}),
    "translate": frozenset({"sentencepiece", "sacrebleu", "numpy"}),
    "infer": frozenset({"pyarrow"}),
    # `huggingface_hub` is what every `from_pretrained` in this entrypoint downloads NLLB
    # through, and it shares `BENCH_PIP_PACKAGES` with `translate`, which asks for it by name
    # because `resolve_weights` calls `snapshot_download` itself.
    "synth": frozenset({"sentencepiece", "huggingface_hub", "numpy"}),
    "llm": frozenset({"sentencepiece", "accelerate"}),
    # `flash-linear-attention` and `causal-conv1d` are imported by transformers'
    # `modeling_qwen3_5`, not by our own code, so no AST closure can see them.
    # `cut-cross-entropy` backs `agbalu.llm.loss`, which `_step` does not call yet: the
    # trainer still passes `labels=` and takes transformers' own loss, so no AST closure
    # from this entrypoint reaches it. Wiring it is what lets `MICRO_BATCH` go back above 2.
    "jugurtha": frozenset(
        {
            "sentencepiece",
            "accelerate",
            "flash-linear-attention",
            "causal-conv1d",
            "cut_cross_entropy",
        }
    ),
    # `accelerate` is what `from_pretrained` dispatches to for device placement, and
    # `huggingface_hub` arrives with transformers rather than being asked for.
    # Shares the base training image. Boulifa is character-level over its own table, so it
    # reaches none of the corpus stack the other entrypoints on that image pay for.
    "boulifa": frozenset({"numpy", "pydantic", "sentencepiece", "yaml"}),
    # `accelerate` is `from_pretrained`'s device-placement path and `safetensors` is what
    # `save_pretrained` writes through; the e5 backbone's tokenizer is a SentencePiece model
    # that transformers loads, not our code. `numpy` is what `encode` hands back.
    "simohand": frozenset({"accelerate", "numpy", "safetensors", "sentencepiece"}),
    # Same two, for the same reason: the DeiT encoder arrives through `from_pretrained`.
    "ocr": frozenset({"accelerate", "safetensors"}),
    "asr": frozenset({"accelerate", "https://github.com/kpu/kenlm/archive/master.zip"}),
    # The baseline runs on the ASR stack unchanged: `VitsModel` is transformers' own, and
    # the decoder it is scored through is Fadhma's. `tts_image` adds only `torchaudio`,
    # which the restoration front end imports and the closure therefore reaches.
    "tts": frozenset({"accelerate", "https://github.com/kpu/kenlm/archive/master.zip"}),
    # The Kokoro recipe runs as a subprocess out of the cloned StyleTTS2 tree, so nothing
    # in this repository's import closure reaches its stack — `munch`, `einops`, the
    # monotonic-align extension, the WavLM discriminator's transformers path, or the mel
    # front end. They are required by the thing being launched, not by the launcher, and
    # the AST scan is looking at the launcher.
    "matoub": frozenset(
        {
            "accelerate",
            "https://github.com/kpu/kenlm/archive/master.zip",
            "librosa",
            "soundfile",
            "pyctcdecode",
            "munch",
            "pydub",
            "nltk",
            "matplotlib",
            "einops",
            "einops-exts",
            "tensorboard",
            "monotonic-align @ git+https://github.com/resemble-ai/monotonic_align.git",
            "numpy",
            "torchaudio",
            "transformers",
            "pandas",
            "click",
        }
    ),
    # The script model is trained from scratch on characters, so it reaches no tokenizer
    # library; the sentiment benchmark loads the encoder's SentencePiece vocabulary but
    # through `agbalu.tokenizer.evaluate`, whose import the closure does see.
    "tifinagh": frozenset({"sentencepiece", "numpy"}),
    "sentiment": frozenset({"sacrebleu", "huggingface_hub", "sentencepiece"}),
}
"""Required at runtime but imported by no module in the closure, so the AST scan cannot
see them. `numpy` is on the bench image because the sentiment benchmark loads the encoder
through `agbalu.model.data`, and it is unreached by the three older entrypoints that share
that image. The OPUS-MT tokenizers load `source.spm`/`target.spm`, and sacrebleu's
`flores200` tokenizer — the one spBLEU is defined over — does `import sentencepiece` and
raises with an install hint when it is absent."""


def module_path(module: str) -> Path | None:
    relative = Path(module.replace(".", "/"))
    candidates = (
        ROOT / "src" / relative.with_suffix(".py"),
        ROOT / "src" / relative / "__init__.py",
        ROOT / relative.with_suffix(".py"),
        ROOT / relative / "__init__.py",
    )
    return next((c for c in candidates if c.is_file()), None)


def imported_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def ancestors(module: str) -> list[str]:
    """Every parent package of `module`, which Python executes before the module itself.

    `import agbalu.bench.translate` runs `agbalu/bench/__init__.py` first, and that file
    reaches `pyarrow` through `contamination` -> `extract` -> `readers`. Walking only the
    named module missed it, and the gap was paid for with a failed GPU container.
    """
    parts = module.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def import_closure(entrypoint: str) -> tuple[set[str], set[str]]:
    """Local modules reachable from `entrypoint`, and the third-party tops they import."""
    seen: set[str] = set()
    third_party: set[str] = set()
    pending = [entrypoint]
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        path = module_path(module)
        if path is None:
            continue
        pending.extend(parent for parent in ancestors(module) if parent not in seen)
        for name in imported_names(ast.parse(path.read_text(encoding="utf-8"))):
            top = name.split(".")[0]
            if top in LOCAL_PACKAGES:
                pending.append(name)
            elif top not in sys.stdlib_module_names:
                third_party.add(top)
    return seen, third_party


def distribution_name(requirement: str) -> str:
    return requirement.split("==")[0].split(">=")[0].split("[")[0].strip()


def installed_modules(packages: tuple[str, ...]) -> set[str]:
    names = (distribution_name(r) for r in packages)
    return {IMPORT_NAME.get(name, name) for name in names}


@pytest.mark.parametrize(("label", "entrypoint", "packages"), IMAGES, ids=[i[0] for i in IMAGES])
def test_the_entrypoint_is_reachable(
    label: str, entrypoint: str, packages: tuple[str, ...]
) -> None:
    assert module_path(entrypoint) is not None, label
    assert packages


def test_the_closure_reaches_the_training_stack() -> None:
    local, _ = import_closure(ENTRYPOINT)
    assert "agbalu.model.trainer" in local
    assert "agbalu.normalise.rules" in local, "the transitive reach this test exists for"


def test_the_bench_closure_reaches_the_scoring_stack() -> None:
    local, _ = import_closure("modal_app.bench")
    assert "agbalu.bench.mt" in local
    assert "agbalu.bench.translate" in local


def test_the_closure_includes_the_parent_packages_python_executes() -> None:
    """`from agbalu.bench.translate import BASELINES` runs `agbalu/bench/__init__.py`
    first, and that file reaches pyarrow. Walking only the named module made the bench
    image look complete and it failed on the GPU instead."""
    local, third_party = import_closure("modal_app.bench")
    assert "agbalu.bench" in local
    assert "agbalu.extract.readers" in local
    assert "pyarrow" in third_party


@pytest.mark.parametrize(("label", "entrypoint", "packages"), IMAGES, ids=[i[0] for i in IMAGES])
def test_every_reached_third_party_is_installed(
    label: str, entrypoint: str, packages: tuple[str, ...]
) -> None:
    _, third_party = import_closure(entrypoint)
    missing = sorted(third_party - installed_modules(packages) - ALWAYS_PRESENT)
    assert not missing, f"the {label} image does not install: {missing}"


@pytest.mark.parametrize(("label", "entrypoint", "packages"), IMAGES, ids=[i[0] for i in IMAGES])
def test_no_package_is_installed_without_being_reached(
    label: str, entrypoint: str, packages: tuple[str, ...]
) -> None:
    _, third_party = import_closure(entrypoint)
    exempt = third_party | ALWAYS_PRESENT | RUNTIME_ONLY[label]
    unused = sorted(installed_modules(packages) - exempt)
    assert not unused, f"the {label} image installs but never reaches: {unused}"


PREINSTALLED: Final = frozenset({"cd", "cp", "echo", "ln", "mkdir", "mv", "rm", "sh"})
"""Programs `debian_slim` already carries, so a build command may use them unasked. A
compiler, a fetcher or a build system is not on this list and must be installed by name."""


def build_programs(source: str) -> dict[str, tuple[set[str], set[str]]]:
    """Per `*_image`, the packages it apt-installs and the programs its build commands run.

    Read from the source rather than from the `Image` object: modal exposes the layers it
    will build as an opaque chain, and the question here — does step 1 have the binary it
    calls — is answerable from the text.
    """
    found: dict[str, tuple[set[str], set[str]]] = {}
    for node in ast.walk(ast.parse(source)):
        # Every image here is declared `name: Final = (...)`, which is an `AnnAssign`.
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if not name.endswith("_image") or node.value is None:
            continue
        packages: set[str] = set()
        programs: set[str] = set()
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            values = [a.value for a in call.args if isinstance(a, ast.Constant)]
            literals = [value for value in values if isinstance(value, str)]
            if call.func.attr == "apt_install":
                packages.update(literals)
            elif call.func.attr == "run_commands":
                programs.update(
                    step.split()[0] for command in literals for step in command.split("&&")
                )
        if programs:
            found[name] = (packages, programs)
    return found


def test_an_image_installs_every_program_its_build_commands_run() -> None:
    """`git clone` as step 1 of an image with no `git` fails at build time, on the
    operator's account, minutes into a layer that has already been paid for."""
    images = build_programs((ROOT / "modal_app" / "common.py").read_text(encoding="utf-8"))
    assert images, "no image declares run_commands; this check would pass vacuously"

    missing = {
        name: sorted(programs - packages - PREINSTALLED)
        for name, (packages, programs) in images.items()
        if programs - packages - PREINSTALLED
    }
    assert not missing, f"build commands call programs the image never installs: {missing}"


def test_the_resources_directory_shipped_to_the_container_exists() -> None:
    assert (ROOT / RESOURCES / "homoglyphs.yaml").is_file()


def test_the_deploy_module_registers_every_function_without_a_name_collision() -> None:
    """Every module shares one `modal.App`, and `modal deploy -m modal_app.deploy` imports
    all of them at once. A second function of the same name does **not** raise — modal 1.4.2
    logs a warning and *overrides* the first, so the deployed app silently loses one. The
    assertion is therefore equality, not containment: an override shrinks the set."""
    deploy = importlib.import_module("modal_app.deploy")
    assert set(deploy.app.registered_functions) == {
        "jugurtha_pack",
        "jugurtha_train",
        "pretrain",
        "pretrain_smoke",
        "finetune",
        "trim",
        "mt_baselines",
        "inventory",
        "predict",
        "mt_predict",
        "fetch_translations",
        "synthesise",
        "baseline",
        "asr_fetch",
        "asr_finetune",
        "asr_pipeline",
        "asr_repack",
        "asr_score",
        "asr_lm",
        "repack_shard_task",
        "tifinagh_train",
        "tifinagh_evaluate",
        "punctuation_train",
        "punctuation_evaluate",
        "sentiment_label",
        "sentiment_benchmark",
        "tts_baseline",
        "tts_voices",
        "tts_corpus",
        "tts_voice",
        "tts_results",
        "matoub_infer",
        "matoub_prepare",
        "matoub_train",
        "boulifa_prepare",
        "boulifa_train",
        "simohand_prepare",
        "simohand_train",
        "simohand_eval",
        "feraoun_train",
        "feraoun_smoke",
    }

    entrypoints = set(deploy.app.registered_entrypoints)
    assert {
        "upload_corpus",
        "upload_mt",
        "upload_bench",
        "upload_llm",
        "score_mt",
        "fill_mask",
        "run_baseline",
        "run_asr",
        "pack",
    } <= entrypoints
