"""Staging a release so it loads without this repository installed."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from tools.stage_hub import (
    ARCHITECTURES,
    AUTO_MAP,
    HUB,
    Repo,
    StagingError,
    build_char_tokenizer,
    copy_modules,
    stage,
    write_config,
)

from agbalu.tifinagh.tokenizer import VOCAB_CHARS, CharTokenizer

STAGED: list[Repo] = sorted(AUTO_MAP)


@pytest.mark.parametrize("repo", STAGED)
def test_no_staged_module_imports_this_project(repo: Repo) -> None:
    """The one property the whole package exists for. A single `agbalu` import makes the
    published repository unloadable on every machine that is not this one."""
    for path in (HUB / repo).glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = [
            name
            for node in ast.walk(tree)
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom) and node.level == 0
                else []
            )
        ]
        assert not [name for name in imported if name.split(".")[0] == "agbalu"], path


@pytest.mark.parametrize("repo", STAGED)
def test_every_auto_map_target_names_a_module_that_is_staged(repo: Repo) -> None:
    """`auto_map` is resolved on the downloader's machine against the files in the repo, so
    a target naming a module the staging does not copy fails only after publication."""
    staged = {path.name for path in (HUB / repo).glob("*.py")}
    for target in AUTO_MAP[repo].values():
        module, _, symbol = target.partition(".")
        assert f"{module}.py" in staged
        assert symbol
    assert ARCHITECTURES[repo] in {target.split(".")[-1] for target in AUTO_MAP[repo].values()}


@pytest.mark.parametrize("repo", STAGED)
def test_the_modules_are_copied_verbatim(repo: Repo, tmp_path: Path) -> None:
    """Byte-identical, or the published copy is a fork that the suite no longer covers."""
    copied = copy_modules(repo, tmp_path)
    assert copied
    for name in copied:
        assert (tmp_path / name).read_bytes() == (HUB / repo / name).read_bytes()


def test_staging_a_directory_that_holds_no_export_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StagingError, match=re.escape("no config.json")):
        write_config("juba", tmp_path)


def test_staging_a_missing_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StagingError, match="no staged release"):
        stage("juba", tmp_path / "nowhere")


def test_the_written_config_carries_the_auto_map_and_keeps_the_manifest(
    tmp_path: Path,
) -> None:
    """Masinissa's `config.json` is also its training record — the card promises the full
    validation curve is in it — so staging must add to the manifest, not replace it."""
    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 64, "hidden_size": 16, "training": {"step": 4500}}),
        encoding="utf-8",
    )
    write_config("masinissa", tmp_path)

    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert written["auto_map"] == AUTO_MAP["masinissa"]
    assert written["architectures"] == [ARCHITECTURES["masinissa"]]
    assert written["model_type"] == "masinissa"
    assert written["training"] == {"step": 4500}
    assert written["vocab_size"] == 64


def test_restaging_does_not_nest_the_auto_map_from_the_previous_run(tmp_path: Path) -> None:
    """A staged directory is staged again on every release, and `auto_map` read back in as
    a config field would be written out as a nested one."""
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 64}), encoding="utf-8")
    write_config("masinissa", tmp_path)
    write_config("masinissa", tmp_path)
    assert (
        json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["auto_map"]
        == (AUTO_MAP["masinissa"])
    )


UNICODE_CASES = [
    "ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?",
    "AZUL Fell-awen",
    "",
    " ",
    "\t\r\n",
    "line\nbreak",
    "tecfiḍ",
    VOCAB_CHARS,
    VOCAB_CHARS.upper(),
    chr(0x0130) + "stanbul",
    chr(0xFEFF) + "bom",
    chr(0x200B) + "zwsp",
    chr(0) + "null",
    "日本語",
    "\U0001f389",
    "é",
    "ǄǄ",
    "ẞ",
    "ﬁ",
    "a" * 300,
    "ⵜ" * 400,
]


@pytest.mark.parametrize("text", UNICODE_CASES)
def test_the_published_character_tokenizer_is_the_model_s_own(text: str) -> None:
    """Id for id with `CharTokenizer`, which is what the weights were trained against.
    The ids are positions in an embedding table: a tokenizer that disagrees decodes to
    plausible text in the wrong characters and nothing raises."""
    published = build_char_tokenizer()
    assert published(text)["input_ids"] == CharTokenizer().encode(text)


def test_the_only_codepoint_whose_lowercase_expands_is_handled() -> None:
    """U+0130 is the sole codepoint in Unicode where a `Lowercase` normalizer and a
    per-codepoint `str.lower()` disagree, and it is why the normalizer has two steps."""
    expanding = [point for point in range(0x110000) if len(chr(point).lower()) > 1]
    assert expanding == [0x0130]


def test_the_character_tokenizer_decodes_back_to_the_text() -> None:
    published = build_char_tokenizer()
    text = "tecfiḍ fell-i ?"
    assert published.decode(CharTokenizer().encode(text), skip_special_tokens=True) == text
