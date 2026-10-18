"""The full model rebuilt from a trained text tower: every tensor accounted for, both ways."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from agbalu.llm import assemble, merge

TEXT = merge.TEXT_PREFIX
EMBED = f"{TEXT}embed_tokens.weight"
A_LOG = f"{TEXT}layers.0.linear_attn.A_log"


def full(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        EMBED: torch.randn(4, 3, generator=generator).to(torch.bfloat16),
        A_LOG: torch.randn(3, generator=generator),
        "model.visual.patch.weight": torch.randn(2, 2, generator=generator).to(torch.bfloat16),
        "mtp.fc.weight": torch.randn(3, 6, generator=generator).to(torch.bfloat16),
    }


def trained(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        EMBED: torch.randn(4, 3, generator=generator).to(torch.bfloat16),
        A_LOG: torch.randn(3, generator=generator).to(torch.bfloat16),
    }


def write_full(root: Path, values: dict[str, torch.Tensor]) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"architectures": ["Full"]}), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    names = sorted(values)
    weight_map: dict[str, str] = {}
    for shard, group in (("a.safetensors", names[:2]), ("b.safetensors", names[2:])):
        save_file({name: values[name] for name in group}, root / shard)
        weight_map.update(dict.fromkeys(group, shard))
    (root / merge.INDEX_FILE).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return root


def write_tower(root: Path, values: dict[str, torch.Tensor]) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"architectures": ["Text"]}), encoding="utf-8")
    save_file(values, root / merge.SINGLE_FILE)
    return root


def read(root: Path, name: str) -> torch.Tensor:
    checkpoint = merge.Checkpoint.read(root)
    with safe_open(root / checkpoint.shards[name], framework="pt") as handle:
        tensor: torch.Tensor = handle.get_tensor(name)
    return tensor


class TestAssembly:
    def test_the_tower_replaces_the_text_and_everything_else_is_copied(
        self, tmp_path: Path
    ) -> None:
        template = write_full(tmp_path / "merge", full(1))
        tower_values = trained(2)
        tower = write_tower(tmp_path / "best", tower_values)
        out = tmp_path / "run" / "model"
        report = assemble.assemble(template, tower, out)

        assert report["tensors"] == {"from_tower": 2, "copied": 2, "total": 4}
        assert torch.equal(read(out, EMBED), tower_values[EMBED])
        assert read(out, A_LOG).dtype == torch.float32
        assert torch.equal(read(out, A_LOG), tower_values[A_LOG].float())
        assert report["recast_to_template_dtype"] == {"count": 1, "first": [A_LOG]}
        for name in ("model.visual.patch.weight", "mtp.fc.weight"):
            assert torch.equal(read(out, name), full(1)[name])
        assert (out / "chat_template.jinja").read_text(encoding="utf-8") == "{{ messages }}"
        assert json.loads((out / "config.json").read_text(encoding="utf-8")) == {
            "architectures": ["Full"]
        }
        assert (out / assemble.REPORT_FILE).is_file()
        assert not out.with_name(f".{out.name}.partial").exists()


class TestRefusals:
    def test_a_tower_missing_a_text_tensor_is_refused_by_name(self, tmp_path: Path) -> None:
        template = write_full(tmp_path / "merge", full(1))
        tower = write_tower(tmp_path / "best", {EMBED: trained(2)[EMBED]})
        with pytest.raises(merge.MergeError, match=r"missing \[.*A_log"):
            assemble.assemble(template, tower, tmp_path / "out")

    def test_a_tower_with_an_extra_tensor_is_refused_by_name(self, tmp_path: Path) -> None:
        template = write_full(tmp_path / "merge", full(1))
        extra = {**trained(2), "lm_head.weight": torch.zeros(4, 3, dtype=torch.bfloat16)}
        tower = write_tower(tmp_path / "best", extra)
        with pytest.raises(merge.MergeError, match=r"extra \[lm_head\.weight\]"):
            assemble.assemble(template, tower, tmp_path / "out")

    def test_a_changed_shape_is_refused(self, tmp_path: Path) -> None:
        template = write_full(tmp_path / "merge", full(1))
        values = {**trained(2), EMBED: torch.zeros(5, 3, dtype=torch.bfloat16)}
        tower = write_tower(tmp_path / "best", values)
        with pytest.raises(merge.MergeError, match="trained shape"):
            assemble.assemble(template, tower, tmp_path / "out")

    def test_a_non_finite_trained_value_is_refused(self, tmp_path: Path) -> None:
        template = write_full(tmp_path / "merge", full(1))
        values = trained(2)
        values[EMBED][0, 0] = float("nan")
        tower = write_tower(tmp_path / "best", values)
        with pytest.raises(merge.MergeError, match="non-finite"):
            assemble.assemble(template, tower, tmp_path / "out")

    def test_an_existing_output_is_never_overwritten(self, tmp_path: Path) -> None:
        template = write_full(tmp_path / "merge", full(1))
        tower = write_tower(tmp_path / "best", trained(2))
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(merge.MergeError, match="never overwrites"):
            assemble.assemble(template, tower, out)
