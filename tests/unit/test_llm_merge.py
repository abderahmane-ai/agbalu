"""The task-vector merge: exactness at the controls, the arithmetic, and every refusal."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from agbalu.llm import merge

TEXT = merge.TEXT_PREFIX
ADDED = [{"id": 2, "content": "<x>"}]
MODEL: dict[str, object] = {"vocab": {"a": 0, "ɣ": 1}}
TOKENIZER: Mapping[str, object] = {"model": MODEL, "added_tokens": ADDED}

Three = tuple[Path, Path, Path]


def tensors(seed: int, *, text_only: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    values = {
        f"{TEXT}embed_tokens.weight": torch.randn(4, 3, generator=generator),
        f"{TEXT}layers.0.mlp.weight": torch.randn(3, 3, generator=generator),
        f"{TEXT}norm.weight": torch.randn(3, generator=generator),
    }
    if not text_only:
        values["model.visual.patch.weight"] = torch.randn(2, 2, generator=generator)
        values["mtp.fc.weight"] = torch.randn(3, 6, generator=generator)
    return {name: value.to(torch.bfloat16) for name, value in values.items()}


def write(
    root: Path,
    values: dict[str, torch.Tensor],
    *,
    sharded: bool = True,
    tied: bool = True,
    tokenizer: Mapping[str, object] = TOKENIZER,
) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"tie_word_embeddings": tied}), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps(dict(tokenizer)), encoding="utf-8")
    if not sharded:
        save_file(values, root / merge.SINGLE_FILE)
        return root
    names = sorted(values)
    first, second = names[: len(names) // 2], names[len(names) // 2 :]
    weight_map: dict[str, str] = {}
    for shard, group in (("part-1.safetensors", first), ("part-2.safetensors", second)):
        save_file({name: values[name] for name in group}, root / shard)
        weight_map.update(dict.fromkeys(group, shard))
    index = {"metadata": {}, "weight_map": weight_map}
    (root / merge.INDEX_FILE).write_text(json.dumps(index), encoding="utf-8")
    return root


def read(root: Path, name: str) -> torch.Tensor:
    checkpoint = merge.Checkpoint.read(root)
    with safe_open(root / checkpoint.shards[name], framework="pt") as handle:
        tensor: torch.Tensor = handle.get_tensor(name)
    return tensor


def run(base: Path, cpt: Path, instruct: Path, alpha: float, out: Path) -> dict[str, object]:
    return merge.merge(merge.Inputs(base, cpt, instruct), alpha, out)


def checkpoints(
    tmp_path: Path,
    *,
    cpt: dict[str, torch.Tensor] | None = None,
    instruct: dict[str, torch.Tensor] | None = None,
    base_tied: bool = True,
    instruct_tokenizer: Mapping[str, object] = TOKENIZER,
) -> Three:
    return (
        write(tmp_path / "base", tensors(1), tied=base_tied),
        write(tmp_path / "cpt", cpt or tensors(2, text_only=True), sharded=False),
        write(tmp_path / "instruct", instruct or tensors(3), tokenizer=instruct_tokenizer),
    )


@pytest.fixture
def three(tmp_path: Path) -> Three:
    return checkpoints(tmp_path)


class TestControls:
    def test_alpha_zero_reproduces_instruct_exactly(self, three: Three, tmp_path: Path) -> None:
        run(*three, 0.0, tmp_path / "out")
        assert merge.same_tensors(tmp_path / "out", three[2]) == []

    def test_an_unchanged_cpt_reproduces_instruct_at_any_alpha(self, tmp_path: Path) -> None:
        base, cpt, instruct = checkpoints(tmp_path, cpt=tensors(1, text_only=True))
        run(base, cpt, instruct, 1.0, tmp_path / "out")
        assert merge.same_tensors(tmp_path / "out", instruct) == []

    def test_the_comparison_sees_every_changed_tensor(self, three: Three, tmp_path: Path) -> None:
        run(*three, 0.25, tmp_path / "out")
        changed = sorted(name for name in tensors(3) if name.startswith(TEXT))
        assert merge.same_tensors(tmp_path / "out", three[2]) == changed


class TestArithmetic:
    def test_text_tensors_are_instruct_plus_alpha_delta(self, three: Three, tmp_path: Path) -> None:
        base, cpt, instruct = three
        run(base, cpt, instruct, 0.5, tmp_path / "out")
        name = f"{TEXT}layers.0.mlp.weight"
        i, b, c = (read(root, name).float() for root in (instruct, base, cpt))
        expected = (i + 0.5 * (c - b)).to(torch.bfloat16)
        assert torch.equal(read(tmp_path / "out", name), expected)

    def test_vision_and_mtp_are_copied_from_instruct(self, three: Three, tmp_path: Path) -> None:
        run(*three, 1.0, tmp_path / "out")
        for name in ("model.visual.patch.weight", "mtp.fc.weight"):
            assert torch.equal(read(tmp_path / "out", name), read(three[2], name))

    def test_report_accounts_for_every_tensor(self, three: Three, tmp_path: Path) -> None:
        report = run(*three, 0.75, tmp_path / "out")
        assert report["tensors"] == {"merged": 3, "copied": 2, "total": 5}
        stored = json.loads((tmp_path / "out" / merge.REPORT_FILE).read_text(encoding="utf-8"))
        assert stored["alpha"] == 0.75
        assert set(stored["outputs"]) == {"part-1.safetensors", "part-2.safetensors"}
        assert (tmp_path / "out" / merge.INDEX_FILE).is_file()
        assert "config.json" in stored["files_copied"]

    def test_the_change_tally_is_zero_at_alpha_zero(self, three: Three, tmp_path: Path) -> None:
        change = run(*three, 0.0, tmp_path / "out")["change"]
        assert isinstance(change, dict)
        assert change["largest_relative_change"] == 0.0

    def test_a_stale_partial_directory_is_replaced(self, three: Three, tmp_path: Path) -> None:
        stale = tmp_path / ".out.partial"
        stale.mkdir()
        (stale / "junk").write_text("x", encoding="utf-8")
        run(*three, 0.0, tmp_path / "out")
        assert not stale.exists()
        assert not (tmp_path / "out" / "junk").exists()


class TestTokenizers:
    def test_instruct_may_add_tokens_and_spell_out_a_default(self, tmp_path: Path) -> None:
        model = {**MODEL, "ignore_merges": False}
        added = [*ADDED, {"id": 3, "content": "<think>"}]
        roots = checkpoints(tmp_path, instruct_tokenizer={"model": model, "added_tokens": added})
        report = run(*roots, 0.5, tmp_path / "out")
        assert report["instruct_added_tokens"] == ["<think>"]

    def test_a_default_spelled_differently_is_a_difference(self, tmp_path: Path) -> None:
        model = {**MODEL, "ignore_merges": True}
        roots = checkpoints(tmp_path, instruct_tokenizer={"model": model, "added_tokens": ADDED})
        with pytest.raises(merge.MergeError, match="ignore_merges"):
            run(*roots, 0.5, tmp_path / "out")


class TestKeyGaps:
    def test_the_tied_head_is_not_a_gap(self) -> None:
        assert merge.key_gaps({"a", merge.TIED_HEAD}, {"a"}) == (set(), set())

    def test_both_directions_are_reported(self) -> None:
        assert merge.key_gaps({"a", "b"}, {"a", "c"}) == ({"c"}, {"b"})

    def test_a_tensor_the_class_declares_dropped_is_not_a_gap(self) -> None:
        stored = {"a", "mtp.fc.weight", "mtp.layers.0.x"}
        assert merge.key_gaps({"a"}, stored, [r"^mtp.*"]) == (set(), set())

    def test_an_undeclared_drop_is_still_a_gap(self) -> None:
        stored = {"a", "mtp.fc.weight", "model.visual.patch.weight"}
        assert merge.key_gaps({"a"}, stored, [r"^mtp.*"]) == ({"model.visual.patch.weight"}, set())

    def test_the_text_tower_is_named_as_the_text_class_loads_it(self) -> None:
        names = {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
        assert merge.text_names(names) == {"model.embed_tokens.weight", "model.norm.weight"}

    def test_file_names_against_text_class_names_are_all_gaps(self) -> None:
        stored = {"model.language_model.norm.weight"}
        assert merge.key_gaps({"model.norm.weight"}, merge.text_names(stored)) == (set(), set())
        assert merge.key_gaps({"model.norm.weight"}, stored) == (stored, {"model.norm.weight"})

    def test_without_declarations_mtp_is_a_gap(self) -> None:
        assert merge.key_gaps({"a"}, {"a", "mtp.fc.weight"}) == ({"mtp.fc.weight"}, set())


class TestRefusals:
    @pytest.mark.parametrize("alpha", [-0.1, 1.5, math.nan, math.inf])
    def test_alpha_outside_the_range(self, three: Three, tmp_path: Path, alpha: float) -> None:
        with pytest.raises(merge.MergeError, match="outside"):
            run(*three, alpha, tmp_path / "out")

    def test_an_existing_output(self, three: Three, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        with pytest.raises(merge.MergeError, match="never overwrites"):
            run(*three, 0.5, tmp_path / "out")

    def test_a_text_tensor_missing_from_cpt_is_named(self, tmp_path: Path) -> None:
        values = tensors(2, text_only=True)
        del values[f"{TEXT}norm.weight"]
        with pytest.raises(merge.MergeError, match=r"norm\.weight"):
            run(*checkpoints(tmp_path, cpt=values), 0.5, tmp_path / "out")

    def test_a_shape_difference(self, tmp_path: Path) -> None:
        values = tensors(2, text_only=True)
        values[f"{TEXT}norm.weight"] = torch.zeros(4, dtype=torch.bfloat16)
        with pytest.raises(merge.MergeError, match="shape or dtype"):
            run(*checkpoints(tmp_path, cpt=values), 0.5, tmp_path / "out")

    def test_a_dtype_difference_in_instruct(self, tmp_path: Path) -> None:
        values = tensors(3)
        values[f"{TEXT}norm.weight"] = values[f"{TEXT}norm.weight"].float()
        with pytest.raises(merge.MergeError, match="instruct: 1 tensors differ"):
            run(*checkpoints(tmp_path, instruct=values), 0.5, tmp_path / "out")

    def test_a_non_float_dtype_difference_in_cpt(self, tmp_path: Path) -> None:
        values = tensors(2, text_only=True)
        values[f"{TEXT}norm.weight"] = values[f"{TEXT}norm.weight"].to(torch.int32)
        with pytest.raises(merge.MergeError, match="cpt: 1 tensors differ"):
            run(*checkpoints(tmp_path, cpt=values), 0.5, tmp_path / "out")


class TestRecast:
    NAME = f"{TEXT}norm.weight"

    PRECISE = torch.tensor([0.1234567, -0.7654321, 0.3333333], dtype=torch.float32)
    """Values bf16 cannot hold exactly, as Qwen3.5 stores `A_log` in fp32."""

    def three(self, tmp_path: Path, cpt_value: torch.Tensor) -> Three:
        """Base and instruct hold the tensor in fp32; the CPT holds `cpt_value` in bf16, as
        training saved it."""
        base, instruct, cpt = tensors(1), tensors(3), tensors(1, text_only=True)
        base[self.NAME], instruct[self.NAME] = self.PRECISE, self.PRECISE + 1
        cpt[self.NAME] = cpt_value
        return (
            write(tmp_path / "base", base),
            write(tmp_path / "cpt", cpt, sharded=False),
            write(tmp_path / "instruct", instruct),
        )

    def test_an_untrained_value_rounded_to_bf16_moves_nothing(self, tmp_path: Path) -> None:
        three = self.three(tmp_path, self.PRECISE.to(torch.bfloat16))
        report = run(*three, 1.0, tmp_path / "out")
        assert merge.same_tensors(tmp_path / "out", three[2]) == []
        assert report["cpt_dtype_differs"] == {"count": 1, "first": [self.NAME]}
        assert read(tmp_path / "out", self.NAME).dtype == torch.float32

    def test_a_trained_value_moves_by_its_delta_from_the_rounded_base(self, tmp_path: Path) -> None:
        rounded = self.PRECISE.to(torch.bfloat16)
        three = self.three(tmp_path, rounded + 0.5)
        run(*three, 0.5, tmp_path / "out")
        delta = (rounded + 0.5).float() - rounded.float()
        assert torch.equal(read(tmp_path / "out", self.NAME), self.PRECISE + 1 + 0.5 * delta)

    def test_untied_embeddings(self, tmp_path: Path) -> None:
        with pytest.raises(merge.MergeError, match="tie_word_embeddings"):
            run(*checkpoints(tmp_path, base_tied=False), 0.5, tmp_path / "out")

    def test_an_lm_head_beside_tied_embeddings(self, tmp_path: Path) -> None:
        values = tensors(3)
        values[merge.TIED_HEAD] = torch.zeros(4, 3, dtype=torch.bfloat16)
        with pytest.raises(merge.MergeError, match="lm_head"):
            run(*checkpoints(tmp_path, instruct=values), 0.5, tmp_path / "out")

    def test_a_cpt_tensor_outside_the_text_tower(self, tmp_path: Path) -> None:
        values = tensors(2, text_only=True)
        values["mtp.fc.weight"] = torch.zeros(3, 6, dtype=torch.bfloat16)
        with pytest.raises(merge.MergeError, match="outside the text tower"):
            run(*checkpoints(tmp_path, cpt=values), 0.5, tmp_path / "out")

    def test_different_tokenizers(self, tmp_path: Path) -> None:
        other = {"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": ADDED}
        roots = checkpoints(tmp_path, instruct_tokenizer=other)
        with pytest.raises(merge.MergeError, match=r"differ in \['model'\]: \['vocab'\]"):
            run(*roots, 0.5, tmp_path / "out")

    def test_an_added_token_of_the_base_changed_in_instruct(self, tmp_path: Path) -> None:
        other = {"model": MODEL, "added_tokens": [{"id": 2, "content": "<y>"}]}
        roots = checkpoints(tmp_path, instruct_tokenizer=other)
        with pytest.raises(merge.MergeError, match=r"\['added_tokens'\] at ids \[2\]"):
            run(*roots, 0.5, tmp_path / "out")

    def test_a_non_finite_result_writes_nothing(self, tmp_path: Path) -> None:
        values = tensors(2, text_only=True)
        values[f"{TEXT}norm.weight"][0] = math.inf
        with pytest.raises(merge.MergeError, match="not finite"):
            run(*checkpoints(tmp_path, cpt=values), 0.5, tmp_path / "out")
        assert not (tmp_path / "out").exists()

    def test_a_shard_named_in_the_index_but_absent(self, tmp_path: Path) -> None:
        root = write(tmp_path / "instruct", tensors(3))
        (root / "part-2.safetensors").unlink()
        with pytest.raises(merge.MergeError, match="absent"):
            merge.Checkpoint.read(root)

    def test_a_directory_without_config(self, tmp_path: Path) -> None:
        tmp_path.joinpath("empty").mkdir()
        with pytest.raises(merge.MergeError, match=r"no config\.json"):
            merge.Checkpoint.read(tmp_path / "empty")
