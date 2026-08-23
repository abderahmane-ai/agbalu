from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from agbalu.bench.lid import Example, write_evalset
from agbalu.lid.cli import DEFAULTS, held_out_records, main, parse_args, resolve_device
from agbalu.lid.config import ModelConfig, TrainConfig
from agbalu.lid.dataset import CorpusError, Record
from agbalu.lid.model import CharClassifier

LANGUAGES = ("kab_Latn", "rif_Latn", "tzm_Latn")


def small_config() -> ModelConfig:
    return ModelConfig(
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        intermediate_dim=64,
        conv_dim=8,
        projection_dim=16,
        max_length=64,
        classes=LANGUAGES,
    )


def write_split(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "text": r.text,
                    "language": r.language,
                    "domain": r.domain,
                    "group": r.group,
                    "source": r.source,
                }
            )
            for r in records
        ),
        encoding="utf-8",
    )


def records(count: int = 30) -> list[Record]:
    return [
        Record(
            text=f"{language} line number {index} of the corpus here",
            language=language,
            domain=domain,
            group=f"{language}/{domain}",
            source=f"{language}.{domain}",
        )
        for index in range(count)
        for language in LANGUAGES
        for domain in ("web", "scripture")
    ]


def stage(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data"
    write_split(data / "train.jsonl", records())
    write_split(data / "dev.jsonl", records(4))
    checkpoints = tmp_path / "run"
    checkpoints.mkdir()
    bundle = torch.nn.Module()
    bundle.add_module("model", CharClassifier(small_config()))
    torch.save(
        {"model": bundle.state_dict(), "optimizer": {}, "state": {"step": 1}, "rng": {}},
        checkpoints / "best.pt",
    )
    (checkpoints / "config.json").write_text(
        json.dumps({"model": asdict(small_config()), "train": asdict(TrainConfig())}),
        encoding="utf-8",
    )
    evalset = tmp_path / "lid-eval.jsonl"
    write_evalset(
        [
            Example(f"{language} held out line {index}", language, "s", "web")
            for index in range(5)
            for language in LANGUAGES
        ],
        evalset,
    )
    return data, checkpoints, evalset


class TestDefaults:
    def test_the_defaults_come_from_an_instance_not_the_slotted_class(self) -> None:
        """`slots=True` makes class-level access return the slot descriptor, which
        argparse would pass straight into `torch.manual_seed`."""
        assert isinstance(DEFAULTS.seed, int)
        assert isinstance(DEFAULTS.epochs, int)
        assert isinstance(DEFAULTS.learning_rate, float)

    def test_the_parser_carries_those_values(self) -> None:
        args = parse_args(["train"])
        assert args.seed == DEFAULTS.seed
        assert args.epochs == DEFAULTS.epochs


class TestDeviceResolution:
    def test_an_explicit_device_is_honoured(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_resolves_to_something_torch_accepts(self) -> None:
        assert resolve_device("auto").type in {"cpu", "cuda"}


class TestHeldOutRecords:
    def test_the_domain_becomes_the_group(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.jsonl"
        write_evalset([Example("a line here", "rif_Latn", "tatoeba.rif", "everyday")], path)
        assert held_out_records(path)[0].group == "rif_Latn/everyday"

    def test_a_missing_set_names_the_command_that_builds_it(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusError, match="make bench TASK=lid"):
            held_out_records(tmp_path / "absent.jsonl")

    def test_an_empty_set_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "eval.jsonl").write_text("\n\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="carried no rows"):
            held_out_records(tmp_path / "eval.jsonl")


class TestCommands:
    def test_evaluate_writes_a_report_with_per_group_recall(self, tmp_path: Path) -> None:
        data, checkpoints, evalset = stage(tmp_path)
        code = main(
            [
                "--checkpoint-dir",
                str(checkpoints),
                "--data-dir",
                str(data),
                "--device",
                "cpu",
                "evaluate",
                "--eval-path",
                str(evalset),
            ]
        )
        assert code == 0
        report = json.loads((checkpoints / "held-out-report.json").read_text(encoding="utf-8"))
        assert report["per_group"]
        assert set(report["per_class"]) == set(LANGUAGES)

    def test_predict_classifies_without_a_calibration(self, tmp_path: Path) -> None:
        data, checkpoints, _ = stage(tmp_path)
        code = main(
            [
                "--checkpoint-dir",
                str(checkpoints),
                "--data-dir",
                str(data),
                "--device",
                "cpu",
                "predict",
                "--text",
                "Ur ten-teg ca da",
                "azul fell-awen ay imdukal",
            ]
        )
        assert code == 0

    def test_a_missing_checkpoint_returns_one_rather_than_a_traceback(self, tmp_path: Path) -> None:
        code = main(
            [
                "--checkpoint-dir",
                str(tmp_path / "absent"),
                "--device",
                "cpu",
                "predict",
                "--text",
                "awal",
            ]
        )
        assert code == 1

    def test_calibrate_writes_the_quantile_beside_the_checkpoint(self, tmp_path: Path) -> None:
        data, checkpoints, _ = stage(tmp_path)
        write_split(data / "dev.jsonl", records(20))
        code = main(
            [
                "--checkpoint-dir",
                str(checkpoints),
                "--data-dir",
                str(data),
                "--device",
                "cpu",
                "calibrate",
                "--coverage",
                "0.9",
            ]
        )
        assert code == 0
        stored = json.loads((checkpoints / "calibration.json").read_text(encoding="utf-8"))
        assert 0.0 <= stored["q_hat"] <= 1.0
        assert stored["fitted_on"] > 0
        assert stored["measured_on"] > 0
