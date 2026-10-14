"""The trainer on a tiny real Qwen3.5: steps, checkpoints, resume, locks and the guard."""

from __future__ import annotations

import json
import os
import signal
import socket
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from agbalu.llm import train
from tests.unit import tiny_lm

CPU = torch.device("cpu")
LENGTH = 16


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_lm.save(tmp_path_factory.mktemp("tiny"))


def blocks(path: Path, count: int = 12, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    array = rng.integers(1, tiny_lm.VOCABULARY, size=(count, LENGTH), dtype=np.uint32)
    array[:, ::5] = tiny_lm.EOS
    np.save(path, array)
    return path


def config(tmp_path: Path, model_dir: Path) -> train.RunConfig:
    return train.RunConfig(
        run="run",
        model=str(model_dir),
        blocks=blocks(tmp_path / "blocks.npy"),
        optimizer="adamw",
        blocks_per_step=2,
        micro_blocks=1,
        total_steps=4,
        checkpoint_every=2,
        eval_every=0,
        probe=False,
        keep=1,
        workers=0,
        runs_root=tmp_path / "runs",
    )


def fit(cfg: train.RunConfig) -> dict[str, object]:
    return train.train(cfg, tiny_lm.load, tiny_lm.CharTokenizer(), CPU)


class TestRun:
    def test_a_run_reaches_its_schedule_and_accounts_for_its_tokens(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        cfg = config(tmp_path, model_dir)
        result = fit(cfg)
        assert (result["step"], result["ended"]) == (4, "schedule")
        assert result["tokens_total"] == 4 * 2 * LENGTH
        losses = train.losses(cfg.directory / "log.jsonl")
        assert sorted(losses) == [1, 2, 3, 4]
        assert all(np.isfinite(list(losses.values())))
        assert [p.name for p in train.checkpoints(cfg.directory)] == ["step-0000004"]
        assert json.loads((cfg.directory / "summary.json").read_text())["step"] == 4
        assert not (cfg.directory / "lock").exists()

    def test_loss_falls_on_repeated_data(self, tmp_path: Path, model_dir: Path) -> None:
        path = tmp_path / "same.npy"
        np.save(path, np.tile(np.arange(LENGTH, dtype=np.uint32) % 7 + 1, (40, 1)))
        cfg = replace(config(tmp_path, model_dir), blocks=path, total_steps=20, peak_lr=3e-3)
        fit(cfg)
        losses = train.losses(cfg.directory / "log.jsonl")
        assert losses[20] < losses[1] - 0.5

    def test_a_restarted_run_matches_an_uninterrupted_one(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        straight = replace(config(tmp_path, model_dir), run="straight")
        fit(straight)
        stopped = replace(config(tmp_path, model_dir), run="stopped", stop_after=2)
        first = fit(stopped)
        assert first["ended"] == "stop_after"
        fit(replace(stopped, stop_after=None))
        a = train.losses(straight.directory / "log.jsonl")
        b = train.losses(stopped.directory / "log.jsonl")
        assert a.keys() == b.keys()
        for step in (3, 4):
            assert b[step] == pytest.approx(a[step], abs=1e-5)
        assert abs(b[4] - a[3]) > 1e-3, "the shifted control must differ"

    def test_a_resume_with_a_changed_config_is_refused(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        cfg = replace(config(tmp_path, model_dir), stop_after=2)
        fit(cfg)
        with pytest.raises(train.TrainError, match="peak_lr"):
            fit(replace(cfg, stop_after=None, peak_lr=1e-3))

    def test_more_steps_than_the_blocks_hold_is_refused(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        with pytest.raises(train.TrainError, match="holds 6"):
            fit(replace(config(tmp_path, model_dir), total_steps=7))

    def test_a_signal_ends_the_run_with_a_checkpoint(self, tmp_path: Path, model_dir: Path) -> None:
        stop = train.StopRequest()
        stop.handle(signal.SIGTERM, None)
        cfg = config(tmp_path, model_dir)
        result = train.train(cfg, tiny_lm.load, tiny_lm.CharTokenizer(), CPU, stop=stop)
        assert result["ended"] == f"signal {signal.SIGTERM}"
        assert train.checkpoints(cfg.directory)

    def test_evaluation_keeps_the_best_and_sets_the_reference(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        heldout = tmp_path / "heldout"
        heldout.mkdir()
        for language in ("kab", "eng", "fra"):
            rows = [{"text": f"azul {language} {i} fell-awen"} for i in range(5)]
            (heldout / f"heldout-{language}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
            )
        cfg = replace(config(tmp_path, model_dir), eval_every=2, heldout_dir=heldout)
        fit(cfg)
        lines = (cfg.directory / "log.jsonl").read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines]
        evals = [e for e in events if e["event"] == "eval"]
        assert [e["step"] for e in evals] == [0, 2, 4]
        assert (cfg.directory / "best" / "best.json").is_file()


class TestPieces:
    def test_blocks_returns_the_scheduled_slice(self, tmp_path: Path) -> None:
        path = blocks(tmp_path / "b.npy")
        dataset = train.Blocks(path, 2, tiny_lm.EOS, 1, 3)
        expected = np.load(path)[2:4].reshape(-1)
        assert len(dataset) == 2
        assert dataset[0].input_ids[0].tolist() == expected.tolist()
        with pytest.raises(IndexError):
            dataset[2]

    def test_weight_decay_spares_vectors(self, model_dir: Path) -> None:
        model = tiny_lm.load(model_dir, torch.float32)
        decay, exempt = train.parameter_groups(model)
        params = decay["params"]
        assert isinstance(params, list)
        assert params
        assert all(isinstance(p, torch.Tensor) and p.ndim >= 2 for p in params)
        assert exempt["weight_decay"] == 0.0

    def test_the_guard_allows_five_percent(self) -> None:
        reference = {"eng": 1.0, "fra": 2.0}
        assert train.guard_passes({"kab": 9, "eng": 1.05, "fra": 2.1}, reference)
        assert not train.guard_passes({"kab": 0, "eng": 1.06, "fra": 2.0}, reference)

    def test_a_live_lock_is_refused_and_a_dead_one_taken(self, tmp_path: Path) -> None:
        log = train.EventLog(tmp_path / "log.jsonl")
        path = tmp_path / "lock"
        holder = {"pid": os.getpid(), "host": socket.gethostname(), "boot": train.boot_id()}
        path.write_text(json.dumps(holder))
        with pytest.raises(train.TrainError, match="running pid"), train.RunLock(path, log):
            pass
        path.write_text(json.dumps({**holder, "pid": 2**22 + 7}))
        with train.RunLock(path, log):
            assert json.loads(path.read_text())["pid"] == os.getpid()
        assert not path.exists()

    def test_a_lock_from_before_a_restart_is_taken_even_if_its_pid_is_alive(
        self, tmp_path: Path
    ) -> None:
        log = train.EventLog(tmp_path / "log.jsonl")
        path = tmp_path / "lock"
        holder = {"pid": os.getpid(), "host": socket.gethostname(), "boot": "an earlier boot"}
        path.write_text(json.dumps(holder))
        with train.RunLock(path, log):
            assert json.loads(path.read_text())["boot"] == train.boot_id()

    def test_a_bad_config_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(train.TrainError):
            train.RunConfig(run="r", model="m", blocks=tmp_path, blocks_per_step=3, micro_blocks=2)

    def test_a_blocks_file_of_the_wrong_type_is_refused(self, tmp_path: Path) -> None:
        np.save(tmp_path / "x.npy", np.zeros((2, 4), dtype=np.int64))
        with pytest.raises(train.TrainError, match="uint32"):
            train.block_shape(tmp_path / "x.npy")
