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

from agbalu.llm import sft_rows, train
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


STORED_CPT_CONFIG = {
    "run": "jugurtha-4b-v1",
    "model": "Qwen/Qwen3.5-4B-Base",
    "blocks": "data/processed/llm/cpt-v2.npy",
    "optimizer": "adamw8bit",
    "peak_lr": 2e-05,
    "blocks_per_step": 64,
    "micro_blocks": 2,
    "total_steps": None,
    "checkpoint_every": 20,
    "eval_every": 50,
    "heldout_limit": None,
    "probe": True,
    "keep": 2,
}
"""`artifacts/runs/jugurtha-4b-v1/config.json` as the running full CPT stored it, 2026-10-15.
Every restart resumes against it; a field added to `RunConfig` must leave this identity alone."""


class TestIdentity:
    def test_the_running_cpt_run_still_resumes(self) -> None:
        cfg = train.RunConfig(
            run="jugurtha-4b-v1",
            model="Qwen/Qwen3.5-4B-Base",
            blocks=Path("data/processed/llm/cpt-v2.npy"),
        )
        assert cfg.identity() == STORED_CPT_CONFIG

    def test_an_sft_identity_names_its_objective_and_validation(self) -> None:
        cfg = train.RunConfig(
            run="s", model="m", blocks=Path("r.npz"), objective="sft", validation=Path("v.npz")
        )
        identity = cfg.identity()
        assert (identity["objective"], identity["validation"]) == ("sft", "v.npz")

    def test_an_sft_run_without_validation_is_refused(self) -> None:
        with pytest.raises(train.TrainError, match="validation rows"):
            train.RunConfig(run="s", model="m", blocks=Path("r.npz"), objective="sft")

    def test_the_schedule_follows_the_objective(self) -> None:
        cpt = train.RunConfig(run="c", model="m", blocks=Path("b.npy"))
        sft = replace(cpt, objective="sft", validation=Path("v.npz"))
        assert (cpt.warmup_fraction, cpt.weight_decay) == (0.01, 0.1)
        assert (sft.warmup_fraction, sft.weight_decay) == (0.03, 0.01)


def sft_segments(count: int, seed: int) -> list[sft_rows.Segment]:
    rng = np.random.default_rng(seed)
    segments = []
    for index in range(count):
        prompt = rng.integers(1, tiny_lm.VOCABULARY, size=5).tolist()
        answer = rng.integers(1, tiny_lm.VOCABULARY, size=4).tolist()
        ids = tuple(int(t) for t in [*prompt, *answer])
        mask = tuple([False] * 5 + [True] * 4)
        segments.append(sft_rows.Segment(ids, mask, sft_rows.NO_THINK, f"c{index}"))
    return segments


def sft_config(tmp_path: Path, model_dir: Path) -> train.RunConfig:
    rows = tmp_path / "sft.npz"
    held = tmp_path / "sft-validation.npz"
    sft_rows.write(sft_rows.pack(sft_segments(24, 0), capacity=LENGTH * 2, seed=0), rows)
    sft_rows.write(sft_rows.pack(sft_segments(4, 1), capacity=LENGTH * 2, seed=0), held)
    return replace(
        config(tmp_path, model_dir),
        run="sft",
        blocks=rows,
        validation=held,
        objective="sft",
        total_steps=None,
        peak_lr=1e-3,
    )


class TestSft:
    def test_an_sft_run_trains_on_supervised_tokens_only(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        cfg = sft_config(tmp_path, model_dir)
        rows = sft_rows.Rows.read(cfg.blocks)
        result = fit(cfg)
        steps = len(train.sft_order(len(rows))) // cfg.micros_per_step
        assert (result["step"], result["ended"]) == (steps, "schedule")
        state = json.loads(
            (train.checkpoints(cfg.directory)[-1] / train.STATE_FILE).read_text(encoding="utf-8")
        )
        used = train.sft_order(len(rows))[: steps * cfg.micros_per_step]
        assert state["targets"] == sum(rows.packed(i).targets for i in used)
        assert state["targets"] < state["tokens"] / 2

    def test_the_best_checkpoint_is_chosen_on_validation_loss(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        cfg = replace(sft_config(tmp_path, model_dir), eval_every=2)
        fit(cfg)
        events = [
            json.loads(line)
            for line in (cfg.directory / "log.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        evals = [e for e in events if e["event"] == "eval"]
        assert all("validation_loss" in e for e in evals)
        best = json.loads((cfg.directory / "best" / "best.json").read_text(encoding="utf-8"))
        assert best["validation_loss"] == min(e["validation_loss"] for e in evals)

    def test_a_resumed_sft_run_reads_the_same_rows(self, tmp_path: Path, model_dir: Path) -> None:
        straight = replace(sft_config(tmp_path, model_dir), run="straight")
        fit(straight)
        stopped = replace(straight, run="stopped", stop_after=2)
        fit(stopped)
        fit(replace(stopped, stop_after=None))
        a = train.losses(straight.directory / "log.jsonl")
        b = train.losses(stopped.directory / "log.jsonl")
        assert a.keys() == b.keys()
        for step in sorted(a)[2:]:
            assert b[step] == pytest.approx(a[step], abs=1e-5)

    def test_the_row_order_is_a_seeded_permutation_per_epoch(self) -> None:
        order = train.sft_order(10, epochs=2, seed=3)
        assert sorted(order[:10]) == list(range(10))
        assert sorted(order[10:]) == list(range(10))
        assert order[:10] != order[10:]
        assert order == train.sft_order(10, epochs=2, seed=3)

    def test_a_model_on_disk_is_held_to_its_expected_count(
        self, tmp_path: Path, model_dir: Path
    ) -> None:
        cfg = replace(sft_config(tmp_path, model_dir), expected_parameters=7)
        with pytest.raises(train.TrainError, match="trainable parameters"):
            fit(cfg)


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
        decay, exempt = train.parameter_groups(model, 0.1)
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
