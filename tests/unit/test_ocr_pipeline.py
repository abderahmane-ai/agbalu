"""The dataset, the checkpoint round trip, and the guards a resumed run rests on."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from agbalu.ocr.config import ModelConfig, TrainConfig
from agbalu.ocr.dataset import (
    SyntheticDataset,
    build_dual_script_lines,
    chunk_text_into_lines,
    collate_lines,
)
from agbalu.ocr.infer import Recognizer, segment_page_into_lines
from agbalu.ocr.models import VisionEncoderDecoder
from agbalu.ocr.trainer import CHECKPOINT_KEYS, ResumeError, Trainer


def test_chunk_text_into_lines() -> None:
    sentences = ["Azul fell-awen amek tellam, taqbaylit d tutlayt tayemmat."]
    lines = chunk_text_into_lines(sentences, min_chars=10, max_chars=30)
    assert len(lines) >= 1
    assert all(len(line) > 0 for line in lines)


def test_synthetic_dataset_and_collate() -> None:
    lines = ["Taqbaylit d tutlayt nneɣ.", "Awal yettazzal deg udlis."]
    dataset = SyntheticDataset(lines, augment=False)
    assert len(dataset) == 2

    pixel_val, tok_ids, text = dataset[0]
    assert pixel_val.shape == (3, 64, 512)
    assert len(tok_ids) > 0
    assert text == lines[0]

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_lines)
    batch = next(iter(loader))
    assert batch["pixel_values"].shape == (2, 3, 64, 512)
    assert batch["labels"].shape[0] == 2


def test_recognizer_and_segmentation() -> None:
    config = ModelConfig(
        hidden_size=64,
        num_decoder_layers=2,
        num_decoder_heads=2,
        intermediate_size=128,
    )
    model = VisionEncoderDecoder(config=config, use_pretrained_encoder=False)
    recognizer = Recognizer(model=model, device=torch.device("cpu"))

    canvas = Image.new("RGB", (512, 64), color="white")
    text = recognizer.recognize_line(canvas)
    assert isinstance(text, str)

    page = Image.new("RGB", (600, 800), color="white")
    lines = segment_page_into_lines(page)
    assert len(lines) >= 1


def test_build_dual_script_lines_ratio() -> None:
    latin = ["Azul fell-awen amek tellam."] * 50
    tifinagh = ["ⴰⵣⵓⵍ ⴼⵍⵍ-ⴰⵡⴻⵏ ⴰⵎⴻⴽ ⵜⴻⵍⵍⴰⵎ."] * 50
    mixed = build_dual_script_lines(latin, tifinagh, tifinagh_ratio=0.25, max_lines=20)
    assert len(mixed) == 20
    tif_count = sum(1 for line in mixed if any("ⴰ" <= c <= "ⵥ" for c in line))
    assert tif_count == 5


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=64,
        num_decoder_layers=2,
        num_decoder_heads=2,
        intermediate_size=128,
    )


def _tiny_trainer(
    tmp_path: Path,
    *,
    resume: Path | None = None,
    epochs: int = 1,
    max_steps: int | None = 1,
    saved: list[Path] | None = None,
) -> Trainer:
    model = VisionEncoderDecoder(config=_tiny_config(), use_pretrained_encoder=False)
    loader = DataLoader(
        SyntheticDataset(["Azul fell-awen", "Taqbaylit d tutlayt"], augment=False),
        batch_size=1,
        collate_fn=collate_lines,
    )
    config = TrainConfig(
        output_dir=tmp_path / "runs",
        resume_checkpoint=resume,
        epochs=epochs,
        max_steps=max_steps,
        eval_every_steps=1,
        eval_batches=1,
        save_every_steps=1,
        log_every_steps=1,
        device="cpu",
    )
    return Trainer(
        model,
        loader,
        loader,
        config,
        on_checkpoint=None if saved is None else saved.append,
    )


def test_a_weights_only_checkpoint_is_refused_rather_than_resumed_from(tmp_path: Path) -> None:
    """Loaded loosely, a weights-only file restores the weights and silently discards the
    optimizer, the schedule and the step count — a run that reports itself as a
    continuation and is a restart at full learning rate."""
    model = VisionEncoderDecoder(config=_tiny_config(), use_pretrained_encoder=False)
    weights_only = tmp_path / "mock_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, weights_only)

    with pytest.raises(ResumeError, match="optimizer_state_dict"):
        _tiny_trainer(tmp_path, resume=weights_only).fit()


def test_a_run_writes_a_resumable_checkpoint_and_the_hook_sees_every_one(
    tmp_path: Path,
) -> None:
    saved: list[Path] = []
    results = _tiny_trainer(tmp_path, saved=saved).fit()

    assert results["total_steps"] == 1
    assert results["steps_this_run"] == 1
    assert [path.name for path in saved] == ["best.pt", "latest.pt", "final.pt"]

    written = torch.load(tmp_path / "runs" / "final.pt", map_location="cpu", weights_only=False)
    assert set(CHECKPOINT_KEYS) <= set(written)


def test_a_resumed_run_continues_the_step_count_it_was_interrupted_at(tmp_path: Path) -> None:
    """Continuing means raising `epochs` and pointing at the checkpoint, as every other
    trainer here does. The step count carries over; it does not restart at one."""
    first = _tiny_trainer(tmp_path, epochs=1, max_steps=None).fit()
    assert first["total_steps"] == 2
    assert first["steps_this_run"] == 2

    resumed = _tiny_trainer(
        tmp_path,
        resume=tmp_path / "runs" / "final.pt",
        epochs=2,
        max_steps=None,
    ).fit()
    assert resumed["total_steps"] == 4
    assert resumed["steps_this_run"] == 2


def test_a_resumed_run_already_at_max_steps_trains_nothing_and_says_so(tmp_path: Path) -> None:
    """Without the guard it returns the checkpoint's counters as this run's measurement."""
    _tiny_trainer(tmp_path, max_steps=2).fit()
    again = _tiny_trainer(tmp_path, resume=tmp_path / "runs" / "final.pt", max_steps=2).fit()

    assert again["total_steps"] == 2
    assert again["steps_this_run"] == 0


def test_an_unmeasured_best_cer_is_none_and_never_zero(tmp_path: Path) -> None:
    """0.0 is a perfect transcription. A run whose evaluation never fired has not measured
    one, and the two must not print the same."""
    trainer = _tiny_trainer(tmp_path, max_steps=1)
    object.__setattr__(trainer.config, "eval_every_steps", 10_000)
    assert trainer.fit()["best_cer"] is None
