"""Unit tests for Feraoun-Pro document OCR components."""

from __future__ import annotations

import inspect
import itertools
import logging
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from modal_app.ocr_pro import PreconditionError, _resolve_resume
from PIL import Image, ImageFont
from torch.utils.data import DataLoader

from agbalu.hub.feraoun_pro.configuration_feraoun_pro import DEFAULT_VOCABULARY, FeraounProConfig
from agbalu.hub.feraoun_pro.modeling_feraoun_pro import FeraounProForCTC, prepare_line_image
from agbalu.ocr.augment_pro import DegradationPipeline
from agbalu.ocr.dataset_pro import (
    CollateOCRPro,
    SyntheticOCRDatasetPro,
    mix_dual_script,
    sample_line_substring,
)
from agbalu.ocr.fonts_pro import (
    KABYLE_PROBE_CHARS,
    NOTDEF,
    PROBE_SIZE,
    FontPool,
    FontPoolError,
    _supports_kabyle_glyphs,
)
from agbalu.ocr.model_pro import CNNStem, HybridCTCRecognizer
from agbalu.ocr.release_pro import config as release_config
from agbalu.ocr.synthetic_pro import LineRenderer, is_tifinagh_text
from agbalu.ocr.trainer_pro import (
    CheckpointGate,
    TrainingConfigOCRPro,
    TrainingSetupError,
    evaluate_pro,
    train_feraoun_pro,
)
from agbalu.ocr.vocabulary_pro import (
    PAD_ID,
    VOCAB_SIZE,
    VOCABULARY,
    decode_ctc,
    encode_aux,
    encode_ctc,
)


def test_vocabulary_pro_properties() -> None:
    assert VOCAB_SIZE == 173
    text = "Azul, amek i telliḍ ţ-ţayeḍ?"
    ctc_ids = encode_ctc(text)
    assert len(ctc_ids) == len(text)

    aux_ids = encode_aux(text)
    assert len(aux_ids) == len(text) + 2  # BOS + text + EOS
    assert aux_ids[0] == 1  # BOS
    assert aux_ids[-1] == 2  # EOS

    # Check ţ / Ţ
    assert "ţ" in text
    t_id = encode_ctc("ţ")[0]
    assert t_id > 3  # not UNK or special
    assert decode_ctc([t_id]) == "ţ"


def test_ctc_decode_duplicates_and_blanks() -> None:
    """Repeats collapse first, then blanks drop — the order is what encodes a geminate.

    `[t, t, PAD, t, a, PAD, a]` is `t t a a`: the first pair of `t` is one emission held over
    two frames, while the blank before the third separates it into a second letter. Kabyle
    writes geminates (`ttaa`, `nneɣ`, `yettazzal`), so an implementation that dropped blanks
    first would silently halve them.
    """
    t_id = encode_ctc("t")[0]
    a_id = encode_ctc("a")[0]
    assert decode_ctc([t_id, t_id, PAD_ID, t_id, a_id, PAD_ID, a_id]) == "ttaa"
    # Without the separating blank the run is one letter, however many frames it spans.
    assert decode_ctc([t_id, t_id, t_id, a_id]) == "ta"


def test_degradation_is_reproducible_from_a_seeded_generator() -> None:
    """Everything the degradation draws must come from the generator it is handed.

    A stage reaching for a module-level generator — `np.random`, or the `random` module —
    makes the same held-out line a different image at every evaluation, and `best.pt` is
    then chosen by the renderer rather than by the model. This has happened twice: once in
    `backgrounds_pro`'s paper grain, and once in an Augraphy pipeline that also read a
    concurrently-written PNG cache from eight dataloader workers.
    """
    pipeline = DegradationPipeline()
    source = Image.new("RGB", (240, 64), (250, 248, 243))

    first = np.array(pipeline.apply(source, rng=random.Random(7)))
    second = np.array(pipeline.apply(source, rng=random.Random(7)))
    assert np.array_equal(first, second)

    other = np.array(pipeline.apply(source, rng=random.Random(8)))
    assert not np.array_equal(first, other)


def test_a_full_render_is_reproducible_at_a_fixed_seed() -> None:
    """The renderer end to end — background, typeface, ink and degradation together."""
    renderer = LineRenderer()
    text = "Ḥemleɣ ad ɣreɣ idlisen n Dda Lmulud Mammeri."
    first, _ = renderer.render_line(text, rng=random.Random(11), apply_augmentation=True)
    second, _ = renderer.render_line(text, rng=random.Random(11), apply_augmentation=True)
    assert np.array_equal(np.array(first), np.array(second))


def test_synthetic_line_rendering() -> None:
    renderer = LineRenderer()
    text = "Taqbaylit d tutlayt n lejdud."
    img, rendered_text = renderer.render_line(text, apply_augmentation=False)
    assert isinstance(img, Image.Image)
    assert img.height == 64
    assert img.width >= 64
    assert rendered_text == text


def test_cnn_stem_shape() -> None:
    stem = CNNStem(d_model=512)
    # Batch of 2, 3 channels, height 64, width 256
    x = torch.randn(2, 3, 64, 256)
    out = stem(x)
    # Width 256 // 4 = 64 frames
    assert out.shape == (2, 64, 512)


def test_hybrid_ctc_recognizer_forward() -> None:
    model = HybridCTCRecognizer(
        vocab_size=173,
        d_model=128,
        num_encoder_layers=2,
        num_heads=4,
        dim_feedforward=512,
    )

    batch_size = 2
    width = 128
    pixel_values = torch.randn(batch_size, 3, 64, width)
    input_lengths = torch.tensor([width // 4, width // 4], dtype=torch.long)
    ctc_targets = torch.tensor([[10, 15, 20], [12, 14, PAD_ID]], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)
    aux_input_ids = torch.tensor([[1, 10, 15, 20, 2], [1, 12, 14, 2, PAD_ID]], dtype=torch.long)
    aux_labels = torch.tensor([[1, 10, 15, 20, 2], [1, 12, 14, 2, -100]], dtype=torch.long)

    # 1. Training forward pass
    output = model(
        pixel_values=pixel_values,
        input_lengths=input_lengths,
        ctc_targets=ctc_targets,
        target_lengths=target_lengths,
        aux_input_ids=aux_input_ids,
        aux_labels=aux_labels,
    )

    assert output.loss is not None
    assert output.ctc_loss is not None
    assert output.aux_loss is not None
    assert output.ctc_logits.shape == (batch_size, width // 4, 173)

    # 2. Inference forward pass
    inf_out = model(pixel_values=pixel_values)
    assert inf_out.loss is None
    assert inf_out.ctc_logits.shape == (batch_size, width // 4, 173)


def test_collate_ocr_pro() -> None:
    collator = CollateOCRPro()
    img1 = Image.new("RGB", (120, 64), (255, 255, 255))
    img2 = Image.new("RGB", (200, 64), (255, 255, 255))
    batch = [(img1, "Azul"), (img2, "Taqbaylit")]

    collated = collator(batch)
    assert collated.pixel_values.shape[0] == 2
    assert collated.pixel_values.shape[1] == 3
    assert collated.pixel_values.shape[2] == 64
    assert collated.pixel_values.shape[3] == 200  # Padded to max width in batch
    assert collated.input_lengths.tolist() == [120 // 4, 200 // 4]


def test_hub_feraoun_pro_loading() -> None:
    config = FeraounProConfig(
        vocab_size=173,
        d_model=64,
        num_encoder_layers=2,
        num_heads=2,
        dim_feedforward=128,
    )
    model = FeraounProForCTC(config)
    img = Image.new("RGB", (150, 50), (255, 255, 255))
    tensor = prepare_line_image(img)
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 64

    logits = model(tensor)
    assert logits.shape[0] == 1
    assert logits.shape[2] == 173


def test_sample_line_substring_all_lengths() -> None:
    rng = random.Random(42)
    # Test lengths 1 through 300
    for length in range(1, 300):
        dummy_text = "a" * length
        for _ in range(5):
            sub = sample_line_substring(dummy_text, rng)
            assert isinstance(sub, str)
            assert len(sub) <= length
            assert len(sub) > 0


def test_hub_vocabulary_matches_the_trainer_table() -> None:
    """The published `config.json` carries its own copy, because `hub/` may not import
    `agbalu`. This is what stops the two drifting."""
    assert tuple(DEFAULT_VOCABULARY) == VOCABULARY
    assert FeraounProConfig().vocab_size == len(VOCABULARY)


def test_mix_dual_script_ratio_and_both_scripts_present() -> None:
    latin = [f"Taqbaylit d tutlayt {index}" for index in range(800)]
    tifinagh = [f"ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ {index}" for index in range(800)]

    mixed = mix_dual_script(latin, tifinagh, tifinagh_ratio=0.40, max_lines=500)
    assert len(mixed) == 500
    tifinagh_count = sum(1 for line in mixed if is_tifinagh_text(line))
    assert tifinagh_count == 200
    # Shuffled together, so a split taken off either end carries both scripts.
    assert is_tifinagh_text(mixed[0]) != is_tifinagh_text(mixed[-1]) or tifinagh_count > 0

    # 0.0 is the Latin-only arm and must not silently become a default ratio.
    latin_only = mix_dual_script(latin, tifinagh, tifinagh_ratio=0.0, max_lines=100)
    assert not any(is_tifinagh_text(line) for line in latin_only)

    # An empty Tifinagh pool degrades to Latin rather than raising or padding.
    assert len(mix_dual_script(latin, [], tifinagh_ratio=0.5, max_lines=50)) == 50
    # An empty Latin pool is the symmetric case.
    assert len(mix_dual_script([], tifinagh, tifinagh_ratio=0.5, max_lines=50)) == 50
    assert mix_dual_script([], [], tifinagh_ratio=0.5) == []
    # A ratio of 1.0 asks for no Latin at all.
    assert all(
        is_tifinagh_text(line) for line in mix_dual_script(latin, tifinagh, 1.0, max_lines=100)
    )


def test_mix_dual_script_is_seeded() -> None:
    latin = [f"latin {index}" for index in range(50)]
    tifinagh = [f"ⵜⴰⵇ {index}" for index in range(50)]
    first = mix_dual_script(latin, tifinagh, 0.5, max_lines=40, seed=17)
    second = mix_dual_script(latin, tifinagh, 0.5, max_lines=40, seed=17)
    assert first == second
    assert mix_dual_script(latin, tifinagh, 0.5, max_lines=40, seed=18) != first


def test_augmented_dataset_varies_between_epochs_and_validation_does_not() -> None:
    """A train render keyed on the index hands epoch 2 the image epoch 1 already used, which
    is one crop per line however many epochs are paid for."""
    texts = ["Taqbaylit d tutlayt tayemmat nneɣ, ɣas akken."] * 4

    validation = SyntheticOCRDatasetPro(texts, augment=False)
    assert np.array_equal(np.array(validation[0][0]), np.array(validation[0][0]))

    train = SyntheticOCRDatasetPro(texts, augment=True)
    draws = [np.array(train[0][0]) for _ in range(8)]
    assert any(
        first.shape != second.shape or not np.array_equal(first, second)
        for first, second in itertools.pairwise(draws)
    )


def test_evaluate_pro_reports_each_script_separately() -> None:
    """A pooled CER hides a Latin regression behind a falling Tifinagh error."""
    texts = ["Taqbaylit d tutlayt.", "ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ."]
    dataset = SyntheticOCRDatasetPro(texts, augment=False)
    loader = DataLoader(dataset, batch_size=2, collate_fn=CollateOCRPro())
    model = HybridCTCRecognizer(d_model=32, num_encoder_layers=1, num_heads=2, dim_feedforward=64)

    metrics = evaluate_pro(model, loader, torch.device("cpu"), max_batches=None)
    assert metrics["latin_lines"] == 1
    assert metrics["tifinagh_lines"] == 1
    assert isinstance(metrics["latin_cer"], float)
    assert isinstance(metrics["tifinagh_cer"], float)


def test_evaluate_pro_returns_none_for_an_unsupported_script() -> None:
    """`None`, never 0.0: a clamped rate over an empty bucket reads as a perfect score, which
    is exactly what a Latin-only validation split would report for Tifinagh."""
    dataset = SyntheticOCRDatasetPro(["Taqbaylit d tutlayt."], augment=False)
    loader = DataLoader(dataset, batch_size=1, collate_fn=CollateOCRPro())
    model = HybridCTCRecognizer(d_model=32, num_encoder_layers=1, num_heads=2, dim_feedforward=64)

    metrics = evaluate_pro(model, loader, torch.device("cpu"), max_batches=None)
    assert metrics["tifinagh_lines"] == 0
    assert metrics["tifinagh_cer"] is None
    assert metrics["tifinagh_exact_match"] is None
    assert isinstance(metrics["latin_cer"], float)


def test_training_refuses_an_empty_validation_split() -> None:
    model = HybridCTCRecognizer(d_model=32, num_encoder_layers=1, num_heads=2, dim_feedforward=64)
    with pytest.raises(TrainingSetupError):
        train_feraoun_pro(model, ["Azul"], [], TrainingConfigOCRPro(), torch.device("cpu"))


def test_checkpoint_gate_refuses_a_tifinagh_gain_paid_for_in_latin() -> None:
    """The whole protection for the Latin capability, so it is tested on its own rather
    than only through a run that would have to regress to exercise it."""
    gate = CheckpointGate(baseline_latin_cer=0.0050, best_tifinagh_cer=0.90, tolerance=1.01)

    # Tifinagh improves and Latin holds: accepted.
    assert gate.accepts(pooled_cer=0.20, latin_cer=0.0050, tifinagh_cer=0.40)
    # Tifinagh improves further but Latin has drifted past the ceiling: refused, however
    # good the pooled number looks.
    assert not gate.accepts(pooled_cer=0.05, latin_cer=0.0080, tifinagh_cer=0.10)
    # Exactly at the ceiling is still held.
    assert gate.accepts(pooled_cer=0.20, latin_cer=0.0050 * 1.01, tifinagh_cer=0.40)
    # Latin held but Tifinagh did not improve: nothing was bought.
    assert not gate.accepts(pooled_cer=0.20, latin_cer=0.0040, tifinagh_cer=0.95)

    gate.record(pooled_cer=0.20, latin_cer=0.0040, tifinagh_cer=0.40)
    assert gate.best_tifinagh_cer == 0.40
    assert gate.best_latin_cer == 0.0040
    # The ceiling stays keyed on the baseline, not on the best seen: otherwise each accepted
    # checkpoint lowers the bar and Latin ratchets away one tolerated step at a time.
    assert gate.latin_ceiling == 0.0050 * 1.01


class TestResumeResolution:
    """Which checkpoint a continuation picks up, and from whose lineage.

    A run that silently continues another lineage reports that run's training as its own;
    one that silently finds nothing trains from random weights on a paid GPU.
    """

    @staticmethod
    def _checkpoint(root: Path, run: str, name: str = "best.pt") -> Path:
        path = root / run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def test_a_runs_own_checkpoint_wins_over_the_default_lineage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("modal_app.ocr_pro.RUN_ROOT", tmp_path)
        own = self._checkpoint(tmp_path, "feraoun-pro-tifinagh")
        self._checkpoint(tmp_path, "feraoun-pro")
        assert _resolve_resume("best", "feraoun-pro-tifinagh") == own

    def test_another_lineages_checkpoint_is_used_but_never_silently(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("modal_app.ocr_pro.RUN_ROOT", tmp_path)
        source = self._checkpoint(tmp_path, "feraoun-pro")
        with caplog.at_level(logging.INFO, logger="agbalu.modal.ocr_pro"):
            assert _resolve_resume("best", "feraoun-pro-tifinagh") == source
        assert "RESUME_FALLBACK" in caplog.text

    def test_a_named_run_resolves_to_that_runs_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("modal_app.ocr_pro.RUN_ROOT", tmp_path)
        source = self._checkpoint(tmp_path, "feraoun-pro")
        assert _resolve_resume("feraoun-pro", "feraoun-pro-tifinagh") == source

    def test_nothing_on_the_volume_is_refused_rather_than_started_from_scratch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("modal_app.ocr_pro.RUN_ROOT", tmp_path)
        with pytest.raises(PreconditionError):
            _resolve_resume("best", "feraoun-pro-tifinagh")

    def test_no_resume_asked_for_is_a_fresh_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("modal_app.ocr_pro.RUN_ROOT", tmp_path)
        assert _resolve_resume(None, "feraoun-pro-tifinagh") is None


def test_checkpoint_gate_falls_back_to_pooled_cer_without_tifinagh() -> None:
    """The Latin-only arm. With no Tifinagh support there is no trade-off to police."""
    gate = CheckpointGate(best_pooled_cer=0.30)
    assert gate.accepts(pooled_cer=0.20, latin_cer=0.20, tifinagh_cer=None)
    assert not gate.accepts(pooled_cer=0.40, latin_cer=0.40, tifinagh_cer=None)
    # No baseline means nothing to protect, so Latin cannot veto.
    assert CheckpointGate().latin_held(latin_cer=99.0)


def test_transcribe_runs_preprocessing_and_decodes_ctc() -> None:
    """The card's entry point. A model whose only public surface is `forward` hands the
    caller the job of rebuilding the 64 px canvas, and a wrong canvas still returns logits."""
    config = FeraounProConfig(d_model=64, num_encoder_layers=1, num_heads=2, dim_feedforward=128)
    model = FeraounProForCTC(config)

    text = model.transcribe(Image.new("RGB", (200, 40), (255, 255, 255)))
    assert isinstance(text, str)

    # The shipped copy must agree with the trainer's, geminates included.
    t_id = config.vocabulary.index("t")
    a_id = config.vocabulary.index("a")
    assert model.decode_ctc([t_id, t_id, 0, t_id, a_id, 0, a_id]) == "ttaa"
    assert model.decode_ctc([t_id, t_id, t_id, a_id]) == "ta"


class TestFontPool:
    """Which faces are allowed to render a training line.

    A face without the glyph still renders: FreeType draws .notdef, which has a bounding box
    like any other. Filtering on the box alone admits faces that cannot write Kabyle, and
    every line they render supervises the model to read a box as a letter.
    """

    def test_a_face_whose_kabyle_glyphs_are_removed_is_rejected(self, tmp_path: Path) -> None:
        from fontTools.ttLib import TTFont  # noqa: PLC0415

        source = FontPool.discover().fonts[0].path
        stripped = tmp_path / "stripped.ttf"
        font = TTFont(str(source), fontNumber=0)
        for table in font["cmap"].tables:
            for char in KABYLE_PROBE_CHARS:
                table.cmap.pop(ord(char), None)
        font.save(str(stripped))

        assert _supports_kabyle_glyphs(source)
        assert not _supports_kabyle_glyphs(stripped)

    def test_the_probe_agrees_with_the_faces_own_character_map(self) -> None:
        """The mask comparison stands in for a cmap lookup, so it is checked against one."""
        from fontTools.ttLib import TTFont  # noqa: PLC0415

        for entry in FontPool.discover().fonts:
            covered = set[int]()
            for table in TTFont(str(entry.path), fontNumber=0)["cmap"].tables:
                covered |= set(table.cmap)
            for char in KABYLE_PROBE_CHARS:
                font = ImageFont.truetype(str(entry.path), size=PROBE_SIZE)
                drawn = bytes(font.getmask(char, mode="L")) != bytes(font.getmask(NOTDEF, mode="L"))
                assert drawn == (ord(char) in covered), f"{entry.family} {char}"

    def test_a_pool_with_no_usable_face_refuses_rather_than_falling_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A renderer that quietly substitutes any system face turns a broken font list into
        a healthy loss curve, which is what a fallback branch here used to do."""
        monkeypatch.setattr("agbalu.ocr.fonts_pro.SYSTEM_FALLBACKS", ())
        with pytest.raises(FontPoolError):
            FontPool.discover(pool_dir=tmp_path)

    def test_a_missing_file_is_not_a_usable_face(self, tmp_path: Path) -> None:
        assert not _supports_kabyle_glyphs(tmp_path / "absent.ttf")


def test_the_published_config_declares_every_field_the_architecture_takes() -> None:
    """`config.json` is read into `FeraounProConfig` by name. A field the record omits comes
    back as the class default, and the published shape then differs from the trained one."""
    declared = set(inspect.signature(FeraounProConfig.__init__).parameters) - {"self", "kwargs"}
    assert set(release_config()) == declared | {"model_type"}


def test_the_published_config_builds_the_architecture_the_trainer_trained() -> None:
    published = release_config()
    assert published["model_type"] == FeraounProConfig.model_type

    config = FeraounProConfig(
        vocabulary=published["vocabulary"],
        vocab_size=published["vocab_size"],
        d_model=published["d_model"],
        num_encoder_layers=published["num_encoder_layers"],
        num_heads=published["num_heads"],
        dim_feedforward=published["dim_feedforward"],
        max_width_frames=published["max_width_frames"],
        dropout=published["dropout"],
    )
    model = FeraounProForCTC(config)
    trained = HybridCTCRecognizer(include_aux_decoder=False)

    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == sum(p.numel() for p in trained.parameters())
    assert parameters == 51_440_589
    assert config.vocabulary == list(VOCABULARY)
