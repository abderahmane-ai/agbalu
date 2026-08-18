"""The training loop against the real encoder checkpoint and the real tokenizer.

Every defect this package has had so far was found by running it, not by the unit suite: an
asynchronous host-to-device copy reading freed memory, and a learning rate that logged as
zero. Both are only visible when a real step runs on a real device.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from agbalu.punctuation.corpus import Row  # noqa: E402
from agbalu.punctuation.dataset import Tokenizer, encode_corpus  # noqa: E402
from agbalu.punctuation.infer import predict  # noqa: E402
from agbalu.punctuation.labels import CASE, PUNCTUATION, annotate  # noqa: E402
from agbalu.punctuation.model import build  # noqa: E402
from agbalu.punctuation.train import (  # noqa: E402
    Trainer,
    TrainSettings,
    Validation,
    schedule,
)
from agbalu.tokenizer.evaluate import load  # noqa: E402

pytestmark = pytest.mark.integration

ENCODER = Path("artifacts/runs/agbalu-encoder-v1")
TOKENIZER = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")

SENTENCES = [
    "Azul fell-awen, amek tellid?",
    "Tegzem yiwen n useklu deg tebḥirt.",
    "D acu i txedmed assa?",
    "Ur yessaweḍ ad d-yawey akayad-nni.",
    "Yenna-as: ad nruḥ ass-a.",
    "Ḥasan d amdakel-iw n tidet.",
]


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    if not TOKENIZER.is_file():
        pytest.skip(f"no tokenizer at {TOKENIZER}")
    processor: Tokenizer = load(TOKENIZER)
    return processor


@pytest.fixture(scope="module")
def encoder_present() -> None:
    if not (ENCODER / "best.pt").is_file():
        pytest.skip(f"no encoder checkpoint under {ENCODER}")


def test_the_objective_carries_no_class_weighting() -> None:
    """Weighting was removed, not tuned down. A 20x cap produced comma recall 0.919 at
    precision 0.381; the benchmark this task is measured on is fine-tuned unweighted."""
    assert not hasattr(TrainSettings(), "max_class_weight")
    assert not hasattr(Trainer, "punctuation_weight")


def test_schedule_warms_up_then_decays() -> None:
    assert schedule(0, 100, 10) == pytest.approx(0.1)
    assert schedule(9, 100, 10) == pytest.approx(1.0)
    assert schedule(10, 100, 10) == pytest.approx(1.0)
    assert schedule(100, 100, 10) == pytest.approx(0.0, abs=1e-9)
    assert schedule(200, 100, 10) == pytest.approx(0.0, abs=1e-9)


def test_two_real_steps_move_the_loss_and_leave_a_resumable_checkpoint(
    tmp_path: Path, tokenizer: Tokenizer, encoder_present: None
) -> None:
    _ = encoder_present
    rows = [Row(text, "fixture") for text in SENTENCES]
    corpus = encode_corpus(rows, tokenizer, 64)
    device = torch.device("cpu")

    model = build(ENCODER, device=device)
    settings = TrainSettings(epochs=2, batch_size=3, encoder_lr=1e-4, head_lr=1e-2)
    trainer = Trainer(model, corpus, corpus, settings, tmp_path, device)
    summary = trainer.run()

    assert summary.steps_this_run == trainer.total_steps
    assert summary.labelled_this_run > 0
    assert summary.labels_per_second is not None
    assert summary.best is not None
    assert len(summary.history) >= 2
    assert summary.history[-1]["loss"] < summary.history[0]["loss"]
    for name in ("best.pt", "latest.pt", "best.pt.sha256", "checkpoints.jsonl"):
        assert (tmp_path / name).is_file()


def test_a_resumed_run_at_max_steps_trains_nothing_and_says_so(
    tmp_path: Path, tokenizer: Tokenizer, encoder_present: None
) -> None:
    """It has happened in two other entrypoints: the counters get reported as a measurement."""
    _ = encoder_present
    corpus = encode_corpus([Row(text, "fixture") for text in SENTENCES], tokenizer, 64)
    device = torch.device("cpu")
    settings = TrainSettings(epochs=1, batch_size=3)

    first = Trainer(build(ENCODER, device=device), corpus, corpus, settings, tmp_path, device)
    first.run()

    second = Trainer(build(ENCODER, device=device), corpus, corpus, settings, tmp_path, device)
    assert second.maybe_resume()
    assert second.state.step == first.total_steps
    summary = second.run()
    assert summary.steps_this_run == 0
    assert summary.labels_per_second is None


def test_predictions_cover_every_input_and_restore_readable_text(
    tokenizer: Tokenizer, encoder_present: None
) -> None:
    _ = encoder_present
    model = build(ENCODER, device=torch.device("cpu"))
    texts = [annotate(text).text for text in SENTENCES]
    restorations = predict(model, tokenizer, [*texts, ""], device=torch.device("cpu"))

    assert len(restorations) == len(texts) + 1
    assert restorations[-1].words == ()
    for text, restoration in zip(texts, restorations[:-1], strict=True):
        assert restoration.words == annotate(text).words
        assert all(0 <= label < len(PUNCTUATION) for label in restoration.punctuation)
        assert all(0 <= label < len(CASE) for label in restoration.case)
        assert annotate(restoration.text).text == text


def test_the_word_embeddings_train(tokenizer: Tokenizer, encoder_present: None) -> None:
    """Tied to the masked-token classifier, so freezing that head by `parameters()` froze
    6,144,000 embedding parameters without anyone choosing it. The freeze skips the shared
    tensor by identity."""
    _ = encoder_present
    _ = tokenizer
    model = build(ENCODER, device=torch.device("cpu"))

    assert model.encoder.embedding.word_embedding.weight.requires_grad
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {
        "encoder.classifier.dense.weight",
        "encoder.classifier.dense.bias",
        "encoder.classifier.decoder.bias",
    }


def test_selection_prefers_macro_f1_over_loss() -> None:
    """The two disagreed on the first run: loss chose 0.649 macro-F1 over 0.670."""
    worse_loss = Validation(loss=0.2280, punctuation_macro_f1=0.670, case_noninitial_f1=0.90)
    better_loss = Validation(loss=0.2270, punctuation_macro_f1=0.649, case_noninitial_f1=0.90)

    assert better_loss.loss < worse_loss.loss
    assert worse_loss.score > better_loss.score


def test_freezing_the_encoder_leaves_only_the_heads_trainable(
    tokenizer: Tokenizer, encoder_present: None
) -> None:
    _ = encoder_present
    _ = tokenizer
    model = build(ENCODER, device=torch.device("cpu"))
    before = model.trainable_parameters()
    model.freeze_encoder()
    heads = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(("punctuation.", "case."))
    )
    assert model.trainable_parameters() == heads < before
