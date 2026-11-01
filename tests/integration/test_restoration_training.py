"""Compare stopped and continuous restoration training on real-tokenizer data."""

from pathlib import Path

import pytest
import sentencepiece as spm
import torch

from agbalu.encoder import checkpoint as encoder_checkpoint
from agbalu.encoder.model import Encoder
from agbalu.restoration.annotations import annotate
from agbalu.restoration.checkpoint import latest, load
from agbalu.restoration.inference import Inference, Stream
from agbalu.restoration.model import Restorer
from agbalu.restoration.training import Trainer
from tests.integration.test_restoration_corpus import TOKENIZER, build

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not TOKENIZER.exists(), reason="real Mammeri artifact is absent")
def test_full_training_resume_preserves_next_update(tmp_path: Path) -> None:
    locations, settings = build(tmp_path)
    locations.encoder.mkdir()
    encoder = Encoder(settings.geometry)
    optimizer = torch.optim.AdamW(encoder.parameters())
    snapshot = encoder_checkpoint.save(
        locations.encoder, encoder, optimizer, encoder_checkpoint.State(), "test-encoder"
    )
    encoder_checkpoint.promote(locations.encoder, snapshot, "best_mlm", {})
    continuous = Trainer(settings, locations, "continuous", torch.device("cpu"))
    continuous.run(stop_after=2, smoke=True)
    stopped = Trainer(settings, locations, "resumed", torch.device("cpu"))
    stopped.run(stop_after=1, smoke=True)
    resumed = Trainer(settings, locations, "resumed", torch.device("cpu"))
    report = resumed.run(stop_after=2, smoke=True)
    assert report["steps_this_run"] == 1
    for expected, observed in zip(
        continuous.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    saved = latest(resumed.directory)
    assert saved is not None
    assert load(saved, resumed.model, resumed.optimizer, resumed.identity).step == 2


@pytest.mark.skipif(not TOKENIZER.exists(), reason="real Mammeri artifact is absent")
def test_training_selects_a_loadable_full_document_restorer(tmp_path: Path) -> None:
    locations, settings = build(tmp_path)
    locations.encoder.mkdir()
    encoder = Encoder(settings.geometry)
    optimizer = torch.optim.AdamW(encoder.parameters())
    snapshot = encoder_checkpoint.save(
        locations.encoder, encoder, optimizer, encoder_checkpoint.State(), "test-encoder"
    )
    encoder_checkpoint.promote(locations.encoder, snapshot, "best_mlm", {})
    trainer = Trainer(settings, locations, "fit", torch.device("cpu"))
    result = trainer.run(stop_after=1)
    assert result["steps_this_run"] == 1
    inference = Inference.load(trainer.directory, torch.device("cpu"))
    text = "azul fell-awen " * 400
    restored = inference.predict(text)
    assert restored.words == tuple(text.split())
    assert len(restored.casing) == 800
    assert len(restored.marks) == 801


@pytest.mark.skipif(not TOKENIZER.exists(), reason="real Mammeri artifact is absent")
def test_stream_commits_each_word_once_and_flushes_the_tail(tmp_path: Path) -> None:
    locations, settings = build(tmp_path)
    model = Restorer(settings.geometry, ("@",)).eval()
    with torch.no_grad():
        model.punctuation.weight.zero_()
        model.punctuation.bias.fill_(-20)
        model.punctuation.bias[0] = 20
    inference = Inference(
        model, spm.SentencePieceProcessor(model_file=str(locations.tokenizer)), settings
    )
    stream = Stream(inference)
    words = ("azul", "fell-awen") * 400
    assert stream.push("azul") == ""
    first = stream.push(" ".join(words[:400]))
    committed = stream.committed
    assert 0 < committed < 400
    with pytest.raises(ValueError, match="already committed"):
        stream.push("axxam " + " ".join(words[1:400]))
    last = stream.push(" ".join(words), final=True)
    assert annotate(first + last).words == words
    assert stream.committed == len(words)
    with pytest.raises(ValueError, match="already closed"):
        stream.push(" ".join(words), final=True)
