"""The staged releases, loaded the way a downloader loads them.

The unit tests prove the standalone modules equal the trained ones on random weights. This
proves it on the published ones, through `AutoModel`/`AutoTokenizer` with
`trust_remote_code=True` and no part of this package in the path the loader takes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import sentencepiece as spm
import torch
from safetensors.torch import load_file
from tools.stage_hub import stage
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
)

from agbalu.model.config import PRESETS
from agbalu.model.modeling import Encoder
from agbalu.tifinagh.infer import Transliterator

RELEASE = Path("artifacts/release")

pytestmark = pytest.mark.integration


def _staged(name: str, repo: str, tmp_path: Path) -> Path:
    source = RELEASE / name
    if not (source / "model.safetensors").is_file():
        pytest.skip(f"{source} is not staged; run `make release REPO=...`")
    target = tmp_path / name
    shutil.copytree(source, target)
    stage(repo, target)
    return target


def test_juba_decodes_exactly_as_the_transliterator_does(tmp_path: Path) -> None:
    """Free-running, both decodings. A teacher-forced comparison would pass on a model
    that cannot decode at all, which is the defect this project has already shipped once."""
    directory = _staged("Juba-27M", "juba", tmp_path)
    reference = Transliterator.load(RELEASE / "Juba-27M")
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(directory, trust_remote_code=True).eval()

    for text in ("ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?", "ⴰⵣⵓⵍ ⴼⵍⵍⴰⵡⵏ, ⴰⵎⴽ ⵜⵜⵉⵍⵉⴹ ?", "ⵜⴰⵎⵓⵔⵜ ⵏⵏⵖ"):
        encoded = tokenizer(text, return_tensors="pt")
        assert encoded["input_ids"][0].tolist() == reference.tokenizer.encode(text)
        with torch.inference_mode():
            beams = model.generate(**encoded, max_length=256, num_beams=4)
            greedy = model.generate(**encoded, max_length=256, num_beams=1)
        assert tokenizer.decode(beams[0], skip_special_tokens=True) == reference.transliterate(
            text, num_beams=4
        )
        assert (
            tokenizer.decode(greedy[0], skip_special_tokens=True)
            == reference.greedy_batch([text])[0]
        )


def test_masinissa_produces_the_encoder_s_own_representations(tmp_path: Path) -> None:
    directory = _staged("Masinissa-31M", "masinissa", tmp_path)
    reference = Encoder(PRESETS["kab"])
    reference.load_state_dict(
        load_file(RELEASE / "Masinissa-31M" / "model.safetensors"), strict=False
    )
    reference.eval()

    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(directory, trust_remote_code=True).eval()
    assert tokenizer.mask_token_id == 4

    ids = torch.tensor([tokenizer("Azul fell-awen, amek i tellam?")["input_ids"]])
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        expected = reference.classifier(reference.contextualise(ids, mask.bool()))
        produced = model(input_ids=ids, attention_mask=mask).logits
    assert torch.equal(expected, produced)


def test_masinissa_s_published_tokenizer_is_the_vocabulary_it_was_trained_on(
    tmp_path: Path,
) -> None:
    """The converted `tokenizer.json` must be id for id with the SentencePiece model, and
    it is over every sentence but one: a whitespace-only input, which the processor encodes
    to nothing and the converted tokenizer to a lone metaspace piece."""
    directory = _staged("Masinissa-31M", "masinissa", tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    processor = spm.SentencePieceProcessor(
        model_file=str(RELEASE / "Masinissa-31M" / "agbalu-tok-base-16k.model")
    )
    for text in (
        "Azul fell-awen, amek i tellam?",
        "Aman d tudert.",
        "Tamurt n Leqbayel",
        "123 456",
        "日本語",
        "",
    ):
        assert tokenizer(text)["input_ids"] == [2, *processor.encode(text), 3], text


def test_fadhma_loads_its_processor_and_its_ctc_head() -> None:
    """No remote code here — the architecture is native. What was missing was a tokenizer,
    without which `AutoProcessor` and the ASR pipeline have nothing to decode with."""
    source = RELEASE / "Fadhma-300M"
    if not (source / "model.safetensors").is_file():
        pytest.skip(f"{source} is not staged; run `make release REPO=fadhma`")

    config = AutoConfig.from_pretrained(source)
    assert config.architectures == ["Wav2Vec2ForCTC"]
    assert config.vocab_size == 40

    processor = AutoProcessor.from_pretrained(source)
    assert processor.feature_extractor.do_normalize is True
    assert processor.feature_extractor.sampling_rate == 16_000
    assert processor.tokenizer.pad_token == "[PAD]"
    assert processor.tokenizer.word_delimiter_token == "|"
