"""The architecture a published CTC model needs beside its weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agbalu.speech.release import ARCHITECTURE, CTC_OVERRIDES, ReleaseError, write_config
from agbalu.speech.vocabulary import PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER


def test_the_training_loop_imports_the_overrides_rather_than_repeating_them() -> None:
    """Two copies of a config are one config that will disagree with the weights, and the
    disagreement is only visible to whoever downloads the model. Asserted on the source
    because an identity check would pass on two modules that both defined their own."""
    source = Path("modal_app/asr.py").read_text(encoding="utf-8")
    assert "from agbalu.speech.release import BASE_MODEL, CTC_OVERRIDES" in source
    assert "BASE_MODEL: Final" not in source
    assert "CTC_OVERRIDES: Final" not in source


def test_the_overrides_are_the_ones_the_model_was_trained_under() -> None:
    assert CTC_OVERRIDES["ctc_zero_infinity"] is True
    assert CTC_OVERRIDES["layerdrop"] == 0.0
    assert CTC_OVERRIDES["mask_time_prob"] == 0.05
    assert CTC_OVERRIDES["add_adapter"] is False


def test_the_vocabulary_supplies_the_size_and_the_blank_and_is_not_hard_coded() -> None:
    """`vocab_size` and `pad_token_id` are absent from the overrides on purpose: they come
    from the built vocabulary, and a second copy is what would drift from the head."""
    assert "vocab_size" not in CTC_OVERRIDES
    assert "pad_token_id" not in CTC_OVERRIDES


def test_a_vocabulary_without_a_blank_class_is_refused(tmp_path: Path) -> None:
    """CTC needs a blank. Without one the config names a class the head does not have."""
    with pytest.raises(ReleaseError, match="blank class"):
        write_config({"a": 0, "b": 1}, tmp_path)


def test_the_published_architecture_is_the_head_the_weights_have() -> None:
    """`Wav2Vec2Config.from_pretrained(BASE_MODEL)` carries the SSL base's
    `Wav2Vec2ForPreTraining`, which is not this model. Inherited unchanged it publishes a
    config that builds the wrong architecture for a `pipeline` with nothing raising."""
    assert ARCHITECTURE == "Wav2Vec2ForCTC"


def test_the_release_recipe_stages_every_file_the_release_command_writes() -> None:
    """`vocab.json` alone is not a tokenizer, and a file written but not staged is one the
    downloader does not get. The recipe names them; this is what keeps the two lists one."""
    recipe = Path("Makefile").read_text(encoding="utf-8").split("release-fadhma:")[1]
    recipe = recipe.split("\nrelease-")[0]
    for name in (
        "config.json",
        "preprocessor_config.json",
        "vocab.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "5gram.klm",
    ):
        assert f"artifacts/asr/{name}" in recipe, name


@pytest.mark.integration
def test_the_written_config_describes_the_published_model() -> None:
    """Reads the base model's config from the Hub, so it is integration-marked."""
    vocabulary = json.loads(Path("artifacts/asr/vocab.json").read_text(encoding="utf-8"))
    written = write_config(vocabulary, Path("artifacts/asr"))
    assert {path.name for path in written} >= {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
    }

    config = json.loads(Path("artifacts/asr/config.json").read_text(encoding="utf-8"))
    assert config["vocab_size"] == len(vocabulary) == 40
    assert config["pad_token_id"] == vocabulary[PAD_TOKEN]
    assert config["model_type"] == "wav2vec2"
    assert config["architectures"] == [ARCHITECTURE]
    for key, value in CTC_OVERRIDES.items():
        assert config[key] == value

    # The whole point of writing a tokenizer: `Wav2Vec2CTCTokenizer` defaults to `<unk>`
    # and `<pad>` where this vocabulary writes `[UNK]` and `[PAD]`, so a downloader who
    # builds a default one decodes against specials the model never emits.
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("artifacts/asr")
    assert processor.tokenizer.pad_token == PAD_TOKEN
    assert processor.tokenizer.unk_token == UNK_TOKEN
    assert processor.tokenizer.word_delimiter_token == WORD_DELIMITER
    # `|` decodes to a space and the repeated `l` collapses, which is CTC's own rule.
    ids = [4, 29, 24, 15, 2, 9, 8, 15, 15, 3, 4, 26, 8, 17]
    assert processor.tokenizer.decode(ids) == "azul fel-awen"

    # `do_normalize` is why the extractor is published at all: a caller who builds a default
    # one feeds the model a distribution it never saw, and nothing raises.
    extractor = json.loads(
        Path("artifacts/asr/preprocessor_config.json").read_text(encoding="utf-8")
    )
    assert extractor["do_normalize"] is True
    assert extractor["sampling_rate"] == 16_000
