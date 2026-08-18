"""The character model: its shapes, its vocabulary, and its loss counters."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from agbalu.tifinagh.config import MIN_VOCAB_SIZE, ConfigError, ModelConfig, TrainConfig
from agbalu.tifinagh.model import CharTransformer
from agbalu.tifinagh.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    VOCAB_CHARS,
    CharTokenizer,
    TokenizerError,
)

SMALL = ModelConfig(
    vocab_size=128,
    hidden_size=16,
    intermediate_size=48,
    num_attention_heads=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
)


def test_the_release_config_reports_the_published_parameter_count() -> None:
    """26,901,888 is on the card. A count derived from the shapes must agree with a
    count taken from the built module, or one of the two is fiction."""
    built = sum(
        parameter.numel()
        for name, parameter in CharTransformer(SMALL).named_parameters(remove_duplicate=False)
        if name != "lm_head.weight"
    )
    assert built == SMALL.parameters
    assert ModelConfig().parameters == 26_901_888


def test_a_head_count_that_does_not_divide_the_hidden_size_is_refused() -> None:
    with pytest.raises(ConfigError, match="divisible"):
        ModelConfig(hidden_size=384, num_attention_heads=5)


def test_a_vocabulary_too_small_to_spell_anything_is_refused() -> None:
    with pytest.raises(ConfigError, match="at least"):
        ModelConfig(vocab_size=MIN_VOCAB_SIZE - 1)


def test_the_config_is_frozen_so_a_loaded_checkpoint_cannot_be_reshaped_in_place() -> None:
    with pytest.raises(AttributeError):
        ModelConfig().hidden_size = 512  # type: ignore[misc]


def test_the_output_projection_is_the_embedding() -> None:
    model = CharTransformer(SMALL)
    assert model.lm_head.weight is model.embed.weight


def test_the_release_vocabulary_fits_inside_the_release_embedding() -> None:
    """The embedding is padded to 128 and the inventory is smaller. An inventory that
    outgrew it would index past the table and raise only at the first unlucky character."""
    assert CharTokenizer().vocab_size <= ModelConfig().vocab_size


def test_every_declared_character_has_its_own_id() -> None:
    tokenizer = CharTokenizer()
    assert len(set(VOCAB_CHARS)) == len(VOCAB_CHARS)
    assert tokenizer.vocab_size == len(VOCAB_CHARS) + 4


def test_encoding_brackets_the_text_and_decoding_removes_the_brackets() -> None:
    tokenizer = CharTokenizer()
    ids = tokenizer.encode("azul")
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert tokenizer.decode(ids) == "azul"


def test_case_is_folded_rather_than_reaching_unk() -> None:
    tokenizer = CharTokenizer()
    assert tokenizer.encode("AZUL", add_special_tokens=False) == tokenizer.encode(
        "azul", add_special_tokens=False
    )


def test_a_character_outside_the_inventory_becomes_unk() -> None:
    assert CharTokenizer().encode("茶", add_special_tokens=False) == [UNK_ID]


def test_specials_can_be_kept_so_a_degenerate_decode_is_visible() -> None:
    tokenizer = CharTokenizer()
    assert tokenizer.decode([BOS_ID, PAD_ID], skip_special_tokens=False) == "[BOS][PAD]"


def test_an_unknown_id_decodes_to_nothing_rather_than_raising() -> None:
    assert CharTokenizer().decode([9_999]) == ""


def test_the_empty_string_round_trips() -> None:
    tokenizer = CharTokenizer()
    assert tokenizer.decode(tokenizer.encode("")) == ""


def test_a_saved_vocabulary_reloads_identically(tmp_path: Path) -> None:
    original = CharTokenizer()
    path = tmp_path / "vocab.json"
    original.save(path)
    assert CharTokenizer.load(path).char_to_id == original.char_to_id


def test_a_vocabulary_file_that_is_not_a_str_to_int_table_is_refused(tmp_path: Path) -> None:
    """The ids are positions in an embedding table: a wrong mapping decodes to
    plausible text in the wrong characters and nothing raises."""
    path = tmp_path / "vocab.json"
    path.write_text('{"char_to_id": {"a": "one"}}', encoding="utf-8")
    with pytest.raises(TokenizerError, match="str -> int"):
        CharTokenizer.load(path)

    path.write_text('{"vocab_size": 4}', encoding="utf-8")
    with pytest.raises(TokenizerError, match="no char_to_id"):
        CharTokenizer.load(path)


def test_teacher_forced_logits_have_one_row_per_target_position() -> None:
    model = CharTransformer(SMALL).eval()
    source = torch.randint(4, SMALL.vocab_size, (2, 7))
    prefix = torch.randint(4, SMALL.vocab_size, (2, 5))
    with torch.no_grad():
        logits = model(source, prefix)
    assert logits.shape == (2, 5, SMALL.vocab_size)


def test_the_source_and_the_target_may_differ_in_length() -> None:
    """Schwa restoration makes the target longer than the source by construction, so a
    decoder that assumed one position per source character could not do the task."""
    model = CharTransformer(SMALL).eval()
    source = torch.zeros((1, 3), dtype=torch.long)
    prefix = torch.zeros((1, 11), dtype=torch.long)
    with torch.no_grad():
        assert model(source, prefix).shape[1] == 11


def test_the_loss_counts_only_the_positions_it_scored() -> None:
    model = CharTransformer(SMALL)
    source = torch.randint(4, SMALL.vocab_size, (2, 6))
    prefix = torch.randint(4, SMALL.vocab_size, (2, 4))
    labels = torch.tensor([[5, 6, 7, PAD_ID], [8, 9, PAD_ID, PAD_ID]])
    output, logits = model.loss_with_logits(source, prefix, labels)
    assert int(output.tokens) == 5
    assert 0 <= int(output.correct) <= 5
    assert logits.shape == (2, 4, SMALL.vocab_size)


def test_the_loss_returns_counts_rather_than_an_accuracy() -> None:
    """Dividing inside `forward` is a host synchronisation per micro-batch."""
    model = CharTransformer(SMALL)
    output, _ = model.loss_with_logits(
        torch.zeros((1, 3), dtype=torch.long),
        torch.zeros((1, 3), dtype=torch.long),
        torch.ones((1, 3), dtype=torch.long),
    )
    assert output.correct.device == output.tokens.device
    assert output.tokens.dtype == torch.int64


def test_a_batch_of_only_padding_labels_scores_nothing_and_does_not_raise() -> None:
    model = CharTransformer(SMALL)
    output, _ = model.loss_with_logits(
        torch.zeros((1, 3), dtype=torch.long),
        torch.zeros((1, 3), dtype=torch.long),
        torch.full((1, 3), PAD_ID),
    )
    assert int(output.tokens) == 0


def test_the_training_recipe_is_the_one_the_release_was_fitted_under() -> None:
    recipe = TrainConfig()
    assert recipe.max_steps == 15_500
    assert recipe.micro_batch_size * recipe.gradient_accumulation_steps == 128
    assert recipe.seed == 42
