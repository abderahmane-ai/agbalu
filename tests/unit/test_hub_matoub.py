"""Matoub-82M as a published repository: the front end, the tokenizer, and the export."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from tools.export_matoub import fuse_weight_norm, strip_module_prefix
from tools.stage_hub import HUB, TOKENIZER_AUTO_MAP

from agbalu.hub.matoub import tokenization_matoub as hub
from agbalu.hub.matoub.configuration_matoub import MatoubConfig
from agbalu.hub.matoub.modeling_matoub import MatoubForTextToWaveform
from agbalu.hub.matoub.tokenization_matoub import MatoubTokenizer, PhonemeError
from agbalu.tts import g2p, kokoro
from agbalu.tts.vocabulary import Vocabulary

SENTENCES = [
    "Azul fell-awen, amek i telliḍ taṣebḥit-a?",
    "Aɣbalu d anegraw ameqqran n tutlayt taqbaylit.",
    "Tameṭṭut-nni tessawal s taqbaylit i warraw-is.",
    "Yečča aksum, yeswa aman, yeṭṭes.",
    "Ur ẓriɣ ara acu ara xedmeɣ azekka.",
    "ţ-ţamurt-nneɣ",
    "annect-a",
    "axxam",
    "wexxam",
    "a",
    "aḍ",
    "ɣef",
    "«Azul»…",
]
"""Every conditioned rule the table has: gemination on `b d g k t ḍ`, `a`-backing beside
each trigger, nasal assimilation before `f m y q x`, the legacy tense `t`, and the clitic
hyphen. The last three exercise a one-character word, a backing neighbour with no vowel
before it, and punctuation that must be stripped rather than phonemised."""

FRAMES_PER_SAMPLE = 600
"""One predictor frame is `gen_istft_hop_size * prod(upsample_rates)` samples through the
generator, doubled by the decoder's own upsampling block."""


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary.load()


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory, vocabulary: Vocabulary) -> MatoubTokenizer:
    directory = tmp_path_factory.mktemp("matoub-tokenizer")
    path = MatoubTokenizer.write_vocabulary(vocabulary.symbols, directory)
    return MatoubTokenizer(vocab_file=str(path))


@pytest.fixture(scope="module")
def small_model() -> MatoubForTextToWaveform:
    """The decoder's channel widths are fixed by the checkpoint, so only PL-BERT, the
    duration stack and the resblock count can shrink."""
    config = MatoubConfig(
        plbert_hidden_size=64,
        plbert_num_attention_heads=1,
        plbert_intermediate_size=64,
        plbert_num_hidden_layers=1,
        plbert_max_position_embeddings=64,
        num_layers=1,
        max_duration=4,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    )
    torch.manual_seed(0)
    model = MatoubForTextToWaveform(config)
    model.eval()
    return model


@pytest.mark.parametrize("text", SENTENCES)
def test_the_shipped_front_end_is_the_project_front_end(text: str) -> None:
    """The copy under `hub/` and `agbalu.tts.g2p` are two files that must agree. Nothing
    else keeps them together: the published repository cannot import the second."""
    assert hub.phonemize(text) == kokoro.fold(g2p.phonemize(text))


def test_the_shipped_rule_tables_are_the_project_tables() -> None:
    assert dict(g2p.PLAIN) == hub.PLAIN
    assert dict(g2p.SPIRANTS) == hub.SPIRANTS
    assert dict(g2p.NASAL_ASSIMILATION) == hub.NASAL_ASSIMILATION
    assert hub.VOWELS == g2p.VOWELS
    assert hub.BACKING_TRIGGERS == g2p.BACKING_TRIGGERS
    assert dict(kokoro.FOLD) == hub.FOLD
    assert hub.LEGACY_TENSE_T == g2p.LEGACY_TENSE_T
    assert hub.LENGTH == g2p.LENGTH
    assert hub.BOUNDARY == g2p.BOUNDARY


@pytest.mark.parametrize(
    "text",
    ["\u200b", "helló", "azul\x00", "日本語", "azul 42"],
)
def test_a_character_with_no_rule_raises_in_both_copies(text: str) -> None:
    """Dropping is the defect the front end exists to remove; the copy must refuse where
    the original refuses, or the published model is fitted to a string nobody chose."""
    with pytest.raises(PhonemeError):
        hub.phonemize(text)
    with pytest.raises(g2p.PhonemeError):
        kokoro.fold(g2p.phonemize(text))


@pytest.mark.parametrize("codepoint", [0xFEFF, 0x200F, 0x2060])
def test_an_invisible_format_character_raises_rather_than_vanishing(codepoint: int) -> None:
    """A byte-order mark, a right-to-left mark and a word joiner. Written as codepoints
    because a literal one is unreadable in a diff and indistinguishable from a typo."""
    with pytest.raises(PhonemeError):
        hub.phonemize(f"{chr(codepoint)}azul")


def test_a_non_breaking_space_is_a_word_boundary_not_a_phoneme() -> None:
    assert hub.phonemize("azul fell") == hub.phonemize("azul fell")


def test_empty_and_punctuation_only_text_phonemise_to_nothing() -> None:
    assert hub.phonemize("") == ""
    assert hub.phonemize("   ") == ""
    assert hub.phonemize("«…»") == ""


def test_every_symbol_the_front_end_emits_has_an_embedding_row(vocabulary: Vocabulary) -> None:
    emitted = {symbol for text in SENTENCES for symbol in hub.phonemize(text)}
    assert emitted
    assert not emitted - set(vocabulary.symbols)


def test_the_tokenizer_wraps_the_sequence_in_the_boundary_symbol(
    tokenizer: MatoubTokenizer, vocabulary: Vocabulary
) -> None:
    """`meldataset` prepends and appends token 0 to every training target."""
    text = "Azul fell-awen"
    ids = tokenizer(text)["input_ids"]
    pad = vocabulary.pad_index
    assert ids[0] == pad
    assert ids[-1] == pad
    assert ids[1:-1] == [vocabulary.symbols[symbol] for symbol in hub.phonemize(text)]


def test_the_tokenizer_agrees_with_the_projects_encoder(
    tokenizer: MatoubTokenizer, vocabulary: Vocabulary
) -> None:
    for text in SENTENCES:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        assert ids == list(vocabulary.encode(hub.phonemize(text)))


def test_the_tokenizer_refuses_a_symbol_with_no_row(
    tokenizer: MatoubTokenizer, vocabulary: Vocabulary
) -> None:
    """The tie bar is the one symbol the front end can produce and the table cannot place,
    which is why `fold` exists. Reaching the tokenizer with it is a fold that did not run."""
    tie_bar = "͡"
    assert tie_bar not in vocabulary.symbols
    with pytest.raises(KeyError):
        tokenizer.convert_tokens_to_ids([tie_bar])


def test_the_tokenizer_round_trips_through_save_and_load(
    tokenizer: MatoubTokenizer, tmp_path: Path
) -> None:
    tokenizer.save_pretrained(tmp_path)
    reloaded = MatoubTokenizer.from_pretrained(tmp_path)
    text = "Azul fell-awen, amek i telliḍ?"
    assert reloaded(text)["input_ids"] == tokenizer(text)["input_ids"]
    assert reloaded.phonemize(text) == tokenizer.phonemize(text)


def test_the_tokenizer_auto_map_names_a_module_the_staging_copies() -> None:
    """Resolved on the downloader's machine, so a wrong target fails only after publication."""
    for repo, mapping in TOKENIZER_AUTO_MAP.items():
        module, _, symbol = str(mapping[0]).partition(".")
        assert (HUB / repo / f"{module}.py").is_file()
        assert symbol
        assert mapping[1] is None


def test_the_configuration_round_trips_through_json(tmp_path: Path) -> None:
    config = MatoubConfig()
    config.save_pretrained(tmp_path)
    written = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    reloaded = MatoubConfig(**{k: v for k, v in written.items() if k != "architectures"})
    assert reloaded.resblock_dilation_sizes == config.resblock_dilation_sizes
    assert reloaded.upsample_rates == config.upsample_rates
    assert reloaded.max_token_length == 510


def test_synthesis_returns_audio_whose_length_is_the_predicted_durations(
    small_model: MatoubForTextToWaveform,
) -> None:
    torch.manual_seed(0)
    ids = torch.randint(1, 170, (1, 9))
    out = small_model(ids)
    assert out.durations.shape == (1, 9)
    assert int(out.durations.sum()) * FRAMES_PER_SAMPLE == out.waveform.shape[-1]
    assert out.waveform_lengths.tolist() == [out.waveform.shape[-1]]
    assert torch.isfinite(out.waveform).all()


def test_a_padded_batch_synthesises_each_item_from_its_own_tokens(
    small_model: MatoubForTextToWaveform,
) -> None:
    """Padding is the boundary symbol, so a row synthesised with its padding attached is a
    different utterance — and it still produces a tensor of the right rank."""
    short = torch.randint(1, 170, (1, 5))
    padded = torch.cat([short, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 5, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)], 1)

    torch.manual_seed(0)
    alone = small_model(short).waveform_lengths
    torch.manual_seed(0)
    batched = small_model(padded, attention_mask=mask).waveform_lengths
    assert alone.tolist() == batched.tolist()


def test_a_batch_is_right_padded_to_its_longest_item(
    small_model: MatoubForTextToWaveform,
) -> None:
    ids = torch.cat(
        [
            torch.cat([torch.randint(1, 170, (1, 4)), torch.zeros(1, 5, dtype=torch.long)], dim=1),
            torch.randint(1, 170, (1, 9)),
        ]
    )
    mask = torch.tensor([[1] * 4 + [0] * 5, [1] * 9])
    out = small_model(ids, attention_mask=mask)
    lengths = out.waveform_lengths
    assert out.waveform.shape == (2, int(lengths.max()))
    tail = out.waveform[0, int(lengths[0]) :]
    assert torch.equal(tail, torch.zeros_like(tail))
    assert tail.numel() > 0


def test_speed_stretches_the_predicted_durations(small_model: MatoubForTextToWaveform) -> None:
    ids = torch.randint(1, 170, (1, 9))
    torch.manual_seed(0)
    normal = int(small_model(ids).durations.sum())
    torch.manual_seed(0)
    slow = int(small_model(ids, speed=0.5).durations.sum())
    assert slow > normal


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"speed": 0.0}, "speed must be positive"),
        ({"voice": torch.zeros(1, 8)}, "256-dim style vector"),
    ],
)
def test_synthesis_refuses_arguments_it_cannot_honour(
    small_model: MatoubForTextToWaveform, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        small_model(torch.randint(1, 170, (1, 5)), **kwargs)


def test_synthesis_refuses_more_tokens_than_the_encoder_can_position(
    small_model: MatoubForTextToWaveform,
) -> None:
    limit = small_model.config.max_token_length
    with pytest.raises(ValueError, match="exceeds"):
        small_model(torch.randint(1, 170, (1, limit + 1)))


def test_synthesis_refuses_a_tensor_that_is_not_a_batch_of_sequences(
    small_model: MatoubForTextToWaveform,
) -> None:
    with pytest.raises(ValueError, match="batch, tokens"):
        small_model(torch.randint(1, 170, (5,)))


def test_fusing_weight_normalisation_reproduces_the_parametrised_weight() -> None:
    """The export ships plain weights. If the fusion is wrong every convolution in the
    vocoder is scaled, which sounds like a quieter model rather than a broken one."""
    torch.manual_seed(0)
    for module in (
        torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(6, 4, 3)),
        torch.nn.utils.parametrizations.weight_norm(torch.nn.ConvTranspose1d(4, 6, 3)),
    ):
        with torch.no_grad():
            module.parametrizations.weight.original0.normal_()
            module.parametrizations.weight.original1.normal_()
        fused = fuse_weight_norm(dict(module.state_dict()))
        assert set(fused) == {"weight", "bias"}
        assert torch.allclose(fused["weight"], module.weight, atol=1e-6)


def test_fusing_leaves_an_unparametrised_state_dict_alone() -> None:
    state = {"conv.weight": torch.ones(2, 2), "conv.bias": torch.zeros(2)}
    assert fuse_weight_norm(state) == state


def test_the_distributed_prefix_is_stripped_from_every_key() -> None:
    stripped = strip_module_prefix({"module.a.weight": torch.ones(1), "b.bias": torch.zeros(1)})
    assert sorted(stripped) == ["a.weight", "b.bias"]


def test_the_frames_constant_matches_the_configured_upsampling() -> None:
    config = MatoubConfig()
    assert config.gen_istft_hop_size * math.prod(config.upsample_rates) * 2 == FRAMES_PER_SAMPLE
