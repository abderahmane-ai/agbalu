"""Make a staged release directory loadable by `transformers` alone.

The exporters write weights and a card. That is not enough for a downloader: neither
architecture is native to `transformers`, so without the modelling code beside them
`from_pretrained` has nothing to construct, and neither repository shipped a tokenizer at
all — Juba's alphabet was a Python constant in this package and Masinissa's vocabulary was
a bare SentencePiece model no `AutoTokenizer` can read.

This copies `agbalu.hub`'s standalone modules in, builds a `tokenizer.json`, and rewrites
`config.json` with the `auto_map` that points at them. It then loads the directory back
through the real `AutoModel`/`AutoTokenizer` path with `trust_remote_code=True` and refuses
to leave a directory that does not load, because a staged directory that looks published
and is not is how a broken repository reaches the Hub.

    python3 -m tools.stage_hub --repo juba --dir artifacts/release/Juba-27M
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict

if TYPE_CHECKING:
    from transformers import (
        PreTrainedConfig,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
    )

Repo = Literal[
    "belaid", "boulifa", "dihya", "feraoun", "feraoun_pro", "juba", "masinissa", "matoub"
]

HUB: Final = Path("src/agbalu/hub")

TIFINAGH_PROBE: Final = "ⵜⴰⵇⴱⴰⵢⵍⵉⵜ ⴷ ⵜⵓⵜⵍⴰⵢⵜ ⵜⴰⵢⴻⵎⵎⴰⵜ ⵏⵏⴻⵖ."
"""Feraoun-Pro's second script. Its Latin probe cannot fail on a Tifinagh-blind checkpoint."""

DOTTED_CAPITAL_I: Final = chr(0x0130)
"""The only codepoint in Unicode whose lowercase expands to two, so the only one where a
`Lowercase` normalizer and `CharTokenizer`'s per-codepoint `str.lower()` disagree. Mapped
aside before lowercasing, which sends it to `[UNK]` down both paths."""

OUT_OF_ALPHABET: Final = chr(0x0001)
"""Absent from the character vocabulary, so it resolves to `[UNK]`."""

AUTO_MAP: Final[dict[Repo, dict[str, str]]] = {
    "belaid": {
        "AutoConfig": "configuration_belaid.BelaidConfig",
        "AutoModelForTokenClassification": "modeling_belaid.BelaidForTokenClassification",
    },
    "boulifa": {
        "AutoConfig": "configuration_boulifa.BoulifaConfig",
        "AutoModelForSeq2SeqLM": "modeling_boulifa.BoulifaForSeq2SeqLM",
    },
    "dihya": {
        "AutoConfig": "configuration_dihya.DihyaConfig",
        "AutoModel": "modeling_dihya.DihyaForSequenceClassification",
        "AutoModelForSequenceClassification": ("modeling_dihya.DihyaForSequenceClassification"),
    },
    "feraoun": {
        "AutoConfig": "configuration_feraoun.FeraounConfig",
        "AutoModel": "modeling_feraoun.FeraounForVision2Seq",
    },
    "feraoun_pro": {
        "AutoConfig": "configuration_feraoun_pro.FeraounProConfig",
        "AutoModel": "modeling_feraoun_pro.FeraounProForCTC",
    },
    "juba": {
        "AutoConfig": "configuration_juba.JubaConfig",
        "AutoModelForSeq2SeqLM": "modeling_juba.JubaForSeq2SeqLM",
    },
    "masinissa": {
        "AutoConfig": "configuration_masinissa.MasinissaConfig",
        "AutoModel": "modeling_masinissa.MasinissaModel",
        "AutoModelForMaskedLM": "modeling_masinissa.MasinissaForMaskedLM",
    },
    "matoub": {
        "AutoConfig": "configuration_matoub.MatoubConfig",
        "AutoModel": "modeling_matoub.MatoubForTextToWaveform",
        "AutoModelForTextToWaveform": "modeling_matoub.MatoubForTextToWaveform",
    },
}

TOKENIZER_AUTO_MAP: Final[dict[Repo, list[str | None]]] = {
    "matoub": ["tokenization_matoub.MatoubTokenizer", None],
}
"""Repositories whose tokenizer is remote code too. `AutoTokenizer` reads this from
`tokenizer_config.json` and expects the pair `[slow, fast]`; Matoub has no fast half,
because its front end is a rule table rather than a merge table."""

ARCHITECTURES: Final[dict[Repo, str]] = {
    "belaid": "BelaidForTokenClassification",
    "boulifa": "BoulifaForSeq2SeqLM",
    "dihya": "DihyaForSequenceClassification",
    "feraoun": "FeraounForVision2Seq",
    "feraoun_pro": "FeraounProForCTC",
    "juba": "JubaForSeq2SeqLM",
    "masinissa": "MasinissaForMaskedLM",
    "matoub": "MatoubForTextToWaveform",
}

MATOUB_PROBE: Final = "Azul fell-awen, amek i telliḍ taṣebḥit-a?"

DIHYA_PROBES: Final[tuple[tuple[str, str], ...]] = (
    ("Azul fell-awen, amek tettilim ass-a?", "kab_Latn"),
    ("Mayemmi ur teccid ticemrarin nnecm a gma?", "rif_Latn"),
    ("Bonjour tout le monde, comment allez-vous?", "NOT_AMAZIGH"),
)
"""Kabyle, Tarifit and French, checked against the label each must get.

A logits *shape* is identical for a working classifier and for a head that collapsed to
one class, and three of this model's seven classes are the ones a Kabyle-only probe can
never exercise. The rejection class is in the probe because a filter that cannot say
`NOT_AMAZIGH` is the one failure that silently contaminates a corpus."""

MATOUB_SILENCE_PEAK: Final = 0.01
"""Below this the directory synthesised silence, which is what a mis-slotted style vector
or a half-loaded decoder produces — and a shape check cannot see it."""

MATOUB_MIN_SECONDS: Final = 0.5
"""A 40-phoneme probe under half a second is a duration head that clamped to its floor."""

PIECE_REPOS: Final[frozenset[Repo]] = frozenset({"belaid", "masinissa"})
"""Repositories whose vocabulary is the Mammeri SentencePiece model rather than an alphabet."""


class StagingError(Exception):
    """A directory that would not load on a machine without this repository."""


def build_boulifa_tokenizer() -> PreTrainedTokenizerFast:
    """Boulifa's 128-character inventory as a `tokenizer.json`."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers import processors as tokenizer_processors
    from transformers import PreTrainedTokenizerFast

    from agbalu.standardise.tokenizer import Tokenizer as StandardTokenizer

    reference = StandardTokenizer.build()
    backend = Tokenizer(models.WordLevel(vocab=dict(reference.char_to_id), unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.post_processor = tokenizer_processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s>:0 $A:0 </s>:0 <s>:1 $B:1 </s>:1",
        special_tokens=[("<s>", reference.bos_id), ("</s>", reference.eos_id)],
    )
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=512,
    )


def build_feraoun_tokenizer() -> PreTrainedTokenizerFast:
    """Feraoun's 171-symbol table as a `tokenizer.json`, id for id with `agbalu.ocr`.

    No normalizer: the model is defined over the glyphs on the page, so folding case or
    stripping a combining mark here would change what a decoded id means.
    """
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers import processors as tokenizer_processors
    from transformers import PreTrainedTokenizerFast

    from agbalu.ocr.vocabulary import BOS_ID, BOS_TOKEN, EOS_ID, EOS_TOKEN, VOCABULARY

    vocab = {symbol: index for index, symbol in enumerate(VOCABULARY)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.post_processor = tokenizer_processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, BOS_ID), (EOS_TOKEN, EOS_ID)],
    )
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        model_max_length=256,
    )


def build_feraoun_pro_tokenizer() -> PreTrainedTokenizerFast:
    """Feraoun-Pro's 173-symbol table, id for id with `agbalu.ocr.vocabulary_pro`."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers import processors as tokenizer_processors
    from transformers import PreTrainedTokenizerFast

    from agbalu.ocr.vocabulary_pro import BOS_ID, BOS_TOKEN, EOS_ID, EOS_TOKEN, VOCABULARY

    vocab = {symbol: index for index, symbol in enumerate(VOCABULARY)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.post_processor = tokenizer_processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, BOS_ID), (EOS_TOKEN, EOS_ID)],
    )
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        model_max_length=1024,
    )


BYTE_VALUES: Final = 256

PRINTABLE_RUNS: Final[tuple[tuple[int, int], ...]] = (
    (ord("!"), ord("~")),
    (ord("\xa1"), ord("\xac")),
    (ord("\xae"), ord("\xff")),
)
"""GPT-2's byte-level alphabet: the byte values that already have a printable, non-space
character. Every other byte is displaced into the private-use range above U+0100, which is
what lets a `str`-only tokenizer pipeline carry arbitrary bytes."""


def byte_to_character() -> dict[int, str]:
    """The byte-level table, byte value to the character `ByteLevel` represents it with.

    Written out rather than read from `ByteLevel.alphabet()`, which returns the characters
    with no byte attached — the position of a character in that list is not its byte value,
    so building the vocabulary from it would map most of the alphabet to the wrong id.
    """
    printable = [value for low, high in PRINTABLE_RUNS for value in range(low, high + 1)]
    table = {value: chr(value) for value in printable}
    displaced = 0
    for value in range(BYTE_VALUES):
        if value not in table:
            table[value] = chr(BYTE_VALUES + displaced)
            displaced += 1
    return table


def build_byte_tokenizer() -> PreTrainedTokenizerFast:
    """Dihya's vocabulary: the 256 UTF-8 byte values, shifted past `[PAD]` and `[UNK]`.

    Built through `ByteLevel`, whose byte-to-character table is the only way to reach raw
    bytes from a `tokenizers` pipeline, which works on `str`. The table is a bijection, so
    `id - 2` is the byte — which is what `agbalu.lid.dataset.encode` produces, and
    `tests/unit/test_hub_dihya.py` checks the two agree over a corpus rather than over a
    fixture of ASCII.

    `use_regex=False` because the GPT-2 word-splitting regex would isolate whitespace runs
    into their own pieces; here every byte is its own piece and nothing is grouped.
    """
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from agbalu.lid.config import BYTE_OFFSET, PAD_ID, UNK_ID

    table = byte_to_character()
    vocab = {"[PAD]": PAD_ID, "[UNK]": UNK_ID}
    vocab.update({character: value + BYTE_OFFSET for value, character in table.items()})
    if len(vocab) != BYTE_VALUES + BYTE_OFFSET:
        message = f"the byte vocabulary holds {len(vocab)} entries, not {BYTE_VALUES + 2}"
        raise StagingError(message)

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated"),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_max_length=256,
    )


def build_char_tokenizer() -> PreTrainedTokenizerFast:
    """Juba's alphabet as a `tokenizer.json`, id for id with `CharTokenizer`."""
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    from tokenizers import processors as tokenizer_processors
    from transformers import PreTrainedTokenizerFast

    from agbalu.tifinagh.tokenizer import BOS_ID, EOS_ID, CharTokenizer

    reference = CharTokenizer()
    backend = Tokenizer(models.WordLevel(vocab=dict(reference.char_to_id), unk_token="[UNK]"))
    backend.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex(DOTTED_CAPITAL_I), OUT_OF_ALPHABET),
            normalizers.Lowercase(),
        ]
    )
    # `[\s\S]` rather than `.`, which does not match a newline.
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.post_processor = tokenizer_processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", BOS_ID), ("[EOS]", EOS_ID)],
    )
    backend.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=256,
    )


def build_piece_tokenizer(model_file: Path) -> PreTrainedTokenizerFast:
    """Masinissa's SentencePiece unigram vocabulary as a `tokenizer.json`.

    Converted rather than shipped as-is because `AutoTokenizer` cannot read a bare
    `.model`, and rebuilt from the processor's own pieces and scores rather than through a
    slow-tokenizer class, whose internals were reworked in transformers 5.
    """
    import sentencepiece as spm
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers import processors as tokenizer_processors
    from transformers import PreTrainedTokenizerFast

    processor = spm.SentencePieceProcessor(model_file=str(model_file))
    vocabulary = [
        (processor.id_to_piece(index), processor.get_score(index))
        for index in range(processor.get_piece_size())
    ]
    backend = Tokenizer(models.Unigram(vocabulary, unk_id=1, byte_fallback=True))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always", split=True),
            # `split_digits=True` in the build spec.
            pre_tokenizers.Digits(individual_digits=True),
        ]
    )
    backend.decoder = decoders.Sequence(
        [
            decoders.Metaspace(replacement="\u2581", prepend_scheme="always", split=True),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(content=" ", left=1),
        ]
    )
    backend.post_processor = tokenizer_processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=512,
    )


def build_matoub_tokenizer(directory: Path) -> PreTrainedTokenizerBase:
    """Matoub's phoneme table, written where the tokenizer's own loader will find it."""
    from agbalu.hub.matoub.tokenization_matoub import MatoubTokenizer
    from agbalu.tts.vocabulary import Vocabulary

    vocabulary = Vocabulary.load()
    path = MatoubTokenizer.write_vocabulary(vocabulary.symbols, directory)
    return MatoubTokenizer(vocab_file=str(path))


def write_tokenizer_auto_map(repo: Repo, directory: Path) -> None:
    """Point `AutoTokenizer` at the module the repository ships.

    `save_pretrained` records the class name and nothing that resolves it, so without this
    a downloader gets `Tokenizer class MatoubTokenizer does not exist` — on the Hub, where
    it is visible only to whoever tries.
    """
    mapping = TOKENIZER_AUTO_MAP.get(repo)
    if mapping is None:
        return
    path = directory / "tokenizer_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["auto_map"] = {"AutoTokenizer": mapping}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_modules(repo: Repo, directory: Path) -> list[str]:
    """The standalone modelling code, verbatim. Byte-identical or the copy is a fork."""
    source = HUB / repo
    if not source.is_dir():
        message = f"no hub package at {source}"
        raise StagingError(message)
    copied: list[str] = []
    for path in sorted(source.glob("*.py")):
        shutil.copyfile(path, directory / path.name)
        copied.append(path.name)
    return copied


def write_config(repo: Repo, directory: Path) -> None:
    """The existing manifest, plus what makes an `AutoClass` able to build it.

    The manifest is read and passed through rather than replaced: Masinissa's carries the
    training summary and the full validation curve, which the card promises are there.
    """
    from agbalu.hub.belaid.configuration_belaid import BelaidConfig
    from agbalu.hub.boulifa.configuration_boulifa import BoulifaConfig
    from agbalu.hub.dihya.configuration_dihya import DihyaConfig
    from agbalu.hub.feraoun.configuration_feraoun import FeraounConfig
    from agbalu.hub.feraoun_pro.configuration_feraoun_pro import FeraounProConfig
    from agbalu.hub.juba.configuration_juba import JubaConfig
    from agbalu.hub.masinissa.configuration_masinissa import MasinissaConfig
    from agbalu.hub.matoub.configuration_matoub import MatoubConfig

    path = directory / "config.json"
    if not path.is_file():
        message = f"no config.json in {directory}; export the weights first"
        raise StagingError(message)
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing.pop("auto_map", None)
    existing.pop("architectures", None)

    classes = {
        "belaid": BelaidConfig,
        "boulifa": BoulifaConfig,
        "dihya": DihyaConfig,
        "feraoun": FeraounConfig,
        "feraoun_pro": FeraounProConfig,
        "juba": JubaConfig,
        "masinissa": MasinissaConfig,
        "matoub": MatoubConfig,
    }
    config = classes[repo](**existing)
    config.auto_map = dict(AUTO_MAP[repo])
    config.architectures = [ARCHITECTURES[repo]]
    config.save_pretrained(directory)


def refresh_manifest(directory: Path) -> None:
    """Bring `export.stats.json`'s file list back in line with the directory.

    It is written by the exporter, which runs before this and so records a directory that
    no longer exists. A manifest listing fewer files than were published is the same class
    of defect as a card claiming what a checkpoint does not hold.
    """
    import hashlib

    path = directory / "export.stats.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = [
        {
            "name": entry.name,
            "bytes": entry.stat().st_size,
            "sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
        }
        for entry in sorted(directory.iterdir())
        if entry.name != "export.stats.json"
    ]
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class _Prediction(TypedDict):
    """One row of `DihyaForSequenceClassification.identify`."""

    language: str
    confidence: float
    prediction_set: tuple[str, ...]
    probabilities: dict[str, float]


class _Identifier(Protocol):
    """What `verify_dihya` needs of the remote-code class `AutoModel` returned.

    A protocol rather than the local `DihyaForSequenceClassification`: the loaded object is
    built from the *staged* module, so it is a structurally identical class of a different
    identity, and annotating it with the local one would be a claim staging exists to test.
    """

    @property
    def config(self) -> PreTrainedConfig: ...

    def identify(self, texts: list[str]) -> list[_Prediction]: ...


def verify_dihya(model: _Identifier, tokenizer: PreTrainedTokenizerBase, directory: Path) -> str:
    """Classify the probes and check the labels, and check the tokenizer against `encode`.

    Neither check has a shape that could fail instead. A byte vocabulary built through the
    wrong table encodes every string to ids in range, and the model then reads a different
    language than the one on the page; a head that has collapsed to one class returns
    logits of exactly the right rank.

    Raises:
        StagingError: the tokenizer disagrees with the training encoder, or a probe came
            back as the wrong language.
    """
    from agbalu.lid.dataset import encode

    for text, _ in DIHYA_PROBES:
        ours = encode(text, model.config.max_position_embeddings)
        theirs = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ours != theirs:
            message = (
                f"{directory}: the staged tokenizer disagrees with the training encoder "
                f"on {text!r} — {theirs[:8]} against {ours[:8]}"
            )
            raise StagingError(message)

    predicted = model.identify([text for text, _ in DIHYA_PROBES])
    wrong = [
        f"{text!r} -> {got['language']}, expected {expected}"
        for (text, expected), got in zip(DIHYA_PROBES, predicted, strict=True)
        if got["language"] != expected
    ]
    if wrong:
        message = f"{directory} misclassified its probe: {'; '.join(wrong)}"
        raise StagingError(message)
    return " | ".join(
        f"{row['language']} {row['confidence']:.4f} set={list(row['prediction_set'])}"
        for row in predicted
    )


def verify(repo: Repo, directory: Path) -> dict[str, object]:
    """Load the directory the way a downloader would, and report what it produced."""
    import torch
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTextToWaveform,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    loaders = {
        "belaid": AutoModelForTokenClassification,
        "boulifa": AutoModelForSeq2SeqLM,
        "dihya": AutoModelForSequenceClassification,
        "feraoun": AutoModel,
        "feraoun_pro": AutoModel,
        "juba": AutoModelForSeq2SeqLM,
        "masinissa": AutoModelForMaskedLM,
        "matoub": AutoModelForTextToWaveform,
    }
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    model = loaders[repo].from_pretrained(directory, trust_remote_code=True)
    model.eval()

    probes = {
        "belaid": "azul fell-awen amek i tellam",
        "boulifa": "achimi ur d-thekhedmedh ara tamazight g l'ecole?",
        "feraoun": "Aḍris n uḥric ɣef tɛeṛṛamt d uẓekka.",
        "dihya": " | ".join(text for text, _ in DIHYA_PROBES),
        "feraoun_pro": "Aḍris n uḥric ɣef tɛeṛṛamt d uẓekka ţ-ţili.",
        # Both halves of the probe are reported, so the staging record shows what each
        # script produced rather than an average that hides one of them.
        "juba": "ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?",
        "masinissa": "Azul fell-awen, amek i tellam?",
        "matoub": MATOUB_PROBE,
    }
    probe = probes[repo]
    with torch.inference_mode():
        if repo == "belaid":
            # The end-to-end call, not a logits shape: the labels are per word and the model
            # sees subwords, so an aligner that is wrong still produces a plausible tensor.
            produced = model.restore(probe, tokenizer)[0]
        elif repo == "feraoun":
            # Rendered and read back, because the preprocessing is half of this model: a
            # canvas built at the wrong scale still gives logits of the right shape. Seeded,
            # so the typeface is fixed and two stagings of the same weights are comparable.
            import random

            from agbalu.ocr.synthetic import render_text_line

            image = render_text_line(probe, augment=False, rng=random.Random(3))
            produced = model.transcribe([image], tokenizer)[0]
        elif repo == "feraoun_pro":
            # Both scripts, for the same reason and one more: this checkpoint's Neo-Tifinagh
            # arm was added after its Latin one, and a staging that exercises only Latin
            # cannot tell a working dual-script release from a half-trained one.
            import random

            from agbalu.ocr.synthetic_pro import LineRenderer

            renderer = LineRenderer()
            produced = " | ".join(
                model.transcribe(
                    renderer.render_line(line, rng=random.Random(3), apply_augmentation=False)[0]
                )
                for line in (probe, TIFINAGH_PROBE)
            )
        elif repo == "matoub":
            # Synthesised and measured, not a shape: a mis-slotted style vector, a decoder
            # loaded with half its weights and a duration head that collapsed all produce a
            # tensor of exactly the right rank. What separates them is the audio.
            audio = model(**tokenizer(probe, return_tensors="pt")).waveform[0]
            peak = float(audio.abs().max())
            seconds = audio.shape[-1] / model.config.sampling_rate
            if peak < MATOUB_SILENCE_PEAK:
                message = f"{directory} synthesised silence: peak {peak:.5f}"
                raise StagingError(message)
            if seconds < MATOUB_MIN_SECONDS:
                message = (
                    f"{directory} gave {len(tokenizer(probe)['input_ids'])} phonemes "
                    f"{seconds:.2f} s of audio; the duration head is not predicting"
                )
                raise StagingError(message)
            rms = float(audio.pow(2).mean().sqrt())
            produced = (
                f"{tokenizer.phonemize(probe)}  ->  {seconds:.2f} s at "
                f"{model.config.sampling_rate} Hz, peak {peak:.3f}, "
                f"rms {20 * math.log10(rms):.1f} dBFS, "
                f"{int((audio.abs() >= 1.0).sum())} clipped"
            )
        elif repo == "dihya":
            produced = verify_dihya(model, tokenizer, directory)
        elif repo in ("boulifa", "juba"):
            produced = tokenizer.decode(
                model.generate(
                    **tokenizer(probe, return_tensors="pt"), max_length=256, num_beams=1
                )[0],
                skip_special_tokens=True,
            )
        else:
            AutoModel.from_pretrained(directory, trust_remote_code=True)
            encoded = tokenizer(probe, return_tensors="pt")
            produced = str(tuple(model(**encoded).logits.shape))
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer": len(tokenizer),
        "probe": probe,
        "produced": produced,
    }


def stage(repo: Repo, directory: Path) -> dict[str, object]:
    if not directory.is_dir():
        message = f"no staged release at {directory}"
        raise StagingError(message)

    copied = copy_modules(repo, directory)
    tokenizer: PreTrainedTokenizerBase
    if repo == "matoub":
        tokenizer = build_matoub_tokenizer(directory)
    elif repo == "dihya":
        tokenizer = build_byte_tokenizer()
    elif repo == "boulifa":
        tokenizer = build_boulifa_tokenizer()
    elif repo == "feraoun":
        tokenizer = build_feraoun_tokenizer()
    elif repo == "feraoun_pro":
        tokenizer = build_feraoun_pro_tokenizer()
    elif repo not in PIECE_REPOS:
        tokenizer = build_char_tokenizer()
    else:
        pieces = sorted(directory.glob("*.model"))
        if len(pieces) != 1:
            message = f"expected exactly one SentencePiece model in {directory}, found {pieces}"
            raise StagingError(message)
        tokenizer = build_piece_tokenizer(pieces[0])
    tokenizer.save_pretrained(directory)
    write_tokenizer_auto_map(repo, directory)
    write_config(repo, directory)
    refresh_manifest(directory)

    try:
        report = verify(repo, directory)
    except Exception as error:
        message = f"{directory} does not load as a standalone model: {error}"
        raise StagingError(message) from error
    return {"modules": copied, **report}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, choices=sorted(AUTO_MAP))
    parser.add_argument("--dir", type=Path, required=True, dest="directory")
    args = parser.parse_args(argv)

    try:
        report = stage(args.repo, args.directory)
    except StagingError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"{args.directory}")
    print(f"  modules      {report['modules']}")
    print(f"  parameters   {int(str(report['parameters'])):,}")
    print(f"  tokenizer    {report['tokenizer']} tokens")
    print(f"  {report['probe']}  ->  {report['produced']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
