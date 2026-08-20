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
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

Repo = Literal["belaid", "boulifa", "feraoun", "juba", "masinissa"]

HUB: Final = Path("src/agbalu/hub")

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
    "feraoun": {
        "AutoConfig": "configuration_feraoun.FeraounConfig",
        "AutoModel": "modeling_feraoun.FeraounForVision2Seq",
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
}

ARCHITECTURES: Final[dict[Repo, str]] = {
    "belaid": "BelaidForTokenClassification",
    "boulifa": "BoulifaForSeq2SeqLM",
    "feraoun": "FeraounForVision2Seq",
    "juba": "JubaForSeq2SeqLM",
    "masinissa": "MasinissaForMaskedLM",
}

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
    from agbalu.hub.feraoun.configuration_feraoun import FeraounConfig
    from agbalu.hub.juba.configuration_juba import JubaConfig
    from agbalu.hub.masinissa.configuration_masinissa import MasinissaConfig

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
        "feraoun": FeraounConfig,
        "juba": JubaConfig,
        "masinissa": MasinissaConfig,
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


def verify(repo: Repo, directory: Path) -> dict[str, object]:
    """Load the directory the way a downloader would, and report what it produced."""
    import torch
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    loaders = {
        "belaid": AutoModelForTokenClassification,
        "boulifa": AutoModelForSeq2SeqLM,
        "feraoun": AutoModel,
        "juba": AutoModelForSeq2SeqLM,
        "masinissa": AutoModelForMaskedLM,
    }
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    model = loaders[repo].from_pretrained(directory, trust_remote_code=True)
    model.eval()

    probes = {
        "belaid": "azul fell-awen amek i tellam",
        "boulifa": "achimi ur d-thekhedmedh ara tamazight g l'ecole?",
        "feraoun": "Aḍris n uḥric ɣef tɛeṛṛamt d uẓekka.",
        "juba": "ⵜⵛⴼⵉⴹ ⴼⵍⵍⵉ ?",
        "masinissa": "Azul fell-awen, amek i tellam?",
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
    if repo == "boulifa":
        tokenizer = build_boulifa_tokenizer()
    elif repo == "feraoun":
        tokenizer = build_feraoun_tokenizer()
    elif repo not in PIECE_REPOS:
        tokenizer = build_char_tokenizer()
    else:
        pieces = sorted(directory.glob("*.model"))
        if len(pieces) != 1:
            message = f"expected exactly one SentencePiece model in {directory}, found {pieces}"
            raise StagingError(message)
        tokenizer = build_piece_tokenizer(pieces[0])
    tokenizer.save_pretrained(directory)
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
