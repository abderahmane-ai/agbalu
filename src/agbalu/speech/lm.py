"""A KenLM n-gram over AƔBALU-Text v1, and the CTC decoder it is shallow-fused into.

CTC emits characters independently per frame, so it has no way to prefer a spelling that
is a word over one that is not. An n-gram supplies exactly that and nothing else, which is
why it is the right amount of language model here: Fadhma's output enters the corpus, and a
decoder with a *neural* language model in it is the thing this project rejected Whisper for.

The commands are built rather than run here, so what produced the published binary is a
value the suite can assert on rather than a shell line in a docstring.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pyctcdecode.decoder import BeamSearchDecoderCTC

log: Final = logging.getLogger("agbalu.speech.lm")

LM_ORDER: Final = 5

LM_PRUNE: Final = "0 1 2 3"
"""What built `artifacts/asr/5gram.klm`: keep every unigram, drop bigrams seen once,
trigrams seen twice and 4-grams and above seen three times or fewer. Over 3,041,989
sentences that is 186 MB; unpruned it does not fit a container's memory beside the model."""

ALPHA: Final = 0.5
BETA: Final = 1.5
"""LM weight and word-insertion bonus. The defaults `pyctcdecode` documents, unswept: a
sweep needs a dev split decoded per setting, which is GPU time this has not been given."""

LOCAL_LM_DIR: Final = Path("artifacts/asr")
LOCAL_LM_ARPA: Final = LOCAL_LM_DIR / "5gram.arpa"
LOCAL_LM_BINARY: Final = LOCAL_LM_DIR / "5gram.klm"

BLANK: Final = ""
"""pyctcdecode reads label 0 as the CTC blank and wants it empty."""

DELIMITER: Final = " "
"""And a literal space as the word delimiter, where the CTC vocabulary writes `|`."""


class LanguageModelError(Exception):
    """A language model that cannot be built."""


def extract_unigrams(sentences: Iterable[str]) -> list[str]:
    """The distinct words of the training targets, sorted.

    Sorted rather than in corpus order because beam search constrains itself to this list,
    and an unordered set makes two runs of one decoder produce different beams.
    """
    words: set[str] = set()
    for sentence in sentences:
        words.update(sentence.split())
    return sorted(words)


def arpa_command(text_path: Path, arpa_path: Path, *, order: int = LM_ORDER) -> list[str]:
    """The `lmplz` invocation, as a value.

    `--skip_symbols` is not optional: AƔBALU-Text v1 contains lines carrying `<s>` and
    `</s>`, and `lmplz` aborts on them rather than skipping them.
    """
    return [
        "lmplz",
        "--order",
        str(order),
        "--prune",
        *LM_PRUNE.split(),
        "--skip_symbols",
        "--discount_fallback",
        "--text",
        str(text_path),
        "--arpa",
        str(arpa_path),
    ]


def binary_command(arpa_path: Path, binary_path: Path) -> list[str]:
    """The `build_binary` invocation. `trie` is array-compressed; `probing` is not."""
    return ["build_binary", "trie", str(arpa_path), str(binary_path)]


def _run(command: Sequence[str], what: str) -> None:
    log.info("%s: %s", what, " ".join(command))
    try:
        subprocess.run(list(command), check=True)  # noqa: S603
    except FileNotFoundError as error:
        message = (
            f"{command[0]} is not on PATH. Build KenLM from https://github.com/kpu/kenlm, "
            f"or run `make modal-asr TASK=lm` to build the model in a container instead."
        )
        raise LanguageModelError(message) from error
    except subprocess.CalledProcessError as error:
        message = f"{what} failed with exit {error.returncode}"
        raise LanguageModelError(message) from error


def build_arpa(text_path: Path, arpa_path: Path, *, order: int = LM_ORDER) -> Path:
    """One normalised Kabyle sentence per line in, an ARPA model out."""
    arpa_path.parent.mkdir(parents=True, exist_ok=True)
    _run(arpa_command(text_path, arpa_path, order=order), f"building a {order}-gram")
    log.info("wrote %s (%.1f MB)", arpa_path, arpa_path.stat().st_size / 1e6)
    return arpa_path


def compile_binary(arpa_path: Path, binary_path: Path) -> Path:
    """The ARPA model compiled to the binary trie the decoder memory-maps."""
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    _run(binary_command(arpa_path, binary_path), "compiling to a trie")
    log.info("wrote %s (%.1f MB)", binary_path, binary_path.stat().st_size / 1e6)
    return binary_path


def decoder_labels(vocabulary: Mapping[str, int]) -> list[str]:
    """The CTC classes as pyctcdecode expects them, ordered by id.

    Two substitutions, both load-bearing. `[PAD]` is the blank and must be the empty
    string, and `|` is the word delimiter and must be a space — passed through verbatim
    they appear in every hypothesis, and every error rate computed from one is wrong.
    """
    labels = [token for token, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    labels[0] = BLANK
    return [DELIMITER if label == "|" else label for label in labels]


def build_ctc_decoder(
    vocabulary: Mapping[str, int],
    unigrams: Sequence[str] | None = None,
    kenlm_path: Path | str | None = None,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> BeamSearchDecoderCTC:
    """A beam-search decoder, with the n-gram fused in when its file is on disk.

    A missing file warns and decodes without it rather than raising: the binary lives on a
    volume the container may not have, and a scoring pass that reports a greedy number is
    more useful than one that dies — as long as the log says which it did.
    """
    from pyctcdecode.decoder import build_ctcdecoder  # noqa: PLC0415

    resolved: str | None = None
    if kenlm_path is not None:
        candidate = Path(kenlm_path)
        if candidate.is_file():
            resolved = str(candidate)
            log.info("shallow fusion active: %s alpha=%.2f beta=%.2f", candidate.name, alpha, beta)
        else:
            log.warning("no language model at %s — decoding without one", candidate)

    return build_ctcdecoder(
        labels=decoder_labels(vocabulary),
        kenlm_model_path=resolved,
        unigrams=list(unigrams) if unigrams is not None else None,
        alpha=alpha,
        beta=beta,
    )
