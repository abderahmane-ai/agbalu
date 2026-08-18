"""The transformers configuration a published CTC model needs beside its weights.

A resumable training checkpoint carries the optimizer, the scheduler and the RNG, and
carries no `config.json` at all — the training loop builds the config from the base model
and never writes it down. A downloader has neither, so `from_pretrained` on the weights
alone cannot construct the architecture.

This writes the files that close that gap, from the same constants the training loop uses.
Every override here is `build_model`'s, imported rather than repeated: two copies of a
config are one config that will disagree with the weights.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Final

BASE_MODEL: Final = "ylacombe/omniASR_W2V_300M_SSL"
"""Omnilingual ASR's 300M SSL encoder, ported to transformers. Apache-2.0.

`model_type` is **`wav2vec2`**, not `wav2vec2-bert` — the card says "almost the same usage
as w2v-bert-2.0", which is loose: this one takes a raw 16 kHz waveform through
`Wav2Vec2FeatureExtractor`, where w2v-BERT takes 80 mel bins. Read from its own
`config.json`; `Wav2Vec2ForCTC` accepts it and comes to 315,479,720 parameters with a
40-class head."""

CTC_OVERRIDES: Final[dict[str, object]] = {
    "ctc_loss_reduction": "mean",
    # An infeasible row would otherwise return inf and poison the batch. `corpus` already
    # rejects those, so this is the second line of defence, not the first: resampling can
    # shorten a clip by a frame.
    "ctc_zero_infinity": True,
    "add_adapter": False,
    "mask_time_prob": 0.05,
    "layerdrop": 0.0,
}
"""What the fine-tune changes about the base configuration. `vocab_size` and
`pad_token_id` are not here because they come from the built vocabulary, which is the one
thing that must not be hard-coded twice."""

ARCHITECTURE: Final = "Wav2Vec2ForCTC"
"""Overridden rather than inherited. `Wav2Vec2Config.from_pretrained(BASE_MODEL)` carries
the SSL base's `Wav2Vec2ForPreTraining`, which is not the head these weights have: a
`pipeline` or a bare `AutoModel` built from that config gets the wrong architecture with
nothing raising."""

log: Final = logging.getLogger("agbalu.speech.release")


class ReleaseError(Exception):
    """A configuration that cannot describe the published weights."""


def write_config(vocabulary: Mapping[str, int], out: Path) -> list[Path]:
    """The configuration, the feature extractor and the tokenizer for the published model.

    The feature extractor is written and not as a courtesy: the base sets
    `do_normalize: true`, so a caller who builds a default extractor feeds the model audio
    on a distribution it never saw and gets a plausible, worse transcript with nothing
    raising.

    The tokenizer likewise. `vocab.json` alone is not a tokenizer — `Wav2Vec2CTCTokenizer`
    reads `tokenizer_config.json` for the special tokens, and its defaults are `<unk>` and
    `<pad>` where this vocabulary writes `[UNK]` and `[PAD]`. Without it `AutoProcessor`
    and the ASR pipeline either fail or decode against the wrong specials.
    """
    from transformers import (  # noqa: PLC0415
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
    )

    from agbalu.speech.vocabulary import (  # noqa: PLC0415
        PAD_TOKEN,
        UNK_TOKEN,
        WORD_DELIMITER,
    )

    for required in (PAD_TOKEN, UNK_TOKEN, WORD_DELIMITER):
        if required not in vocabulary:
            message = f"the vocabulary has no {required!r}, so the blank class is unknown"
            raise ReleaseError(message)

    out.mkdir(parents=True, exist_ok=True)
    vocabulary_path = out / "vocab.json"
    vocabulary_path.write_text(
        json.dumps(dict(vocabulary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    config = Wav2Vec2Config.from_pretrained(BASE_MODEL)
    config.update(
        {**CTC_OVERRIDES, "vocab_size": len(vocabulary), "pad_token_id": vocabulary[PAD_TOKEN]}
    )
    config.architectures = [ARCHITECTURE]
    config.save_pretrained(out)
    Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL).save_pretrained(out)
    Wav2Vec2CTCTokenizer(
        str(vocabulary_path),
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        word_delimiter_token=WORD_DELIMITER,
        do_lower_case=False,
    ).save_pretrained(out)

    written = sorted(path for path in out.iterdir() if path.suffix == ".json")
    log.info("wrote %s", [path.name for path in written])
    return written
