"""The deploy entrypoint: every long-running function, on the one shared app.

`modal.App("agbalu")` is a single object that `train`, `mt`, `bench` and `infer` all
register against, and a deploy publishes only the functions its entrypoint module imported.
Deploying `modal_app.train` alone leaves `finetune` unreachable by name; deploying
`modal_app.mt` alone drops `pretrain` from the same app. Importing all of them here is what
makes `modal_app.launch --function` able to address any of them.

    modal deploy -m modal_app.deploy
"""

from __future__ import annotations

from modal_app import (
    asr,
    bench,
    boulifa,
    infer,
    jugurtha,
    llm,
    matoub,
    mt,
    ocr,
    punctuation,
    sentiment,
    simohand,
    synth,
    tifinagh,
    train,
    translate,
    tts,
)
from modal_app.common import app

__all__ = [
    "app",
    "asr",
    "bench",
    "boulifa",
    "infer",
    "jugurtha",
    "llm",
    "matoub",
    "mt",
    "ocr",
    "punctuation",
    "sentiment",
    "simohand",
    "synth",
    "tifinagh",
    "train",
    "translate",
    "tts",
]
