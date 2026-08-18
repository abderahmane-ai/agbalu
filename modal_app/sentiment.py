"""Build `agbalu/KabSentiment`, and score the encoder on it.

`label` is the only part that needs a GPU: Kabyle has no sentiment-annotated text and no
classifier, so the label is taken from the **English side of a Tatoeba pair** and carried
across. That is a projection, not an annotation, and everything downstream inherits its
error — the same relationship that put the previously published Kabyle POS tagger at
63.69% on gold against the 94.8% it reported against its own projection.

`benchmark` fits the frozen probe and the full fine-tune and scores both once on test.

The Kabyle side goes through `agbalu.normalise`, not a hand-written substitution. The
first build used a single `ε`→`ɛ` replacement and shipped 34 rows of 15,000 still
carrying Greek capital gamma, Cyrillic `Ԑ` and non-breaking spaces — 0.23% of a corpus
whose reason for existing is that the alternative is corrupt.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Final

from modal_app.common import BENCH_TIMEOUT, BENCH_VOLUMES, app, bench_image

log: Final = logging.getLogger("agbalu.sentiment")

GPU: Final = "A10G"
CLASSIFIER: Final = "cardiffnlp/twitter-roberta-base-sentiment-latest"
"""Three-class English sentiment. Its label order is negative, neutral, positive, which is
`agbalu.bench.sentiment.LABEL_NAMES` — asserted rather than assumed, below."""

CONFIDENCE: Final = 0.80
"""Softmax floor for keeping a labelled pair. The projection is the weakest link in the
corpus, so the gate is on the teacher's own confidence and it is deliberately high: at
0.80 roughly half the candidates are discarded."""

PER_CLASS: Final = 5_000
MIN_WORDS: Final = 4
MAX_WORDS: Final = 25
"""English word bounds. Below four there is not enough text for sentiment to be carried
by anything but a single word; above twenty-five a Tatoeba row is usually a fragment of
something longer whose Kabyle side was translated loosely."""

BATCH: Final = 256
SEED: Final = 42

TATOEBA: Final = Path("data/raw/tatoeba/tatoeba_kab_eng_2026-08-05.tsv")
TATOEBA_COLUMNS: Final = 4
"""id, Kabyle, id, English. A shorter row is a truncated export line, not a pair."""
OUT_DIR: Final = Path("data/tasks/sentiment")
RESULTS: Final = Path("data/processed/bench/sentiment.json")
ENCODER_RUN: Final = Path("artifacts/runs/agbalu-encoder-v1")
TOKENIZER: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")

NOISE: Final = re.compile(r"https?://|www\.|[@#$£€]|\b\d+\b")
"""Rows the English classifier reads badly and the Kabyle side rarely translates: URLs,
handles, currency and bare numerals."""

benchmark_image: Final = (
    bench_image.add_local_dir(OUT_DIR, remote_path=f"/root/{OUT_DIR}")
    .add_local_dir(ENCODER_RUN, remote_path=f"/root/{ENCODER_RUN}")
    .add_local_file(TOKENIZER, remote_path=f"/root/{TOKENIZER}")
)
"""The splits, the encoder and its vocabulary, mounted rather than uploaded to a volume:
together they are 400 MB, they change only when the encoder is retrained, and a mount is
what keeps the benchmark reproducible from a checkout."""


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", force=True
    )


@app.function(image=bench_image, gpu=GPU, timeout=BENCH_TIMEOUT)
def sentiment_label(sentences: list[str]) -> list[tuple[int, float]]:
    """The classifier's argmax and its confidence, one per input, in the input's order.

    Returns every row rather than filtering here, so the confidence gate is applied once,
    where it is documented, instead of once here and once in the caller.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _configure_logging()
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER).to("cuda").eval()

    from agbalu.bench.sentiment import LABEL_NAMES

    ordered = tuple(model.config.id2label[index].lower() for index in range(len(LABEL_NAMES)))
    if ordered != LABEL_NAMES:
        message = f"{CLASSIFIER} orders its labels {ordered}, not {LABEL_NAMES}"
        raise ValueError(message)

    scored: list[tuple[int, float]] = []
    with torch.no_grad():
        for start in range(0, len(sentences), BATCH):
            window = sentences[start : start + BATCH]
            encoded = tokenizer(
                window, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to("cuda")
            probabilities = model(**encoded).logits.softmax(dim=-1)
            confidence, index = probabilities.max(dim=-1)
            scored.extend(zip(index.tolist(), confidence.tolist(), strict=True))
            log.info("labelled %d/%d", len(scored), len(sentences))
    return scored


@app.function(image=benchmark_image, gpu=GPU, volumes=BENCH_VOLUMES, timeout=BENCH_TIMEOUT)
def sentiment_benchmark() -> dict[str, object]:
    """Frozen probe and full fine-tune over the built splits, each scored once on test."""
    import torch

    from agbalu.bench import sentiment
    from agbalu.model.infer import load_encoder
    from agbalu.tokenizer.evaluate import load as load_tokenizer

    _configure_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {name: sentiment.read_split(name, OUT_DIR) for name in ("train", "dev", "test")}
    log.info("splits %s", {name: len(items) for name, items in splits.items()})

    reports = {}
    for setting in ("probe", "finetune"):
        system = sentiment.System(
            encoder=load_encoder(ENCODER_RUN, "kab", name="best.pt", device=device),
            tokenizer=load_tokenizer(TOKENIZER),
        )
        reports[setting] = sentiment.run(system, splits, setting, device=device).as_dict()
    return {"dataset": "agbalu/KabSentiment", "test_sentences": len(splits["test"]), **reports}


def _candidates() -> list[tuple[str, str]]:
    """Deduplicated, normalised Kabyle-English pairs inside the length and noise gates."""
    import csv

    from agbalu.normalise import Normaliser

    normaliser = Normaliser()
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    with TATOEBA.open(encoding="utf-8-sig", newline="") as handle:
        # `QUOTE_NONE` is not optional: default quoting silently drops 5,904 rows of this
        # export, because Kabyle sentences contain unbalanced apostrophes.
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < TATOEBA_COLUMNS:
                continue
            kabyle = normaliser.normalise(row[1].strip())
            english = row[3].strip()
            if not kabyle or kabyle in seen:
                continue
            if not MIN_WORDS <= len(english.split()) <= MAX_WORDS or NOISE.search(english):
                continue
            seen.add(kabyle)
            pairs.append((kabyle, english))
    return pairs


@app.local_entrypoint()
def run_sentiment(task: str = "benchmark") -> None:
    """`build` writes the splits from Tatoeba; `benchmark` scores the encoder on them."""
    if task == "benchmark":
        print(json.dumps(sentiment_benchmark.remote(), ensure_ascii=False, indent=2))
        return
    if task != "build":
        message = f"unknown task {task!r}; expected build or benchmark"
        raise SystemExit(message)

    from agbalu.bench.sentiment import LABEL_NAMES

    pairs = _candidates()
    print(f"{len(pairs):,} candidate pairs")
    scored = sentiment_label.remote([english for _, english in pairs])

    by_class: dict[int, list[dict[str, object]]] = {index: [] for index in range(len(LABEL_NAMES))}
    for (kabyle, english), (index, confidence) in zip(pairs, scored, strict=True):
        if confidence < CONFIDENCE:
            continue
        by_class[index].append(
            {
                "text_kab": kabyle,
                "text_en": english,
                "label": index,
                "label_name": LABEL_NAMES[index],
                "confidence_score": round(confidence, 4),
                "source": f"tatoeba-kab-eng projected by {CLASSIFIER}",
            }
        )
    for index, name in enumerate(LABEL_NAMES):
        print(f"  {name}: {len(by_class[index]):,} above {CONFIDENCE}")

    keep = min(PER_CLASS, *(len(rows) for rows in by_class.values()))
    shuffler = random.Random(SEED)
    balanced: list[dict[str, object]] = []
    for rows in by_class.values():
        shuffler.shuffle(rows)
        balanced.extend(rows[:keep])
    shuffler.shuffle(balanced)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_start, test_start = int(0.8 * len(balanced)), int(0.9 * len(balanced))
    bounds = {
        "train": (0, dev_start),
        "dev": (dev_start, test_start),
        "test": (test_start, len(balanced)),
    }
    for split, (start, end) in bounds.items():
        rows = balanced[start:end]
        for position, row in enumerate(rows):
            row["id"] = f"kab_sent_{split}_{position:05d}"
        path = OUT_DIR / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        print(f"  {split}: {len(rows):,} -> {path}")
