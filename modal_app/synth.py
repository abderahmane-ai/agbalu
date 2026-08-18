"""Generate Kabyle bitext for languages the corpus does not cover (pivot synthesis).

`agbalu.mt.pivot` selects Kabyle sentences whose English or French side survived language
identification. This translates that pivot side into a third language with **stock** NLLB —
never the trimmed fine-tune, whose 52,209 rows hold no Arabic script — and pairs the result
with the human Kabyle.

Only `X->kab` is produced. The Kabyle side is authentic and the third language is machine
made, which is the back-translation arrangement: what the model learns to *generate* is
real, so quality is not capped by the teacher. The reverse would train it to reproduce
NLLB's own output and could never exceed it; serve `kab->X` by pivoting through English at
inference instead.

Where a sentence has both pivot sides, both are translated and the pair is kept only if the
two agree above `--threshold` chrF. Two independent teachers agreeing is a precision
signal; one teacher is a guess. The filtering itself is `agbalu.mt.pivot.combine`, which is
pure and tested without a GPU.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

from agbalu.mt.pivot import CombineStats
from modal_app.common import (
    BENCH_VOLUMES,
    DATA_PATH,
    MT_TIMEOUT,
    RETRIES,
    app,
    bench_image,
    data_volume,
)

GPU: Final = "A10"

LOCAL_PIVOT: Final = Path("data/processed/mt/pivot.jsonl")
REMOTE_PIVOT: Final = "mt/pivot.jsonl"
REMOTE_SYNTH: Final = "synth"

TARGETS: Final[tuple[str, ...]] = ("arb_Arab", "spa_Latn", "deu_Latn")
"""Modern Standard Arabic first: it is Algeria's official language, so it is the direction
Kabyle speakers actually need. Spanish and German follow because their pivot sides already
exist — each extra language costs one generation pass, not a new corpus."""

BASE: Final = "facebook/nllb-200-distilled-1.3B"

BATCH: Final = 128
NUM_BEAMS: Final = 1
"""Greedy. Bulk synthesis is throughput-bound, and beam search buys little on a teacher
whose output is filtered by agreement anyway."""
MAX_LENGTH: Final = 256
PROGRESS_EVERY: Final = 20_000

THRESHOLD: Final = 50.0

log: Final = logging.getLogger("agbalu.synth")


def by_length(indexed: Sequence[tuple[int, str]], size: int) -> Iterator[Sequence[tuple[int, str]]]:
    """Batches of near-equal length, each carrying its record index.

    `DataCollator`-style padding costs `batch x longest`, and the pivot side has a mean of
    20 tokens against a 256 cap. Sorting by length before batching cuts padded tokens
    **5.0x** against corpus order at this batch size, measured on 21,564 pivot strings.
    Character length is the sort key: it correlates with token length and costs no extra
    tokenisation pass. The index rides along, so the caller restores order by writing into
    a dict rather than by position.
    """
    ordered = sorted(indexed, key=lambda pair: len(pair[1]))
    for start in range(0, len(ordered), size):
        yield ordered[start : start + size]


def chrf(first: str, second: str) -> float:
    """chrF between two teachers' translations of the same sentence."""
    import sacrebleu

    return float(sacrebleu.sentence_chrf(first, [second]).score)


@app.function(
    image=bench_image, gpu=GPU, volumes=BENCH_VOLUMES, timeout=MT_TIMEOUT, retries=RETRIES
)
def synthesise(
    targets: str = "",
    threshold: float = THRESHOLD,
    limit: int = 0,
    two_teacher_only: bool = False,
) -> list[CombineStats]:
    """Translate every pivot side into each target, filter, write one file per target.

    `targets` is a comma-separated list of NLLB codes, as `modal_app.bench.score_mt` takes
    its models: modal's CLI cannot parse a `list[str] | None` annotation, and the flag has
    to work from both `modal run` and the launcher.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from agbalu.mt.pivot import BOTH_TEACHERS, Policy, combine, read, write_pairs

    chosen = [code.strip() for code in targets.split(",") if code.strip()] or list(TARGETS)
    records = read(Path(DATA_PATH) / REMOTE_PIVOT)
    if two_teacher_only:
        records = [r for r in records if r.teachers == BOTH_TEACHERS]
    if limit:
        records = records[:limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The checkpoint is fp32 and `dtype` defaults to the checkpoint's, so bf16 has to be
    # asked for. SDPA is the fused attention path M2M100 declares support for.
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            BASE,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    tokenizers = {
        side: AutoTokenizer.from_pretrained(BASE, src_lang=code)
        for side, code in (("eng", "eng_Latn"), ("fra", "fra_Latn"))
    }
    log.info(
        "pivot %s records | %s | targets %s | threshold %.1f",
        f"{len(records):,}",
        device,
        chosen,
        threshold,
    )

    def translate(side: str, target: str) -> dict[int, str]:
        tokenizer = tokenizers[side]
        forced = tokenizer.convert_tokens_to_ids(target)
        if not isinstance(forced, int) or forced == tokenizer.unk_token_id:
            message = f"{target!r} is not a token of the NLLB tokenizer"
            raise RuntimeError(message)
        indexed = [
            (i, text) for i, r in enumerate(records) if (text := getattr(r, side)) is not None
        ]
        out: dict[int, str] = {}
        for chunk in by_length(indexed, BATCH):
            texts = [text for _, text in chunk]
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.inference_mode():
                produced = model.generate(
                    **encoded.to(device),
                    forced_bos_token_id=forced,
                    num_beams=NUM_BEAMS,
                    max_length=MAX_LENGTH,
                )
            decoded = tokenizer.batch_decode(produced, skip_special_tokens=True)
            for (index, _), text in zip(chunk, decoded, strict=True):
                out[index] = text
            if len(out) % PROGRESS_EVERY < BATCH:
                log.info("  %s->%s %s of %s", side, target, f"{len(out):,}", f"{len(indexed):,}")
        return out

    results: list[CombineStats] = []
    for target in chosen:
        english = translate("eng", target)
        french = translate("fra", target)
        policy = Policy(language=target, threshold=threshold, keep_single=not two_teacher_only)
        pairs, stats = combine(records, english, french, policy, chrf)
        out = Path(DATA_PATH) / REMOTE_SYNTH / f"kab-{target}.jsonl"
        write_pairs(pairs, out)
        (out.with_suffix(".stats.json")).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        data_volume.commit()
        results.append(stats)
        log.info("wrote %s | %s", out, json.dumps(stats, ensure_ascii=False))

    return results


@app.local_entrypoint()
def upload_pivot() -> None:
    """Push the pivot job to the data volume."""
    if not LOCAL_PIVOT.exists():
        message = f"{LOCAL_PIVOT} missing; run `make pivot-data` first"
        raise SystemExit(message)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(LOCAL_PIVOT, f"/{REMOTE_PIVOT}")
    print(f"uploaded the pivot job, {LOCAL_PIVOT.stat().st_size / 1e6:.1f} MB")
