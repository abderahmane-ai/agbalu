"""Translate a sentence or a document with the MT fine-tune, by hand.

Separate from `modal_app.infer` because that module runs on the training image, which
carries neither transformers nor the scoring stack; and on `bench_image` rather than
`mt_image` because generation is what the bench container already holds — the same path
task 8.4 was scored through, so what this prints is what the benchmark measured, trimmed-id
remap and forced target token included.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

from agbalu.bench.mt import DIRECTIONS, Direction
from agbalu.mt.finetune import DEFAULT_RUN_NAME as MT_RUN_NAME
from modal_app.bench import resolve_weights
from modal_app.common import (
    BENCH_TIMEOUT,
    BENCH_VOLUMES,
    CHECKPOINT_PATH,
    DATA_PATH,
    app,
    bench_image,
    data_volume,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from agbalu.bench.translate import Decoder, Generator, Remapper

GPU: Final = "A10"

DEFAULT_TEXT: Final = "The old man of the village went out to the great market."

RETRY_NGRAM_SIZE: Final = 4
RETRY_REPETITION_PENALTY: Final = 1.05
"""Applied on the second pass only, to the segments `agbalu.mt.quality` calls failures.

Not globally, for two reasons. They move the decode away from the one configuration every
published number was measured under, and n-gram blocking damages text that repeats by
design — the King James genealogies are `X begat Y; Y begat Z` for hundreds of words, which
is exactly what a 3-gram block forbids. Segmentation removed every degeneration event from
two whole documents on its own, so the first pass has nothing to fix; this is what catches
what it misses, on the handful of segments where it happens."""

DOCUMENTS: Final = Path("data/documents")
"""Sources, one directory per source language: `data/documents/eng/dracula.txt`. The
language lives in the path rather than the filename so one work translated from two
languages keeps one basename, which is what makes the two outputs comparable."""

TRANSLATIONS: Final = Path("artifacts/translations")
"""Outputs, one directory per direction, mirroring `DOCUMENTS` basename for basename."""

REMOTE_TRANSLATIONS: Final = Path(DATA_PATH) / "translations"
"""The same layout on the data volume, so a spawned document survives the client that asked
for it. `pull_translations` copies from here into `TRANSLATIONS`."""

LENGTH_ALLOWANCE: Final = 2.0
LENGTH_MARGIN: Final = 24
"""The decoder cap, as `allowance x longest source in the batch + margin`.

A fixed cap is wrong in both directions: 256 truncates a long segment's tail, and 512 lets a
short segment that has started to loop fill half a thousand tokens before anything stops it.
NLLB's own decoder was trained to 512, so that is the ceiling the ratio is clamped to.
Twice the source is generous — Kabyle runs about 0.87 of English by word — and the margin
covers the short segments where a ratio alone is too tight."""

TRAINED_TARGET_LENGTH: Final = 512
"""NLLB's own training ceiling for the decoder as well as the encoder
([arXiv 2207.04672](https://arxiv.org/abs/2207.04672) §8.2.4), so no ratio may exceed it."""

DOCUMENT_BATCH_SIZE: Final = 32
"""Larger than the scoring path's 16 because the document path sorts by length before it
batches, so a batch is near-uniform and the padding that made 16 the sensible number is
mostly gone."""

log: Final = logging.getLogger("agbalu.bench")


def parse_direction(value: str) -> Direction:
    """A direction from the command line, checked against the harness's own list.

    Unchecked it reaches `NLLB_CODE` as a `KeyError` on the worker, minutes after launch
    and naming a dict rather than the flag that was mistyped.
    """
    for direction in DIRECTIONS:
        if value == direction:
            return direction
    message = f"unknown direction {value!r}; have {list(DIRECTIONS)}"
    raise ValueError(message)


def default_weights() -> str:
    """Where `modal_app.mt.finetune` saves, which is one directory deeper than the run."""
    return str(Path(CHECKPOINT_PATH) / "mt" / MT_RUN_NAME / "final")


def default_output(source: Path, direction: Direction) -> Path:
    """Where a document's translation belongs.

    Derived rather than typed, so an output cannot land beside the sources it was read from
    or under a name that does not say which direction produced it. The stem carries across
    unchanged: `data/documents/eng/dracula.txt` becomes
    `artifacts/translations/eng-kab/dracula.txt`.
    """
    return TRANSLATIONS / direction / f"{source.stem}.txt"


def by_length[T](
    texts: Sequence[str],
    model: Generator[T],
    encode: Decoder[T],
    forced: int | None,
    *,
    device: str,
    tables: Remapper[T] | None,
    count: Callable[[str], int],
    retry: bool,
) -> list[str]:
    """One decoding pass, batched longest-first, returned in input order.

    Sorting by length is what `bench.translate.generate` refuses to do and is right here:
    scoring pairs hypothesis *i* with reference *i* and must never reorder, while a document
    is reassembled by index either way. A near-uniform batch also spends almost nothing on
    padding, which is most of the cost of a nine-thousand-segment novel.

    The decoder cap is per batch and proportional, so a short segment cannot fill hundreds of
    tokens before anything stops it, and a long one still has room for its tail.
    """
    from agbalu.bench.translate import generate

    order = sorted(range(len(texts)), key=lambda index: -count(texts[index]))
    produced: dict[int, str] = {}
    for start in range(0, len(order), DOCUMENT_BATCH_SIZE):
        chunk = order[start : start + DOCUMENT_BATCH_SIZE]
        longest = max(count(texts[index]) for index in chunk)
        cap = min(int(LENGTH_ALLOWANCE * longest) + LENGTH_MARGIN, TRAINED_TARGET_LENGTH)
        hypotheses = generate(
            [texts[index] for index in chunk],
            model,
            encode,
            forced,
            device=device,
            batch_size=DOCUMENT_BATCH_SIZE,
            tables=tables,
            max_length=cap,
            no_repeat_ngram_size=RETRY_NGRAM_SIZE if retry else None,
            repetition_penalty=RETRY_REPETITION_PENALTY if retry else None,
        )
        produced.update(zip(chunk, hypotheses, strict=True))
    return [produced[index] for index in range(len(texts))]


def two_pass[T](
    sources: Sequence[str],
    model: Generator[T],
    encode: Decoder[T],
    forced: int | None,
    *,
    device: str,
    tables: Remapper[T] | None,
    count: Callable[[str], int],
    prepare_sources: Callable[[list[str]], list[str]],
    label: str,
) -> list[str]:
    """Decode as the benchmark does, then decode the failures again with the penalties on.

    A rate too small to see in a benchmark is still several ruined paragraphs in a novel, and
    the alternative — penalties on every segment — moves the whole document off the one
    configuration whose quality has been measured.
    """
    from agbalu.mt.quality import failed

    hypotheses = by_length(
        prepare_sources(list(sources)),
        model,
        encode,
        forced,
        device=device,
        tables=tables,
        count=count,
        retry=False,
    )
    broken = [n for n, source in enumerate(sources) if failed(source, hypotheses[n])]
    if not broken:
        log.info("%s | %d segments, none failed", label, len(sources))
        return hypotheses

    log.warning(
        "%s | %d of %d segments failed, decoding them again", label, len(broken), len(sources)
    )
    again = by_length(
        prepare_sources([sources[n] for n in broken]),
        model,
        encode,
        forced,
        device=device,
        tables=tables,
        count=count,
        retry=True,
    )
    unrecovered = 0
    for n, hypothesis in zip(broken, again, strict=True):
        if failed(sources[n], hypothesis):
            unrecovered += 1
            log.warning("%s | still failing | %s", label, sources[n][:120])
        hypotheses[n] = hypothesis
    log.warning("%s | %d of %d still failing after the retry", label, unrecovered, len(broken))
    return hypotheses


@app.function(image=bench_image, gpu=GPU, volumes=BENCH_VOLUMES, timeout=BENCH_TIMEOUT)
def mt_predict(
    text: str,
    direction: str = "eng-kab",
    weights: str = "",
    compare: bool = False,
    name: str = "",
) -> dict[str, str]:
    """The fine-tune's translation, and the base model's beside it when asked.

    The source is folded onto what NLLB's vocabulary can represent before it is split. A
    typeset document is otherwise 7.55% `<unk>` by token on *Alice* and 2.05% on *Dracula*,
    and the marks that produce it are the quotation marks around every line of dialogue.

    `name` is the document's basename, and passing it makes the translation land on the data
    volume as well as coming back in the result. That is what lets a long document be
    `spawn`ed instead of run: *Dracula* is 9,547 segments and about thirteen GPU-minutes, and
    a `modal run` client's disconnect is an input cancellation — it died at ten minutes with
    nothing kept. A result only the caller can see is lost with the caller.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from agbalu.bench.translate import (
        BASELINES,
        sources_for,
        target_token_id,
        tokenizer_options,
    )
    from agbalu.mt.casing import restore, soften
    from agbalu.mt.finetune import BASE_MODEL
    from agbalu.mt.segment import assemble, plan
    from agbalu.mt.typography import prepare, translatable
    from agbalu.mt.vocab import Remap, read_keep

    chosen = parse_direction(direction)
    text = prepare(text)
    path = resolve_weights(weights, Path(default_weights()))

    spec = next(s for s in BASELINES if s.repo == BASE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(spec.repo, **tokenizer_options(spec, chosen))
    forced = target_token_id(spec, chosen, tokenizer)

    def count(piece: str) -> int:
        # Without specials, which is what `MAX_SOURCE_TOKENS` already subtracts.
        return len(tokenizer(piece, add_special_tokens=False)["input_ids"])

    # Segmented rather than truncated. `generate` encodes with `truncation=True`, so a source
    # past 1,024 positions loses its tail with nothing in the output to say so — and NLLB is
    # sentence-level anyway, so units are how it is meant to be fed.
    segments = plan(text, count)
    if not segments:
        message = "nothing to translate: the text is empty or only whitespace"
        raise ValueError(message)
    sendable = [index for index, segment in enumerate(segments) if translatable(segment.text)]

    tables = None
    if (path / "keep.json").is_file():
        unk = tokenizer.unk_token_id
        if unk is None:
            message = f"{spec.repo} has no unknown token, so a trimmed model cannot be run"
            raise RuntimeError(message)
        tables = Remap(read_keep(path / "keep.json")).tables(len(tokenizer), unk)

    log.info(
        "translate %s | %s | %s | trimmed=%s | %d segment(s), %d sent, longest %d tokens",
        path,
        chosen,
        device,
        tables is not None,
        len(segments),
        len(sendable),
        max(count(segment.text) for segment in segments),
    )

    def run[T](engine: Generator[T], remap: Remapper[T] | None, label: str) -> str:
        sources = [segments[index].text for index in sendable]
        hypotheses = two_pass(
            sources,
            engine,
            tokenizer,
            forced,
            device=device,
            tables=remap,
            count=count,
            # Headings reach the model in title case and the reader in the case they were
            # written in. `soften` is inside `prepare_sources`, so `two_pass` still compares
            # the *original* source against the hypothesis when deciding what to decode again.
            prepare_sources=lambda texts: sources_for(spec, chosen, [soften(t) for t in texts]),
            label=label,
        )
        pieces = [segment.text for segment in segments]
        for index, hypothesis in zip(sendable, hypotheses, strict=True):
            pieces[index] = restore(segments[index].text, hypothesis)
        return assemble(segments, pieces)

    model = AutoModelForSeq2SeqLM.from_pretrained(str(path)).to(device).eval()
    result = {
        "direction": chosen,
        "source": text,
        "finetuned": run(model, tables, "fine-tuned"),
        "segments": str(len(segments)),
    }
    log.info("  fine-tuned | %s", result["finetuned"][:160])

    if compare:
        base = AutoModelForSeq2SeqLM.from_pretrained(spec.repo).to(device).eval()
        result["base"] = run(base, None, "base")
        log.info("  base       | %s", result["base"][:160])

    if name:
        landed = REMOTE_TRANSLATIONS / chosen / f"{name}.txt"
        landed.parent.mkdir(parents=True, exist_ok=True)
        landed.write_text(result["finetuned"], encoding="utf-8")
        data_volume.commit()
        log.info("  wrote %s", landed)
    return result


@app.local_entrypoint()
def translate(
    text: str = DEFAULT_TEXT,
    direction: str = "eng-kab",
    weights: str = "",
    compare: bool = False,
    file: str = "",
    out: str = "",
) -> None:
    """One sentence from `--text`, or a whole document from `--file`.

    A document is what `--file` exists for: passing one through `--text` means the shell's
    argument limit and its quoting decide what the model receives. `--out` is optional for a
    document and `default_output` is where it goes without one.
    """
    chosen = parse_direction(direction)
    if file:
        source = Path(file)
        if not source.is_file():
            message = f"{source} does not exist"
            raise SystemExit(message)
        text = source.read_text(encoding="utf-8")
        out = out or str(default_output(source, chosen))

    result = mt_predict.remote(
        text=text,
        direction=direction,
        weights=weights,
        compare=compare,
        name=Path(file).stem if file else "",
    )
    pieces = result.get("segments", "1")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(result["finetuned"], encoding="utf-8")
        print(f"\n{result['direction']} | {pieces} segment(s) -> {out}\n")
        return

    print(f"\n{result['direction']} | {pieces} segment(s)")
    print(f"  source      {result['source']}")
    if "base" in result:
        print(f"  base        {result['base']}")
    print(f"  fine-tuned  {result['finetuned']}\n")


@app.function(image=bench_image, volumes=BENCH_VOLUMES, timeout=BENCH_TIMEOUT)
def fetch_translations() -> dict[str, str]:
    """Every translation on the volume, keyed `<direction>/<name>.txt`.

    Read inside a container rather than through the local `Volume` API: those methods are
    annotated as async generators, and this needs no GPU, so the cost is one cached
    container start.
    """
    if not REMOTE_TRANSLATIONS.is_dir():
        return {}
    return {
        str(path.relative_to(REMOTE_TRANSLATIONS)): path.read_text(encoding="utf-8")
        for path in sorted(REMOTE_TRANSLATIONS.rglob("*.txt"))
    }


TITLE_FIELD: Final[dict[str, str]] = {
    "eng-kab": "kabyle_title",
    "fra-kab": "kabyle_title",
    "arb-kab": "kabyle_title",
    "kab-eng": "english_title",
}
"""Which sidecar field names a document's title in the direction's target language.

A published title is an editorial choice, not a measurement: a work has a name in Kabyle
that a sentence-level model has no way to know, and `Dracula` coming back as a transliterated
noun phrase is wrong in a way no chrF++ would show. The substitution is therefore opt-out
and recorded, and it never touches anything but the first line."""


def retitle(text: str, title: str) -> str:
    """`text` with its first line replaced by `title`, keeping the file's line ending."""
    lines = text.splitlines()
    if not lines:
        return text
    lines[0] = title.upper()
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def official_titles(root: Path = Path("data/documents")) -> dict[str, dict[str, str]]:
    """Sidecar contents by document stem.

    A sidecar that will not parse raises here rather than being skipped: it is the file
    that carries the document's provenance, and one silently ignored is one whose licence
    nobody checked.
    """
    found: dict[str, dict[str, str]] = {}
    for sidecar in sorted(root.rglob("*.meta.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        found[sidecar.name.removesuffix(".meta.json")] = payload
    return found


@app.local_entrypoint()
def pull_translations(retitle_documents: bool = True) -> None:
    """Copy every translation the volume holds into `artifacts/translations/`."""
    landed = fetch_translations.remote()
    if not landed:
        print("nothing on the volume yet — `make modal-status FUNCTION=mt_predict`")
        return

    sidecars = official_titles() if retitle_documents else {}
    retitled = 0
    for relative, text in sorted(landed.items()):
        local = TRANSLATIONS / relative
        local.parent.mkdir(parents=True, exist_ok=True)

        direction = relative.split("/")[0]
        sidecar = sidecars.get(Path(relative).stem, {})
        title = sidecar.get(TITLE_FIELD.get(direction, ""), "")
        body = retitle(text, title) if title else text
        retitled += bool(title)

        local.write_text(body, encoding="utf-8")
        print(f"  {relative} -> {local}{'  (title from sidecar)' if title else ''}")
    print(f"\npulled {len(landed)} translation(s), {retitled} retitled, into {TRANSLATIONS}\n")
