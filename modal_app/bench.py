"""Score public MT systems on a Modal GPU (task 7.7).

The weights stay on a volume and the corrected reference is uploaded once, so a rerun
costs the GPU minutes and nothing else. Every result is scored under both conditions of
`docs/benchmark.md` §2, because FLORES+ `kab_Latn` is 16.2% corrupt and the raw number
alone understates any system that spells Kabyle correctly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from modal_app.common import (
    BENCH_TIMEOUT,
    BENCH_VOLUMES,
    CHECKPOINT_PATH,
    DATA_PATH,
    REMOTE_FLORES,
    RETRIES,
    app,
    bench_image,
    data_volume,
    models_volume,
)

if TYPE_CHECKING:
    from agbalu.bench.mt import Direction, Result
    from agbalu.bench.translate import ModelSpec

GPU: Final = "A10"

LOCAL_FLORES: Final = Path("data/raw/hf.flores-plus-kab")
LOCAL_CORRECTED: Final = Path("data/processed/bench/flores-plus-kab_Latn-corrected.jsonl")

REMOTE_CORRECTED: Final = "bench/flores-plus-kab_Latn-corrected.jsonl"

SPLIT: Final = "devtest"
"""devtest, not dev: dev is the split anyone would have tuned on."""

log: Final = logging.getLogger("agbalu.bench")


HUB_WEIGHTS: Final = "agbalu/Amrouche-1.3B"
"""The published fine-tune, for a container whose checkpoint volume does not carry it.

Verified live: public, and it holds `keep.json` (0.38 MB) beside `model.safetensors`
(4.65 GB), which is the pair a *trimmed* model needs — 52,209 of NLLB's 256,206 tokens are
kept, so the weights alone would produce fluent output in the wrong rows of the vocabulary."""


def resolve_weights(weights: str, default: Path) -> Path:
    """The checkpoint to load: what was asked for, or the published one fetched from the Hub.

    An **explicitly requested** path is never substituted. Falling back there would turn a
    mistyped `--weights` into a silent translation by a different model, which is the class of
    failure this file's other guard exists to prevent.

    The fallback covers the case that actually happened: the operator moved Modal accounts, so
    the new checkpoint volume had never been written and every document died on a path that
    could not exist. Nothing was lost — the weights have been on the Hub since the release.

    A directory is returned either way, so nothing downstream needs to know which path was
    taken; `from_pretrained` and `read_keep` both take it. `HF_HOME` is on the models volume,
    so the 4.65 GB is fetched once per volume rather than once per call.
    """
    if weights:
        chosen = Path(weights)
        check_weights(chosen)
        return chosen
    if default.is_dir() and (default / "config.json").is_file():
        return default

    from huggingface_hub import snapshot_download

    log.info("no checkpoint at %s; loading the published %s instead", default, HUB_WEIGHTS)
    return Path(snapshot_download(HUB_WEIGHTS))


def check_weights(weights: Path) -> None:
    """Fail on a checkpoint path that does not resolve, before anything tries to load it.

    `from_pretrained` treats a local path it cannot find as a Hub repo id, so a checkpoint
    volume that is not mounted surfaces as `Repo id must be in the form ...` from inside
    `huggingface_hub` — an error naming neither the path nor the mount.
    """
    if not weights.is_dir():
        message = (
            f"{weights} is not a directory in this container; "
            f"{CHECKPOINT_PATH} must be mounted and the run must have written it"
        )
        raise FileNotFoundError(message)
    if not (weights / "config.json").is_file():
        message = f"{weights} holds no config.json, so it is not a saved checkpoint"
        raise FileNotFoundError(message)


def requested_directions(directions: list[str] | None, known: Sequence[str]) -> frozenset[str]:
    """The directions to score, with an unknown name rejected here rather than silently
    scoring nothing — an empty intersection reads as a finished run with no rows."""
    if directions is None:
        return frozenset(known)
    unknown = [d for d in directions if d not in known]
    if unknown:
        message = f"unknown direction(s) {unknown}; have {list(known)}"
        raise ValueError(message)
    if not directions:
        message = "no directions requested"
        raise ValueError(message)
    return frozenset(directions)


def results_stem(weights: str | None, wanted: frozenset[str], known: Sequence[str]) -> str:
    """The results filename, which must name the run and its scope.

    Both halves are collision fixes. `Path(weights).name` is `final` for every checkpoint,
    so two fine-tunes wrote the same file; and a direction-filtered sweep would otherwise
    overwrite the full one with a subset of its rows.
    """
    if weights is None:
        stem = "mt-baselines"
    else:
        path = Path(weights)
        stem = f"mt-finetuned-{path.parent.name}-{path.name}"
    if wanted != frozenset(known):
        stem += "-" + "+".join(sorted(wanted))
    return stem


class BaselineRow(TypedDict):
    """One model in one direction. A TypedDict rather than a dataclass because the value
    crosses the Modal boundary and lands in a JSON artifact, and stays a plain dict in both."""

    model: str
    family: str
    direction: str
    split: str
    sentences: int
    reference: str
    normaliser_version: str
    raw: dict[str, float]
    normalised: dict[str, float]
    chrf_gap: float
    bleu_gap: float


def check_corpora(directions: Sequence[Direction], corrected: bool) -> None:
    """Every file every scheduled direction reads, before a single model is downloaded.

    A language added to `LANGUAGE_CODE` but not re-uploaded fails inside `read_split` on the
    direction that needs it — which, on a seven-direction sweep, is after the first four have
    been scored and paid for. The check is a stat call; the failure it prevents is GPU
    minutes, times `RETRIES`.
    """
    from agbalu.bench.flores import split_path
    from agbalu.bench.mt import LANGUAGE_CODE, source_language, target_language

    root = Path(DATA_PATH) / REMOTE_FLORES
    needed: set[Path] = set()
    for direction in directions:
        needed.add(split_path(root, SPLIT, LANGUAGE_CODE[source_language(direction)]))
        if corrected and target_language(direction) == "kab":
            needed.add(Path(DATA_PATH) / REMOTE_CORRECTED)
        else:
            needed.add(split_path(root, SPLIT, LANGUAGE_CODE[target_language(direction)]))

    missing = sorted(str(path) for path in needed if not path.is_file())
    if missing:
        message = (
            f"{len(missing)} corpus file(s) absent from the volume, so "
            f"{sorted(directions)} cannot all be scored: {missing}. Run `make modal-upload`."
        )
        raise FileNotFoundError(message)


def sources_and_references(direction: Direction, corrected: bool) -> tuple[list[str], list[str]]:
    """Source and reference strings in `(split, id)` order.

    FLORES+ ids restart at 0 per split, so both sides are sorted on the pair.
    """
    from agbalu.bench.flores import read_split
    from agbalu.bench.mt import LANGUAGE_CODE, references_for, source_language, target_language

    root = Path(DATA_PATH) / REMOTE_FLORES
    source = references_for(read_split(root, SPLIT, LANGUAGE_CODE[source_language(direction)]))
    target_code = LANGUAGE_CODE[target_language(direction)]

    if corrected and target_language(direction) == "kab":
        path = Path(DATA_PATH) / REMOTE_CORRECTED
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        reference = [
            row["text"]
            for row in sorted((r for r in rows if r["split"] == SPLIT), key=lambda r: r["id"])
        ]
    else:
        reference = references_for(read_split(root, SPLIT, target_code))

    if len(source) != len(reference):
        message = f"{direction}: {len(source)} sources against {len(reference)} references"
        raise ValueError(message)
    return source, reference


def _score_one(
    spec: ModelSpec, direction: Direction, corrected: bool, weights: str | None = None
) -> Result:
    """`weights` points at a fine-tuned checkpoint; `spec.repo` still supplies the
    tokenizer, which the trim deliberately leaves alone."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from agbalu.bench.mt import score
    from agbalu.bench.translate import generate, sources_for, target_token_id, tokenizer_options
    from agbalu.mt.vocab import Remap, read_keep

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(spec.repo, **tokenizer_options(spec, direction))
    model = AutoModelForSeq2SeqLM.from_pretrained(weights or spec.repo).to(device).eval()

    tables = None
    if weights is not None and (Path(weights) / "keep.json").is_file():
        unk = tokenizer.unk_token_id
        if unk is None:
            message = f"{spec.repo} has no unknown token, so a trimmed model cannot be scored"
            raise RuntimeError(message)
        tables = Remap(read_keep(Path(weights) / "keep.json")).tables(len(tokenizer), unk)

    sources, references = sources_and_references(direction, corrected)
    prepared = sources_for(spec, direction, sources)
    forced = target_token_id(spec, direction, tokenizer)

    log.info(
        "translate %s | %s | %d sentences | %s | forced=%s | trimmed=%s",
        weights or spec.repo,
        direction,
        len(prepared),
        device,
        forced,
        tables is not None,
    )
    hypotheses = generate(prepared, model, tokenizer, forced, device=device, tables=tables)
    return score(hypotheses, references, direction, SPLIT)


@app.function(
    image=bench_image, gpu=GPU, volumes=BENCH_VOLUMES, timeout=BENCH_TIMEOUT, retries=RETRIES
)
def mt_baselines(
    repos: list[str] | None = None,
    corrected: bool = True,
    weights: str | None = None,
    directions: list[str] | None = None,
) -> list[BaselineRow]:
    """Every declared direction of every baseline, scored under both conditions.

    `weights` scores a fine-tuned checkpoint under the same protocol as the baselines it
    has to beat, which is the only way the two numbers are comparable. Its `repos` must
    name the single base model the checkpoint came from, so the tokenizer matches.

    `directions` restricts the sweep. Adding a language to `DIRECTIONS` would otherwise
    make every rerun re-pay for directions already scored.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    from agbalu.bench.mt import DIRECTIONS
    from agbalu.bench.translate import BASELINES

    specs = [s for s in BASELINES if repos is None or s.repo in repos]
    if not specs:
        message = f"no baseline matches {repos}; have {[s.repo for s in BASELINES]}"
        raise ValueError(message)
    if weights is not None and len(specs) != 1:
        message = f"scoring {weights} needs exactly one base model, got {[s.repo for s in specs]}"
        raise ValueError(message)
    if weights is not None:
        check_weights(Path(weights))

    wanted = requested_directions(directions, DIRECTIONS)
    scheduled = [(s, d) for s in specs for d in s.directions if d in wanted]
    if not scheduled:
        message = f"none of {sorted(wanted)} is declared by {[s.repo for s in specs]}"
        raise ValueError(message)
    check_corpora([d for _, d in scheduled], corrected)

    out = Path(DATA_PATH) / "bench" / f"{results_stem(weights, wanted, DIRECTIONS)}-{SPLIT}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    results: list[BaselineRow] = []
    for spec in specs:
        for direction in (d for d in spec.directions if d in wanted):
            result = _score_one(spec, direction, corrected, weights)
            row: BaselineRow = {
                "model": weights or spec.repo,
                "family": spec.family,
                "direction": direction,
                "split": result.split,
                "sentences": result.sentences,
                "reference": "corrected" if corrected else "published",
                "normaliser_version": result.normaliser_version,
                "raw": {m.name: m.score for m in result.raw.metrics},
                "normalised": {m.name: m.score for m in result.normalised.metrics},
                "chrf_gap": round(result.gap("chrf++"), 4),
                "bleu_gap": round(result.gap("bleu"), 4),
            }
            results.append(row)
            log.info(
                "score %s | %s | chrf++ raw %.2f norm %.2f | bleu raw %.2f norm %.2f",
                row["model"],
                direction,
                row["raw"]["chrf++"],
                row["normalised"]["chrf++"],
                row["raw"]["bleu"],
                row["normalised"]["bleu"],
            )
            # After each direction, not at the end: a failure on the fifth would otherwise
            # discard four completed scores and the GPU minutes that bought them.
            out.write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            data_volume.commit()
        models_volume.commit()
    data_volume.commit()
    log.info("wrote %d rows to %s", len(results), out)
    return results


@app.local_entrypoint()
def upload_bench() -> None:
    """Push FLORES+ and the corrected reference to the data volume. 5.7 MB in total."""
    from agbalu.bench.mt import LANGUAGE_CODE

    for path in (LOCAL_FLORES, LOCAL_CORRECTED):
        if not path.exists():
            message = f"{path} missing; run `make bench TASK=audit` first"
            raise SystemExit(message)

    wanted = [
        LOCAL_FLORES / split / f"{language}.jsonl"
        for split in ("dev", "devtest")
        for language in sorted(set(LANGUAGE_CODE.values()))
    ]
    missing = [path for path in wanted if not path.is_file()]
    if missing:
        message = (
            f"{len(missing)} FLORES+ file(s) absent, so the directions needing them cannot be "
            f"scored: {[str(p) for p in missing]}. `make acquire-flores` fetches them."
        )
        raise SystemExit(message)

    with data_volume.batch_upload(force=True) as batch:
        for path in wanted:
            batch.put_file(path, f"/{REMOTE_FLORES}/{path.parent.name}/{path.name}")
        batch.put_file(LOCAL_CORRECTED, f"/{REMOTE_CORRECTED}")
    print(f"uploaded {len(wanted)} FLORES+ splits and the corrected reference")


@app.local_entrypoint()
def score_mt(
    models: str = "", published: bool = False, weights: str = "", directions: str = ""
) -> None:
    """`--models` is a comma-separated repo filter; empty runs every baseline.

    `--weights` scores a fine-tuned checkpoint on the volume instead of the base weights,
    under the same protocol, and needs `--models` to name its single base model.
    `--directions` is a comma-separated scope; empty scores every direction the spec declares.
    """
    repos = [m.strip() for m in models.split(",") if m.strip()] or None
    wanted = [d.strip() for d in directions.split(",") if d.strip()] or None
    rows = mt_baselines.remote(
        repos=repos, corrected=not published, weights=weights or None, directions=wanted
    )
    print(f"\n{'model':<52}{'direction':<11}{'chrf++':>9}{'bleu':>8}{'spbleu':>8}{'gap':>7}")
    for row in rows:
        print(
            f"{row['model'][:50]:<52}{row['direction']:<11}"
            f"{row['normalised']['chrf++']:>9.2f}{row['normalised']['bleu']:>8.2f}"
            f"{row['normalised']['spbleu']:>8.2f}{row['chrf_gap']:>7.2f}"
        )
