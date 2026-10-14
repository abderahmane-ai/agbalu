"""English, French, maths and code replay for continued pretraining (task 11.17).

Short parallel sentences do not protect general ability: with little replay, AfriqueLLM's
Qwen3.5-4B lost 13.84 points of French MGSM and 9.52 of French MMLU (arXiv 2601.06395,
Tables 16-25), and adding maths and code raised AfriMGSM by 2.84. Replay is a third of the
stream; `recipe.REPLAY_SHARE` owns the number and `budgets` derives each source's tokens.

A dataset of billions of rows cannot be reservoir-sampled from here, so each source is read
shard by shard in a seeded order at a pinned revision, every row kept with a seeded
probability, until its token budget is met. The shards used and the revision are written
beside the sample, which is what makes it reproducible.
"""

from __future__ import annotations

import json
import random
from collections import Counter as Tally
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol, TextIO

from agbalu.llm.corpus import TokenCounter
from agbalu.llm.decontaminate import Decontaminator
from agbalu.llm.progress import Tracker, stage

Format = Literal["parquet", "jsonl"]

REPLAY_SEED: Final = 20260815
COUNT_BATCH: Final = 512
SHARE_TOLERANCE: Final = 1e-9
"""Float slack when checking that the source shares sum to one."""


class ReplayError(Exception):
    """A replay source cannot be listed, read, or filled to its budget."""


@dataclass(frozen=True, slots=True)
class ReplaySource:
    name: str
    repo: str
    prefix: str
    format: Format
    field: str
    licence: str
    share: float
    """This source's fraction of the replay budget."""

    keep: float
    """Probability a row is taken, low enough that a budget spans several shards."""


REPLAY_SOURCES: Final = (
    ReplaySource(
        "fineweb-edu",
        "HuggingFaceFW/fineweb-edu",
        "sample/10BT/",
        "parquet",
        "text",
        "odc-by",
        0.45,
        0.25,
    ),
    ReplaySource(
        "fineweb2-hq-fra",
        "epfml/FineWeb2-HQ",
        "fra_Latn/",
        "parquet",
        "text",
        "odc-by",
        0.30,
        0.25,
    ),
    ReplaySource(
        "finemath-4plus",
        "HuggingFaceTB/finemath",
        "finemath-4plus/",
        "parquet",
        "text",
        "odc-by",
        0.125,
        0.25,
    ),
    ReplaySource(
        "opc-algorithmic-code",
        "OpenCoder-LLM/opc-annealing-corpus",
        "algorithmic_corpus/",
        "jsonl",
        "text",
        "odc-by",
        0.125,
        0.25,
    ),
)


def budgets(total: int, sources: Sequence[ReplaySource] = REPLAY_SOURCES) -> dict[str, int]:
    """Each source's tokens out of `total`. The shares must sum to one."""
    if total < 1:
        message = f"the replay budget must be positive, got {total}"
        raise ReplayError(message)
    shares = sum(source.share for source in sources)
    if abs(shares - 1.0) > SHARE_TOLERANCE:
        message = f"replay shares sum to {shares}, not 1"
        raise ReplayError(message)
    return {source.name: int(total * source.share) for source in sources}


def shard_order(files: Sequence[str], seed: int) -> list[str]:
    order = sorted(files)
    random.Random(seed).shuffle(order)
    return order


def _rows(path: Path, layout: Format, name: str) -> Iterator[object]:
    if layout == "parquet":
        import pyarrow.parquet as pq

        handle = pq.ParquetFile(path)
        if name not in handle.schema_arrow.names:
            message = f"{path}: no column {name!r}; has {handle.schema_arrow.names}"
            raise ReplayError(message)
        for batch in handle.iter_batches(columns=[name], batch_size=2048):
            yield from batch.column(name).to_pylist()
        return
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                message = f"{path}:{number} is not a JSON object"
                raise ReplayError(message)
            yield row.get(name)


@dataclass(slots=True)
class ReplayTally:
    name: str
    repo: str
    revision: str
    budget: int
    shards: list[str] = field(default_factory=list)
    read: int = 0
    kept: int = 0
    tokens: int = 0
    skipped: int = 0
    rejected: Tally[str] = field(default_factory=Tally)

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.name,
            "repo": self.repo,
            "revision": self.revision,
            "budget": self.budget,
            "shards": self.shards,
            "read": self.read,
            "kept": self.kept,
            "tokens": self.tokens,
            "not_sampled": self.skipped,
            "rejected": dict(sorted(self.rejected.items())),
        }


class Fetcher(Protocol):
    """A dataset's files at one revision: listed by prefix, fetched to a local path."""

    @property
    def revision(self) -> str: ...

    def files(self, prefix: str, suffix: str) -> list[str]: ...

    def fetch(self, name: str) -> Path: ...


class ShardFetcher:
    """Lists and downloads one dataset repository's files at a pinned revision."""

    def __init__(self, repo: str, cache: Path) -> None:
        from huggingface_hub import HfApi

        self.repo = repo
        self.cache = cache
        self.api = HfApi()
        self.revision = str(self.api.dataset_info(repo).sha)

    def files(self, prefix: str, suffix: str) -> list[str]:
        names = self.api.list_repo_files(self.repo, repo_type="dataset", revision=self.revision)
        return [n for n in names if n.startswith(prefix) and n.endswith(suffix)]

    def fetch(self, name: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.repo,
                name,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cache,
            )
        )


def sample(
    source: ReplaySource,
    budget: int,
    fetcher: Fetcher,
    decontaminator: Decontaminator,
    counter: TokenCounter,
    sink: TextIO,
    *,
    seed: int = REPLAY_SEED,
) -> ReplayTally:
    """Fill one source's budget, writing rows as they are counted."""
    tally = ReplayTally(source.name, source.repo, fetcher.revision, budget)
    suffix = ".parquet" if source.format == "parquet" else ".jsonl"
    shards = shard_order(fetcher.files(source.prefix, suffix), seed)
    if not shards:
        message = f"{source.repo}: no {suffix} file under {source.prefix!r}"
        raise ReplayError(message)
    draw = random.Random(f"{seed}:{source.name}")
    pending: list[str] = []

    def flush() -> None:
        counts = counter.count(pending)
        for text, tokens in zip(pending, counts, strict=True):
            tally.kept += 1
            tally.tokens += tokens
            row = {"text": text, "source": source.name, "licence": source.licence}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        pending.clear()

    tracker = Tracker(f"replay {source.name} tokens", budget)
    for shard in shards:
        if tally.tokens >= budget:
            break
        tally.shards.append(shard)
        with stage("fetch shard", source=source.name, shard=shard):
            path = fetcher.fetch(shard)
        for value in _rows(path, source.format, source.field):
            tally.read += 1
            if draw.random() >= source.keep:
                tally.skipped += 1
                continue
            if not isinstance(value, str) or not value.strip():
                tally.rejected["empty"] += 1
                continue
            if decontaminator.matches(value):
                tally.rejected["contaminated"] += 1
                continue
            pending.append(value)
            if len(pending) >= COUNT_BATCH:
                before = tally.tokens
                flush()
                tracker.advance(tally.tokens - before, rows_read=tally.read)
                if tally.tokens >= budget:
                    break
        if pending:
            flush()

    if tally.tokens < budget:
        message = f"{source.name}: {tally.tokens:,} tokens after every shard, budget {budget:,}"
        raise ReplayError(message)
    rejected = sum(tally.rejected.values())
    if tally.read != tally.kept + tally.skipped + rejected:
        message = f"{source.name}: {tally.read} rows read do not equal kept + skipped + rejected"
        raise ReplayError(message)
    return tally
