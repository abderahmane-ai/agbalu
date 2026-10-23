"""Phase 11: Jugurtha's corpus, training and evaluation.

python3 -m agbalu.llm.cli documents|aligned|replay|pack|train|smoke|evaluate|criterion
python3 -m agbalu.llm.cli fertility|holdout|merge|select
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

from agbalu.llm import (
    documents,
    expand,
    holdout,
    mixture,
    pack,
    replay,
    sft_build,
    sft_data,
    sft_files,
    teacher,
)
from agbalu.llm.bases import BASE, INSTRUCT, REVISIONS, TEXT_TOWER_PARAMETERS
from agbalu.llm.chat import ChatError
from agbalu.llm.corpus import HOLDOUT_RATE, CorpusError, Source, is_held_out
from agbalu.llm.decontaminate import (
    DecontaminationError,
    Decontaminator,
    evaluation_sentences,
    first_indexable,
    heldout_sentences,
)
from agbalu.llm.documents import DocumentError
from agbalu.llm.evaluate import PARTS, EvaluationError, HubTokenizer, Scorer, exit_criterion
from agbalu.llm.expand import ExpandError
from agbalu.llm.fertility import Encoder, FertilityError, report, sample
from agbalu.llm.merge import ALPHAS, MergeError
from agbalu.llm.mixture import MixtureError
from agbalu.llm.pack import PackError
from agbalu.llm.progress import Heartbeat, stage
from agbalu.llm.provenance import ProvenanceError, code_stamp
from agbalu.llm.recipe import SFT_PEAK_LR, SFT_ROWS_PER_STEP, replay_budget
from agbalu.llm.replay import ReplayError
from agbalu.llm.seeds import CONVERSATIONS_PER_CARD, Card, SeedError
from agbalu.llm.selection import SelectionError
from agbalu.llm.sft_build import BuildError
from agbalu.llm.sft_data import DataError
from agbalu.llm.sft_files import FilesError
from agbalu.llm.sft_rows import RowsError
from agbalu.llm.signals import SignalError
from agbalu.llm.smoke import SMOKE_PARTS
from agbalu.llm.teacher import TeacherError
from agbalu.llm.translations import TranslationError

if TYPE_CHECKING:
    import torch

    from agbalu.llm.forward import LanguageModel

FERTILITY_SEED: Final = 20260810

CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
VOCABULARY: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
FERTILITY_OUT: Final = Path("data/processed/llm/fertility.json")
SAMPLE: Final = 3000

LLM_DIR: Final = Path("data/processed/llm")
RUNS: Final = Path("artifacts/runs")
MERGE_RUNS: Final = tuple(RUNS / f"merge-a{round(alpha * 100):03d}" for alpha in ALPHAS)
"""Where `make merge RUN=merge-a050 ALPHA=0.5` and its evaluation write, per coefficient."""

SFT_DIR: Final = LLM_DIR / "sft"
SFT_CONVERSATIONS: Final = SFT_DIR / "sft-v1.jsonl"
SFT_ROWS: Final = SFT_DIR / "sft-rows.npz"
SFT_VALIDATION: Final = SFT_DIR / "sft-validation.npz"
SFT_EVERY: Final = 10
"""Validation and a checkpoint every 10 of the 91 SFT updates."""
"""Checkpoint and validation cadence in updates (`docs/llm_post_training_design.md` §5)."""
DOCUMENTS_OUT: Final = LLM_DIR / "kab-docs.jsonl"
ALIGNED_OUT: Final = LLM_DIR / "aligned.jsonl"
REPLAY_OUT: Final = LLM_DIR / "replay.jsonl"
BLOCKS_OUT: Final = LLM_DIR / "cpt-v2.npy"
HOLDOUT_STATS: Final = LLM_DIR / "heldout.stats.json"
REGISTRY: Final = Path("resources/corpus_registry.yaml")
DIHYA_RUN: Final = Path("artifacts/runs/dihya")
REPLAY_CACHE: Final = Path("artifacts/hf-replay")
IDENTIFIER_BATCH: Final = 1024
"""Dihya scored 3,594 texts/s at this batch on the HPC's H100 slice, 2026-10-14."""

PACK_WORKERS: Final = 12
"""The pod's loader ceiling (`~/.claude/standards/hpc.md` §9)."""

TEXT: Final = Source(name="agbalu-text-v1", path=CORPUS, kind="kabyle", fields=("text",))
MT_TRAIN: Final = Source(
    name="mt-train",
    path=Path("data/processed/mt/train.jsonl"),
    kind="aligned",
    fields=("source", "target"),
    direction_field="direction",
)
PARALLEL_KAB_FRA: Final = Source(
    name="parallel-kab-fra",
    path=Path("data/interim/parallel/agbalu-parallel-v1.kab-fra.jsonl"),
    kind="aligned",
    fields=("kab", "fra"),
    code="fra_Latn",
)

HOLDOUT_SOURCES: Final = (TEXT, MT_TRAIN, PARALLEL_KAB_FRA)
"""`parallel-kab-fra` gives the French held-out set its size; its pairs never train."""

ALIGNED_SOURCES: Final = (MT_TRAIN,)
"""The 542,729 non-mined pairs. 975,655 of `parallel-kab-fra`'s 1.2M pairs are NLLB-mined,
and the identifier that mined them calls 87-95% of three sibling languages Kabyle."""


def stats_path(path: Path) -> Path:
    return path.with_suffix(".stats.json")


def write_stats(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"code": code_stamp(), **payload}
    path.write_text(json.dumps(stamped, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@contextmanager
def atomic(path: Path) -> Iterator[TextIO]:
    """A text file that appears under its name only once it has been written in full."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w", encoding="utf-8") as handle:
        yield handle
    partial.replace(path)


class HubEncoder:
    """A `transformers` tokenizer, without its special tokens."""

    def __init__(self, repo: str) -> None:
        from transformers import AutoTokenizer

        self._repo = repo
        self._tokenizer = AutoTokenizer.from_pretrained(repo)

    @property
    def name(self) -> str:
        return self._repo

    @property
    def unknown_id(self) -> int | None:
        unknown = self._tokenizer.unk_token_id
        return int(unknown) if isinstance(unknown, int) else None

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encoded = self._tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return [[int(i) for i in ids] for ids in encoded]


class SentencePieceEncoder:
    """One of the `Mammeri-Tok` vocabularies."""

    def __init__(self, path: Path) -> None:
        import sentencepiece as spm

        if not path.is_file():
            message = f"vocabulary not found: {path}"
            raise FertilityError(message)
        self._path = path
        self._processor = spm.SentencePieceProcessor()
        self._processor.load(str(path))

    @property
    def name(self) -> str:
        return self._path.stem

    @property
    def unknown_id(self) -> int | None:
        return int(self._processor.unk_id())

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [[int(i) for i in self._processor.encode(text)] for text in texts]


class HubCounter:
    """Tokens per text in the base's vocabulary, counted exactly as `pack` will encode them."""

    def __init__(self, repo: str) -> None:
        self._tokenizer = HubTokenizer(repo)

    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(ids) for ids in self._tokenizer.encode_batch(texts)]


class DihyaScorer:
    """Dihya-5M's prior-adjusted class probabilities, one mapping per text."""

    def __init__(self, run: Path, device: str) -> None:
        from agbalu.lid.infer import Identifier

        self.identifier = Identifier.load(run, device=device)

    def probabilities(self, texts: Sequence[str]) -> list[dict[str, float]]:
        rows = self.identifier.probabilities(texts, batch_size=IDENTIFIER_BATCH).tolist()
        return [dict(zip(self.identifier.classes, row, strict=True)) for row in rows]


def identifier_device() -> str:
    """Dihya scores 61 texts/s on the laptop CPU and 121 on its GPU, measured 2026-10-14;
    the three million sentence rows need CUDA."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def decontaminator() -> tuple[Decontaminator, str]:
    """The index over every evaluation set, refused unless it fires on one of its own."""
    sentences = evaluation_sentences()
    index = Decontaminator.build(sentences, whole=heldout_sentences())
    control = first_indexable(sentences)
    index.require_positive(control)
    return index, control


def index_stats(index: Decontaminator, control: str) -> dict[str, object]:
    return {"indexed": index.indexed, "too_short": index.too_short, "control": control}


def command_fertility(args: argparse.Namespace) -> int:
    if not args.corpus.is_file():
        print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    sentences = sample(args.corpus, args.sample, seed=args.seed)
    ours = SentencePieceEncoder(args.vocabulary)
    encoders: list[Encoder] = [HubEncoder(args.base), ours]
    measured = report(encoders, sentences, reference=ours.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base": args.base,
        "corpus": str(args.corpus),
        "seed": args.seed,
        **measured.as_dict(),
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{measured.sentences:,} sentences, seed {args.seed}")
    for row in measured.vocabularies:
        ratio = row.ratio_to_reference
        suffix = "" if ratio is None else f"  {ratio}x reference"
        print(
            f"  {row.fertility.name:26} {row.fertility.tokens_per_word:6.3f} tokens/word  "
            f"UNK {100 * row.fertility.unknown_share:.4f}%{suffix}"
        )
    print(f"-> {args.out}")
    return 0


def command_holdout(args: argparse.Namespace) -> int:
    """Write the evaluation sets the corpus withholds, and what they are drawn from."""
    result = holdout.build(HOLDOUT_SOURCES, args.out, rate=args.rate, cap=args.cap)
    write_stats(args.stats, result.as_dict())
    for tally in result.tallies:
        print(
            f"  {tally.source:22} {tally.language:9} {tally.documents:>6,} docs "
            f"{tally.characters:>10,} chars"
        )
    print(f"-> {' '.join(str(p) for p in result.paths)}  {args.stats}")
    return 0


def command_documents(args: argparse.Namespace) -> int:
    from agbalu.registry.loader import load_registry

    registry = load_registry(args.registry)
    index, control = decontaminator()
    scorer = DihyaScorer(args.dihya, args.device or identifier_device())
    rejects = args.out.with_suffix(".rejects.jsonl")
    sentences = expand.merge_expanded_sentences(documents.read_sentences(args.corpus))
    with atomic(args.out) as sink, atomic(rejects) as dropped:
        builder = documents.Builder(scorer, index, HubCounter(args.base), sink, dropped)
        tallies = documents.build(documents.DOCUMENT_SOURCES, sentences, builder, registry)
    rows = [t.as_dict() for t in tallies]
    write_stats(
        stats_path(args.out),
        {
            "base": args.base,
            "registry": registry.version,
            "normaliser": builder.normaliser.version,
            "identifier": {"run": str(args.dihya), "revision": scorer.identifier.revision},
            "decontamination": index_stats(index, control),
            "documents": sum(t.kept for t in tallies),
            "characters": sum(t.characters for t in tallies),
            "tokens": sum(t.tokens for t in tallies),
            "sources": rows,
        },
    )
    for tally in tallies:
        counts = f"read {tally.read:>9,} kept {tally.kept:>9,} {tally.tokens:>12,} tok"
        print(f"  {tally.source_id:36} {counts}")
    print(f"-> {args.out}  {rejects}")
    return 0


def command_aligned(args: argparse.Namespace) -> int:
    from itertools import chain

    index, control = decontaminator()
    records = chain(mixture.translation_records(), expand.read_expanded_aligned())
    with atomic(args.out) as sink:
        writer = mixture.AlignedWriter(index, HubCounter(args.base), sink, rate=args.rate)
        tallies = mixture.build(ALIGNED_SOURCES, records, writer)
    write_stats(
        stats_path(args.out),
        {
            "base": args.base,
            "rate": args.rate,
            "decontamination": index_stats(index, control),
            "documents": sum(t.kept for t in tallies),
            "tokens": sum(t.tokens for t in tallies),
            "sources": [t.as_dict() for t in tallies],
        },
    )
    for tally in tallies:
        print(
            f"  {tally.name:22} read {tally.read:>9,} kept {tally.kept:>9,} {tally.tokens:>12,} tok"
        )
    print(f"-> {args.out}")
    return 0


def command_expand_fetch(args: argparse.Namespace) -> int:
    manifest = expand.fetch_all(raw_root=args.raw_root, models_root=args.models_root)
    write_stats(LLM_DIR / "expand-fetch.stats.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def command_expand_pdfs(args: argparse.Namespace) -> int:
    device = args.device or identifier_device()
    scorer = DihyaScorer(args.dihya, device)
    tallies = expand.extract_pdfs_and_text(
        scorer,
        raw_root=args.raw_root,
        models_root=args.models_root,
        out_path=args.out,
        device=device,
        use_ocr=not args.no_ocr,
        use_boulifa=args.boulifa,
        resume=args.resume,
    )
    write_stats(
        stats_path(args.out),
        {
            "documents": sum(t.kept for t in tallies),
            "characters": sum(t.characters for t in tallies),
            "sources": [t.as_dict() for t in tallies],
        },
    )
    for tally in tallies:
        marker = "  (resumed)" if tally.resumed else ""
        print(f"  {tally.source_id:36} read {tally.read:>7,} kept {tally.kept:>7,}{marker}")
    print(f"-> {args.out}")
    return 0


def command_expand_clean(args: argparse.Namespace) -> int:
    """French-line and glued-token cleanup over every expanded document file."""
    device = args.device or identifier_device()
    scorer = DihyaScorer(args.dihya, device)
    for path in args.files:
        if not path.is_file():
            print(f"  {path}: absent, skipped")
            continue
        counts = expand.clean_jsonl(path, scorer)
        for source_id, tally in counts.items():
            print(f"  {source_id:40} {tally}")
        write_stats(stats_path(path), {"sources": counts})
    return 0


def command_expand_audio(args: argparse.Namespace) -> int:
    device = args.device or identifier_device()
    scorer = DihyaScorer(args.dihya, device)
    tallies = expand.extract_audio_corpora(
        scorer,
        raw_root=args.raw_root,
        models_root=args.models_root,
        out_path=args.out,
        device=device,
    )
    write_stats(
        stats_path(args.out),
        {
            "documents": sum(t.kept for t in tallies),
            "characters": sum(t.characters for t in tallies),
            "sources": [t.as_dict() for t in tallies],
        },
    )
    for tally in tallies:
        print(f"  {tally.source_id:36} read {tally.read:>7,} kept {tally.kept:>7,}")
    print(f"-> {args.out}")
    return 0


def command_expand_aligned(args: argparse.Namespace) -> int:
    device = args.device or identifier_device()
    scorer = DihyaScorer(args.dihya, device)
    counts = expand.extract_aligned_records(
        scorer,
        raw_root=args.raw_root,
        out_path=args.out,
    )
    write_stats(
        stats_path(args.out),
        {"pairs": sum(counts.values()), "sources": counts},
    )
    for source_name, count in counts.items():
        print(f"  {source_name:36} {count:>9,} pairs")
    print(f"-> {args.out}")
    return 0


def command_compile(args: argparse.Namespace) -> int:
    """Run documents -> aligned -> replay -> pack end-to-end on the HPC."""
    doc_args = argparse.Namespace(
        base=args.base,
        registry=REGISTRY,
        corpus=CORPUS,
        dihya=DIHYA_RUN,
        device=args.device,
        out=DOCUMENTS_OUT,
    )
    aligned_args = argparse.Namespace(
        base=args.base,
        rate=HOLDOUT_RATE,
        out=ALIGNED_OUT,
    )
    replay_args = argparse.Namespace(
        base=args.base,
        kabyle=DOCUMENTS_OUT,
        aligned=ALIGNED_OUT,
        cache=REPLAY_CACHE,
        out=REPLAY_OUT,
    )
    pack_args = argparse.Namespace(
        base=args.base,
        kabyle=DOCUMENTS_OUT,
        aligned=ALIGNED_OUT,
        replay=REPLAY_OUT,
        workers=PACK_WORKERS,
        out=BLOCKS_OUT,
    )
    for step_name, fn, ns in (
        ("documents", command_documents, doc_args),
        ("aligned", command_aligned, aligned_args),
        ("replay", command_replay, replay_args),
        ("pack", command_pack, pack_args),
    ):
        with stage(f"compile:{step_name}"):
            code = fn(ns)
            if code != 0:
                return code
    return 0


def read_tokens(path: Path) -> int:
    if not path.is_file():
        message = f"{path} not found; build it first"
        raise PackError(message)
    tokens = json.loads(path.read_text(encoding="utf-8")).get("tokens")
    if not isinstance(tokens, int) or tokens < 1:
        message = f"{path} carries no positive token count"
        raise PackError(message)
    return tokens


def command_replay(args: argparse.Namespace) -> int:
    kabyle = read_tokens(stats_path(args.kabyle))
    aligned = read_tokens(stats_path(args.aligned))
    total = replay_budget(kabyle, aligned)
    budgets = replay.budgets(total)
    index, control = decontaminator()
    counter = HubCounter(args.base)
    tallies = []
    with atomic(args.out) as sink:
        for source in replay.REPLAY_SOURCES:
            fetcher = replay.ShardFetcher(source.repo, args.cache)
            tally = replay.sample(source, budgets[source.name], fetcher, index, counter, sink)
            tallies.append(tally)
            print(f"  {source.name:22} {tally.tokens:>12,} tok of {tally.budget:,}", flush=True)
    write_stats(
        stats_path(args.out),
        {
            "base": args.base,
            "kabyle_tokens": kabyle,
            "aligned_tokens": aligned,
            "budget": total,
            "seed": replay.REPLAY_SEED,
            "decontamination": index_stats(index, control),
            "documents": sum(t.kept for t in tallies),
            "tokens": sum(t.tokens for t in tallies),
            "sources": [t.as_dict() for t in tallies],
        },
    )
    print(f"-> {args.out}")
    return 0


def command_pack(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    eos = AutoTokenizer.from_pretrained(args.base).eos_token_id
    if not isinstance(eos, int):
        message = f"{args.base} has no eos token"
        raise PackError(message)
    kinds = pack.kinds(args.kabyle, args.aligned, args.replay)
    parts = []
    sources: list[str] = []
    for code, kind in enumerate(kinds):
        texts, names = pack.read_texts(kind.path)
        tokens, lengths = pack.encode_all(texts, args.base, args.workers, eos)
        parts.append((tokens, lengths, code))
        sources.extend(names)
        print(f"  {kind.name:8} {len(texts):>10,} docs {int(lengths.sum()):>13,} tok", flush=True)
    encoded = pack.assemble(parts)
    passes = [kind.passes for kind in kinds]
    order = pack.schedule(encoded.kind_of, passes)
    written = pack.write_blocks(encoded, order, args.out)
    payload = pack.summary(encoded, order, written, [k.name for k in kinds], passes, sources)
    write_stats(stats_path(args.out), {"base": args.base, "eos": eos, **payload})
    print(f"  {written.blocks:,} blocks of {written.length}, {written.tail:,} tail tokens dropped")
    print(f"-> {args.out}")
    return 0


TOKENIZER_FILES: Final = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def file_tokens(stats: Path) -> int:
    """The token count a corpus build recorded, which sets a replay draw's rate."""
    tokens = json.loads(stats.read_text(encoding="utf-8")).get("tokens")
    if not isinstance(tokens, int) or tokens < 1:
        message = f"{stats}: no positive token count"
        raise BuildError(message)
    return tokens


def command_sft(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    template = snapshot_download(
        INSTRUCT, revision=REVISIONS[INSTRUCT], allow_patterns=list(TOKENIZER_FILES)
    )
    replay_sources = {
        "kabyle": (args.kabyle, file_tokens(stats_path(args.kabyle))),
        "aligned": (args.aligned, file_tokens(stats_path(args.aligned))),
        "replay": (args.replay, file_tokens(stats_path(args.replay))),
    }
    stats = sft_build.build(
        sft_build.read_conversations(args.conversations),
        AutoTokenizer.from_pretrained(template),
        HubTokenizer(BASE),
        replay_sources,
        sft_build.Outputs(args.out, args.validation),
    )
    write_stats(stats_path(args.out), {"conversations_file": str(args.conversations), **stats})
    print(json.dumps(stats, indent=2))
    print(f"-> {args.out}, {args.validation}")
    return 0


def loader(device: torch.device) -> Callable[[str | Path, torch.dtype], LanguageModel]:
    from agbalu.llm import kernels

    def load(path: str | Path, dtype: torch.dtype) -> LanguageModel:
        return kernels.load(path, dtype=dtype, device=device)

    return load


def cuda() -> torch.device:
    import torch

    if not torch.cuda.is_available():
        message = "no CUDA device; this command runs on the HPC (make smoke TARGET=hpc)"
        raise EvaluationError(message)
    return torch.device("cuda")


def command_train(args: argparse.Namespace) -> int:
    from agbalu.llm.train import RunConfig, StopRequest, train

    config = RunConfig(
        run=args.run,
        model=args.model,
        blocks=args.blocks,
        optimizer=args.optimizer,
        total_steps=args.total_steps,
        stop_after=args.stop_after,
    )
    if args.objective == "sft":
        config = replace(
            config,
            objective="sft",
            blocks=args.blocks if args.blocks != BLOCKS_OUT else SFT_ROWS,
            validation=args.validation,
            peak_lr=SFT_PEAK_LR,
            blocks_per_step=SFT_ROWS_PER_STEP,
            micro_blocks=1,
            checkpoint_every=SFT_EVERY,
            eval_every=SFT_EVERY,
            probe=False,
            expected_parameters=TEXT_TOWER_PARAMETERS[INSTRUCT],
        )
    if args.checkpoint_every is not None:
        config = replace(config, checkpoint_every=args.checkpoint_every)
    if args.eval_every is not None:
        config = replace(config, eval_every=args.eval_every)
    if args.workers is not None:
        config = replace(config, workers=args.workers)
    device = cuda()
    stop = StopRequest()
    stop.install()
    result = train(config, loader(device), HubTokenizer(args.tokenizer), device, stop=stop)
    print(json.dumps(result, indent=2))
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    from agbalu.llm import smoke

    device = cuda()
    load = loader(device)
    tokenizer = HubTokenizer(BASE)
    match args.part:
        case "throughput":
            result = smoke.throughput(args.run, args.blocks, load, tokenizer, device)
        case "parity":
            result = smoke.parity(args.run, args.blocks, load, tokenizer, device)
        case "resume":
            result = smoke.resume(args.run, args.blocks, load, tokenizer, device)
        case _:
            result = smoke.evaluation(load, tokenizer, device)
    out = Path("artifacts/runs") / args.run / f"smoke-{args.part}.json"
    write_stats(out, {"part": args.part, **result})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"-> {out}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    import torch

    from agbalu.llm.evaluate import run

    device = cuda()
    model = loader(device)(args.model, torch.bfloat16)
    result = run(model, HubTokenizer(args.tokenizer), args.parts, device=device, limit=args.limit)
    write_stats(args.out, {"model": args.model, **result})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"-> {args.out}")
    return 0


SNAPSHOT_FILES: Final = ("*.safetensors", "*.json", "*.jinja", "merges.txt", "vocab.json")
"""What a merge reads from a published checkpoint: weights, configs, tokenizer, template."""


def snapshot(repo: str) -> Path:
    """`repo` at its pinned revision in the local Hugging Face cache."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, revision=REVISIONS[repo], allow_patterns=list(SNAPSHOT_FILES))
    return Path(path)


def command_merge(args: argparse.Namespace) -> int:
    from agbalu.llm import merge

    sources = {
        "base": f"{BASE}@{REVISIONS[BASE]}",
        "instruct": f"{INSTRUCT}@{REVISIONS[INSTRUCT]}",
        "cpt": str(args.cpt),
    }
    inputs = merge.Inputs(
        base=args.base or snapshot(BASE),
        cpt=args.cpt,
        instruct=args.instruct or snapshot(INSTRUCT),
        sources=sources,
    )
    out = RUNS / args.run / "model"
    report = merge.merge(inputs, args.alpha, out)
    write_stats(RUNS / args.run / merge.REPORT_FILE, report)
    print(json.dumps(report["tensors"]), json.dumps(report["change"]))
    if args.verify:
        with stage("verify the merge loads"):
            loaded = merge.verify(out)
        write_stats(RUNS / args.run / "verify.json", loaded)
        print(json.dumps(loaded, indent=2))
    print(f"-> {out}")
    return 0


def command_assemble(args: argparse.Namespace) -> int:
    from agbalu.llm import assemble, merge

    out = RUNS / args.run / "model"
    report = assemble.assemble(args.template, args.tower, out)
    write_stats(RUNS / args.run / assemble.REPORT_FILE, report)
    print(json.dumps(report["tensors"]), json.dumps(report["recast_to_template_dtype"]))
    with stage("verify the assembly loads"):
        loaded = merge.verify(out)
    write_stats(RUNS / args.run / "verify.json", loaded)
    print(json.dumps(loaded, indent=2))
    print(f"-> {out}")
    return 0


def command_select(args: argparse.Namespace) -> int:
    from agbalu.llm.selection import read_candidate, select_alpha

    verdict = select_alpha([read_candidate(run) for run in args.runs])
    write_stats(args.out, verdict)
    print(json.dumps(verdict, indent=2))
    print(f"-> {args.out}")
    return 0 if verdict["selected_alpha"] is not None else 2


WIKI: Final = "wikimedia.kabwiki"


def command_seeds(args: argparse.Namespace) -> int:
    from agbalu.llm import seeds
    from agbalu.registry.loader import load_registry

    del args
    registry = load_registry(REGISTRY)
    index, control = decontaminator()
    wiki = next(s for s in documents.DOCUMENT_SOURCES if s.source_id == WIKI)
    articles = ((d.key, d.text) for d in documents.read_source(wiki))
    passages = (
        p
        for p in seeds.wiki_passages(articles, registry.by_id(WIKI).licence)
        if not index.matches(p.text)
    )
    rows = sft_files.read_jsonl(MT_TRAIN.path)
    pairs = (
        p
        for p in seeds.pair_passages(rows, lambda source: registry.by_id(source).licence)
        if not is_held_out(p.text)
        and not index.matches(p.text)
        and not index.matches(p.other or "")
    )
    cards = seeds.attach(seeds.allocate(), passages, pairs)
    seeds.write(cards, sft_files.SEEDS)
    stats = seeds.summary(cards)
    write_stats(
        stats_path(sft_files.SEEDS), {"decontamination": index_stats(index, control), **stats}
    )
    print(json.dumps(stats, indent=2))
    return 0


def command_prompts(args: argparse.Namespace) -> int:
    from agbalu.llm import seeds

    prompts = teacher.load_prompts()
    candidates = sft_files.read_candidates()
    replied = {path.stem for path in teacher.REPLY_DIR.glob("batch-*.txt")}
    waiting = {
        asked
        for name, ids, _ in sft_files.manifests(teacher.BATCH_DIR, "batch")
        if name not in replied
        for asked in ids
    }
    first = sft_files.next_number(teacher.BATCH_DIR, "batch")
    pool = seeds.teacher_cards(seeds.read(sft_files.SEEDS))
    current = [c for c in candidates if c.version in prompts.counted]
    todo_cards = teacher.wanting(pool, teacher.coverage(current), waiting)
    size = args.batches * teacher.CARDS_PER_BATCH
    written = teacher.write_batches(todo_cards[:size], prompts, teacher.BATCH_DIR, first)
    for path in written:
        print(f"paste {path}\n  save the reply as {teacher.REPLY_DIR / path.name}")
    print(f"{len(written)} batches; first line of each reply file: {REPLY_HEADER}")
    return 0


REPLY_HEADER: Final = "# model: <name as the interface shows it> | date: YYYY-MM-DD"


def command_ingest(args: argparse.Namespace) -> int:
    from agbalu.llm import seeds

    del args
    cards = {card.id: card for card in seeds.read(sft_files.SEEDS)}
    candidates: list[teacher.Candidate] = []
    rejects: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []
    for name, ids, version in sft_files.manifests(teacher.BATCH_DIR, "batch"):
        path = teacher.REPLY_DIR / f"{name}.txt"
        if not path.is_file():
            batches.append({"batch": name, "waiting": True})
            continue
        asked = {card_id: cards[card_id] for card_id in ids}
        found, dropped, tally = teacher.parse_reply(teacher.read_reply(path), name, asked, version)
        candidates.extend(found)
        rejects.extend(dropped)
        batches.append(
            {
                "batch": name,
                "requested": tally.requested,
                "returned": tally.returned,
                "missing": tally.missing,
                "rejected": dict(tally.rejected),
                "ignored": tally.ignored,
            }
        )
    sft_files.write_jsonl(sft_files.CANDIDATES, sft_files.candidate_rows(candidates))
    sft_files.write_jsonl(sft_files.INTERIM / "rejects.jsonl", rejects)
    covered = teacher.coverage(candidates)
    stats = {
        "candidates": len(candidates),
        "rejected": len(rejects),
        "cards_complete": sum(n >= CONVERSATIONS_PER_CARD for n in covered.values()),
        "batches": batches,
    }
    write_stats(stats_path(sft_files.CANDIDATES), stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "batches"}, indent=2))
    return 0


def command_signals(args: argparse.Namespace) -> int:
    from agbalu.bench.lid_systems import build
    from agbalu.lid.infer import Identifier
    from agbalu.llm import signals

    del args
    done = sft_files.read_signals(sft_files.LANGUAGE_SIGNALS)
    todo = [c for c in sft_files.read_candidates() if c.id not in done]
    found = signals.language_signals(
        todo, build("glotlid"), Identifier.load(DIHYA_RUN, device=identifier_device())
    )
    rows = sft_files.signal_rows({**done, **found})
    count = sft_files.write_jsonl(sft_files.LANGUAGE_SIGNALS, rows)
    print(f"{len(found)} new, {count} in all -> {sft_files.LANGUAGE_SIGNALS}")
    return 0


def command_fluency(args: argparse.Namespace) -> int:
    import torch

    from agbalu.llm import signals

    device = cuda()
    model = loader(device)(args.model, torch.bfloat16)
    model.eval()
    tokenizer = HubTokenizer(BASE)
    scorer = Scorer.of(model, tokenizer.eos, device)
    heldout = holdout.read(LLM_DIR / "heldout-kab.jsonl")
    floor = signals.percentile(
        signals.bits_per_character(scorer, tokenizer, heldout), signals.FLUENCY_FLOOR
    )
    candidates = sft_files.read_candidates()
    texts = [signals.kabyle_text(c) for c in candidates]
    values = signals.bits_per_character(scorer, tokenizer, texts)
    rows = (
        {"candidate": c.id, "fluency_bpc": round(v, 4)}
        for c, v in zip(candidates, values, strict=True)
    )
    sft_files.write_jsonl(sft_files.FLUENCY_SIGNALS, rows)
    stats = {
        "model": args.model,
        "threshold_bpc": floor,
        "heldout": len(heldout),
        "candidates": len(candidates),
    }
    write_stats(stats_path(sft_files.FLUENCY_SIGNALS), stats)
    print(json.dumps(stats, indent=2))
    return 0


def sft_inputs() -> tuple[
    list[teacher.Candidate], dict[str, Card], sft_data.Evidence, sft_data.Text
]:
    """Everything the review sheet and the accepted set are decided on."""
    from agbalu.llm import seeds, signals
    from agbalu.normalise import Normaliser

    fluency_stats = stats_path(sft_files.FLUENCY_SIGNALS)
    if not fluency_stats.is_file():
        message = f"{fluency_stats} not found; run `make data TASK=fluency` on the pod first"
        raise sft_data.DataError(message)
    threshold = json.loads(fluency_stats.read_text(encoding="utf-8")).get("threshold_bpc")
    if not isinstance(threshold, float):
        message = f"{fluency_stats}: no threshold_bpc"
        raise sft_data.DataError(message)
    evidence = sft_data.Evidence(
        signals.merged(
            sft_files.read_signals(sft_files.LANGUAGE_SIGNALS),
            sft_files.read_signals(sft_files.FLUENCY_SIGNALS),
        ),
        sft_data.read_reviews(sft_files.REVIEWS),
        sft_data.Thresholds(threshold),
        teacher.load_prompts().counted,
    )
    index, _ = decontaminator()
    normaliser = Normaliser()
    checks = sft_data.Text(index.matches, normaliser.normalise)
    cards = {card.id: card for card in seeds.read(sft_files.SEEDS)}
    return sft_files.read_candidates(), cards, evidence, checks


def command_review(args: argparse.Namespace) -> int:
    del args
    candidates, cards, evidence, checks = sft_inputs()
    screened = sft_data.screen(candidates, cards, evidence, checks)
    names = [n for n in sft_data.review_sample(screened.clean, cards) if n not in evidence.reviews]
    sheet = sft_data.review_sheet(names, screened.normalised, cards)
    sft_files.write_jsonl(sft_files.REVIEW_SHEET, sheet)
    print(f"{len(sheet)} to review: {sft_files.REVIEW_SHEET}")
    print(f"write each verdict as a line of {sft_files.REVIEWS}:")
    print('  {"candidate": "<id>", "verdict": "accept" | "reject", "note": "<why>"}')
    return 0


def command_accept(args: argparse.Namespace) -> int:
    from agbalu.normalise import Normaliser

    del args
    candidates, cards, evidence, checks = sft_inputs()
    accepted, ledger = sft_data.assemble(candidates, cards, evidence, checks)
    version = Normaliser().version
    teacher_share = sum(cards[c.card].think for c in accepted) / len(accepted)
    rows = [sft_data.accepted_row(c, cards[c.card], version) for c in accepted]
    if not sft_files.TRANSLATIONS.is_file():
        message = f"{sft_files.TRANSLATIONS} not found; run `make llm TASK=translations` first"
        raise sft_data.DataError(message)
    rows.extend(sft_files.read_jsonl(sft_files.TRANSLATIONS))
    sft_files.write_jsonl(SFT_CONVERSATIONS, rows)
    sft_files.write_jsonl(
        sft_files.LEDGER,
        (
            {"candidate": name, "status": status, "reasons": ledger.reasons[name]}
            for name, status in sorted(ledger.status.items())
        ),
    )
    by_task: dict[str, int] = {}
    for row in rows:
        by_task[str(row["task"])] = by_task.get(str(row["task"]), 0) + 1
    stats = {
        "candidates": len(candidates),
        "status": ledger.counts(),
        "accepted_by_task": dict(sorted(by_task.items())),
        "teacher_think_share": round(teacher_share, 4),
        "think_share": round(sum(bool(r["think"]) for r in rows) / len(rows), 4),
        "splits": {
            split: sum(r["split"] == split for r in rows) for split in ("train", "validation")
        },
        "normaliser": version,
    }
    write_stats(stats_path(SFT_CONVERSATIONS), stats)
    print(json.dumps(stats, indent=2))
    print(f"-> {SFT_CONVERSATIONS}")
    return 0


def command_requests(args: argparse.Namespace) -> int:
    from agbalu.llm import translations

    del args
    text = translations.requests_prompt(teacher.load_prompts())
    sft_files.REQUESTS_PROMPT.parent.mkdir(parents=True, exist_ok=True)
    sft_files.REQUESTS_PROMPT.write_text(text, encoding="utf-8")
    print(f"paste {sft_files.REQUESTS_PROMPT}\n  save the reply as {sft_files.REQUESTS_REPLY}")
    print(f"first line of the reply file: {REPLY_HEADER}")
    return 0


def command_translations(args: argparse.Namespace) -> int:
    from agbalu.llm import seeds, translations
    from agbalu.normalise import Normaliser
    from agbalu.registry.loader import load_registry

    del args
    reply = teacher.read_reply(sft_files.REQUESTS_REPLY)
    templates, rejected = translations.parse_requests(reply.lines)
    registry = load_registry(REGISTRY)
    index, control = decontaminator()
    used = {
        card.passage.text
        for card in seeds.read(sft_files.SEEDS)
        if card.passage is not None and card.passage.other is not None
    }
    normaliser = Normaliser()
    provenance = translations.Provenance(
        lambda source: registry.by_id(source).licence, normaliser.normalise, normaliser.version
    )
    rows, stats = translations.build(
        sft_files.read_jsonl(MT_TRAIN.path),
        templates,
        lambda text: not is_held_out(text) and not index.matches(text),
        used,
        provenance,
    )
    count = sft_files.write_jsonl(sft_files.TRANSLATIONS, rows)
    stats = {
        "requests": {"model": reply.model, "date": reply.date, "rejected": dict(rejected)},
        "written": count,
        "decontamination": index_stats(index, control),
        **stats,
    }
    write_stats(stats_path(sft_files.TRANSLATIONS), stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"-> {sft_files.TRANSLATIONS}")
    return 0


def command_flores(args: argparse.Namespace) -> int:
    """§4.2's teacher check: write the 200-sentence prompts, or score their pasted replies."""
    from agbalu.bench.mt import as_direction, score
    from agbalu.llm.evaluate import flores_side

    english = flores_side("eng", "devtest")[: teacher.FLORES_PILOT]
    names = [f"flores-{n:04d}" for n in range(1, teacher.FLORES_PILOT // teacher.FLORES_BATCH + 1)]
    if not args.score:
        prompts = teacher.load_prompts()
        sft_files.PILOT.mkdir(parents=True, exist_ok=True)
        for offset, name in enumerate(names):
            start = offset * teacher.FLORES_BATCH
            chunk = english[start : start + teacher.FLORES_BATCH]
            text = teacher.translation_prompt(chunk, start + 1, prompts)
            (sft_files.PILOT / f"{name}.txt").write_text(text, encoding="utf-8")
            saved = teacher.REPLY_DIR / f"{name}.txt"
            print(f"paste {sft_files.PILOT / name}.txt\n  save the reply as {saved}")
        return 0
    found: dict[int, str] = {}
    rejected: dict[str, int] = {}
    models: set[str] = set()
    for name in names:
        reply = teacher.read_reply(teacher.REPLY_DIR / f"{name}.txt")
        models.add(f"{reply.model} {reply.date}")
        lines, dropped = teacher.parse_translations(reply.lines, range(1, teacher.FLORES_PILOT + 1))
        found.update(lines)
        for reason, count in dropped.items():
            rejected[reason] = rejected.get(reason, 0) + count
    hypotheses = [found.get(i, "") for i in range(1, teacher.FLORES_PILOT + 1)]
    references = flores_side("kab", "devtest")[: teacher.FLORES_PILOT]
    result = score(hypotheses, references, as_direction("eng-kab"), "devtest")
    report = {
        "teacher": sorted(models),
        "sentences": teacher.FLORES_PILOT,
        "missing": teacher.FLORES_PILOT - len(found),
        "rejected": rejected,
        "chrf++": result.raw.get("chrf++").score,
        "signature": result.raw.get("chrf++").signature,
    }
    write_stats(sft_files.PILOT / "flores.json", report)
    print(json.dumps(report, indent=2))
    return 0


def command_criterion(args: argparse.Namespace) -> int:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.candidate, args.base)]
    verdict = exit_criterion(reports[0], reports[1])
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["adopted"] else 2


def add_corpus_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fertility = sub.add_parser("fertility", help="tokens per Kabyle word, the base against ours")
    fertility.add_argument("--base", default=BASE)
    fertility.add_argument("--corpus", type=Path, default=CORPUS)
    fertility.add_argument("--vocabulary", type=Path, default=VOCABULARY)
    fertility.add_argument("--sample", type=int, default=SAMPLE)
    fertility.add_argument("--seed", type=int, default=FERTILITY_SEED)
    fertility.add_argument("--out", type=Path, default=FERTILITY_OUT)

    held = sub.add_parser("holdout", help="write the evaluation sets the corpus withholds")
    held.add_argument("--out", type=Path, default=LLM_DIR)
    held.add_argument("--stats", type=Path, default=HOLDOUT_STATS)
    held.add_argument("--rate", type=int, default=HOLDOUT_RATE)
    held.add_argument("--cap", type=int, default=holdout.CAP)

    docs = sub.add_parser("documents", help="whole Kabyle documents, filtered and deduplicated")
    docs.add_argument("--base", default=BASE)
    docs.add_argument("--registry", type=Path, default=REGISTRY)
    docs.add_argument("--corpus", type=Path, default=CORPUS)
    docs.add_argument("--dihya", type=Path, default=DIHYA_RUN)
    docs.add_argument("--device", help="identifier device; CUDA when present, else CPU")
    docs.add_argument("--out", type=Path, default=DOCUMENTS_OUT)

    aligned = sub.add_parser("aligned", help="each human parallel pair once, as one document")
    aligned.add_argument("--base", default=BASE)
    aligned.add_argument("--rate", type=int, default=HOLDOUT_RATE)
    aligned.add_argument("--out", type=Path, default=ALIGNED_OUT)

    sample_replay = sub.add_parser("replay", help="English, French, maths and code replay")
    sample_replay.add_argument("--base", default=BASE)
    sample_replay.add_argument("--kabyle", type=Path, default=DOCUMENTS_OUT)
    sample_replay.add_argument("--aligned", type=Path, default=ALIGNED_OUT)
    sample_replay.add_argument("--cache", type=Path, default=REPLAY_CACHE)
    sample_replay.add_argument("--out", type=Path, default=REPLAY_OUT)

    packer = sub.add_parser("pack", help="tokenise, schedule, shuffle and write the blocks")
    packer.add_argument("--base", default=BASE)
    packer.add_argument("--kabyle", type=Path, default=DOCUMENTS_OUT)
    packer.add_argument("--aligned", type=Path, default=ALIGNED_OUT)
    packer.add_argument("--replay", type=Path, default=REPLAY_OUT)
    packer.add_argument("--workers", type=int, default=PACK_WORKERS)
    packer.add_argument("--out", type=Path, default=BLOCKS_OUT)

    sft = sub.add_parser("sft", help="SFT rows from accepted conversations and replay")
    sft.add_argument("--conversations", type=Path, default=SFT_CONVERSATIONS)
    sft.add_argument("--kabyle", type=Path, default=DOCUMENTS_OUT)
    sft.add_argument("--aligned", type=Path, default=ALIGNED_OUT)
    sft.add_argument("--replay", type=Path, default=REPLAY_OUT)
    sft.add_argument("--out", type=Path, default=SFT_ROWS)
    sft.add_argument("--validation", type=Path, default=SFT_VALIDATION)


def add_expansion_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    exp_fetch = sub.add_parser(
        "expand-fetch", help="download new PDF, audio, text and parallel datasets on the HPC"
    )
    exp_fetch.add_argument("--raw-root", type=Path, default=expand.RAW_V2_ROOT)
    exp_fetch.add_argument("--models-root", type=Path, default=expand.HUB_MODELS_ROOT)

    exp_clean = sub.add_parser(
        "expand-clean",
        help="drop French lines and glued documents from the expanded files, re-check with Dihya",
    )
    exp_clean.add_argument("--dihya", type=Path, default=DIHYA_RUN)
    exp_clean.add_argument("--device", help="CUDA when present, else CPU")
    exp_clean.add_argument(
        "--files",
        type=Path,
        nargs="+",
        default=[expand.EXPANDED_DOCS_OUT, expand.EXPANDED_AUDIO_OUT],
    )
    exp_pdfs = sub.add_parser(
        "expand-pdfs",
        help="extract PDFs (PyMuPDF + Feraoun-Pro OCR + Boulifa + Dihya) and theses on the HPC",
    )
    exp_pdfs.add_argument("--raw-root", type=Path, default=expand.RAW_V2_ROOT)
    exp_pdfs.add_argument("--models-root", type=Path, default=expand.HUB_MODELS_ROOT)
    exp_pdfs.add_argument("--dihya", type=Path, default=DIHYA_RUN)
    exp_pdfs.add_argument("--device", help="CUDA when present, else CPU")
    exp_pdfs.add_argument("--no-ocr", action="store_true")
    exp_pdfs.add_argument("--boulifa", action="store_true")
    exp_pdfs.add_argument(
        "--resume",
        action="store_true",
        help="append to an existing --out, skipping the sources it already holds whole",
    )
    exp_pdfs.add_argument("--out", type=Path, default=expand.EXPANDED_DOCS_OUT)

    exp_audio = sub.add_parser(
        "expand-audio",
        help="transcribe Chaine 2 and Tughalin with Fadhma-300M + Belaid + Boulifa + Dihya",
    )
    exp_audio.add_argument("--raw-root", type=Path, default=expand.RAW_V2_ROOT)
    exp_audio.add_argument("--models-root", type=Path, default=expand.HUB_MODELS_ROOT)
    exp_audio.add_argument("--dihya", type=Path, default=DIHYA_RUN)
    exp_audio.add_argument("--device", help="CUDA when present, else CPU")
    exp_audio.add_argument("--out", type=Path, default=expand.EXPANDED_AUDIO_OUT)

    exp_aligned = sub.add_parser(
        "expand-aligned", help="extract and Dihya-filter expanded parallel pairs"
    )
    exp_aligned.add_argument("--raw-root", type=Path, default=expand.RAW_V2_ROOT)
    exp_aligned.add_argument("--dihya", type=Path, default=DIHYA_RUN)
    exp_aligned.add_argument("--device", help="CUDA when present, else CPU")
    exp_aligned.add_argument("--out", type=Path, default=expand.EXPANDED_ALIGNED_OUT)

    comp = sub.add_parser(
        "compile", help="compile documents, aligned, 10-source replay and packed blocks"
    )
    comp.add_argument("--base", default=BASE)
    comp.add_argument("--device", help="CUDA when present, else CPU")


def add_model_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    trainer = sub.add_parser("train", help="continued pretraining, resuming the run if it exists")
    trainer.add_argument("--run", required=True)
    trainer.add_argument("--model", default=BASE)
    trainer.add_argument("--tokenizer", default=BASE)
    trainer.add_argument("--blocks", type=Path, default=BLOCKS_OUT)
    trainer.add_argument("--optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit")
    trainer.add_argument("--objective", choices=("cpt", "sft"), default="cpt")
    trainer.add_argument("--validation", type=Path, default=SFT_VALIDATION)
    trainer.add_argument("--total-steps", type=int)
    trainer.add_argument("--stop-after", type=int)
    trainer.add_argument("--checkpoint-every", type=int)
    trainer.add_argument("--eval-every", type=int)
    trainer.add_argument("--workers", type=int)

    smoke = sub.add_parser("smoke", help="one part of the HPC smoke (task 11.19)")
    smoke.add_argument("--part", choices=SMOKE_PARTS, required=True)
    smoke.add_argument("--run", required=True)
    smoke.add_argument("--blocks", type=Path, default=BLOCKS_OUT)

    evaluate = sub.add_parser("evaluate", help="the evaluation suite on one model")
    evaluate.add_argument("--model", default=BASE)
    evaluate.add_argument("--tokenizer", default=BASE)
    evaluate.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--out", type=Path, required=True)

    criterion = sub.add_parser("criterion", help="the exit criterion: candidate against base")
    criterion.add_argument("--candidate", type=Path, required=True)
    criterion.add_argument("--base", type=Path, required=True)

    sub.add_parser("seeds", help="the 1,200 seed cards with their passages and pairs")
    prompter = sub.add_parser("prompts", help="the next Gemini batches to paste")
    prompter.add_argument("--batches", type=int, default=10)
    sub.add_parser("ingest", help="parse every pasted reply into candidates")
    sub.add_parser("signals", help="GlotLID, Dihya and Boulifa on every candidate")
    fluent = sub.add_parser("fluency", help="CPT-model bits per character per candidate (GPU)")
    fluent.add_argument("--model", default=str(RUNS / "jugurtha-4b-v1" / "best"))
    sub.add_parser("review", help="the review sheet for the operator")
    sub.add_parser("accept", help="the accepted SFT set, sft-v1.jsonl")
    sub.add_parser("requests", help="the one prompt for translation request templates")
    sub.add_parser("translations", help="translation conversations from human pairs")
    checker = sub.add_parser("flores", help="the teacher's 200-sentence FLORES+ check")
    checker.add_argument("--score", action="store_true", help="score the pasted replies")

    chooser = sub.add_parser("select", help="the merge coefficient by the fixed rule")
    chooser.add_argument("--runs", type=Path, nargs="+", default=list(MERGE_RUNS))
    chooser.add_argument("--out", type=Path, default=RUNS / "merge-selection.json")


def add_checkpoint_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    merger = sub.add_parser("merge", help="instruct + alpha * (cpt - base), task 11.21")
    merger.add_argument("--run", required=True)
    merger.add_argument("--alpha", type=float, required=True)
    merger.add_argument("--cpt", type=Path, required=True)
    merger.add_argument("--base", type=Path, help="default: the pinned base from the HF cache")
    merger.add_argument("--instruct", type=Path, help="default: the pinned instruct model")
    merger.add_argument("--no-verify", dest="verify", action="store_false")

    assembler = sub.add_parser("assemble", help="a trained text tower back into its full model")
    assembler.add_argument("--run", required=True)
    assembler.add_argument("--tower", type=Path, required=True)
    assembler.add_argument("--template", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-llm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_corpus_commands(sub)
    add_expansion_commands(sub)
    add_model_commands(sub)
    add_checkpoint_commands(sub)
    return parser


COMMANDS: Final[dict[str, Callable[[argparse.Namespace], int]]] = {
    "fertility": command_fertility,
    "holdout": command_holdout,
    "documents": command_documents,
    "aligned": command_aligned,
    "replay": command_replay,
    "pack": command_pack,
    "sft": command_sft,
    "expand-fetch": command_expand_fetch,
    "expand-pdfs": command_expand_pdfs,
    "expand-audio": command_expand_audio,
    "expand-clean": command_expand_clean,
    "expand-aligned": command_expand_aligned,
    "compile": command_compile,
    "train": command_train,
    "smoke": command_smoke,
    "evaluate": command_evaluate,
    "criterion": command_criterion,
    "merge": command_merge,
    "assemble": command_assemble,
    "select": command_select,
    "seeds": command_seeds,
    "prompts": command_prompts,
    "ingest": command_ingest,
    "signals": command_signals,
    "fluency": command_fluency,
    "review": command_review,
    "accept": command_accept,
    "flores": command_flores,
    "requests": command_requests,
    "translations": command_translations,
}

ERRORS: Final = (
    DataError,
    ExpandError,
    FilesError,
    TranslationError,
    SeedError,
    SignalError,
    TeacherError,
    BuildError,
    ChatError,
    RowsError,
    CorpusError,
    DecontaminationError,
    DocumentError,
    EvaluationError,
    FertilityError,
    MergeError,
    MixtureError,
    SelectionError,
    PackError,
    ProvenanceError,
    ReplayError,
)


OWN_HEARTBEAT: Final = frozenset({"train", "smoke", "evaluate"})
"""Commands whose trainer or evaluator beats with its own status; the rest get a plain one."""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with stage(args.command):
            if args.command in OWN_HEARTBEAT:
                return COMMANDS[args.command](args)
            with Heartbeat(args.command):
                return COMMANDS[args.command](args)
    except ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
