"""Phase 11: Jugurtha's corpus, training and evaluation.

python3 -m agbalu.llm.cli documents|aligned|replay|pack|train|smoke|evaluate|criterion
python3 -m agbalu.llm.cli fertility|holdout
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

from agbalu.llm import documents, holdout, mixture, pack, replay
from agbalu.llm.bases import BASE
from agbalu.llm.corpus import HOLDOUT_RATE, CorpusError, Source
from agbalu.llm.decontaminate import (
    DecontaminationError,
    Decontaminator,
    evaluation_sentences,
    first_indexable,
    heldout_sentences,
)
from agbalu.llm.documents import DocumentError
from agbalu.llm.evaluate import PARTS, EvaluationError, HubTokenizer, exit_criterion
from agbalu.llm.fertility import Encoder, FertilityError, report, sample
from agbalu.llm.mixture import MixtureError
from agbalu.llm.pack import PackError
from agbalu.llm.provenance import ProvenanceError, code_stamp
from agbalu.llm.recipe import replay_budget
from agbalu.llm.replay import ReplayError
from agbalu.llm.smoke import SMOKE_PARTS

if TYPE_CHECKING:
    import torch

    from agbalu.llm.forward import LanguageModel

FERTILITY_SEED: Final = 20260810

CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
VOCABULARY: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
FERTILITY_OUT: Final = Path("data/processed/llm/fertility.json")
SAMPLE: Final = 3000

LLM_DIR: Final = Path("data/processed/llm")
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
    with atomic(args.out) as sink, atomic(rejects) as dropped:
        builder = documents.Builder(scorer, index, HubCounter(args.base), sink, dropped)
        tallies = documents.build(
            documents.DOCUMENT_SOURCES, documents.read_sentences(args.corpus), builder, registry
        )
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
    index, control = decontaminator()
    with atomic(args.out) as sink:
        writer = mixture.AlignedWriter(index, HubCounter(args.base), sink, rate=args.rate)
        tallies = mixture.build(ALIGNED_SOURCES, mixture.translation_records(), writer)
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


def add_model_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    trainer = sub.add_parser("train", help="continued pretraining, resuming the run if it exists")
    trainer.add_argument("--run", required=True)
    trainer.add_argument("--model", default=BASE)
    trainer.add_argument("--tokenizer", default=BASE)
    trainer.add_argument("--blocks", type=Path, default=BLOCKS_OUT)
    trainer.add_argument("--optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-llm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_corpus_commands(sub)
    add_model_commands(sub)
    return parser


COMMANDS: Final[dict[str, Callable[[argparse.Namespace], int]]] = {
    "fertility": command_fertility,
    "holdout": command_holdout,
    "documents": command_documents,
    "aligned": command_aligned,
    "replay": command_replay,
    "pack": command_pack,
    "train": command_train,
    "smoke": command_smoke,
    "evaluate": command_evaluate,
    "criterion": command_criterion,
}

ERRORS: Final = (
    CorpusError,
    DecontaminationError,
    DocumentError,
    EvaluationError,
    FertilityError,
    MixtureError,
    PackError,
    ProvenanceError,
    ReplayError,
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except ERRORS as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
