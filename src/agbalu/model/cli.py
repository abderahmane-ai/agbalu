"""Model CLI — data preparation and local training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

from agbalu.model.checkpoint import BEST, LATEST
from agbalu.model.config import DEFAULT_RUN_NAME, PRESETS, ModelError, RunConfig, TrainConfig
from agbalu.model.data import PackedDataset, TokenisedCorpus, tokenise_corpus
from agbalu.model.infer import TOP_K
from agbalu.tokenizer.corpus import DEFAULT_CORPUS
from agbalu.tokenizer.spec import CLS_ID, PAD_ID, SEP_ID

log: Final = logging.getLogger("agbalu.model")

DEFAULT_DATA_DIR: Final = Path("artifacts/model-data")
DEFAULT_TOKENIZER: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
DEFAULT_RUNS_DIR: Final = Path("artifacts/runs")
DEFAULT_RUN_DIR: Final = DEFAULT_RUNS_DIR / DEFAULT_RUN_NAME
"""`train` writes to `<runs>/<name>`, so every command that reads a checkpoint has to
address the run, not the directory holding the runs."""

TOP_NEIGHBOURS: Final = 10
"""More than `fill-mask`'s 5: an analogy's own operands crowd the head of the ranking,
so a short list can be all inputs and no answer."""

N_SPECIAL_TOKENS: Final = 5
MASK_TOKEN_ID: Final = 4
PREVIEW_SENTENCES: Final = 3


def command_data(args: argparse.Namespace) -> int:
    corpus = tokenise_corpus(args.corpus, args.tokenizer, args.out)
    stats = json.loads(corpus.stats.read_text(encoding="utf-8"))
    print(json.dumps(stats, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    from agbalu.model.data import iter_batches
    from agbalu.model.preview import Previewer
    from agbalu.model.trainer import Trainer
    from agbalu.tokenizer.evaluate import load

    corpus = TokenisedCorpus(
        train=args.data / "train.bin",
        validation=args.data / "validation.bin",
        stats=args.data / "tokenised.stats.json",
    )
    if not corpus.train.is_file():
        msg = f"{corpus.train} missing; run `make model-data`"
        raise ModelError(msg)

    base = TrainConfig()
    config = RunConfig(
        model=PRESETS[args.preset],
        train=replace(base, max_steps=args.steps) if args.steps else base,
        name=args.name,
    )

    def dataset(split: str) -> PackedDataset:
        return PackedDataset(
            corpus.load_split(split),
            config.train,
            config.model.vocab_size,
            mask_token_id=MASK_TOKEN_ID,
            n_special_tokens=N_SPECIAL_TOKENS,
        )

    validation = dataset("validation")
    batch = next(iter_batches(validation, PREVIEW_SENTENCES, seed=config.train.seed, epoch=0))
    trainer = Trainer(
        config,
        dataset("train"),
        validation,
        args.out / args.name,
        previewer=Previewer(
            batch,
            load(args.tokenizer),
            stop_ids=frozenset({SEP_ID, PAD_ID}),
            skip_ids=frozenset({CLS_ID}),
        ),
    )
    state = trainer.train()
    print(
        f"{state.step} steps / {state.tokens_seen:,} tokens / "
        f"best validation {state.best_validation_loss:.4f}"
    )
    return 0


def command_fill_mask(args: argparse.Namespace) -> int:
    from agbalu.model.infer import fill_mask, load_encoder
    from agbalu.tokenizer.evaluate import load

    model = load_encoder(args.run, args.preset, name=args.checkpoint)
    predictions = fill_mask(args.text, model, load(args.tokenizer), top_k=args.top_k)
    payload = {
        "text": args.text,
        "checkpoint": str(args.run / args.checkpoint),
        "masks": [
            {
                "index": prediction.index,
                "candidates": [
                    {"piece": c.piece, "probability": round(c.probability, 6)}
                    for c in prediction.candidates
                ],
            }
            for prediction in predictions
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_neighbours(args: argparse.Namespace) -> int:
    from agbalu.model.embeddings import neighbours, piece_for
    from agbalu.model.infer import load_encoder
    from agbalu.tokenizer.evaluate import load

    tokenizer = load(args.tokenizer)
    model = load_encoder(args.run, args.preset, name=args.checkpoint)
    table = model.embedding.word_embedding.weight.detach()
    row = piece_for(args.word, tokenizer)
    found = neighbours(table[row], table, tokenizer, top_k=args.top_k, exclude=[row])
    payload = {
        "word": args.word,
        "piece": tokenizer.id_to_piece(row),
        "piece_id": row,
        "neighbours": [{"piece": n.piece, "similarity": round(n.similarity, 4)} for n in found],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_analogy(args: argparse.Namespace) -> int:
    from agbalu.model.embeddings import analogy
    from agbalu.model.infer import load_encoder
    from agbalu.tokenizer.evaluate import load

    tokenizer = load(args.tokenizer)
    model = load_encoder(args.run, args.preset, name=args.checkpoint)
    table = model.embedding.word_embedding.weight.detach()
    words = (args.a, args.b, args.c)
    found, ids = analogy(words, table, tokenizer, top_k=args.top_k)
    payload = {
        "expression": f"{args.a} - {args.b} + {args.c}",
        "pieces": [tokenizer.id_to_piece(i) for i in ids],
        "ranked": [{"piece": n.piece, "similarity": round(n.similarity, 4)} for n in found],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agbalu-model", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="tokenise the corpus into train/validation streams")
    data.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    data.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    data.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR)
    data.set_defaults(func=command_data)

    train = sub.add_parser("train", help="train locally (CPU/MPS/CUDA)")
    train.add_argument("--preset", choices=sorted(PRESETS), default="kab")
    train.add_argument("--steps", type=int, default=0)
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR)
    train.add_argument("--name", default=DEFAULT_RUN_NAME)
    train.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    train.set_defaults(func=command_train)

    mask = sub.add_parser("fill-mask", help="predict a [MASK] with a trained checkpoint")
    mask.add_argument("text", help="text containing [MASK], written exactly so")
    mask.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    mask.add_argument("--checkpoint", default=BEST, help=f"{BEST} or {LATEST}")
    mask.add_argument("--preset", choices=sorted(PRESETS), default="kab")
    mask.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    mask.add_argument("--top-k", type=int, default=TOP_K)
    mask.set_defaults(func=command_fill_mask)

    for name, help_text in (
        ("neighbours", "nearest vocabulary rows to a word, by cosine similarity"),
        ("analogy", "a - b + c over the word-embedding table"),
    ):
        vectors = sub.add_parser(name, help=help_text)
        if name == "neighbours":
            vectors.add_argument("word")
        else:
            vectors.add_argument("a")
            vectors.add_argument("b")
            vectors.add_argument("c")
        vectors.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
        vectors.add_argument("--checkpoint", default=BEST, help=f"{BEST} or {LATEST}")
        vectors.add_argument("--preset", choices=sorted(PRESETS), default="kab")
        vectors.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
        vectors.add_argument("--top-k", type=int, default=TOP_NEIGHBOURS)
        vectors.set_defaults(func=command_neighbours if name == "neighbours" else command_analogy)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except ModelError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
