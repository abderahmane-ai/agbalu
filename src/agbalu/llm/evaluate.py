"""The Phase 11 evaluation suite (task 11.15).

One model load answers the three halves of the exit criterion `docs/ROADMAP.md` fixes before
training: held-out Kabyle bits per character, eng→kab 5-shot chrF++ on corrected FLORES+
devtest, and the English and French retention suite. SIB-200 topic accuracy is reported
beside them and decides nothing.

Bits per character runs through `agbalu.llm.forward` and `agbalu.llm.loss`, the trainer's
own path, over whole documents packed into variable-length rows. Each document is scored as
if alone (`agbalu.llm.packing`), so the figure is the training loss on text never trained on.
Translation prompts use the tag format the aligned corpus trains on, `kab_Latn: …` then
`eng_Latn: …`, so the CPT model is asked in the shape it learned. Every Kabyle side, shot or
test sentence, comes from the corrected FLORES+ file.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from agbalu.bench.flores import Split, read_split
from agbalu.bench.mt import LANGUAGE_CODE, as_direction, references_for, score
from agbalu.llm.baseline import Scored
from agbalu.llm.corpus import LANGUAGE_TAG
from agbalu.llm.decontaminate import FLORES_CORRECTED, FLORES_ROOT, HELDOUT_DIR, SIB_ROOT
from agbalu.llm.forward import Backbone, LanguageModel, hidden_states, to_device
from agbalu.llm.holdout import read as read_heldout
from agbalu.llm.loss import logit_softcap, sequence_loss
from agbalu.llm.packing import PackedRow, concatenate, pack_documents, plan_rows
from agbalu.llm.prompting import (
    SHOT_SEED,
    SHOTS,
    Shot,
    length_ordered_batches,
    new_tokens,
    require_tensor,
    shots,
)
from agbalu.llm.recipe import MICRO_BLOCKS, SEQUENCE_LENGTH

if TYPE_CHECKING:
    import torch

Part = Literal["heldout", "mt", "sib", "retention"]
PARTS: Final[tuple[Part, ...]] = ("heldout", "mt", "sib", "retention")

CAPACITY: Final = SEQUENCE_LENGTH * MICRO_BLOCKS
"""Scoring rows are the training row's size, so memory behaves as in a training step."""

HELDOUT_LANGUAGES: Final = ("kab", "eng", "fra")
MT_DIRECTIONS: Final = ("eng-kab", "kab-eng")
"""`eng-kab` is the exit criterion; `kab-eng` shows whether understanding moved with it."""

EXIT_DIRECTION: Final = "eng-kab"
MAX_RETENTION_DROP: Final = 2.0
"""Points of the retention mean the CPT model may lose, fixed before training."""

GENERATION_BATCH: Final = 16
SIB_SEED: Final = 20261014
SIB_SPLITS: Final = ("train", "test")
SAMPLES: Final = 5
"""Translations recorded verbatim per direction, because a score does not show the Kabyle."""

RETENTION_SEED: Final = 20261014
RETENTION_BATCH: Final = 8


class EvaluationError(Exception):
    """A part cannot be run, or a report does not carry what a comparison needs."""


@dataclass(frozen=True, slots=True)
class RetentionTask:
    name: str
    shots: int
    metric: str
    language: str


RETENTION: Final = (
    RetentionTask("mmlu", 5, "acc,none", "eng"),
    RetentionTask("arc_challenge", 0, "acc_norm,none", "eng"),
    RetentionTask("hellaswag", 0, "acc_norm,none", "eng"),
    RetentionTask("gsm8k", 5, "exact_match,strict-match", "eng"),
    RetentionTask("belebele_eng_Latn", 0, "acc,none", "eng"),
    RetentionTask("belebele_fra_Latn", 0, "acc,none", "fra"),
    RetentionTask("global_mmlu_fr", 5, "acc,none", "fra"),
)
"""Fixed before training. gsm8k stays because maths is what little replay lost first."""


class Tokenizer(Protocol):
    """Ids without special tokens, text back from ids, the separator, and the library object
    generation and lm-eval are handed."""

    @property
    def eos(self) -> int: ...

    @property
    def backend(self) -> object: ...

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class HubTokenizer:
    """A `transformers` tokenizer that never turns document text into a special token."""

    def __init__(self, repo: str | Path) -> None:
        from transformers import AutoTokenizer

        self.backend = AutoTokenizer.from_pretrained(str(repo))
        eos = self.backend.eos_token_id
        if not isinstance(eos, int):
            message = f"{repo}: no eos token, so documents cannot be separated"
            raise EvaluationError(message)
        self.eos = eos

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encoded = self.backend(list(texts), add_special_tokens=False, split_special_tokens=True)
        return [[int(token) for token in ids] for ids in encoded["input_ids"]]

    def decode(self, ids: Sequence[int]) -> str:
        return str(self.backend.decode(list(ids), skip_special_tokens=True))


@dataclass(frozen=True, slots=True)
class Scorer:
    """What scoring needs from a model: the stack, the classifier and the separator."""

    backbone: Backbone
    classifier: torch.Tensor
    softcap: float | None
    eos: int
    device: torch.device

    @classmethod
    def of(cls, model: LanguageModel, eos: int, device: torch.device) -> Scorer:
        return cls(model.model, model.lm_head.weight, logit_softcap(model.config), eos, device)


def summed_nll(scorer: Scorer, rows: Sequence[PackedRow]) -> tuple[float, int]:
    """Total nats and scored targets over packed rows, one host synchronisation at the end."""
    import torch

    total = torch.zeros((), dtype=torch.float64, device=scorer.device)
    targets = 0
    with torch.inference_mode():
        for packed in rows:
            row = to_device(packed, scorer.device)
            hidden = hidden_states(scorer.backbone, row)
            loss = sequence_loss(
                hidden, scorer.classifier, row.labels, softcap=scorer.softcap, reduction="sum"
            )
            total += loss.double()
            targets += packed.targets
    return float(total.item()), targets


def held_out(
    scorer: Scorer,
    tokenizer: Tokenizer,
    texts: Sequence[str],
    *,
    language: str,
    capacity: int = CAPACITY,
) -> Scored:
    """One evaluation set's likelihood, each document scored whole after an `eos`."""
    encoded = tokenizer.encode_batch(texts)
    if len(encoded) != len(texts):
        message = f"heldout-{language}: {len(encoded)} encodings for {len(texts)} documents"
        raise EvaluationError(message)
    rows = pack_documents(encoded, scorer.eos, capacity)
    nll, tokens = summed_nll(scorer, rows)
    return Scored(
        name=f"heldout-{language}",
        language=language,
        documents=len(texts),
        tokens=tokens,
        characters=sum(len(text) for text in texts),
        nll=nll,
        truncated=sum(1 for ids in encoded if len(ids) + 1 > capacity),
    )


@dataclass(frozen=True, slots=True)
class Choice:
    context: tuple[int, ...]
    continuation: tuple[int, ...]


def continuation_logprobs(
    scorer: Scorer, choices: Sequence[Choice], capacity: int = CAPACITY
) -> list[float]:
    """Summed log-probability of each continuation given its context, in input order."""
    import torch

    documents = [[scorer.eos, *choice.context, *choice.continuation] for choice in choices]
    for index, (choice, document) in enumerate(zip(choices, documents, strict=True)):
        if not choice.continuation or len(document) > capacity:
            message = f"choice {index}: {len(document)} tokens, continuation {choice.continuation}"
            raise EvaluationError(message)
    scores = [0.0] * len(choices)
    for group in plan_rows([len(document) for document in documents], capacity):
        row = to_device(concatenate([documents[i] for i in group]), scorer.device)
        predictors: list[int] = []
        targets: list[int] = []
        owners: list[int] = []
        start = 0
        for slot, index in enumerate(group):
            first = start + 1 + len(choices[index].context)
            for position in range(first, start + len(documents[index])):
                predictors.append(position - 1)
                targets.append(documents[index][position - start])
                owners.append(slot)
            start += len(documents[index])
        with torch.inference_mode():
            hidden = hidden_states(scorer.backbone, row)[0]
            picked = hidden[torch.tensor(predictors, device=scorer.device)]
            logits = (picked @ scorer.classifier.T).float()
            if scorer.softcap is not None:
                logits = scorer.softcap * torch.tanh(logits / scorer.softcap)
            logprobs = torch.log_softmax(logits, dim=-1)
            chosen = logprobs.gather(1, torch.tensor(targets, device=scorer.device)[:, None])
            sums = torch.zeros(len(group), dtype=torch.float64, device=scorer.device)
            sums.index_add_(0, torch.tensor(owners, device=scorer.device), chosen[:, 0].double())
        for slot, value in enumerate(sums.tolist()):
            scores[group[slot]] = float(value)
    return scores


def kabyle_corrected(split: Split, path: Path = FLORES_CORRECTED) -> list[str]:
    """The corrected FLORES+ Kabyle side of one split, in id order."""
    if not path.is_file():
        message = f"corrected FLORES+ not found at {path}; run `make bench TASK=audit`"
        raise EvaluationError(message)
    rows: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                message = f"{path}:{number} carries no text"
                raise EvaluationError(message)
            if row.get("split") == split:
                rows.append((int(row["id"]), str(row["text"])))
    if not rows:
        message = f"{path} holds no {split} rows"
        raise EvaluationError(message)
    return [text for _, text in sorted(rows)]


def flores_side(code: str, split: Split, root: Path = FLORES_ROOT) -> list[str]:
    if code == "kab":
        return kabyle_corrected(split)
    return references_for(read_split(root, split, LANGUAGE_CODE[code]))


def translation_prompt(examples: Sequence[Shot], source: str, target: str, text: str) -> str:
    """Shots and the test sentence in the aligned corpus's tag format, one blank line apart."""
    blocks = [f"{source}: {shot.source}\n{target}: {shot.target}" for shot in examples]
    blocks.append(f"{source}: {text}\n{target}:")
    return "\n\n".join(blocks)


def first_line(text: str) -> str:
    """The continuation up to the first newline; the next line would be a new pair."""
    return text.strip().split("\n", 1)[0].strip()


class Generator(Protocol):
    """The generation slice of `forward.LanguageModel`."""

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int,
        stop_strings: list[str],
        tokenizer: object,
    ) -> object: ...


def greedy(
    model: Generator,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    budgets: Sequence[int],
    *,
    device: torch.device,
    batch_size: int = GENERATION_BATCH,
) -> list[str]:
    """Greedy continuations in input order, `eos`-prefixed like a training document.

    Left padding is exact here: padded positions enter the Gated DeltaNet layers as zeros,
    so their state stays zero, and rotary attention depends only on relative position.
    `smoke` checks a padded batch against each prompt alone.
    """
    import torch

    if len(prompts) != len(budgets):
        message = f"{len(prompts)} prompts against {len(budgets)} budgets"
        raise EvaluationError(message)
    encoded = [[tokenizer.eos, *ids] for ids in tokenizer.encode_batch(prompts)]
    outputs = [""] * len(prompts)
    for batch in length_ordered_batches([len(ids) for ids in encoded], batch_size):
        width = max(len(encoded[i]) for i in batch)
        input_ids = torch.full((len(batch), width), tokenizer.eos, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for slot, index in enumerate(batch):
            ids = encoded[index]
            input_ids[slot, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention[slot, width - len(ids) :] = 1
        with torch.inference_mode():
            generated = require_tensor(
                model.generate(
                    input_ids=input_ids.to(device),
                    attention_mask=attention.to(device),
                    max_new_tokens=max(budgets[i] for i in batch),
                    do_sample=False,
                    pad_token_id=tokenizer.eos,
                    stop_strings=["\n"],
                    tokenizer=tokenizer.backend,
                ),
                "generate",
            )
        for slot, index in enumerate(batch):
            outputs[index] = tokenizer.decode(generated[slot, width:].tolist())
    return outputs


def translate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    direction: str,
    *,
    device: torch.device,
    limit: int | None = None,
) -> dict[str, object]:
    """One direction, 5-shot from FLORES+ dev, scored on devtest under both conditions."""
    checked = as_direction(direction)
    source, target = checked.split("-")
    examples = shots(flores_side(source, "dev"), flores_side(target, "dev"), SHOTS)
    sources = flores_side(source, "devtest")[:limit]
    references = flores_side(target, "devtest")[:limit]
    tags = LANGUAGE_TAG[source], LANGUAGE_TAG[target]
    prompts = [translation_prompt(examples, *tags, text) for text in sources]
    budgets = [new_tokens([len(ids)]) for ids in tokenizer.encode_batch(sources)]
    hypotheses = [
        first_line(text) for text in greedy(model, tokenizer, prompts, budgets, device=device)
    ]
    result = score(hypotheses, references, checked, "devtest")
    return {
        "direction": direction,
        "sentences": result.sentences,
        "limit": limit,
        "shots": SHOTS,
        "shot_seed": SHOT_SEED,
        "chrf++": result.raw.get("chrf++").score,
        "chrf++_normalised": result.normalised.get("chrf++").score,
        "bleu": result.raw.get("bleu").score,
        "spbleu": result.raw.get("spbleu").score,
        "signature": result.raw.get("chrf++").signature,
        "normaliser": result.normaliser_version,
        "samples": [
            {"source": s, "hypothesis": h, "reference": r}
            for s, h, r in list(zip(sources, hypotheses, references, strict=True))[:SAMPLES]
        ],
    }


@dataclass(frozen=True, slots=True)
class TopicSet:
    texts: tuple[str, ...]
    labels: tuple[int, ...]
    names: tuple[str, ...]


def read_topics(split: str, root: Path = SIB_ROOT) -> TopicSet:
    """A SIB-200 split, label names read from the file's own feature metadata."""
    import pyarrow.parquet as pq

    paths = sorted(root.glob(f"{split}-*.parquet"))
    if len(paths) != 1:
        message = f"expected one {split} parquet under {root}, found {len(paths)}"
        raise EvaluationError(message)
    metadata = pq.read_schema(paths[0]).metadata or {}
    features = json.loads(metadata.get(b"huggingface", b"{}")).get("info", {}).get("features", {})
    names = features.get("label", {}).get("names")
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        message = f"{paths[0]}: no label names in the parquet metadata"
        raise EvaluationError(message)
    table = pq.read_table(paths[0], columns=["text", "label"])
    texts = tuple(str(text) for text in table["text"].to_pylist())
    labels = tuple(int(label) for label in table["label"].to_pylist())
    if any(not 0 <= label < len(names) for label in labels):
        message = f"{paths[0]}: a label falls outside the {len(names)} names"
        raise EvaluationError(message)
    return TopicSet(texts, labels, tuple(names))


def topic_shots(topics: TopicSet, seed: int = SIB_SEED) -> list[tuple[str, str]]:
    """One demonstration per label, drawn and ordered by `seed`."""
    import random

    draw = random.Random(seed)
    chosen: list[tuple[str, str]] = []
    for label, name in enumerate(topics.names):
        members = [
            text for text, own in zip(topics.texts, topics.labels, strict=True) if own == label
        ]
        if not members:
            message = f"no demonstration for label {name!r}"
            raise EvaluationError(message)
        chosen.append((draw.choice(members), name))
    draw.shuffle(chosen)
    return chosen


def topic_prompt(examples: Sequence[tuple[str, str]], text: str) -> str:
    blocks = [f"Text: {shot}\nTopic: {name}" for shot, name in examples]
    blocks.append(f"Text: {text}\nTopic:")
    return "\n\n".join(blocks)


def topic_accuracy(
    scorer: Scorer, tokenizer: Tokenizer, *, limit: int | None = None
) -> dict[str, object]:
    """SIB-200 Kabyle test accuracy by the label's log-likelihood, summed and per byte."""
    train = read_topics("train")
    test = read_topics("test")
    if train.names != test.names:
        message = f"train labels {train.names} differ from test labels {test.names}"
        raise EvaluationError(message)
    examples = topic_shots(train)
    texts = test.texts[:limit]
    contexts = tokenizer.encode_batch([topic_prompt(examples, text) for text in texts])
    endings = tokenizer.encode_batch([f" {name}" for name in test.names])
    choices = [Choice(tuple(context), tuple(ending)) for context in contexts for ending in endings]
    scores = continuation_logprobs(scorer, choices)
    width = len(test.names)
    byte_lengths = [len(f" {name}".encode()) for name in test.names]
    correct = correct_norm = 0
    for row, label in enumerate(test.labels[: len(texts)]):
        options = scores[row * width : (row + 1) * width]
        normalised = [value / size for value, size in zip(options, byte_lengths, strict=True)]
        correct += int(max(range(width), key=options.__getitem__) == label)
        correct_norm += int(max(range(width), key=normalised.__getitem__) == label)
    return {
        "examples": len(texts),
        "limit": limit,
        "shots": len(examples),
        "seed": SIB_SEED,
        "accuracy": round(correct / len(texts), 4),
        "accuracy_per_byte": round(correct_norm / len(texts), 4),
        "chance": round(1 / width, 4),
    }


def read_metric(output: object, task: RetentionTask) -> float:
    """The task's metric out of an lm-eval result, or a named error listing what is there."""
    results = output.get("results") if isinstance(output, dict) else None
    row = results.get(task.name) if isinstance(results, dict) else None
    if not isinstance(row, dict):
        message = f"lm-eval returned no result row for {task.name!r}"
        raise EvaluationError(message)
    value = row.get(task.metric)
    if not isinstance(value, int | float) or not math.isfinite(value):
        message = f"{task.name}: no finite {task.metric!r}; the row has {sorted(row)}"
        raise EvaluationError(message)
    return float(value)


def retention(
    model: LanguageModel,
    tokenizer: Tokenizer,
    tasks: Sequence[RetentionTask] = RETENTION,
    *,
    limit: int | None = None,
    batch_size: int = RETENTION_BATCH,
) -> dict[str, object]:
    """The English and French suite through lm-eval, each task at its own shot count."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    harness = HFLM(pretrained=model, tokenizer=tokenizer.backend, batch_size=batch_size)
    rows: dict[str, dict[str, object]] = {}
    for task in tasks:
        output = simple_evaluate(
            model=harness,
            tasks=[task.name],
            num_fewshot=task.shots,
            limit=limit,
            bootstrap_iters=0,
            random_seed=RETENTION_SEED,
            numpy_random_seed=RETENTION_SEED,
            torch_random_seed=RETENTION_SEED,
            fewshot_random_seed=RETENTION_SEED,
        )
        rows[task.name] = {
            "score": read_metric(output, task),
            "metric": task.metric,
            "shots": task.shots,
            "language": task.language,
        }
    return {"tasks": rows, "limit": limit, "mean_points": retention_mean(rows)}


def retention_mean(rows: Mapping[str, Mapping[str, object]]) -> float:
    """The unweighted mean over tasks, in points."""
    if not rows:
        message = "no retention task was scored"
        raise EvaluationError(message)
    values: list[float] = []
    for name, row in rows.items():
        value = row.get("score")
        if not isinstance(value, float):
            message = f"{name}: score {value!r} is not a number"
            raise EvaluationError(message)
        values.append(value)
    return round(100 * math.fsum(values) / len(values), 4)


def run(
    model: LanguageModel,
    tokenizer: Tokenizer,
    parts: Sequence[Part],
    *,
    device: torch.device,
    limit: int | None = None,
) -> dict[str, object]:
    """Every requested part on one loaded model."""
    model.eval()
    scorer = Scorer.of(model, tokenizer.eos, device)
    report: dict[str, object] = {"parts": list(parts), "limit": limit}
    if "heldout" in parts:
        report["heldout"] = {
            language: held_out(
                scorer,
                tokenizer,
                read_heldout(HELDOUT_DIR / f"heldout-{language}.jsonl")[:limit],
                language=language,
            ).as_dict()
            for language in HELDOUT_LANGUAGES
        }
    if "mt" in parts:
        report["mt"] = {
            direction: translate(model, tokenizer, direction, device=device, limit=limit)
            for direction in MT_DIRECTIONS
        }
    if "sib" in parts:
        report["sib"] = topic_accuracy(scorer, tokenizer, limit=limit)
    if "retention" in parts:
        report["retention"] = retention(model, tokenizer, limit=limit)
    return report


def number(report: Mapping[str, object], *path: str) -> float:
    """A float at `path` in a report, or a named error saying where it is missing."""
    node: object = report
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            message = f"the report has no {'.'.join(path)}"
            raise EvaluationError(message)
        node = node[key]
    if not isinstance(node, int | float) or isinstance(node, bool):
        message = f"{'.'.join(path)} is {node!r}, not a number"
        raise EvaluationError(message)
    return float(node)


def exit_criterion(
    candidate: Mapping[str, object], base: Mapping[str, object]
) -> dict[str, object]:
    """Whether the CPT model is adopted over the base, by the three rules fixed beforehand."""
    for name, report in (("candidate", candidate), ("base", base)):
        if report.get("limit") is not None:
            message = f"the {name} report was limited to {report['limit']}; it decides nothing"
            raise EvaluationError(message)
    rows: list[set[str]] = []
    for report in (candidate, base):
        retained = report.get("retention")
        names = retained.get("tasks") if isinstance(retained, Mapping) else None
        rows.append({str(name) for name in names} if isinstance(names, Mapping) else set())
    if rows[0] != rows[1] or not rows[0]:
        message = f"retention tasks differ: {sorted(rows[0])} against {sorted(rows[1])}"
        raise EvaluationError(message)
    bpc = [number(r, "heldout", "kab", "bits_per_character") for r in (candidate, base)]
    chrf = [number(r, "mt", EXIT_DIRECTION, "chrf++") for r in (candidate, base)]
    mean = [number(r, "retention", "mean_points") for r in (candidate, base)]
    rules = {
        "kabyle_bpc": {"candidate": bpc[0], "base": bpc[1], "passed": bpc[0] < bpc[1]},
        "eng_kab_chrf": {"candidate": chrf[0], "base": chrf[1], "passed": chrf[0] > chrf[1]},
        "retention": {
            "candidate": mean[0],
            "base": mean[1],
            "drop": round(mean[1] - mean[0], 4),
            "allowed": MAX_RETENTION_DROP,
            "passed": mean[1] - mean[0] <= MAX_RETENTION_DROP,
        },
    }
    return {"rules": rules, "adopted": all(bool(rule["passed"]) for rule in rules.values())}
