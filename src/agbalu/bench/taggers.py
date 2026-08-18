"""The systems scored in task 7.5: one neural tagger, one lexicon lookup, one baseline.

They differ in what they will not answer, so `agbalu.bench.pos` reports coverage
separately from accuracy: the neural tagger labels every position, the lexicon labels
only what it knows.

`torch` and `transformers` are imported here and nowhere else, so the rest of `agbalu`
installs and runs without them.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from agbalu.acquire.manifest import Manifest
from agbalu.lexicon.analyser import Analyser
from agbalu.treebank import Sentence

DEFAULT_RAW: Final = Path("data/raw")
DEFAULT_MODEL_SOURCE: Final = "hf.boffire.kabyle-pos-v2"
DEFAULT_MODEL_REPO: Final = "boffire/kabyle-pos-v2"
DEFAULT_LEXICON: Final = Path("data/processed/lexicon/agbalu-lexicon-v1.jsonl")

BATCH_SIZE: Final = 32
UNKNOWN_REVISION: Final = "unknown"


class TaggerError(Exception):
    """A tagger could not be built, or could not run."""


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def manifest_revision(source_id: str, raw: Path = DEFAULT_RAW) -> str:
    """The upstream commit the local copy was fetched at.

    Admissibility (`docs/benchmark.md` §4) wants a revision, not a model name. The
    acquisition manifest is the only record of which one is on disk.
    """
    entries = Manifest(raw).by_source(source_id)
    revisions = {entry.revision for entry in entries if entry.revision}
    if len(revisions) != 1:
        return UNKNOWN_REVISION
    return revisions.pop()


class NeuralTagger:
    """A HuggingFace token-classification model over pre-tokenised words.

    The label of a word is read off its **first** subword. Averaging the pieces
    would mix an XLM-R vocabulary that has no `ɛ` as an atomic unit — the same
    fragmentation measured at +17.8–21.3% tokens in `docs/prior_art.md` — into a
    single prediction, and the model was trained with first-subword labels.
    """

    def __init__(
        self,
        path: Path,
        name: str,
        revision: str,
        batch_size: int = BATCH_SIZE,
        device: torch.device | None = None,
    ) -> None:
        if not path.is_dir():
            msg = f"model directory not found: {path}"
            raise TaggerError(msg)
        self._name = name
        self._revision = revision
        self.batch_size = batch_size
        self.device = device if device is not None else _device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(path))
        self.model.to(self.device)
        self.model.eval()
        self.id2label: dict[int, str] = dict(self.model.config.id2label.items())
        self.max_length = int(self.tokenizer.model_max_length)

    @classmethod
    def load(
        cls,
        path: Path,
        repo: str = DEFAULT_MODEL_REPO,
        source_id: str = DEFAULT_MODEL_SOURCE,
        raw: Path = DEFAULT_RAW,
    ) -> NeuralTagger:
        return cls(path=path, name=repo, revision=manifest_revision(source_id, raw))

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    def _tag_batch(self, batch: list[list[str]]) -> list[list[str | None]]:
        encoded = self.tokenizer(batch, is_split_into_words=True, padding=True, return_tensors="pt")
        width = int(encoded["input_ids"].shape[1])
        if width > self.max_length:
            msg = f"a sentence encodes to {width} subwords, over the model's {self.max_length}"
            raise TaggerError(msg)

        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        best = logits.argmax(-1).to("cpu").tolist()

        labels: list[list[str | None]] = []
        for row, words in enumerate(batch):
            # A word whose every subword is dropped keeps `None`: an abstention is
            # the truth, and a default label here would be scored as an answer.
            row_labels: list[str | None] = [None] * len(words)
            for position, word in enumerate(encoded.word_ids(row)):
                if word is None or row_labels[word] is not None:
                    continue
                row_labels[word] = self.id2label[int(best[row][position])]
            labels.append(row_labels)
        return labels

    def tag(self, sentences: Sequence[Sequence[str]]) -> list[list[str | None]]:
        """Longest first, so a batch pads to its own longest sentence and not the
        corpus maximum. Input order is restored before returning."""
        order = sorted(range(len(sentences)), key=lambda index: -len(sentences[index]))
        labels: list[list[str | None]] = [[] for _ in sentences]
        for start in range(0, len(order), self.batch_size):
            indices = [
                index for index in order[start : start + self.batch_size] if sentences[index]
            ]
            if not indices:
                continue
            batch = [list(sentences[index]) for index in indices]
            for index, row in zip(indices, self._tag_batch(batch), strict=True):
                labels[index] = row
        return labels


class LexiconTagger:
    """AƔBALU-Lexicon v1 through `Analyser.upos`.

    Answers only where the lexicon has an unambiguous reading. Its silence is the
    measurement `lexicon-coverage.json` reports, not a failure to run.
    """

    def __init__(self, analyser: Analyser, name: str, revision: str) -> None:
        self.analyser = analyser
        self._name = name
        self._revision = revision

    @classmethod
    def load(cls, path: Path = DEFAULT_LEXICON) -> LexiconTagger:
        if not path.is_file():
            msg = f"lexicon not found: {path}"
            raise TaggerError(msg)
        stats = path.with_suffix(".stats.json")
        revision = UNKNOWN_REVISION
        if stats.is_file():
            revision = str(json.loads(stats.read_text(encoding="utf-8"))["normaliser_version"])
        return cls(analyser=Analyser.from_lexicon(path), name=path.stem, revision=revision)

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    def tag(self, sentences: Sequence[Sequence[str]]) -> list[list[str | None]]:
        return [[self.analyser.upos(form) for form in sentence] for sentence in sentences]


class MostFrequentTagger:
    """Each form's most frequent gold tag; the treebank's majority tag otherwise.

    The floor any real system has to clear. Without it, an accuracy figure for a
    language with this tag distribution — NOUN alone is 18.6% — reads as better
    than it is.
    """

    def __init__(self, table: dict[str, str], fallback: str, revision: str) -> None:
        self.table = table
        self.fallback = fallback
        self._revision = revision

    @classmethod
    def fit(cls, sentences: Iterable[Sentence]) -> MostFrequentTagger:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        overall: Counter[str] = Counter()
        splits: set[str] = set()
        for sentence in sentences:
            splits.add(sentence.split)
            for word in sentence.words:
                counts[word.form][word.upos] += 1
                counts[word.form.casefold()][word.upos] += 1
                overall[word.upos] += 1
        if not overall:
            msg = "nothing to fit the baseline on"
            raise TaggerError(msg)
        # Ties break on the tag name so a refit over the same data gives the same
        # tagger; `Counter.most_common` alone leaves them to insertion order.
        table = {
            form: min(tags.most_common(), key=lambda item: (-item[1], item[0]))[0]
            for form, tags in counts.items()
        }
        fallback = min(overall.most_common(), key=lambda item: (-item[1], item[0]))[0]
        return cls(table=table, fallback=fallback, revision=f"fit:{'+'.join(sorted(splits))}")

    @property
    def name(self) -> str:
        return "most-frequent-tag"

    @property
    def revision(self) -> str:
        return self._revision

    def _lookup(self, form: str) -> str:
        if form in self.table:
            return self.table[form]
        return self.table.get(form.casefold(), self.fallback)

    def tag(self, sentences: Sequence[Sequence[str]]) -> list[list[str | None]]:
        return [[self._lookup(form) for form in sentence] for sentence in sentences]
