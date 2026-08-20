"""Stage the publishable dataset repositories.

Five repos. `KabBench` carries the evaluation data every Kabyle number in this project is
measured on, `KabLex` the lexical layer, `KabSentiment`, `KabInflect` and `KabTifinagh` a
task each. All are staged the same way and for the same reason the model exports are: what
is published must be a filtered, provenance-carrying cut of a built artifact, never the
artifact itself.

Staging is one path because the checks are the point, and three of these repos were first
published around it. Two shipped a *pipeline* tag in `task_ids`, which is the defect
`check_tasks` was written for after `KabLex` shipped one in `task_categories` — the Hub
accepts it at upload and warns only on the rendered page.

The filtering is the point rather than a formality. The lexicon holds three redistribution
classes and only one of them can be published without imposing an obligation on whoever
downloads it, so the permissive cut is taken **by code** — which is the guarantee CLAUDE.md
§2.1 rule 2 exists to make, and which only holds because `odbl` is classed share-alike.

Every check runs before anything is written: a refused export must not leave a directory
that looks published.

    python3 -m tools.export_datasets --out artifacts/release
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

BENCH: Final = Path("data/processed/bench")
TASKS: Final = Path("data/tasks")
LEXICON: Final = Path("data/processed/lexicon")
PUNCT: Final = Path("data/processed/punctuation")
PRONUNCIATIONS: Final = LEXICON / "agbalu-pronunciations-v1.jsonl"
CARDS: Final = Path("docs/cards")

TASK_CATEGORIES: Final = frozenset(
    {
        "any-to-any",
        "audio-classification",
        "audio-text-to-text",
        "audio-to-audio",
        "automatic-speech-recognition",
        "depth-estimation",
        "document-question-answering",
        "feature-extraction",
        "fill-mask",
        "graph-ml",
        "image-classification",
        "image-feature-extraction",
        "image-segmentation",
        "image-text-to-image",
        "image-text-to-text",
        "image-text-to-video",
        "image-to-3d",
        "image-to-image",
        "image-to-text",
        "image-to-video",
        "keypoint-detection",
        "mask-generation",
        "multiple-choice",
        "object-detection",
        "other",
        "question-answering",
        "reinforcement-learning",
        "robotics",
        "sentence-similarity",
        "summarization",
        "table-question-answering",
        "table-to-text",
        "tabular-classification",
        "tabular-regression",
        "tabular-to-text",
        "text-classification",
        "text-generation",
        "text-ranking",
        "text-retrieval",
        "text-to-3d",
        "text-to-audio",
        "text-to-image",
        "text-to-speech",
        "text-to-video",
        "time-series-forecasting",
        "token-classification",
        "translation",
        "unconditional-image-generation",
        "video-classification",
        "video-text-to-text",
        "video-to-video",
        "visual-document-retrieval",
        "visual-question-answering",
        "voice-activity-detection",
        "zero-shot-classification",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
    }
)
"""The Hub's closed vocabulary for `task_categories`.

A name outside it is not rejected at upload — the repo publishes, and the warning appears on
the *rendered card*, where only someone looking at the page sees it. `text2text-generation`
reached `KabLex` that way and sat on a live public dataset. Checked here instead, because a
staging-time failure is one the operator sees before the push."""

TASK_IDS: Final = frozenset(
    {
        "language-identification",
        "lemmatization",
        "part-of-speech",
        "sentiment-analysis",
    }
)
"""The `task_ids` these repos use. The Hub validates this field the same way it validates
`task_categories` — on render, silently — and its allowed values are not specified in the
dataset-card schema, so this is an allow-list of what has been seen to render clean rather
than a copy of an upstream table. Adding a fifth repo means adding its id here on purpose.
`text2text-generation` is what this exists to stop: a *pipeline* tag, valid nowhere in a
dataset card, which reached two cards after the same mistake had already been caught once
in `task_categories`."""

Row = Mapping[str, object]


class ExportError(Exception):
    """A dataset cannot be staged as described."""


@dataclass(frozen=True, slots=True)
class Config:
    """One config of one repo: its rows, already split, and the fields it publishes."""

    name: str
    splits: Mapping[str, Sequence[Row]]
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.splits:
            message = f"{self.name}: a config with no split publishes nothing"
            raise ExportError(message)
        for split, rows in self.splits.items():
            if not rows:
                message = f"{self.name}/{split}: no rows"
                raise ExportError(message)
            # A field absent from *some* rows is ordinary — `glosses` and `lemma` are both
            # optional. A field absent from *every* row is a misspelling that would publish
            # a column of nulls, so it is checked over the whole split, not over the first.
            empty = [f for f in self.fields if all(row.get(f) is None for row in rows)]
            if empty:
                message = f"{self.name}/{split}: no row carries {empty}"
                raise ExportError(message)

    @property
    def rows(self) -> int:
        return sum(len(r) for r in self.splits.values())


@dataclass(frozen=True, slots=True)
class Repo:
    """A staged dataset repository.

    `layout` is `"flat"` when one config writes `train.parquet` at the root and `"nested"`
    when each config gets a directory. It is not cosmetic: the card's `configs:` block names
    the exact paths, and `stage` refuses a card whose declared paths it did not write.
    """

    name: str
    card: Path
    configs: tuple[Config, ...]
    suffix: str = "jsonl"
    layout: str = "nested"

    def __post_init__(self) -> None:
        if not self.configs:
            message = f"{self.name}: nothing to publish"
            raise ExportError(message)
        if not self.card.is_file():
            message = f"{self.name}: card not found at {self.card}"
            raise ExportError(message)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        message = f"not found: {path}. Run its `make` target first."
        raise ExportError(message)
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"{path}:{number} is not JSON"
            raise ExportError(message) from exc
        if not isinstance(row, dict):
            message = f"{path}:{number} is not a JSON object"
            raise ExportError(message)
        rows.append(row)
    if not rows:
        message = f"{path} is empty"
        raise ExportError(message)
    return rows


def project(rows: Sequence[Row], fields: Sequence[str]) -> list[dict[str, object]]:
    """Rows reduced to the published fields, in the declared order.

    Every row carries every declared key, `None` where the source has nothing, so the
    published schema is rectangular and a reader does not have to guess whether an absent
    key means "unknown" or "not in this row".
    """
    return [{field: row.get(field) for field in fields} for row in rows]


def partition(rows: Sequence[Row], key: str) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            message = f"a row carries no usable {key!r}"
            raise ExportError(message)
        grouped.setdefault(value, []).append(row)
    return grouped


def bench() -> Repo:
    """The evaluation suite: the corrected MT reference, and the sibling LID set.

    The POS test set is deliberately absent. `pos-kab_adpt-test.json` holds *scores* — 16
    runs over four systems — and the sentences behind them are `UD_Kabyle-ADPT` itself,
    which is someone else's treebank and gains nothing from being republished verbatim. The
    contribution there is the protocol and the numbers, so they live in the card.
    """
    corrected = read_jsonl(BENCH / "flores-plus-kab_Latn-corrected.jsonl")
    by_split = partition(corrected, "split")
    mt_fields = ("id", "split", "text", "corrected")

    lid = read_jsonl(BENCH / "lid-eval.jsonl")
    # `source` repeats `language` in every row; publishing both would imply they can differ.
    lid_fields = ("text", "language")

    return Repo(
        name="KabBench",
        card=CARDS / "kabbench.md",
        configs=(
            Config(
                name="mt",
                splits={s: project(r, mt_fields) for s, r in sorted(by_split.items())},
                fields=mt_fields,
            ),
            Config(
                name="lid",
                splits={"test": project(lid, lid_fields)},
                fields=lid_fields,
            ),
        ),
    )


def permissive(rows: Sequence[Row]) -> list[Row]:
    """The rows that can be redistributed without imposing an obligation.

    `redistribution` is derived from the licence by `registry.models.redistribution_class`,
    so this filter inherits that table rather than restating it — including that `odbl` is
    share-alike, which is what keeps the ODbL toponyms out of a CC-BY-4.0 release.
    """
    return [row for row in rows if row.get("redistribution") == "permissive"]


def lexicon() -> Repo:
    """The lexical layer, cut to what a permissive licence can cover."""
    entries = read_jsonl(LEXICON / "agbalu-lexicon-v1.jsonl")
    kept = permissive(entries)
    if not kept:
        message = "no permissive lexicon entries; refusing to stage an empty release"
        raise ExportError(message)
    if len(kept) == len(entries):
        message = (
            "every lexicon entry is permissive, which contradicts the registry: "
            "the ODbL and unclear sources should have been filtered out"
        )
        raise ExportError(message)

    entry_fields = ("form", "lemma", "upos", "feats", "glosses", "source", "licence")
    sounds = read_jsonl(LEXICON / "agbalu-pronunciations-v1.jsonl")
    sound_fields = ("word", "ipa", "variants", "repaired")

    return Repo(
        name="KabLex",
        card=CARDS / "kablex.md",
        configs=(
            Config(
                name="lexicon",
                splits={"train": project(kept, entry_fields)},
                fields=entry_fields,
            ),
            Config(
                name="pronunciations",
                splits={"train": project(sounds, sound_fields)},
                fields=sound_fields,
            ),
        ),
    )


def read_parquet(path: Path, fields: Sequence[str]) -> list[dict[str, object]]:
    """One built split, reduced to the published fields."""
    import pyarrow.parquet as pq

    if not path.is_file():
        message = f"not found: {path}. Run its `make` target first."
        raise ExportError(message)
    rows: list[dict[str, object]] = pq.read_table(path, columns=list(fields)).to_pylist()
    return rows


def _task_repo(name: str, card: str, built: Path, configs: Mapping[str, tuple[str, ...]]) -> Repo:
    """A repo whose configs are subdirectories of one built task directory.

    The built directory is passed rather than derived from `name`: `KabInflect` is built
    under `inflection`, so a rule that lowercases the name silently resolves to a path that
    does not exist and the failure names the config rather than the path.
    """
    return Repo(
        name=name,
        card=CARDS / card,
        configs=tuple(
            Config(
                name=config,
                splits={
                    split: read_parquet(built / config / f"{split}.parquet", fields)
                    for split in ("train", "dev", "test")
                    if (built / config / f"{split}.parquet").is_file()
                },
                fields=fields,
            )
            for config, fields in configs.items()
        ),
        suffix="parquet",
    )


def sentiment() -> Repo:
    """Three balanced classes, projected from the English side of a Tatoeba pair.

    Read from JSONL rather than parquet: this is the only task corpus the builder writes as
    JSONL, because the confidence score and the projection's source belong beside every row
    and a reader should be able to grep them.
    """
    fields = ("id", "text_kab", "text_en", "label", "label_name", "confidence_score", "source")
    splits = {
        split: project(read_jsonl(TASKS / "sentiment" / f"{split}.jsonl"), fields)
        for split in ("train", "dev", "test")
    }
    return Repo(
        name="KabSentiment",
        card=CARDS / "kabsentiment.md",
        configs=(Config(name="default", splits=splits, fields=fields),),
        suffix="parquet",
        layout="flat",
    )


def inflection() -> Repo:
    """Verb morphology, split by lemma so no test paradigm was seen in any cell."""
    return _task_repo(
        "KabInflect",
        "kabinflect.md",
        TASKS / "inflection",
        {
            "inflection": ("id", "lemma", "feats", "tense_raw", "person_raw", "form"),
            "analysis": ("id", "form", "lemma", "feats", "tense_raw", "person_raw"),
            "paradigms": (
                "id",
                "name",
                "translation",
                "is_irregular",
                "is_derived",
                "pattern_verb",
                "imperative",
                "aorist",
                "preterite",
                "negative_preterite",
                "aorist_participle",
                "preterite_participle",
                "negative_preterite_participle",
                "intensive_forms",
            ),
        },
    )


def transliteration() -> Repo:
    """Kabyle in both scripts, and the two trilingual alignments beside it."""
    return _task_repo(
        "KabTifinagh",
        "kabtifinagh.md",
        TASKS / "tifinagh",
        {
            "script_conversion": ("id", "text_latn", "text_tfng"),
            "trilingual_en": ("id", "text_latn", "text_tfng", "text_en"),
            "trilingual_fr": ("id", "text_latn", "text_tfng", "text_fr"),
        },
    )


def declared_configs(card: Path) -> set[str]:
    """Config names the card's YAML front matter declares to the Hub.

    The front matter is what makes a multi-config repo loadable; a config staged but not
    declared is invisible, and one declared but not staged is a load error for whoever
    downloads it. Parsed by hand rather than with a YAML dependency, because the shape is
    fixed and this runs before anything is written.
    """
    return {
        line.split(":", 1)[1].strip()
        for line in front_matter(card).splitlines()
        if "config_name:" in line
    }


def front_matter(card: Path) -> str:
    text = card.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        message = f"{card}: no YAML front matter, so the Hub cannot resolve its configs"
        raise ExportError(message)
    return text.split("---\n", 2)[1]


def _listed_under(card: Path, field: str) -> list[str]:
    """The `- value` entries of one top-level list field in the front matter."""
    listed: list[str] = []
    inside = False
    for line in front_matter(card).splitlines():
        if not line.startswith((" ", "-", "\t")) and line.strip():
            inside = line.startswith(f"{field}:")
            continue
        if inside and line.strip().startswith("- "):
            listed.append(line.strip()[2:].strip())
    return listed


def check_tasks(card: Path) -> None:
    """Both task fields against their vocabularies, before anything is written."""
    for field, allowed in (("task_categories", TASK_CATEGORIES), ("task_ids", TASK_IDS)):
        unknown = sorted(set(_listed_under(card, field)) - allowed)
        if unknown:
            message = f"{card}: {unknown} are not valid {field}; the card would warn on render"
            raise ExportError(message)


@dataclass(frozen=True, slots=True)
class Written:
    """One file the staging produced."""

    config: str
    split: str
    rows: int
    path: str
    size: int

    def as_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "split": self.split,
            "rows": self.rows,
            "path": self.path,
            "bytes": self.size,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """What one staged repo contains."""

    repo: str
    files: tuple[Written, ...]

    @property
    def rows(self) -> int:
        return sum(f.rows for f in self.files)

    def as_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "files": [f.as_dict() for f in self.files],
            "rows": self.rows,
        }


def _write_rows(path: Path, rows: Sequence[Row], suffix: str) -> None:
    """One split, in the format the repo's card declares."""
    if suffix == "jsonl":
        with path.open("w", encoding="utf-8") as sink:
            for row in rows:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if suffix != "parquet":
        message = f"{path}: unknown format {suffix!r}"
        raise ExportError(message)

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(list(rows)), path, compression="snappy")


def stage(repo: Repo, out: Path) -> Report:
    """Write one repo, after every check has passed."""
    check_tasks(repo.card)
    declared = declared_configs(repo.card)
    staged = {config.name for config in repo.configs}
    if declared != staged:
        message = (
            f"{repo.name}: the card declares {sorted(declared)} but {sorted(staged)} is "
            f"staged; the Hub would fail to load the difference"
        )
        raise ExportError(message)

    target = out / repo.name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    written: list[Written] = []
    for config in repo.configs:
        for split, rows in config.splits.items():
            stem = f"{split}.{repo.suffix}"
            path = target / stem if repo.layout == "flat" else target / config.name / stem
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_rows(path, rows, repo.suffix)
            written.append(
                Written(
                    config=config.name,
                    split=split,
                    rows=len(rows),
                    path=str(path.relative_to(target)),
                    size=path.stat().st_size,
                )
            )

    shutil.copyfile(repo.card, target / "README.md")
    report = Report(repo=repo.name, files=tuple(written))
    (target / "dataset.stats.json").write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def punctuation() -> Repo:
    """KabPunct: annotated punctuation & casing restoration corpus.

    Each sentence is converted from raw text to a sequence of (word, punct_label, case_label)
    triples. The raw JSONL holds one sentence per row with `text` and `source` fields;
    `annotate` from `agbalu.punctuation.labels` is the one definition of how a sentence
    becomes word tokens and labels, so the published rows are exactly what the model trains on.

    The four splits map to two configs: `default` (train/dev/test — CV-style sentences) and
    `ood` (long-form prose, held out of training). The card's `configs:` block declares both.
    """
    from agbalu.punctuation.labels import CASE, PUNCTUATION, annotate

    def _annotated(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Convert raw sentence rows to the published word-level annotation format."""
        out: list[dict[str, object]] = []
        for row in rows:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            ann = annotate(text)
            if not ann.words:
                continue
            out.append(
                {
                    "words": list(ann.words),
                    "punctuation": [PUNCTUATION[i] for i in ann.punctuation],
                    "case": [CASE[i] for i in ann.case],
                    "source": row.get("source", ""),
                }
            )
        return out

    fields = ("words", "punctuation", "case", "source")

    train_rows = _annotated(read_jsonl(PUNCT / "train.jsonl"))
    dev_rows = _annotated(read_jsonl(PUNCT / "dev.jsonl"))
    test_rows = _annotated(read_jsonl(PUNCT / "test.jsonl"))
    ood_rows = _annotated(read_jsonl(PUNCT / "ood.jsonl"))

    return Repo(
        name="KabPunct",
        card=CARDS / "kabpunct.md",
        configs=(
            Config(
                name="default",
                splits={"train": train_rows, "dev": dev_rows, "test": test_rows},
                fields=fields,
            ),
            Config(
                name="ood",
                splits={"ood": ood_rows},
                fields=fields,
            ),
        ),
    )


def g2p() -> Repo:
    """KabG2P: Kabyle grapheme-to-phoneme pronunciation dictionary.

    25,634 word-IPA pairs recovered by aligning 59,462 Kabyle sentences against
    their IPA transcriptions (99.53% alignment rate, 0% ambiguity). The 8 entries
    whose headword falls outside the Kabyle writing system are excluded.
    """
    outside_writing_system: frozenset[str] = frozenset(
        {
            "3d",
            "androïd",
            "mp3",
            "muḥ€nd",
            "rosé",
            "supermarché",
            "xelleṣ̣",
            "ṭeyyeb\u201f",
        }
    )

    rows = read_jsonl(PRONUNCIATIONS)
    kept = [r for r in rows if r.get("word") not in outside_writing_system]
    if len(kept) == len(rows):
        message = "outside-writing-system filter removed nothing; verify the source data"
        raise ExportError(message)

    fields = ("word", "ipa", "variants", "repaired")
    return Repo(
        name="KabG2P",
        card=CARDS / "kabg2p.md",
        configs=(
            Config(
                name="default",
                splits={"train": project(kept, fields)},
                fields=fields,
            ),
        ),
        layout="nested",
    )


def repositories() -> Iterator[tuple[str, Callable[[], Repo]]]:
    yield "bench", bench
    yield "lex", lexicon
    yield "sentiment", sentiment
    yield "inflect", inflection
    yield "tifinagh", transliteration
    yield "punct", punctuation
    yield "g2p", g2p


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/release"))
    parser.add_argument("--only", choices=[name for name, _ in repositories()])
    args = parser.parse_args(argv)

    if args.out.resolve() in (LEXICON.resolve(), BENCH.resolve()):
        message = f"--out {args.out} is a source directory; staging there would overwrite it"
        raise ExportError(message)

    wanted = [build for name, build in repositories() if args.only in (None, name)]
    for build in wanted:
        report = stage(build(), args.out)
        print(f"{report.repo}: {report.rows:,} rows")
        for entry in report.files:
            print(f"   {entry.path:<28} {entry.rows:>8,} rows  {entry.size:>10,} B")
        print(f"-> {args.out / report.repo}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
