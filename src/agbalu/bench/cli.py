"""Benchmark CLI.

python -m agbalu.bench.cli audit
python -m agbalu.bench.cli contamination
python -m agbalu.bench.cli score --hypotheses out.txt --direction eng-kab --split devtest
python -m agbalu.bench.cli pos --split test
python -m agbalu.bench.cli tifinagh
python -m agbalu.bench.cli inflect
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Final

from agbalu.bench import inflect, lid, lid_systems, tifinagh
from agbalu.bench.audit import AuditReport, audit, describe
from agbalu.bench.contamination import ContaminationReport, scan
from agbalu.bench.flores import (
    DEFAULT_ROOT,
    KABYLE,
    SPLITS,
    FloresError,
    Sentence,
    Split,
    read_all,
    read_split,
)
from agbalu.bench.mt import (
    DIRECTIONS,
    LANGUAGE_CODE,
    Direction,
    Result,
    ScoringError,
    references_for,
    score,
    target_language,
)
from agbalu.bench.pos import (
    CONDITIONS,
    CORRUPTIONS,
    SETTINGS,
    Condition,
    PosScoringError,
    Run,
    Score,
    Setting,
    Tagger,
    items_for,
    run,
)
from agbalu.bench.pos import (
    score as score_pos,
)
from agbalu.model.config import DEFAULT_RUN_NAME, PRESETS
from agbalu.normalise import Normaliser
from agbalu.normalise.rules import load_rules
from agbalu.treebank import DEFAULT_ROOT as DEFAULT_TREEBANK
from agbalu.treebank import Sentence as TreebankSentence
from agbalu.treebank import Split as TreebankSplit
from agbalu.treebank import TreebankError
from agbalu.treebank import read_split as read_treebank_split

DEFAULT_CORPUS: Final = Path("data/processed/text/agbalu-text-v1.jsonl")
DEFAULT_OUT: Final = Path("data/processed/bench")
DEFAULT_MODEL: Final = Path("data/raw/hf.boffire.kabyle-pos-v2")
DEFAULT_LEXICON: Final = Path("data/processed/lexicon/agbalu-lexicon-v1.jsonl")

SYSTEMS: Final[tuple[str, ...]] = ("neural", "lexicon", "baseline", "encoder")

DEFAULT_RAW: Final = Path("data/raw")
LID_PER_LANGUAGE: Final = 250
"""Lines drawn per language. The binding pool is Central Atlas Tamazight at 285
usable lines (2026-08-09), the only Latin-script tzm text the survey found."""

LID_SEED: Final = 20260809

DEFAULT_MT_CORPUS: Final = Path("data/processed/mt/train.jsonl")
LID_PER_SOURCE: Final = 3000
"""Kabyle lines sampled per MT source. Above the point where a source's label
distribution stops moving, and small enough that 18 sources score in seconds."""

DEFAULT_RUN_DIR: Final = Path("artifacts/runs") / DEFAULT_RUN_NAME
DEFAULT_ENCODER_TOKENIZER: Final = Path("artifacts/tokenizer/agbalu-tok-base-16k.model")
CHECKPOINT_NAMES: Final[tuple[str, ...]] = ("best.pt", "latest.pt")
"""Spelled here rather than imported from `model.checkpoint`, which pulls in torch: every
other subcommand of this CLI has to keep working without the optional extra."""
PROBE_EPOCHS: Final = 20

PUNCTUATION: Final[frozenset[str]] = frozenset({"PUNCT"})
"""16.6% of this treebank, and no lexicon holds any of it. Every result is reported
with and without."""

log: Final = logging.getLogger("agbalu.bench")

TOP_CODEPOINTS: Final = 12
TOP_CONFUSIONS: Final = 15


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_audit(
    report: AuditReport, sentences: list[Sentence], version: str, out: Path
) -> tuple[Path, Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    corrected = out / f"flores-plus-{report.language}-corrected.jsonl"
    diff = out / f"flores-plus-{report.language}-diff.jsonl"
    summary = out / f"flores-plus-{report.language}-audit.json"

    # FLORES+ ids restart at 0 in every split, so `id` alone is not unique:
    # all 997 dev ids collide with devtest.
    fixed = {(d.split, d.id): d.corrected for d in report.diffs}
    with corrected.open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            handle.write(
                json.dumps(
                    {
                        "id": sentence.id,
                        "split": sentence.split,
                        "text": fixed.get((sentence.split, sentence.id), sentence.text),
                        "corrected": (sentence.split, sentence.id) in fixed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with diff.open("w", encoding="utf-8") as handle:
        for item in report.diffs:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    _write_json(
        summary,
        {
            "normaliser_version": version,
            "language": report.language,
            "sentences": report.sentences,
            "changed": report.changed,
            "changed_rate": round(report.changed_rate, 6),
            "mean_token_divergence": round(report.mean_token_divergence, 6),
            "protected": report.protected,
            "revisions": report.revisions,
            "by_split": dict(report.by_split),
            "by_kind": dict(report.by_kind),
            "by_codepoint": {describe(c): n for c, n in report.by_codepoint.most_common()},
        },
    )
    return corrected, diff, summary


def command_audit(args: argparse.Namespace) -> int:
    sentences = list(read_all(args.root, args.language))
    engine = Normaliser()
    report = audit(sentences, args.language, engine)
    corrected, diff, summary = _write_audit(report, sentences, engine.version, args.out)

    print(f"FLORES+ {report.language}: {report.sentences:,} sentences")
    print(f"  normaliser         {engine.version}")
    print(f"  revisions          {report.revisions}")
    print(f"  changed            {report.changed:,} ({report.changed_rate:.1%})")
    print(f"  token divergence   {report.mean_token_divergence:.1%} (mean, changed sentences)")
    print(f"  protected          {report.protected:,} (foreign proper nouns, left alone)")
    print(f"  by split           {dict(report.by_split)}")
    print(f"  by change kind     {dict(report.by_kind)}")
    print("  corrupted codepoints:")
    for codepoint, count in report.by_codepoint.most_common(TOP_CODEPOINTS):
        print(f"    {count:>6,}  {describe(codepoint)}")
    print(f"\n  corrected -> {corrected}")
    print(f"  diff      -> {diff}")
    print(f"  summary   -> {summary}")
    return 0


def _write_contamination(
    report: ContaminationReport, language: str, corpus: Path, version: str, out: Path
) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    detail = out / f"contamination-{language}.jsonl"
    summary = out / f"contamination-{language}.json"

    with detail.open("w", encoding="utf-8") as handle:
        for leak in report.leaked:
            handle.write(json.dumps(asdict(leak), ensure_ascii=False) + "\n")

    # A zero-leak run writes an empty detail file, indistinguishable from no run.
    _write_json(
        summary,
        {
            "normaliser_version": version,
            "language": language,
            "corpus": str(corpus),
            "corpus_lines": report.corpus_lines,
            "benchmark_sentences": report.benchmark_sentences,
            "leaked": len(report.leaked),
            "leak_rate": round(report.leak_rate, 6),
            "by_split": dict(report.by_split),
            "by_kind": dict(report.by_kind),
            "by_source": dict(report.by_source.most_common()),
        },
    )
    return detail, summary


def command_contamination(args: argparse.Namespace) -> int:
    sentences = list(read_all(args.root, args.language))
    engine = Normaliser()
    report = scan(args.corpus, sentences, engine)
    path, summary = _write_contamination(
        report, args.language, args.corpus, engine.version, args.out
    )

    print(
        f"{report.benchmark_sentences:,} benchmark sentences vs "
        f"{report.corpus_lines:,} corpus lines"
    )
    print(f"  normaliser         {engine.version}")
    print(f"  leaked             {len(report.leaked):,} ({report.leak_rate:.2%})")
    print(f"  by split           {dict(report.by_split)}")
    print("  by corpus source:")
    for source, count in report.by_source.most_common():
        print(f"    {count:>6,}  {source}")
    print(f"\n  detail  -> {path}")
    print(f"  summary -> {summary}")
    return 0


def _read_hypotheses(path: Path) -> list[str]:
    """One hypothesis per line, in reference order.

    Interior blank lines are kept, so an empty translation does not realign every
    sentence after it. Only one trailing newline is dropped.
    """
    if not path.is_file():
        msg = f"hypotheses not found: {path}"
        raise ScoringError(msg)
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def command_score(args: argparse.Namespace) -> int:
    direction: Direction = args.direction
    split: Split = args.split
    language = LANGUAGE_CODE[target_language(direction)]
    references = references_for(read_split(args.root, split, language))
    hypotheses = _read_hypotheses(args.hypotheses)
    result = score(hypotheses, references, direction, split)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / f"mt-{direction}-{split}.json"
    _write_json(summary, _result_payload(result, args.model, args.hypotheses))

    print(f"{direction} {split}: {result.sentences:,} sentences")
    print(f"  model              {args.model}")
    print(f"  normaliser         {result.normaliser_version}")
    for condition in (result.raw, result.normalised):
        print(f"  {condition.name}")
        for metric in condition.metrics:
            print(f"    {metric.name:8} {metric.score:8.4f}  {metric.signature}")
    print("  orthography gap (normalised - raw)")
    for name in ("chrf++", "bleu", "spbleu"):
        print(f"    {name:8} {result.gap(name):+8.4f}")
    print(f"\n  summary -> {summary}")
    return 0


def _result_payload(result: Result, model: str, hypotheses: Path) -> dict[str, object]:
    return {
        "model": model,
        "hypotheses": str(hypotheses),
        "direction": result.direction,
        "split": result.split,
        "sentences": result.sentences,
        "normaliser_version": result.normaliser_version,
        "conditions": {
            condition.name: {
                metric.name: {"score": metric.score, "signature": metric.signature}
                for metric in condition.metrics
            }
            for condition in (result.raw, result.normalised)
        },
        "orthography_gap": {
            name: round(result.gap(name), 4) for name in ("chrf++", "bleu", "spbleu")
        },
    }


def _score_payload(result: Score) -> dict[str, object]:
    return {
        "scored": result.scored,
        "answered": result.answered,
        "correct": result.correct,
        "accuracy": round(result.accuracy, 6),
        "coverage": round(result.coverage, 6),
        "accuracy_when_answered": round(result.accuracy_when_answered, 6),
        "macro_f1": round(result.macro_f1, 6),
        "by_label": [
            {
                "label": item.label,
                "support": item.support,
                "predicted": item.predicted,
                "precision": round(item.precision, 6),
                "recall": round(item.recall, 6),
                "f1": round(item.f1, 6),
            }
            for item in result.label_scores()
        ],
        "top_confusions": [
            {"gold": gold, "predicted": hypothesis, "count": count}
            for gold, hypothesis, count in result.top_confusions(TOP_CONFUSIONS)
        ],
    }


def _run_payload(result: Run) -> dict[str, object]:
    return {
        "system": result.tagger,
        "revision": result.revision,
        "setting": result.setting,
        "condition": result.condition,
        "sentences": result.sentences,
        "units": result.units,
        "unscorable": result.unscorable,
        "unscorable_rate": round(result.unscorable_rate, 6),
        "all_tags": _score_payload(score_pos(result)),
        "without_punct": _score_payload(score_pos(result, ignore=PUNCTUATION)),
    }


def _build_taggers(args: argparse.Namespace, train: list[TreebankSentence]) -> list[Tagger]:
    """Deferred import: `torch` and `transformers` are an optional extra, and the
    other subcommands must keep working without them."""
    from agbalu.bench.taggers import LexiconTagger, MostFrequentTagger, NeuralTagger

    taggers: list[Tagger] = []
    if "neural" in args.systems:
        taggers.append(NeuralTagger.load(args.model, repo=args.model_repo, raw=args.raw))
    if "lexicon" in args.systems:
        taggers.append(LexiconTagger.load(args.lexicon))
    if "baseline" in args.systems:
        taggers.append(MostFrequentTagger.fit(train))
    if "encoder" in args.systems:
        taggers.append(_probe(args, train))
    return taggers


def _probe(args: argparse.Namespace, train: list[TreebankSentence]) -> Tagger:
    """Fit the linear probe on the treebank's train split.

    Fitted here rather than loaded because the head is 17 rows and takes seconds; the
    encoder underneath is the artifact, and it is frozen.
    """
    from agbalu.bench.probe import ProbeConfig, ProbeTagger
    from agbalu.model.checkpoint import load as load_state
    from agbalu.model.config import PRESETS
    from agbalu.model.modeling import Encoder
    from agbalu.model.trainer import select_device
    from agbalu.tokenizer.evaluate import load as load_tokenizer

    device = select_device()
    config = PRESETS[args.preset]
    encoder = Encoder(config).to(device)
    state = load_state(args.run, encoder, None, name=args.checkpoint, restore_rng=False)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    return ProbeTagger.fit(
        encoder,
        load_tokenizer(args.encoder_tokenizer),
        train,
        device=device,
        step=state.step,
        config=ProbeConfig(epochs=args.probe_epochs),
    )


def _write_pos(
    runs: list[Run], args: argparse.Namespace, sentences: list[TreebankSentence], version: str
) -> tuple[Path, Path]:
    args.out.mkdir(parents=True, exist_ok=True)
    summary = args.out / f"pos-kab_adpt-{args.split}.json"
    detail = args.out / f"pos-kab_adpt-{args.split}-predictions.jsonl"

    words = sum(len(s.words) for s in sentences)
    tokens = sum(len(s.tokens) for s in sentences)
    multiword = sum(len(t.words) for s in sentences for t in s.tokens if t.is_multiword)

    _write_json(
        summary,
        {
            "treebank": str(args.treebank),
            "split": args.split,
            "sentences": len(sentences),
            "words": words,
            "surface_tokens": tokens,
            "multiword_words": multiword,
            "multiword_rate": round(multiword / words, 6) if words else 0.0,
            "normaliser_version": version,
            "corruption": CORRUPTIONS,
            "ignored_in_second_view": sorted(PUNCTUATION),
            "runs": [_run_payload(result) for result in runs],
        },
    )

    with detail.open("w", encoding="utf-8") as handle:
        for result in runs:
            for prediction in result.predictions:
                handle.write(
                    json.dumps(
                        {
                            "system": result.tagger,
                            "setting": result.setting,
                            "condition": result.condition,
                            **asdict(prediction),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return summary, detail


def _report_pos(result: Run) -> None:
    overall = score_pos(result)
    without = score_pos(result, ignore=PUNCTUATION)
    print(f"  {result.tagger} @ {result.revision}")
    print(f"    accuracy    {overall.accuracy:7.2%}   without PUNCT {without.accuracy:7.2%}")
    print(f"    macro-F1    {overall.macro_f1:7.2%}   without PUNCT {without.macro_f1:7.2%}")
    print(
        f"    coverage    {overall.coverage:7.2%}   accuracy when answered "
        f"{overall.accuracy_when_answered:7.2%}"
    )


def command_pos(args: argparse.Namespace) -> int:
    split: TreebankSplit = args.split
    sentences = read_treebank_split(args.treebank, split)
    fitted = {"baseline", "encoder"} & set(args.systems)
    train = read_treebank_split(args.treebank, "train") if fitted else []
    if split == "train" and fitted:
        msg = f"{sorted(fitted)} are fitted on the train split and cannot be scored on it"
        raise PosScoringError(msg)

    rules = load_rules()
    taggers = _build_taggers(args, train)

    runs: list[Run] = []
    for setting in SETTINGS:
        setting_name: Setting = setting
        for condition in CONDITIONS:
            condition_name: Condition = condition
            items = items_for(sentences, setting_name, condition_name, rules)
            for tagger in taggers:
                log.info(
                    "tagging %s setting=%s condition=%s", tagger.name, setting_name, condition_name
                )
                runs.append(run(tagger, items, setting_name, condition_name))

    summary, detail = _write_pos(runs, args, sentences, Normaliser().version)

    words = sum(len(s.words) for s in sentences)
    print(f"UD_Kabyle-ADPT {args.split}: {len(sentences):,} sentences / {words:,} words")
    for setting in SETTINGS:
        for condition in CONDITIONS:
            selected = [r for r in runs if r.setting == setting and r.condition == condition]
            if not selected:
                continue
            print(
                f"\n  [{setting} / {condition}]  "
                f"{selected[0].units:,} positions, {selected[0].unscorable:,} unscorable"
            )
            for result in selected:
                _report_pos(result)
    print(f"\n  summary     -> {summary}")
    print(f"  predictions -> {detail}")
    return 0


def _print_matrix(matrix: lid.ConfusionMatrix, *, others: int = 2) -> None:
    """The eval languages as columns, plus the heaviest labels from outside them.

    Printing every predicted label is unreadable: a 2,102-label system scatters
    the tail across ~90 columns of mostly zeros, which buries the diagonal.
    `elsewhere` keeps the rows summing to their support.
    """
    outside: Counter[str] = Counter()
    for (_, predicted), count in matrix.counts.items():
        if predicted not in matrix.labels:
            outside[predicted] += count
    extra = [label for label, _ in outside.most_common(others)]
    columns = [*matrix.labels, *extra]
    width = max(max((len(x) for x in columns), default=8), 9)

    print(" " * 12 + " ".join(f"{x:>{width}}" for x in [*columns, "elsewhere"]))
    for true in matrix.labels:
        shown = [matrix.counts.get((true, pred), 0) for pred in columns]
        rest = matrix.support(true) - sum(shown)
        cells = " ".join(f"{value:>{width}}" for value in [*shown, rest])
        marker = " " if true in matrix.system_labels else "*"
        print(f"{true:<11}{marker}{cells}")
    if matrix.unnameable:
        print(f"  * absent from this system's {len(matrix.system_labels)}-label set")


def command_lid(args: argparse.Namespace) -> int:
    pools = lid.build_pools(args.raw)
    examples = lid.build_evalset(pools, per_language=args.per_language, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    evalset_path = args.out / "lid-eval.jsonl"
    with evalset_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    log.info(
        "evalset languages=%d per_language=%d total=%d path=%s",
        len(pools),
        args.per_language,
        len(examples),
        evalset_path,
    )

    results: dict[str, object] = {
        "per_language": args.per_language,
        "seed": args.seed,
        "pool_sizes": {language: len(pool) for language, pool in sorted(pools.items())},
        "systems": {},
    }
    systems: dict[str, object] = {}
    for name in args.systems:
        identifier = lid_systems.build(name)
        matrix = lid.score(identifier, examples)
        systems[name] = {"revision": identifier.revision, **matrix.to_dict()}
        print(f"\n=== {name} (rev {identifier.revision}) ===")
        print(
            f"accuracy {matrix.accuracy():.4f}   "
            f"macro-F1 over the {len(matrix.discriminable)} nameable "
            f"of {len(matrix.labels)} {matrix.macro_f1():.4f}"
        )
        _print_matrix(matrix)
        absorbed = [
            f"{language} {matrix.absorption(language):.1%}"
            for language in matrix.labels
            if language != lid.KABYLE and matrix.absorption(language) > 0
        ]
        if absorbed:
            print("labelled Kabyle: " + ", ".join(absorbed))
        log.info(
            "scored system=%s accuracy=%.4f macro_f1=%.4f nameable=%d/%d",
            name,
            matrix.accuracy(),
            matrix.macro_f1(),
            len(matrix.discriminable),
            len(matrix.labels),
        )
    results["systems"] = systems

    out = args.out / "lid-confusion.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def command_lid_audit(args: argparse.Namespace) -> int:
    reports: list[dict[str, object]] = []
    for name in args.systems:
        report = lid.audit_marker(
            args.corpus,
            marker=args.marker,
            identifier=lid_systems.build(name),
            limit=args.limit,
        )
        reports.append(report.to_dict())
        print(f"\n=== {name} on {args.marker!r} ===")
        print(
            f"records {report.records}   occurrences {report.occurrences}   "
            f"classified {report.classified}   Kabyle {report.kabyle_share:.1%}"
        )
        for label, count in list(report.labels.items())[:8]:
            print(f"  {label:<12} {count:>6}")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"lid-audit-{args.marker.encode('unicode_escape').decode()}.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def command_lid_sources(args: argparse.Namespace) -> int:
    identifier = lid_systems.build(args.system)
    audits = lid.audit_sources(
        args.corpus, identifier=identifier, per_source=args.per_source, seed=args.seed
    )
    print(f"{identifier.name} (rev {identifier.revision}) on {args.corpus}")
    print(f"{'source':<40} {'judged':>8} {'short':>7} {'Kabyle':>8}  top other labels")
    for entry in audits:
        others = [f"{k} {v}" for k, v in entry.labels.items() if k != lid.KABYLE][:3]
        print(
            f"{entry.source_id:<40} {entry.judged:>8,} {entry.too_short:>7,} "
            f"{entry.kabyle_share:>7.1%}  {', '.join(others)}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "lid-mt-sources.json"
    payload = {
        "corpus": str(args.corpus),
        "system": identifier.name,
        "revision": identifier.revision,
        "per_source": args.per_source,
        "seed": args.seed,
        "sources": [entry.to_dict() for entry in audits],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


def command_tifinagh(args: argparse.Namespace) -> int:
    """What the deterministic character table scores on script conversion.

    The floor the model is read against, and it is produced rather than quoted: a baseline
    computed once and written into a card cannot be checked when the split changes.
    """
    pairs = tifinagh.read_split(args.split, limit=args.limit)
    references = [pair.latin.lower() for pair in pairs]
    hypotheses = [tifinagh.to_latin(pair.tifinagh) for pair in pairs]
    report = {
        "split": args.split,
        "system": "deterministic character table",
        "conversion": tifinagh.score_conversion(references, hypotheses).as_dict(),
        "schwa": tifinagh.score_schwa(references, hypotheses).as_dict(),
        "reverse_conversion": tifinagh.score_conversion(
            [pair.tifinagh for pair in pairs],
            [tifinagh.to_tifinagh(pair.latin) for pair in pairs],
        ).as_dict(),
    }
    return _write_report(report, args.out / "tifinagh-baseline.json")


def command_inflect(args: argparse.Namespace) -> int:
    """What copying the lemma unchanged scores on morphological inflection."""
    entries = inflect.read_split(args.split, limit=args.limit)
    report = {
        "split": args.split,
        "system": "lemma copy",
        "lemmas": len({entry.lemma for entry in entries}),
        "inflection": inflect.copy_floor(entries).as_dict(),
    }
    return _write_report(report, args.out / "inflect-baseline.json")


def _write_report(report: dict[str, object], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    common.add_argument("--language", default=KABYLE)
    common.add_argument("--out", type=Path, default=DEFAULT_OUT)

    parser = argparse.ArgumentParser(prog="agbalu-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", parents=[common], help="orthographic audit of a FLORES+ language")
    contamination = sub.add_parser(
        "contamination", parents=[common], help="benchmark sentences present in the corpus"
    )
    contamination.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    scoring = sub.add_parser("score", parents=[common], help="score MT output, both conditions")
    scoring.add_argument("--hypotheses", type=Path, required=True)
    scoring.add_argument("--direction", choices=DIRECTIONS, required=True)
    scoring.add_argument("--split", choices=SPLITS, default="devtest")
    scoring.add_argument(
        "--model",
        required=True,
        help="repo id and revision, e.g. 'boffire/marianmt-en-kab@a1b2c3d'",
    )

    tagging = sub.add_parser("pos", help="score POS taggers against the gold UD treebank")
    tagging.add_argument("--treebank", type=Path, default=DEFAULT_TREEBANK)
    tagging.add_argument("--split", choices=("train", "test"), default="test")
    tagging.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    tagging.add_argument("--model-repo", default="boffire/kabyle-pos-v2")
    tagging.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    tagging.add_argument("--raw", type=Path, default=Path("data/raw"))
    tagging.add_argument("--out", type=Path, default=DEFAULT_OUT)
    tagging.add_argument("--run", type=Path, default=DEFAULT_RUN_DIR)
    tagging.add_argument("--checkpoint", choices=CHECKPOINT_NAMES, default=CHECKPOINT_NAMES[0])
    tagging.add_argument("--preset", choices=sorted(PRESETS), default="kab")
    tagging.add_argument("--encoder-tokenizer", type=Path, default=DEFAULT_ENCODER_TOKENIZER)
    tagging.add_argument("--probe-epochs", type=int, default=PROBE_EPOCHS)
    tagging.add_argument(
        "--systems",
        nargs="+",
        choices=SYSTEMS,
        default=list(SYSTEMS),
        help="which systems to score",
    )

    identification = sub.add_parser(
        "lid", help="score public language identifiers against the Berber siblings"
    )
    identification.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    identification.add_argument("--out", type=Path, default=DEFAULT_OUT)
    identification.add_argument("--per-language", type=int, default=LID_PER_LANGUAGE)
    identification.add_argument("--seed", type=int, default=LID_SEED)
    identification.add_argument(
        "--systems",
        nargs="+",
        choices=sorted(lid_systems.SYSTEMS),
        default=sorted(lid_systems.SYSTEMS),
        help="which identifiers to score",
    )

    mt_sources = sub.add_parser(
        "lid-sources", help="classify the Kabyle side of an MT corpus, per source"
    )
    mt_sources.add_argument("--corpus", type=Path, default=DEFAULT_MT_CORPUS)
    mt_sources.add_argument("--per-source", type=int, default=LID_PER_SOURCE)
    mt_sources.add_argument("--seed", type=int, default=LID_SEED)
    mt_sources.add_argument("--out", type=Path, default=DEFAULT_OUT)
    mt_sources.add_argument("--system", choices=sorted(lid_systems.SYSTEMS), default="glotlid")

    audit_lines = sub.add_parser(
        "lid-audit", help="classify the corpus lines carrying a suspect character"
    )
    audit_lines.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    audit_lines.add_argument("--marker", default="ṯ", help="the character to audit")
    audit_lines.add_argument("--limit", type=int, default=5000)
    audit_lines.add_argument("--out", type=Path, default=DEFAULT_OUT)
    audit_lines.add_argument(
        "--systems",
        nargs="+",
        choices=sorted(lid_systems.SYSTEMS),
        default=sorted(lid_systems.SYSTEMS),
    )
    _add_floor_parsers(sub)
    return parser


def _add_floor_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The two baselines whose only options are which split and how much of it."""
    for name, help_text in (
        ("tifinagh", "what the character table scores on Tifinagh to Latin conversion"),
        ("inflect", "what copying the lemma scores on morphological inflection"),
    ):
        floor = sub.add_parser(name, help=help_text)
        floor.add_argument("--split", choices=("dev", "test"), default="test")
        floor.add_argument("--limit", type=int)
        floor.add_argument("--out", type=Path, default=DEFAULT_OUT)


COMMANDS: Final[dict[str, Callable[[argparse.Namespace], int]]] = {
    "audit": command_audit,
    "contamination": command_contamination,
    "score": command_score,
    "pos": command_pos,
    "lid": command_lid,
    "lid-audit": command_lid_audit,
    "lid-sources": command_lid_sources,
    "tifinagh": command_tifinagh,
    "inflect": command_inflect,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr
    )
    args = build_parser().parse_args(argv)
    handler = COMMANDS.get(args.command)
    if handler is None:
        print(f"error: unknown command {args.command!r}", file=sys.stderr)
        return 1
    try:
        return handler(args)
    except (
        FloresError,
        ScoringError,
        PosScoringError,
        TreebankError,
        lid.LidError,
        lid_systems.LidModelError,
        tifinagh.BenchmarkError,
        inflect.InflectionError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
