"""Count every codepoint in every readable text artifact under data/raw/.

The orthography spec must be grounded in what the corpus actually contains, not
in what a standard says it should. Run before changing `resources/homoglyphs.yaml`
or the canonical inventory in `docs/orthography.md`.

    python tools/character_census.py [--root data/raw] [--out census.json]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path

csv.field_size_limit(10**9)

SKIP_DIRS = frozenset({".git", ".cache", "__pycache__"})

KAB_FIELD = ("kab", "kabyle", "taqbaylit")
"""Field names that denote the Kabyle side of a multilingual record.

Without this filter the census counts the English, French, Romanian and Hindi
sides of the parallel corpora too, and every "corruption rate" it reports is
really a measure of other languages using their own alphabets correctly.
"""


TEXT_FIELD = frozenset(
    {"text", "raw_text", "sentence", "content", "body", "line", "src", "tgt", "target"}
)
"""Field names that hold running text.

Several sources ship text alongside metadata. `tatoeba-kabyle-mono-cleaned` carries
GlotLID output (`glot_lang`, `berber_status`, `lang_distribution`); counting those
inflates the census with label vocabulary — its `HIGH_CONF` strings alone put
capital P above lowercase p across the entire corpus.
"""


def is_kab_field(name: str) -> bool:
    lowered = name.lower()
    return any(
        k == lowered or lowered.endswith(f"_{k}") or lowered.startswith(f"{k}_") for k in KAB_FIELD
    )


def pick_fields(names: list[str], *, kab_only: bool) -> set[str] | None:
    """Columns to count, or None to count them all.

    Preference: Kabyle-named columns, then text-named ones. A source with neither
    is assumed to be running text throughout.
    """
    if kab_only:
        kab = {n for n in names if is_kab_field(n)}
        if kab:
            return kab
    text = {n for n in names if n.lower() in TEXT_FIELD}
    return text or None


def iter_strings_parquet(path: Path, *, kab_only: bool) -> Iterator[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    wanted = pick_fields(list(table.column_names), kab_only=kab_only)
    for name, column in zip(table.column_names, table.columns, strict=True):
        if not name or (wanted is not None and name not in wanted):
            continue
        for chunk in column.chunks:
            if chunk.type.id not in (13, 14):  # string, large_string
                continue
            for value in chunk.to_pylist():
                if isinstance(value, str):
                    yield value


def iter_strings_jsonl(path: Path, *, kab_only: bool) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(record, str):
                yield record
            elif isinstance(record, dict):
                wanted = pick_fields(list(record), kab_only=kab_only)
                for key, value in record.items():
                    if wanted is not None and key not in wanted:
                        continue
                    if isinstance(value, str):
                        yield value


def iter_strings_delimited(path: Path, delimiter: str) -> Iterator[str]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle, delimiter=delimiter, quoting=csv.QUOTE_NONE):
            yield from row


def iter_strings_plain(path: Path) -> Iterator[str]:
    yield path.read_text(encoding="utf-8", errors="replace")


def iter_strings(path: Path, *, kab_only: bool = False) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from iter_strings_parquet(path, kab_only=kab_only)
    elif suffix in (".jsonl", ".ndjson"):
        yield from iter_strings_jsonl(path, kab_only=kab_only)
    elif suffix == ".tsv":
        yield from iter_strings_delimited(path, "\t")
    elif suffix == ".csv":
        yield from iter_strings_delimited(path, ",")
    elif suffix in (".txt", ".conllu"):
        yield from iter_strings_plain(path)


def census(
    root: Path, *, kab_only: bool = False
) -> tuple[collections.Counter[str], collections.Counter[str], int]:
    chars: collections.Counter[str] = collections.Counter()
    per_source: collections.Counter[str] = collections.Counter()
    failures = 0
    suffixes = {".parquet", ".jsonl", ".ndjson", ".tsv", ".csv", ".txt", ".conllu"}
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in suffixes
        and not (SKIP_DIRS & set(p.relative_to(root).parts))
    ]
    print(f"scanning {len(files)} files under {root}", file=sys.stderr)
    for index, path in enumerate(files, start=1):
        source = path.relative_to(root).parts[0]
        try:
            for value in iter_strings(path, kab_only=kab_only):
                chars.update(value)
                per_source[source] += len(value)
        except Exception as exc:
            failures += 1
            print(f"  skip {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if index % 200 == 0:
            print(f"  {index}/{len(files)} files, {len(chars)} distinct", file=sys.stderr)
    return chars, per_source, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("census.json"))
    parser.add_argument(
        "--kab-only",
        action="store_true",
        help="count only Kabyle-named fields where a record has one",
    )
    args = parser.parse_args(argv)

    chars, per_source, failures = census(args.root, kab_only=args.kab_only)
    total = sum(chars.values())
    print(
        f"\n{len(chars):,} distinct codepoints over {total:,} characters, {failures} files skipped"
    )

    rows = [
        {
            "char": ch,
            "codepoint": f"U+{ord(ch):04X}",
            "count": n,
            "category": unicodedata.category(ch),
            "script_name": unicodedata.name(ch, "<unnamed>"),
        }
        for ch, n in chars.most_common()
    ]
    args.out.write_text(
        json.dumps(
            {"total_characters": total, "distinct": len(chars), "chars": rows}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"written: {args.out}")

    print("\nlargest sources by character count:")
    for source, count in per_source.most_common(12):
        print(f"  {source:46} {count:>14,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
