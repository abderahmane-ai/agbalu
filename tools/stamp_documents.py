"""Recompute every document sidecar's measured fields, and report missing provenance.

`data/documents/**` holds whole works run through the MT stack. The bytes are git-ignored
and the sidecars are not, so **the sidecar is the only provenance that survives a clean
checkout** — §2.1 rule 1.

Two ways it rots, both of which have happened. A document staged in a hurry gets a sidecar
with a title and nothing else. And a document cleaned in place after its sidecar was written
keeps a `sha256` that no longer matches the bytes, which is worse than no hash at all: it
reads as verified.

This recomputes what can be measured from the file and leaves what cannot be measured alone.
It never invents a source, a licence or a retrieval date — a sidecar missing one is reported,
not filled.

    python3 -m tools.stamp_documents            # report
    python3 -m tools.stamp_documents --write    # recompute and rewrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DOCUMENTS: Final = Path("data/documents")

MEASURED: Final = ("sha256", "bytes", "characters", "words", "paragraphs")
"""Fields derived from the file. Rewritten every time, because a stale one is a lie."""

REQUIRED: Final = ("title", "language", "licence", "source", "retrieved", "reproduce")
"""Fields no measurement can supply. A sidecar without them is a bookmark."""

PARAGRAPHS: Final = re.compile(r"\n\s*\n")


class DocumentError(Exception):
    """A document that cannot be described."""


@dataclass(frozen=True, slots=True)
class Report:
    """One sidecar, after stamping."""

    path: Path
    present: bool
    missing: tuple[str, ...]
    stale: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.present and not self.missing and not self.stale


def measure(text_path: Path) -> dict[str, object]:
    """The fields the bytes themselves determine."""
    raw = text_path.read_bytes()
    text = raw.decode("utf-8")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "characters": len(text),
        "words": len(text.split()),
        "paragraphs": len([p for p in PARAGRAPHS.split(text) if p.strip()]),
    }


def stamp(sidecar: Path, *, write: bool) -> Report:
    """Recompute one sidecar's measured fields against the document beside it."""
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        message = f"{sidecar} is not a JSON object"
        raise DocumentError(message)

    text_path = sidecar.with_name(sidecar.name.removesuffix(".meta.json") + ".txt")
    missing = tuple(field for field in REQUIRED if not payload.get(field))
    if not text_path.is_file():
        return Report(path=sidecar, present=False, missing=missing, stale=())

    measured = measure(text_path)
    stale = tuple(field for field in MEASURED if payload.get(field) != measured[field])
    if write and stale:
        payload.update(measured)
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stale = ()
    return Report(path=sidecar, present=True, missing=missing, stale=stale)


def sidecars(root: Path = DOCUMENTS) -> list[Path]:
    return sorted(root.rglob("*.meta.json"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DOCUMENTS)
    parser.add_argument("--write", action="store_true", help="rewrite the measured fields")
    args = parser.parse_args(argv)

    reports = [stamp(path, write=args.write) for path in sidecars(args.root)]
    for report in reports:
        state = "ok" if report.ok else "  "
        note = []
        if not report.present:
            note.append("no bytes on disk")
        if report.stale:
            note.append(f"stale {list(report.stale)}")
        if report.missing:
            note.append(f"missing {list(report.missing)}")
        print(f"{state} {report.path.relative_to(args.root)}  {'; '.join(note)}")

    incomplete = [r for r in reports if r.missing]
    print(f"\n{len(reports)} sidecars, {len(reports) - len(incomplete)} carrying full provenance")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
