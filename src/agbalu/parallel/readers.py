"""Read aligned sentence pairs out of a landed artifact.

Two shapes exist and they need different handling:

- **record files** (parquet, jsonl, tsv, ...) carry both sides on one row, so the
  work is choosing the two fields. `extract.readers` already flattens them.
- **OPUS moses bundles** are a zip of one plain-text file per language, aligned by
  line number. Nothing identifies the pair except position, so a truncated member
  would silently shift every alignment after it — the reader refuses a bundle whose
  sides differ in length rather than zipping to the shorter one.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from agbalu.extract.readers import read_records

KAB_SUFFIX: Final = ".kab"


class PairReadError(Exception):
    """A bundle could not be read as aligned pairs."""


def opus_members(archive: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    """`(kab_member, foreign_member, foreign_code)` for each pair in the bundle.

    The foreign code is read from the bundle stem, not guessed from the extension.
    `bible-uedin.en-kab.kab` names its own pair as `en-kab`, so the sibling can only
    be `.en`. Matching any sibling extension instead paired the Kabyle text against
    `bible-uedin.en-kab.xml`, the alignment file, whose differing line count aborted
    the whole source.
    """
    names = archive.namelist()
    found: list[tuple[str, str, str]] = []
    for kab in names:
        if not kab.endswith(KAB_SUFFIX):
            continue
        stem = kab[: -len(KAB_SUFFIX)].rstrip(".")
        pair_segment = stem.rsplit(".", maxsplit=1)[-1]
        codes = [c for c in pair_segment.split("-") if c and c != "kab"]
        if len(codes) != 1:
            continue
        sibling = f"{stem}.{codes[0]}"
        if sibling in names:
            found.append((kab, sibling, codes[0]))
    return found


def read_opus_zip(path: Path) -> Iterator[tuple[str, str, str]]:
    """`(kabyle, foreign, foreign_code)` from an OPUS moses bundle."""
    with zipfile.ZipFile(path) as archive:
        for kab_name, other_name, code in opus_members(archive):
            with archive.open(kab_name) as kab_handle, archive.open(other_name) as other_handle:
                try:
                    # strict=True turns a truncated member into an error at the point
                    # of mismatch. Zipping to the shorter side would instead emit every
                    # later line against the wrong translation, silently.
                    for kab_line, other_line in zip(kab_handle, other_handle, strict=True):
                        kab = kab_line.decode("utf-8", errors="replace").strip()
                        other = other_line.decode("utf-8", errors="replace").strip()
                        if kab and other:
                            yield kab, other, code
                except ValueError as exc:
                    msg = f"{path}: {kab_name} and {other_name} differ in line count"
                    raise PairReadError(msg) from exc


def read_record_pairs(path: Path, kab_field: str, foreign_field: str) -> Iterator[tuple[str, str]]:
    for record in read_records(path):
        kab = record.get(kab_field)
        foreign = record.get(foreign_field)
        if kab is not None and foreign is not None:
            yield kab, foreign
