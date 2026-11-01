"""Allocate script-consistent, uncontaminated examples to isolated splits."""

from __future__ import annotations

import hashlib
import random
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace

from agbalu.dihya.taxonomy import COMPOSITE_CLASSES, script_error

NORMALIZATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class RawRow:
    """Retain a candidate's annotation and auditable extraction identity."""

    text: str
    label: str
    domain: str
    source: str
    script: str
    origin: str = ""
    family: str = ""
    licence: str | None = None
    retrieved: str | None = None


def text_key(text: str) -> str:
    """Compare canonical Unicode text independently of case and whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def evaluation_key(text: str) -> str:
    """Match evaluation text across punctuation and whitespace variants."""
    canonical = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in canonical).split())


def row_error(row: RawRow) -> str | None:
    """Quarantine unverified variety assignments before script checks."""
    if row.source == "ircam.tarifit_stories":
        return "unverified_variety: IRCAM contains multiple varieties without file annotations"
    if not row.domain or not row.source:
        return "missing_source_or_domain"
    return script_error(row.text, row.label, row.script)


def select_rows(
    rows: Iterable[RawRow],
    cap: int,
    seed: int,
    excluded: set[str],
) -> tuple[list[RawRow], list[tuple[RawRow, str]], dict[str, int]]:
    """Account for every candidate, rejecting conflicting labels before sampling."""
    if cap < 3:
        msg = f"cap must permit three splits, got {cap}"
        raise ValueError(msg)
    rng = random.Random(seed)
    by_text: dict[str, list[RawRow]] = defaultdict(list)
    rejects: list[tuple[RawRow, str]] = []
    inputs = 0
    excluded_families: set[str] = set()
    for ordinal, candidate in enumerate(rows):
        inputs += 1
        row = replace(
            candidate,
            text=unicodedata.normalize("NFC", candidate.text),
            origin=candidate.origin or f"{candidate.source}:candidate:{ordinal}",
        )
        error = row_error(row)
        key = text_key(row.text)
        if error:
            rejects.append((row, error))
        elif evaluation_key(row.text) in excluded:
            rejects.append((row, "external_evaluation_overlap"))
            if row.family:
                excluded_families.add(row.family)
        else:
            by_text[key].append(row)
    pools: dict[str, list[RawRow]] = defaultdict(list)
    for candidates in by_text.values():
        if any(row.family in excluded_families for row in candidates):
            rejects.extend((row, "external_evaluation_family_overlap") for row in candidates)
            continue
        if len({row.label for row in candidates}) > 1:
            rejects.extend((row, "conflicting_labels") for row in candidates)
            continue
        pools[candidates[0].label].append(candidates[0])
        rejects.extend((row, "duplicate_text") for row in candidates[1:])
    selected: list[RawRow] = []
    for label in sorted(pools):
        pool = pools[label]
        # A seeded reservoir sees the complete pool rather than a file-order prefix.
        kept = pool[:cap]
        for index, row in enumerate(pool[cap:], cap):
            position = rng.randrange(index + 1)
            if position < cap:
                rejects.append((kept[position], "class_cap"))
                kept[position] = row
            else:
                rejects.append((row, "class_cap"))
        selected.extend(kept)
    counts = {
        "candidate_rows": inputs,
        "accepted_rows": len(selected),
        "rejected_rows": len(rejects),
    }
    if inputs != len(selected) + len(rejects):
        msg = f"Candidate conservation failed: {counts!r}"
        raise ValueError(msg)
    return selected, rejects, counts


def split_rows(rows: list[RawRow], seed: int) -> dict[str, list[RawRow]]:
    """Stratify on label/domain while keeping aligned renderings in one split."""
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[RawRow]] = defaultdict(list)
    for row in rows:
        cells[row.label, row.domain].append(row)
    splits: dict[str, list[RawRow]] = {"train": [], "dev": [], "test": []}
    assigned: dict[str, str] = {}
    for cell in sorted(cells):
        pool = cells[cell]
        rng.shuffle(pool)
        n = len(pool)
        heldout = max(1, round(n * 0.1)) if n >= 3 else 0
        targets = {"dev": heldout, "test": heldout, "train": n - 2 * heldout}
        remaining = targets.copy()
        for row in pool:
            family = row.family or hashlib.sha256(text_key(row.text).encode()).hexdigest()
            split = assigned.get(family)
            if split is None:
                split = max(remaining, key=lambda name: remaining[name] / max(1, targets[name]))
                assigned[family] = split
            splits[split].append(row)
            remaining[split] -= 1
    for split_pool in splits.values():
        rng.shuffle(split_pool)
    if sum(map(len, splits.values())) != len(rows):
        msg = "Split conservation failed"
        raise ValueError(msg)
    missing = set(COMPOSITE_CLASSES) - {row.label for row in rows}
    if missing:
        msg = f"Missing target classes: {sorted(missing)!r}"
        raise ValueError(msg)
    return splits


def class_counts(rows: Iterable[RawRow]) -> dict[str, int]:
    """Report all configured classes, including zero support."""
    counts = Counter(row.label for row in rows)
    return {label: counts[label] for label in COMPOSITE_CLASSES}
