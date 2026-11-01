"""Independently audit written splits for script errors, support and leakage."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

from agbalu.dihya.prepare import read_records
from agbalu.dihya.taxonomy import COMPOSITE_CLASSES
from agbalu.llm.provenance import file_digest


def canonical(text: str) -> str:
    """Normalize comparison keys independently of the production split allocator."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def verify_conservation(candidate_rows: int, accepted_rows: int, rejected_rows: int) -> None:
    """Reject any unexplained candidate loss."""
    if (
        min(candidate_rows, accepted_rows, rejected_rows) < 0
        or candidate_rows != accepted_rows + rejected_rows
    ):
        msg = (
            f"Conservation: input {candidate_rows} != kept {accepted_rows} "
            f"+ rejected {rejected_rows}"
        )
        raise ValueError(msg)


def audit_directory(directory: Path) -> dict[str, object]:
    """Read actual artifact contents without using dataset validation or split code."""
    errors: list[str] = []
    splits: dict[str, object] = {}
    texts: dict[str, set[str]] = {}
    families: dict[str, set[str]] = {}
    total = 0
    exclusions_path = directory / "exclusions.json"
    exclusions: set[str] = set()
    if exclusions_path.is_file():
        exclusions = set(json.loads(exclusions_path.read_text(encoding="utf-8")))
    for name in ("train", "dev", "test"):
        path = directory / f"{name}.parquet"
        if path.exists():
            rows = read_records(path)
        else:
            path = directory / f"{name}.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        labels: Counter[str] = Counter()
        cells: Counter[str] = Counter()
        sources: Counter[str] = Counter()
        bad_scripts: Counter[str] = Counter()
        seen: set[str] = set()
        family_keys: set[str] = set()
        evaluation_overlap = 0
        for index, row in enumerate(rows):
            text, label, domain, source = (
                row[field] for field in ("text", "label", "domain", "source")
            )
            if not all(isinstance(value, str) for value in (text, label, domain, source)):
                msg = f"{path}:row:{index}: invalid fields {row!r}"
                raise ValueError(msg)
            if (
                not isinstance(text, str)
                or not isinstance(label, str)
                or not isinstance(domain, str)
                or not isinstance(source, str)
            ):
                msg = f"{path}:row:{index}: fields must be strings"
                raise TypeError(msg)
            labels[label] += 1
            cells[f"{label}/{domain}"] += 1
            sources[source] += 1
            key = canonical(text)
            if key in seen:
                errors.append(f"{path}:row:{index}: duplicate text {text!r}")
            seen.add(key)
            reference_key = " ".join(
                re.sub(r"[\W_]", " ", unicodedata.normalize("NFC", text).casefold()).split()
            )
            evaluation_overlap += int(reference_key in exclusions)
            family = row.get("family")
            if isinstance(family, str) and family:
                family_keys.add(family)
            expected = label.rsplit("_", 1)[-1]
            expected_name = (
                "LATIN"
                if expected in {"Latn", "Arabizi"}
                else {"Tfng": "TIFINAGH", "Arab": "ARABIC", "Hani": "CJK"}.get(expected, "INVALID")
            )
            incompatible = []
            for offset, char in enumerate(text):
                if not char.isalpha():
                    continue
                unicode_name = unicodedata.name(char, "")
                if expected_name not in unicode_name and not (
                    expected_name == "LATIN" and char in "ʷʸʼʾʿˤˀ"
                ):
                    incompatible.append((offset, char))
            if incompatible or row.get("script") != expected:
                bad_scripts[f"{label}/{source}"] += 1
        if bad_scripts:
            errors.append(f"{path}: {sum(bad_scripts.values())} script-inconsistent rows")
        if evaluation_overlap:
            errors.append(f"{path}: {evaluation_overlap} external evaluation overlaps")
        missing = set(COMPOSITE_CLASSES) - set(labels)
        if missing:
            errors.append(f"{path}: missing classes {sorted(missing)!r}")
        if set(labels) - set(COMPOSITE_CLASSES):
            errors.append(f"{path}: unknown classes")
        splits[name] = {
            "rows": len(rows),
            "sha256": file_digest(path),
            "classes": dict(labels),
            "cells": dict(cells),
            "sources": dict(sources),
            "script_errors": dict(bad_scripts),
            "external_evaluation_overlap": evaluation_overlap,
            "tiny_cells": {cell: count for cell, count in cells.items() if count < 20},
        }
        texts[name], families[name] = seen, family_keys
        total += len(rows)
    for left, right in combinations(texts, 2):
        for kind, overlap in (
            ("text", texts[left] & texts[right]),
            ("family", families[left] & families[right]),
        ):
            if overlap:
                errors.append(f"{left}/{right}: {len(overlap)} overlapping {kind} keys")
    manifest_path = directory / "dataset.stats.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rejects_path = directory / "rejects.jsonl"
        with rejects_path.open(encoding="utf-8") as stream:
            rejects = sum(1 for line in stream if line.strip())
        inputs = manifest["conservation"]["candidate_rows"]
        verify_conservation(inputs, total, rejects)
        if file_digest(rejects_path) != manifest["rejects_sha256"]:
            errors.append(f"{rejects_path}: digest mismatch")
        errors.extend(
            f"{name}: manifest digest mismatch"
            for name in texts
            if file_digest(directory / f"{name}.parquet") != manifest["splits"][name]["sha256"]
        )
    return {
        "passed": not errors,
        "errors": errors,
        "splits": splits,
        "total_rows": total,
        "script_policy": "all letters must match the declared composite script",
        "release_ready": False,
    }
