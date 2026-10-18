"""The stamp every Phase 11 artifact carries: code revision, config hash, input digests.

The HPC copy of the tree is synced without `.git`, so there the revision comes from the
`STAMP.json` that `tools.launch` writes from the laptop's checkout at every sync.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Final

STAMP_FILE: Final = "STAMP.json"


class ProvenanceError(Exception):
    """The code revision cannot be established."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_stamp(root: Path) -> dict[str, object]:
    """The checkout's revision and whether the working tree differs from it."""
    return {
        "git_sha": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }


def code_stamp(root: Path = Path()) -> dict[str, object]:
    if (root / ".git").exists():
        return git_stamp(root)
    stamp = root / STAMP_FILE
    if stamp.is_file():
        payload = json.loads(stamp.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("git_sha"), str):
            return {"git_sha": payload["git_sha"], "dirty": bool(payload.get("dirty"))}
    message = f"no .git and no {STAMP_FILE} under {root.resolve()}; sync with tools.launch"
    raise ProvenanceError(message)


def config_hash(config: Mapping[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def file_digest(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()
