"""Acquisition CLI.

    python -m agbalu.acquire.cli plan
    python -m agbalu.acquire.cli fetch --tier core
    python -m agbalu.acquire.cli verify

`fetch` is idempotent and restartable: a source already recorded in the manifest
is skipped unless `--force` is given, so an interrupted run resumes by re-running.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, get_args

from huggingface_hub.utils.tqdm import disable_progress_bars

from agbalu.acquire.fetch import FetchError, fetch
from agbalu.acquire.manifest import (
    DEFERRALS_NAME,
    MANIFEST_NAME,
    REMOVALS_NAME,
    Manifest,
)
from agbalu.acquire.models import (
    Deferral,
    ManifestEntry,
    Removal,
    RemovalReason,
    VerifyFinding,
)
from agbalu.acquire.placement import source_target
from agbalu.acquire.storage import sha256_file
from agbalu.registry.loader import RegistryError, load_registry, load_sibling_registry
from agbalu.registry.models import Registry, SiblingRegistry, SourceBase, Tier

type SourceCatalogue = Registry | SiblingRegistry
"""Either registry. Acquisition reads only fields common to both source models."""

DEFAULT_REGISTRY: Final = Path("resources/corpus_registry.yaml")
DEFAULT_SIBLING_REGISTRY: Final = Path("resources/sibling_registry.yaml")
DEFAULT_ROOT: Final = Path("data/raw")

log: Final = logging.getLogger("agbalu.acquire")


def _select(
    registry: SourceCatalogue, *, tier: Tier | None, ids: list[str] | None
) -> list[SourceBase]:
    sources: list[SourceBase] = list(registry.sources)
    if tier is not None:
        sources = [s for s in sources if s.tier == tier]
    if ids:
        wanted = set(ids)
        sources = [s for s in sources if s.id in wanted]
    return sources


def _target_of(source: SourceBase) -> str:
    return source_target(tier=source.tier, modality=source.modality, source_id=source.id)


def command_plan(registry: SourceCatalogue, args: argparse.Namespace) -> int:
    sources = _select(registry, tier=args.tier, ids=args.id)
    counts: dict[str, int] = {"local": 0, "remote": 0, "none": 0}
    declared: dict[str, int] = {"local": 0, "remote": 0, "none": 0}
    for source in sorted(sources, key=lambda s: (s.tier, s.id)):
        target = _target_of(source)
        counts[target] += 1
        declared[target] += source.size.bytes or 0
        print(f"{target:7} {source.tier:14} {source.access:9} {source.id}")
    print("")
    for target in ("local", "remote", "none"):
        print(f"{target:7} {counts[target]:3} sources  {declared[target] / 1e9:8.2f} GB declared")
    return 0


def command_fetch(registry: SourceCatalogue, args: argparse.Namespace) -> int:
    manifest = Manifest(args.root)
    already = manifest.recorded_source_ids()
    sources = _select(registry, tier=args.tier, ids=args.id)
    now = datetime.now(UTC)
    failures = 0

    for source in sorted(sources, key=lambda s: s.id):
        target = _target_of(source)
        if source.id in already and not args.force:
            log.info("skip source=%s reason=already-recorded", source.id)
            continue
        if target == "none":
            continue
        if target == "remote" and not args.include_remote:
            manifest.defer(
                Deferral(
                    source_id=source.id,
                    reason="remote-target",
                    detail=(
                        f"{source.modality} payload targets the remote volume; "
                        f"{(source.size.bytes or 0) / 1e9:.2f} GB not landed locally"
                    ),
                    recorded_at=now,
                )
            )
            log.info("defer source=%s reason=remote-target", source.id)
            continue

        try:
            result = fetch(source, args.root / source.id)
        except (FetchError, OSError) as exc:
            failures += 1
            log.error("fail source=%s error=%s", source.id, exc)
            log.debug("traceback source=%s", source.id, exc_info=exc)
            manifest.defer(
                Deferral(
                    source_id=source.id,
                    reason="unavailable",
                    detail=str(exc)[:500],
                    recorded_at=datetime.now(UTC),
                )
            )
            continue

        for artifact in result.artifacts:
            manifest.append(
                ManifestEntry(
                    source_id=source.id,
                    path=artifact.path,
                    bytes=artifact.bytes,
                    sha256=artifact.sha256,
                    kind=artifact.kind,
                    target=artifact.target,
                    uri=source.uri,
                    fetched_at=datetime.now(UTC),
                    revision=result.revision,
                )
            )
        total = sum(a.bytes for a in result.artifacts)
        log.info(
            "done source=%s artifacts=%d bytes=%d rev=%s",
            source.id,
            len(result.artifacts),
            total,
            (result.revision or "-")[:12],
        )

    return 1 if failures else 0


MANIFEST_NAMES: Final = frozenset({MANIFEST_NAME, DEFERRALS_NAME, REMOVALS_NAME})
"""The provenance files themselves are not artifacts. Taken from `manifest`, because a
second copy of these names is how `removals.jsonl` came to be reported as stray bytes."""
IGNORED_DIRS: Final = frozenset({".git", ".cache", "__pycache__"})


def _unrecorded(
    root: Path, entries: Sequence[ManifestEntry], registry: SourceCatalogue
) -> list[VerifyFinding]:
    """Bytes on disk that no manifest entry claims.

    The manifest is append-only, so it can only ever prove that recorded files are
    intact. Without this walk in the opposite direction, an interrupted fetch leaves
    provenance-free data that `verify` reports as healthy (CLAUDE.md 2.7 rule 1).

    A `manual` source's declared file is its own provenance — the registry pins it
    by checksum — so the origin is not a stray even though the copy under the
    source id is what the manifest records.
    """
    known = {(root / e.source_id / e.path).resolve() for e in entries}
    known |= {Path(s.uri).resolve() for s in registry.sources if s.access == "manual"}
    findings: list[VerifyFinding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in MANIFEST_NAMES:
            continue
        relative = path.relative_to(root)
        if IGNORED_DIRS & set(relative.parts):
            continue
        if path.resolve() in known:
            continue
        head, _, tail = relative.as_posix().partition("/")
        findings.append(
            VerifyFinding(
                source_id=head,
                path=tail or relative.as_posix(),
                kind="unrecorded",
                detail=f"{path.stat().st_size:,} bytes on disk with no manifest entry",
            )
        )
    return findings


def command_verify(registry: SourceCatalogue, args: argparse.Namespace) -> int:
    manifest = Manifest(args.root)
    findings: list[VerifyFinding] = []
    entries = manifest.entries()

    removed = manifest.removed_artifacts()

    for entry in entries:
        if entry.target != "local":
            continue
        path = args.root / entry.source_id / entry.path
        if (entry.source_id, entry.path) in removed:
            if path.is_file():
                findings.append(
                    VerifyFinding(
                        source_id=entry.source_id,
                        path=entry.path,
                        kind="resurrected",
                        detail=f"recorded as removed but present at {path}",
                    )
                )
            continue
        if not path.is_file():
            findings.append(
                VerifyFinding(
                    source_id=entry.source_id,
                    path=entry.path,
                    kind="missing",
                    detail=f"recorded in the manifest but absent from {path}",
                )
            )
            continue
        digest, size = sha256_file(path)
        if digest != entry.sha256:
            findings.append(
                VerifyFinding(
                    source_id=entry.source_id,
                    path=entry.path,
                    kind="checksum-mismatch",
                    detail=f"manifest {entry.sha256[:12]}… disk {digest[:12]}… ({size:,} bytes)",
                )
            )

    landed = {e.source_id for e in entries}
    recorded = manifest.recorded_source_ids()
    for source in registry.sources:
        if _target_of(source) == "none" or source.id in recorded:
            continue
        findings.append(
            VerifyFinding(
                source_id=source.id,
                path=None,
                kind="unfetched",
                detail=f"{source.tier}-tier source is neither landed nor deferred",
            )
        )

    findings.extend(_unrecorded(args.root, entries, registry))

    for finding in findings:
        where = f"{finding.source_id}/{finding.path}" if finding.path else finding.source_id
        print(f"{finding.kind:18} {where}: {finding.detail}")

    print("")
    print(
        f"{len(entries):,} artifacts across {len(landed)} sources, "
        f"{len(manifest.active_deferrals())} open deferrals, "
        f"{len(removed)} removed, {len(findings)} findings"
    )
    return 1 if findings else 0


def command_remove(registry: SourceCatalogue, args: argparse.Namespace) -> int:
    """Record that landed bytes were deleted, for artifacts already gone from disk.

    The deletion itself is the operator's; this writes the row that stops `verify`
    reporting the absence forever. An artifact still on disk is refused, so the record
    can never claim a file is gone while it is being read.
    """
    if args.source not in {source.id for source in registry.sources}:
        print(f"error: {args.source} is not in the registry", file=sys.stderr)
        return 1

    manifest = Manifest(args.root)
    wanted = set(args.path)
    entries = [
        entry
        for entry in manifest.entries()
        if entry.source_id == args.source and entry.path in wanted
    ]
    unknown = wanted - {entry.path for entry in entries}
    if unknown:
        print(f"error: not in the manifest under {args.source}: {sorted(unknown)}", file=sys.stderr)
        return 1

    present = [e for e in entries if (args.root / e.source_id / e.path).is_file()]
    if present:
        listing = ", ".join(e.path for e in present)
        print(f"error: still on disk, delete before recording: {listing}", file=sys.stderr)
        return 1

    now = datetime.now(UTC)
    for entry in entries:
        manifest.record_removal(
            Removal(
                source_id=entry.source_id,
                path=entry.path,
                sha256=entry.sha256,
                reason=args.reason,
                detail=args.detail,
                removed_at=now,
            )
        )
        print(f"removed {entry.source_id}/{entry.path}  {entry.bytes:,} bytes  {args.reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared flags live on a parent so they may follow the subcommand, which is
    # the order everyone types: `plan --tier core`, not `--tier core plan`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", type=Path, default=None)
    common.add_argument(
        "--siblings",
        action="store_true",
        help="operate on the Berber sibling registry instead of the Kabyle one",
    )
    common.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    common.add_argument("--tier", choices=("core", "supplementary", "reference", "excluded"))
    common.add_argument("--id", action="append", help="restrict to these source ids")

    parser = argparse.ArgumentParser(prog="agbalu-acquire", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", parents=[common], help="show what would be fetched and where")
    fetch_parser = sub.add_parser(
        "fetch", parents=[common], help="retrieve sources and record provenance"
    )
    fetch_parser.add_argument("--force", action="store_true", help="re-fetch recorded sources")
    fetch_parser.add_argument(
        "--include-remote", action="store_true", help="also fetch remote-target sources locally"
    )
    sub.add_parser("verify", parents=[common], help="re-hash landed bytes against the manifest")
    remove_parser = sub.add_parser(
        "remove", parents=[common], help="record landed bytes as deliberately deleted"
    )
    remove_parser.add_argument("--source", required=True, help="source id owning the artifacts")
    remove_parser.add_argument(
        "--path", action="append", required=True, help="artifact path, repeatable"
    )
    remove_parser.add_argument("--reason", required=True, choices=get_args(RemovalReason))
    remove_parser.add_argument("--detail", required=True, help="why, in one sentence")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stderr
    )
    # One event per line is the contract (CLAUDE.md 3.7). The HTTP stack narrates
    # every redirect, and tqdm redraws a bar per chunk into a non-TTY log file.
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if not sys.stderr.isatty():
        disable_progress_bars()
    args = build_parser().parse_args(argv)
    default = DEFAULT_SIBLING_REGISTRY if args.siblings else DEFAULT_REGISTRY
    path = default if args.registry is None else args.registry
    registry: SourceCatalogue
    try:
        registry = load_sibling_registry(path) if args.siblings else load_registry(path)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    match args.command:
        case "plan":
            return command_plan(registry, args)
        case "fetch":
            return command_fetch(registry, args)
        case "verify":
            return command_verify(registry, args)
        case "remove":
            return command_remove(registry, args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
