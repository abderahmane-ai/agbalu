"""Validate a corpus registry and print a summary.

Usage:
    python -m agbalu.registry.cli resources/corpus_registry.yaml
    python -m agbalu.registry.cli --siblings resources/sibling_registry.yaml
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from agbalu.registry.loader import RegistryError, load_registry, load_sibling_registry
from agbalu.registry.models import Registry, SiblingRegistry


def summarise(registry: Registry | SiblingRegistry) -> str:
    by_modality = collections.Counter(s.modality for s in registry.sources)
    by_tier = collections.Counter(s.tier for s in registry.sources)
    by_redist = collections.Counter(s.redistribution for s in registry.sources)

    lines = [
        f"registry v{registry.version}  surveyed {registry.surveyed}",
        f"{len(registry.sources)} sources",
        "",
        "by modality:      " + ", ".join(f"{k}={v}" for k, v in sorted(by_modality.items())),
        "by tier:          " + ", ".join(f"{k}={v}" for k, v in sorted(by_tier.items())),
        "by redistribution:" + " " + ", ".join(f"{k}={v}" for k, v in sorted(by_redist.items())),
    ]

    hours = sum(s.size.hours or 0.0 for s in registry.sources)
    rows = sum(s.size.rows or 0 for s in registry.sources)
    if hours:
        lines.append(f"speech hours:     {hours:,.2f}")
    if rows:
        lines.append(f"rows (declared):  {rows:,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="path to the registry YAML")
    parser.add_argument(
        "--siblings",
        action="store_true",
        help="validate against the Berber sibling schema, which refuses Kabyle",
    )
    args = parser.parse_args(argv)

    registry: Registry | SiblingRegistry
    try:
        registry = load_sibling_registry(args.path) if args.siblings else load_registry(args.path)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(summarise(registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
