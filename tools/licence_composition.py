"""What licences a built artifact's text actually came under.

A release has to state a licence for weights trained on 42 sources, and the per-source
`kept` counts in a `*.stats.json` are the only record of how much each one contributed.
Joining them to the registry gives the composition the model card has to disclose.

The number that matters is the `unclear` share: it is not a licence, it is the absence of
one, and no permissive grant on the weights can convert it into permission over the text.

    python3 -m tools.licence_composition [--stats PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Final, TypedDict

from agbalu.registry.loader import load_registry

STATS: Final = Path("data/processed/text/agbalu-text-v1.stats.json")
REGISTRY: Final = Path("resources/corpus_registry.yaml")


class Composition(TypedDict):
    stats: str
    total: int
    by_redistribution: dict[str, int]
    by_licence: dict[str, int]
    by_tier: dict[str, int]
    unregistered: list[str]


def compose(stats_path: Path, registry_path: Path = REGISTRY) -> Composition:
    """Sentences kept, grouped by the licence terms of the source they came from."""
    sources = {s.id: s for s in load_registry(registry_path).sources}
    rows = json.loads(stats_path.read_text(encoding="utf-8"))["sources"]

    redistribution: Counter[str] = Counter()
    licence: Counter[str] = Counter()
    tier: Counter[str] = Counter()
    unregistered: list[str] = []

    for row in rows:
        source = sources.get(row["source_id"])
        if source is None:
            unregistered.append(row["source_id"])
            continue
        kept = int(row["kept"])
        redistribution[source.redistribution] += kept
        licence[source.licence] += kept
        tier[source.tier] += kept

    return {
        "stats": str(stats_path),
        "total": sum(redistribution.values()),
        "by_redistribution": dict(redistribution.most_common()),
        "by_licence": dict(licence.most_common()),
        "by_tier": dict(tier.most_common()),
        "unregistered": sorted(unregistered),
    }


def _report(composition: Composition) -> None:
    total = composition["total"]
    print(f"{composition['stats']}\n{total:,} sentences kept\n")
    groups = (
        ("redistribution", composition["by_redistribution"]),
        ("licence", composition["by_licence"]),
        ("tier", composition["by_tier"]),
    )
    for label, counts in groups:
        print(f"-- {label} --")
        for name, count in counts.items():
            print(f"   {name:<22}{count:>10,}{100 * count / total:>8.1f}%")
        print()
    if composition["unregistered"]:
        print("not in the registry:", ", ".join(composition["unregistered"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=STATS)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--json", action="store_true", help="emit the composition as JSON")
    args = parser.parse_args(argv)

    if not args.stats.is_file():
        message = f"{args.stats} not found; build the artifact first"
        raise SystemExit(message)

    composition = compose(args.stats, args.registry)
    if args.json:
        print(json.dumps(composition, indent=2, ensure_ascii=False))
    else:
        _report(composition)
    return 0


if __name__ == "__main__":
    sys.exit(main())
