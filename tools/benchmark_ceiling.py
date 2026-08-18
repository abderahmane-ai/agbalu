"""What a flawless Kabyle translation can score on FLORES+ as published.

The oracle is task 7.0's corrected reference — a perfect translation that also spells
Kabyle correctly. Scoring it against the uncorrected reference gives the ceiling every
published Kabyle MT number sits under.

    python tools/benchmark_ceiling.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from agbalu.bench.flores import DEFAULT_ROOT, KABYLE, SPLITS, Split, read_split
from agbalu.bench.mt import Direction, score

CORRECTED: Final = Path("data/processed/bench/flores-plus-kab_Latn-corrected.jsonl")
OUT: Final = Path("data/processed/bench/ceiling-kab_Latn.json")
DIRECTION: Final[Direction] = "eng-kab"


def oracle(split: Split) -> list[str]:
    rows = [json.loads(line) for line in CORRECTED.read_text(encoding="utf-8").splitlines() if line]
    return [
        r["text"] for r in sorted((r for r in rows if r["split"] == split), key=lambda r: r["id"])
    ]


def main() -> int:
    report: dict[str, object] = {}
    for split in SPLITS:
        references = [
            s.text for s in sorted(read_split(DEFAULT_ROOT, split, KABYLE), key=lambda s: s.id)
        ]
        result = score(oracle(split), references, DIRECTION, split)
        report[split] = {
            "sentences": result.sentences,
            "normaliser_version": result.normaliser_version,
            "ceiling": {m.name: m.score for m in result.raw.metrics},
            "orthography_gap": {n: round(result.gap(n), 4) for n in ("chrf++", "bleu", "spbleu")},
        }
        print(f"{split}: {result.sentences:,} sentences")
        for metric in result.raw.metrics:
            lost = result.gap(metric.name)
            print(f"  {metric.name:8} ceiling {metric.score:8.4f}   lost {lost:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
