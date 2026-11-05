"""Synthesize reference-conditioned speech from the staged model directory."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from agbalu.codec_tts.cli import main as run

    return run(
        [
            "synthesize",
            "--config",
            str(root / "recipe.json"),
            "--inference-config",
            str(root / "inference.json"),
            *sys.argv[1:],
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
