"""Orthographic normalisation for Kabyle Latin script.

The corpus carries a measured 0.0875% codepoint corruption rate, concentrated in
open-e: 11.21% of all `ɛ`-class characters in Kabyle running text are Greek
epsilon. This package specifies and applies the fix.

Two principles govern everything here:

1. **Never delete a character you cannot justify.** Rules that would merge
   distinct phonemes are flagged for review, not applied. See `ţ` in
   `resources/homoglyphs.yaml`.
2. **Normalisation is versioned.** `NORMALISER_VERSION` changes whenever output
   changes for any input, because every downstream artifact depends on it.
"""

from agbalu.normalise.models import Change, Flag, NormalisationResult
from agbalu.normalise.normaliser import NORMALISER_VERSION, Normaliser, normalise

__all__ = [
    "NORMALISER_VERSION",
    "Change",
    "Flag",
    "NormalisationResult",
    "Normaliser",
    "normalise",
]
