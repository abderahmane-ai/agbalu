"""Rewriting a Kabyle reading into the symbols the base model's token table already has.

Kokoro declares `n_token: 178` with 114 symbols mapped, so the 41-symbol Kabyle inventory
is short four symbols against 64 unused rows. Two of the four cost nothing: the affricate
tie bar disappears when `t͡ʃ` and `d͡ʒ` are folded onto the precomposed symbols the table
already carries, which takes the shortfall to three new rows that fine-tuning trains.

The base drops a symbol it cannot represent instead of raising — its English G2P is
constructed with `unk=''` and the documentation says out-of-dictionary words are skipped,
the same silent-deletion defect the published lexicon carried one layer down. Assigning
the three rows needs the base's real token table, which is loaded where synthesis happens;
what 12.4 owes that step is the corpus's **observed** inventory, recorded in its stats
rather than asserted from this table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

FOLD: Final[Mapping[str, str]] = {"t͡ʃ": "ʧ", "d͡ʒ": "ʤ"}
"""Tie-bar sequences the base already carries as single symbols. Folding them is what
takes the shortfall from four symbols to three."""


def fold(ipa: str) -> str:
    """Rewrite a phoneme string into the base's own symbols."""
    for sequence, symbol in FOLD.items():
        ipa = ipa.replace(sequence, symbol)
    return ipa
