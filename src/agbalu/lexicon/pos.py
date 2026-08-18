"""Hunspell's Kabyle part-of-speech labels, mapped to UPOS.

`hunspell-kab`'s `po:` labels are undocumented Kabyle terms. Each mapping was read off
the words carrying the label, not off the label's name; the exemplars are tabulated in
`docs/phases/phase-06-lexicon.md` §POS.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from agbalu.lexicon.models import Upos

HUNSPELL_UPOS: Final[Mapping[str, Upos]] = {
    "isem": "NOUN",
    "amyag": "VERB",
    "isem_n_tigawt": "NOUN",
    "isem_n_umdan": "PROPN",
    "isem_n_tmurt": "PROPN",
    "isem_n_umkan": "PROPN",
    "isem_uzzig": "PROPN",
    "isem_amḍan": "NUM",
    "isem_asinan": "NOUN",
    "isem_n_ssaɛa": "NUM",
    "arbib": "ADJ",
    "amernu": "ADV",
    "tanzeɣt": "ADP",
    "amqim": "PRON",
    "ameskan": "DET",
    "aferdis_n_ubhat": "INTJ",
    "tazelɣa": "PART",
    "tazelɣa_n_usiwel": "PART",
    "awṣil": "X",
    "tissi": "X",
}

UNMAPPED: Final = "X"
"""An unseen label must not inherit the meaning of whichever tag was the default."""


def upos_for(label: str) -> Upos:
    return HUNSPELL_UPOS.get(label, UNMAPPED)


POS_CONFIDENCE: Final[Mapping[str, int]] = {
    "hf.boffire.hunspell-kab": 0,
    "hf.boffire.kabyle-verbs": 1,
    "hf.boffire.kabyle-toponyms": 2,
}
"""Tag authority, lower being better.

Majority voting across sources let the 345,057-entry verb table and the gazetteer
outvote 9,474 hand-written `po:` labels; the verb table can only ever say `VERB` and
the gazetteer only `PROPN`.
"""

DEFAULT_CONFIDENCE: Final = 3


def pos_confidence(source: str) -> int:
    return POS_CONFIDENCE.get(source, DEFAULT_CONFIDENCE)
