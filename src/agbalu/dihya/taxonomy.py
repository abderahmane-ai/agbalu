"""Define language varieties, writing systems, and label spaces for Pan-Amazigh LID.

Provides taxonomic hierarchies, script detection invariants, and label mappings
across 15 Amazigh varieties, 4 scripts, and 5 negative distractor classes.
"""

from __future__ import annotations

import unicodedata
from typing import Final, Literal

Script = Literal["Latn", "Tfng", "Arab", "Arabizi", "Hani"]
Domain = Literal[
    "scripture",
    "encyclopaedia",
    "lexicon",
    "oral_heritage",
    "everyday",
    "interface",
    "web",
    "academic",
    "negative_dialect",
    "negative_standard",
    "negative_foreign",
]

# The 15 recognized Amazigh varieties covering all branches
AMAZIGH_VARIETIES: Final[tuple[str, ...]] = (
    "kab",  # Kabyle (Taqbaylit)
    "shy",  # Shawiya (Tacawit)
    "cnu",  # Shenwa & Dahra (Tachenwit)
    "mzb",  # Mozabite (Tumẓabt)
    "oua",  # Ouargli & Gourara (Teggargrent / Taznatit)
    "shi",  # Tashelhit (Taclḥit)
    "tzm",  # Central Atlas Tamazight
    "rif",  # Tarifit (Tmaziɣt n Arrif)
    "zgh",  # Standard Moroccan Tamazight (IRCAM)
    "taq",  # Tamasheq (Tuareg, Mali/Niger)
    "thv",  # Tamahaq / Tahaggart (Algerian Hoggar / Djanet)
    "jbn",  # Nafusi & Zuwarah (Jebel Nafusa, Libya)
    "gha",  # Ghadamès (Libya)
    "siz",  # Siwi (Siwa Oasis, Egypt)
    "zen",  # Zenaga (Mauritania)
)

SCRIPTS: Final[tuple[Script, ...]] = ("Latn", "Tfng", "Arab", "Arabizi", "Hani")

NEGATIVE_LANGUAGES: Final[tuple[str, ...]] = (
    "arq",  # Algerian Arabic / Darija
    "ary",  # Moroccan Arabic / Darija
    "ara",  # Modern Standard Arabic
    "fra",  # French
    "eng",  # English
    "spa",  # Spanish
    "zho",  # Chinese (Mandarin)
)

NOT_AMAZIGH: Final = "NOT_AMAZIGH"

# Primary composite labels formatted as ISO639-3_ISO15924
COMPOSITE_CLASSES: Final[tuple[str, ...]] = (
    # Northern Algeria
    "kab_Latn",
    "kab_Tfng",
    "shy_Latn",
    "mzb_Latn",
    # Northern Morocco
    "shi_Latn",
    "shi_Tfng",
    "tzm_Latn",
    "tzm_Tfng",
    "rif_Latn",
    "rif_Tfng",
    "zgh_Tfng",
    # Saharan / Tuareg
    "taq_Latn",
    "taq_Tfng",
    # Negatives (North African Dialects & Standard Arabic)
    "arq_Arab",
    "arq_Arabizi",
    "ary_Arab",
    "ary_Arabizi",
    "ara_Arab",
    # Negatives (Foreign World Languages)
    "fra_Latn",
    "eng_Latn",
    "spa_Latn",
    "zho_Hani",
)

# Unicode code point boundaries for deterministic script detection
TIFINAGH_START: Final = 0x2D30
TIFINAGH_END: Final = 0x2D7F
ARABIC_START: Final = 0x0600
ARABIC_END: Final = 0x06FF
HANI_START: Final = 0x4E00
HANI_END: Final = 0x9FFF

# Arabizi numeral mapping frequently used for Arabic consonants in North African social text
ARABIZI_NUMERALS: Final[frozenset[str]] = frozenset({"2", "3", "5", "7", "9"})


def letter_script(char: str) -> str | None:
    """Identify a letter's script without treating punctuation as linguistic evidence."""
    if not unicodedata.category(char).startswith("L"):
        return None
    name = unicodedata.name(char, "")
    for prefix, script in (("LATIN", "Latn"), ("TIFINAGH", "Tfng"), ("ARABIC", "Arab")):
        if prefix in name:
            return script
    if "CJK" in name or "IDEOGRAPHIC" in name:
        return "Hani"
    # These modifier letters encode labialization and glottal articulation in the corpora.
    if char in "ʷʸʼʾʿˤˀ":
        return "Latn"
    return "Other"


def script_error(text: str, label: str, declared: str = "") -> str | None:
    """Return the first incompatible letter and offset for a composite label."""
    if label not in COMPOSITE_CLASSES:
        return f"unknown label {label!r}"
    expected = label.rsplit("_", 1)[1]
    if declared and declared != expected:
        return f"declared script {declared!r} != label script {expected!r}"
    unicode_script = "Latn" if expected == "Arabizi" else expected
    letters = 0
    for offset, char in enumerate(text):
        observed = letter_script(char)
        if observed is None:
            continue
        letters += 1
        if observed != unicode_script:
            return (
                f"letter {char!r} U+{ord(char):04X} at char {offset}: "
                f"script {observed} != {unicode_script}"
            )
    return None if letters else "no script-bearing letters"


def detect_dominant_script(text: str) -> Script:
    """Determine the primary script of a string via character property heuristics.

    Args:
        text: Raw input string.

    Returns:
        One of 'Latn', 'Tfng', 'Arab', 'Arabizi', or 'Hani'.
    """
    counts: dict[str, int] = {"Latn": 0, "Tfng": 0, "Arab": 0, "Hani": 0, "Arabizi_num": 0}
    cased_chars = 0

    for char in text:
        code = ord(char)
        if not char.isalpha():
            if char in ARABIZI_NUMERALS:
                counts["Arabizi_num"] += 1
            continue
        if TIFINAGH_START <= code <= TIFINAGH_END:
            counts["Tfng"] += 1
        elif ARABIC_START <= code <= ARABIC_END:
            counts["Arab"] += 1
        elif HANI_START <= code <= HANI_END:
            counts["Hani"] += 1
        else:
            cat = unicodedata.category(char)
            if cat.startswith("L") and letter_script(char) == "Latn":
                counts["Latn"] += 1
                cased_chars += 1
            elif char in ARABIZI_NUMERALS:
                counts["Arabizi_num"] += 1

    total_script = counts["Tfng"] + counts["Arab"] + counts["Latn"] + counts["Hani"]
    if total_script == 0 or not text:
        return "Latn"

    if counts["Hani"] / total_script > 0.40:
        return "Hani"
    if counts["Tfng"] / total_script > 0.40:
        return "Tfng"
    if counts["Arab"] / total_script > 0.40:
        return "Arab"

    # Identify Arabizi by co-occurrence of Latin characters and digit phonemes
    if counts["Latn"] > 0 and counts["Arabizi_num"] >= 2:
        num_ratio = counts["Arabizi_num"] / (counts["Latn"] + counts["Arabizi_num"])
        if num_ratio > 0.08:
            return "Arabizi"

    return "Latn"


def is_amazigh_label(label: str) -> bool:
    """Check whether a composite label belongs to the Amazigh language family.

    Args:
        label: Composite label string.

    Returns:
        True if the label is Amazigh, False if negative/distractor.
    """
    if label == NOT_AMAZIGH:
        return False
    lang = label.split("_")[0]
    return lang in AMAZIGH_VARIETIES
