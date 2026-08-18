"""Partition the mined NLLB en-kab pool by what two independent filters kept.

`boffire/nllb_en_kab` (2,786,012 pairs, method undocumented) and
`Imsidag-community/nllb_en_kab` (2,484,297, GlotLID v3 at 0.95 plus character
normalisation) both filter the same 4,123,481-pair pool. Neither reports precision, so
neither is gold; the output is a stratification for task 4.2 to sample, not a label.

Matching is on the **pair**, not the Kabyle side: 2,786,012 boffire rows carry only
404,487 distinct Kabyle sides, so a Kabyle-only key made 2.24M pool rows look
"kept by both" and left `imsidag-only` empty. The derivatives normalise Kabyle
characters, so the project normaliser is applied to that side of both; English is
compared as published.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from agbalu.extract.pipeline import fingerprint
from agbalu.extract.readers import read_records
from agbalu.normalise import Normaliser
from agbalu.parallel.quality import HARD_DEFECTS, inspect
from agbalu.parallel.readers import read_opus_zip

log: Final = logging.getLogger("agbalu.parallel")

Stratum = str

BOTH: Final[Stratum] = "kept-by-both"
BOFFIRE_ONLY: Final[Stratum] = "boffire-only"
IMSIDAG_ONLY: Final[Stratum] = "imsidag-only"
NEITHER: Final[Stratum] = "kept-by-neither"

STRATA: Final[tuple[Stratum, ...]] = (BOTH, BOFFIRE_ONLY, IMSIDAG_ONLY, NEITHER)


@dataclass
class StratumStats:
    """Pool pairs in one agreement stratum, and how defective they are."""

    name: Stratum
    pairs: int = 0
    hard_defective: int = 0
    any_defective: int = 0
    by_defect: Counter[str] = field(default_factory=Counter)

    @property
    def hard_rate(self) -> float:
        return self.hard_defective / self.pairs if self.pairs else 0.0

    @property
    def any_rate(self) -> float:
        return self.any_defective / self.pairs if self.pairs else 0.0


@dataclass
class AgreementReport:
    pool: int = 0
    boffire: int = 0
    imsidag: int = 0
    boffire_unmatched: int = 0
    imsidag_unmatched: int = 0
    strata: dict[Stratum, StratumStats] = field(default_factory=dict)

    @property
    def matched_rate(self) -> float:
        """Share of filter output that could be traced back to the raw pool.

        Below 1.0 the partition is incomplete: a derivative pair that does not match
        the pool is not evidence about any pool pair. The gap is normalisation —
        each group normalised Kabyle characters their own way, and ours is a third.
        """
        declared = self.boffire + self.imsidag
        if not declared:
            return 0.0
        return 1.0 - (self.boffire_unmatched + self.imsidag_unmatched) / declared

    @property
    def agreement_rate(self) -> float:
        """Share of the pool the two filters agree about, in either direction."""
        if not self.pool:
            return 0.0
        agreed = self.strata[BOTH].pairs + self.strata[NEITHER].pairs
        return agreed / self.pool

    @property
    def contested(self) -> int:
        return self.strata[BOFFIRE_ONLY].pairs + self.strata[IMSIDAG_ONLY].pairs


def pair_key(kab: str, english: str) -> bytes:
    """Identity of a pair. Both sides, because the Kabyle side is not unique."""
    return fingerprint(kab + "\x00" + english)


def _pair_keys(paths: list[Path], normaliser: Normaliser) -> set[bytes]:
    keys: set[bytes] = set()
    for path in paths:
        for record in read_records(path):
            kab = record.get("kabyle")
            english = record.get("english")
            if kab and english:
                keys.add(pair_key(normaliser.normalise(kab), english))
    return keys


def pool_pairs(zip_path: Path, normaliser: Normaliser) -> Iterator[tuple[str, str, bytes]]:
    """`(kabyle, english, key)` for the raw OPUS NLLB en-kab bundle."""
    for kab, foreign, code in read_opus_zip(zip_path):
        if code != "en":
            continue
        normalised = normaliser.normalise(kab)
        yield normalised, foreign, pair_key(normalised, foreign)


def analyse(
    pool_zip: Path,
    boffire: list[Path],
    imsidag: list[Path],
    normaliser: Normaliser | None = None,
    sample_out: Path | None = None,
) -> AgreementReport:
    engine = normaliser if normaliser is not None else Normaliser()
    kept_boffire = _pair_keys(boffire, engine)
    kept_imsidag = _pair_keys(imsidag, engine)
    log.info("filters boffire=%d imsidag=%d", len(kept_boffire), len(kept_imsidag))

    report = AgreementReport(boffire=len(kept_boffire), imsidag=len(kept_imsidag))
    report.strata = {name: StratumStats(name=name) for name in STRATA}
    pool_keys: set[bytes] = set()
    handle = sample_out.open("w", encoding="utf-8") if sample_out is not None else None

    try:
        for kab, eng, key in pool_pairs(pool_zip, engine):
            report.pool += 1
            pool_keys.add(key)
            in_boffire = key in kept_boffire
            in_imsidag = key in kept_imsidag
            if in_boffire and in_imsidag:
                name = BOTH
            elif in_boffire:
                name = BOFFIRE_ONLY
            elif in_imsidag:
                name = IMSIDAG_ONLY
            else:
                name = NEITHER

            stratum = report.strata[name]
            stratum.pairs += 1
            defects = inspect(kab, eng, "eng")
            if defects.defective:
                stratum.any_defective += 1
                for kind in defects.kinds:
                    stratum.by_defect[kind] += 1
            if any(k in HARD_DEFECTS for k in defects.kinds):
                stratum.hard_defective += 1

            if handle is not None:
                handle.write(
                    json.dumps(
                        {
                            "kab": kab,
                            "eng": eng,
                            "stratum": name,
                            "defects": list(defects.kinds),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if handle is not None:
            handle.close()

    report.boffire_unmatched = len(kept_boffire - pool_keys)
    report.imsidag_unmatched = len(kept_imsidag - pool_keys)
    return report
