"""Auditing the benchmarks everything else is measured against.

FLORES+ is the only Kabyle MT benchmark, and `kab_Latn` has never been revised:
every sentence still reads `last_updated: 1.0`, while the sibling `zgh_Tfng` was
corrected by expert linguists for WMT 2025. An unaudited reference makes every
score reported against it unreliable, so this runs before any model is trained.
"""

from agbalu.bench.audit import AuditReport, SentenceDiff, audit, token_divergence
from agbalu.bench.contamination import ContaminationReport, Leak, scan
from agbalu.bench.flores import Sentence, read_all, read_split, revisions

__all__ = [
    "AuditReport",
    "ContaminationReport",
    "Leak",
    "Sentence",
    "SentenceDiff",
    "audit",
    "read_all",
    "read_split",
    "revisions",
    "scan",
    "token_divergence",
]
