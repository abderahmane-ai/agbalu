"""AƔBALU-Parallel: aligned Kabyle sentence pairs with provenance and defect labels.

Both `kab-eng` and `kab-fra` are first class. Defective pairs are labelled rather
than dropped, because the rate of mechanical defects in the mined pool is the
measurement this layer exists to produce.
"""

from agbalu.parallel.columns import choose_pair
from agbalu.parallel.langid import ForeignLang, identify
from agbalu.parallel.pipeline import (
    ParallelBuilder,
    ParallelSummary,
    SourceStats,
    summary,
)
from agbalu.parallel.quality import PairDefects, inspect, length_ratio
from agbalu.parallel.readers import read_opus_zip, read_record_pairs

__all__ = [
    "ForeignLang",
    "PairDefects",
    "ParallelBuilder",
    "ParallelSummary",
    "SourceStats",
    "choose_pair",
    "identify",
    "inspect",
    "length_ratio",
    "read_opus_zip",
    "read_record_pairs",
    "summary",
]
