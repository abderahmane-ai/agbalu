"""Turn landed raw artifacts into the clean Kabyle text corpus.

An artifact that cannot be parsed is skipped with a recorded error rather than
crashing the build. A column is only accepted as Kabyle on evidence.
"""

from agbalu.extract.columns import choose_field
from agbalu.extract.detect import kabyle_score
from agbalu.extract.pipeline import (
    CorpusBuilder,
    SourceStats,
    fingerprint,
    release_priority,
    summary,
)
from agbalu.extract.readers import read_records

__all__ = [
    "CorpusBuilder",
    "SourceStats",
    "choose_field",
    "fingerprint",
    "kabyle_score",
    "read_records",
    "release_priority",
    "summary",
]
