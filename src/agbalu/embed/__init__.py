"""Kabyle sentence embeddings: the vocabulary repair, the pair corpus, and the harness.

The published Kabyle sentence transformer reports mean cosine similarity between
aligned pairs as its only number, which a collapsed encoder maximises. Every metric
here is reported beside a control that collapse fails.
"""

from agbalu.embed.backbone import CANDIDATES, Repair, widen
from agbalu.embed.corpus import CorpusStats, Pair, build_embed_corpus, split_clusters
from agbalu.embed.vocabulary import Coverage, VocabularyError, assert_covered, coverage

__all__ = [
    "CANDIDATES",
    "CorpusStats",
    "Coverage",
    "Pair",
    "Repair",
    "VocabularyError",
    "assert_covered",
    "build_embed_corpus",
    "coverage",
    "split_clusters",
    "widen",
]
