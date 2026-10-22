"""The speech corpus and the CTC target vocabulary for Fadhma (Phase 5).

Common Voice ships transcripts as raw contributor text, which carries the same
homoglyph substitution as every other Kabyle source. A CTC vocabulary is derived
from the distinct characters of its own transcripts, so an unnormalised inventory
makes Greek epsilon and Latin open-e two separate output classes and the model
reproduces the defect at the scale of the corpus. Normalisation therefore happens
before the inventory is taken, not after training.
"""

from agbalu.speech.corpus import Clip, CorpusReport, SpeechError, build, read_durations
from agbalu.speech.metrics import MetricError, Score, cer, wer
from agbalu.speech.vocabulary import Vocabulary, ctc_target

__all__ = [
    "Clip",
    "CorpusReport",
    "MetricError",
    "Score",
    "SpeechError",
    "Vocabulary",
    "build",
    "cer",
    "ctc_target",
    "read_durations",
    "wer",
]
