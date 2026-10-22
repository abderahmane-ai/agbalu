"""The sovereign Kabyle speech recognition package (Fadhma 2).

Combines:
- Multi-source manifest curation and Dihya-5M language filtering
- 90/5/5 speaker-stratified disjoint partitioning
- Canonical CTC vocabulary with Kabyle orthographic validation
- Multi-condition acoustic training (reverb, noise, telephony, SpecAugment)
- 600M Conformer acoustic model with dual-head CTC and speech-presence supervision
- Autonomous decoding with hard silence blank masking and KenLM shallow fusion
"""

from agbalu.speech.augment import (
    AcousticAugmenter,
    add_noise,
    apply_telephony_filter,
    convolve_rir,
    spec_augment,
)
from agbalu.speech.dataset import (
    DynamicAudioBatchSampler,
    SpeechDataset,
    collate_speech_batch,
    load_audio_file,
)
from agbalu.speech.decode import SpeechDecoder, TranscriptionResult
from agbalu.speech.evaluate import (
    EvaluationScore,
    FullBenchmarkReport,
    SilenceChallengeResult,
    evaluate_manifest,
    evaluate_silence_challenge,
    run_full_benchmark,
)
from agbalu.speech.manifest import (
    AudioRecord,
    ManifestReport,
    filter_with_dihya,
    ingest_common_voice_tsv,
    ingest_hf_parquet_dataset,
    ingest_tatoeba_tsv,
    ingest_tutlayt_jsonl,
    load_manifest,
    save_manifest,
)
from agbalu.speech.metrics import MetricError, Score, cer, wer
from agbalu.speech.model import ConformerSpeechModel, SpeechModelOutput
from agbalu.speech.partition import (
    PartitionReport,
    SplitDataset,
    partition_records,
    verify_speaker_isolation,
)
from agbalu.speech.smoke import SmokeReport, run_smoke_drill
from agbalu.speech.train import TrainingConfig, train
from agbalu.speech.vocabulary import (
    PAD_TOKEN,
    UNK_TOKEN,
    WORD_DELIMITER,
    Vocabulary,
    ctc_target,
    decode,
    encoder,
)

__all__ = [
    "PAD_TOKEN",
    "UNK_TOKEN",
    "WORD_DELIMITER",
    "AcousticAugmenter",
    "AudioRecord",
    "ConformerSpeechModel",
    "DynamicAudioBatchSampler",
    "EvaluationScore",
    "FullBenchmarkReport",
    "ManifestReport",
    "MetricError",
    "PartitionReport",
    "Score",
    "SilenceChallengeResult",
    "SmokeReport",
    "SpeechDataset",
    "SpeechDecoder",
    "SpeechModelOutput",
    "SplitDataset",
    "TrainingConfig",
    "TranscriptionResult",
    "Vocabulary",
    "add_noise",
    "apply_telephony_filter",
    "cer",
    "collate_speech_batch",
    "convolve_rir",
    "ctc_target",
    "decode",
    "encoder",
    "evaluate_manifest",
    "evaluate_silence_challenge",
    "filter_with_dihya",
    "ingest_common_voice_tsv",
    "ingest_hf_parquet_dataset",
    "ingest_tatoeba_tsv",
    "ingest_tutlayt_jsonl",
    "load_audio_file",
    "load_manifest",
    "partition_records",
    "run_full_benchmark",
    "run_smoke_drill",
    "save_manifest",
    "spec_augment",
    "train",
    "verify_speaker_isolation",
    "wer",
]
