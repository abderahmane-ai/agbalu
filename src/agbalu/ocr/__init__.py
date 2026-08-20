"""Document & Book Line OCR (Vision-Encoder-Decoder).

Published as `agbalu/Feraoun-36M`. Provides optical character recognition for scanned
Kabyle books, historical manuscripts, newspapers, and printed archives.
"""

from agbalu.ocr.adlis import (
    BookTranscription,
    ScannedPage,
    discover_adlis_books,
    load_book_pages,
    transcribe_book,
)
from agbalu.ocr.config import ConfigError, ModelConfig, TrainConfig
from agbalu.ocr.dataset import (
    SyntheticDataset,
    build_dual_script_lines,
    chunk_text_into_lines,
    collate_lines,
    load_corpus_sentences,
)
from agbalu.ocr.evaluate import compute_cer, compute_wer, evaluate_ocr_model
from agbalu.ocr.infer import Recognizer, prepare_line_image, segment_page_into_lines
from agbalu.ocr.models import LossOutput, PositionalEncoding, VisionEncoderDecoder
from agbalu.ocr.synthetic import render_text_line
from agbalu.ocr.trainer import ResumeError, Trainer
from agbalu.ocr.vocabulary import VOCAB_SIZE, VOCABULARY, decode, encode

__all__ = [
    "VOCABULARY",
    "VOCAB_SIZE",
    "BookTranscription",
    "ConfigError",
    "LossOutput",
    "ModelConfig",
    "PositionalEncoding",
    "Recognizer",
    "ResumeError",
    "ScannedPage",
    "SyntheticDataset",
    "TrainConfig",
    "Trainer",
    "VisionEncoderDecoder",
    "build_dual_script_lines",
    "chunk_text_into_lines",
    "collate_lines",
    "compute_cer",
    "compute_wer",
    "decode",
    "discover_adlis_books",
    "encode",
    "evaluate_ocr_model",
    "load_book_pages",
    "load_corpus_sentences",
    "prepare_line_image",
    "render_text_line",
    "segment_page_into_lines",
    "transcribe_book",
]
