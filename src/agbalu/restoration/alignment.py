"""Align case components to verified SentencePiece byte spans."""

from collections.abc import Sequence
from dataclasses import dataclass

import sentencepiece as spm
from sentencepiece import sentencepiece_pb2

from agbalu.encoder.corpus import integer_field
from agbalu.restoration.casing import case_units, component_bytes


@dataclass(frozen=True)
class EncodedWord:
    ids: tuple[int, ...]
    byte_spans: tuple[tuple[int, int], ...]


def encode_word(word: str, processor: spm.SentencePieceProcessor) -> EncodedWord:
    serialized = processor.encode(word.lower(), out_type="serialized_proto", enable_sampling=False)
    if not isinstance(serialized, bytes):
        message = f"{word!r}: expected serialized SentencePiece bytes"
        raise TypeError(message)
    proto: object = getattr(sentencepiece_pb2, "SentencePieceText")()
    getattr(proto, "ParseFromString")(serialized)
    pieces: object = getattr(proto, "pieces")
    if not isinstance(pieces, Sequence):
        message = f"{word!r}: missing SentencePiece spans"
        raise TypeError(message)
    return EncodedWord(
        tuple(integer_field(piece, "id") for piece in pieces),
        tuple((integer_field(piece, "begin"), integer_field(piece, "end")) for piece in pieces),
    )


def align_cases(word: str, encoded: EncodedWord) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    lower = word.lower()
    for unit in case_units(word):
        first_byte, last_byte = component_bytes(word, unit)
        indices = [
            index
            for index, (start, end) in enumerate(encoded.byte_spans)
            if start < last_byte and end > first_byte
        ]
        if not indices:
            message = (
                f"{word!r}: case component chars [{unit.char_start},{unit.char_end}) "
                f"bytes [{first_byte},{last_byte}) has no pieces in {encoded.byte_spans!r}"
            )
            raise ValueError(message)
        first_char = len(word[: unit.char_start].lower())
        last_char = len(word[: unit.char_end].lower())
        rows.append((indices[0], indices[-1] + 1, first_char, last_char, len(lower), unit.label))
    return tuple(rows)
