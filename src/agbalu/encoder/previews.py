"""Prepare fixed operator-provided sentences for qualitative masked reconstruction."""

from dataclasses import dataclass
from typing import Final

import sentencepiece as spm
import torch

from agbalu.encoder.configuration import SPECIAL_TOKENS
from agbalu.encoder.corpus import encode_groups
from agbalu.encoder.masking import mask_groups

SENTENCES: Final[tuple[str, ...]] = (
    "Mi ara d-yaweḍ unebdu deg tmurt n Leqbayel, ttuɣalen-d yiminigen ɣer tuddar-nsen akken ad "
    "ẓren imawlan-nsen, ad ḥedren i tmeɣriwin, u ad ssemḥsisen i ccna n teqbaylit deg tɣerɣert "
    "n taddart.",
    "Tutlayt taqbaylit d agerruj i d-ǧǧan lejdud-nneɣ seg zik n zik, ilaq fell-aneɣ ad "
    "tt-id-nessukkes seg tatut s tira, s ulmad deg uɣerbaz, akked useqdec-ines deg tmeddurt "
    "n yal ass.",
    "Argaz-nni amɣar iwala-d aqcic-is mi d-yuɣal seg lexla ɣef tmeddit n wass, yenna-yas-d "
    "s lferḥ d ameqqran belli lɣella n uzemmur n useggas-agi ad tili d tamesbaɣurt nezzeh.",
    "Ur zmireɣ ara ad ttuɣ ussan-nni i nesɛedda lwaḥid deg yidurar n Ǧerjer mi nella d igurdan, "
    "imi dinna i nelmed amek ara nqabel tudert s tissas, s tebɣest, d tegmat gar-aneɣ.",
)
MASK_SEED: Final = 31415
PREVIEW_TOKENS: Final = 512


@dataclass(frozen=True)
class SentencePreview:
    text: str
    original: torch.Tensor
    masked: torch.Tensor
    selected: torch.Tensor


def prepare_sentences(
    processor: spm.SentencePieceProcessor, vocab_size: int
) -> list[SentencePreview]:
    examples: list[SentencePreview] = []
    for index, text in enumerate(SENTENCES):
        ids, groups = encode_groups(text, processor)
        if not ids or any(value < SPECIAL_TOKENS or value >= vocab_size for value in ids):
            message = f"preview sentence {index + 1}: invalid token IDs {ids}"
            raise ValueError(message)
        original = torch.tensor([2, *ids, 3], dtype=torch.long)
        ownership = torch.tensor([-1, *(group + 1 for group in groups), -1], dtype=torch.long)
        chosen = mask_groups(original, ownership, 0.2, MASK_SEED + index, vocab_size)
        selected = chosen.labels >= 0
        # Every qualitative target is visibly masked; training retains its 80/10/10 policy.
        masked = torch.where(selected, 4, original)
        examples.append(SentencePreview(text, original, masked, selected))
    count = sum(example.original.numel() for example in examples)
    if count > PREVIEW_TOKENS:
        message = f"fixed preview sentences need {count} tokens, capacity is {PREVIEW_TOKENS}"
        raise ValueError(message)
    return examples
