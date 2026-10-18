"""Exact and near-duplicate detection, shared by the CPT documents and the SFT candidates.

Exact: a digest of the text's words, so spacing and case of separators do not hide a copy.
Near: MinHash over 5-word shingles, at a Jaccard threshold of 0.8, the setting the CPT
document corpus was built with on 2026-10-14.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Final

from agbalu.llm.decontaminate import words

if TYPE_CHECKING:
    from datasketch import MinHashLSH

NEAR_DUPLICATE_THRESHOLD: Final = 0.8
MINHASH_PERMUTATIONS: Final = 128
SHINGLE_WORDS: Final = 5
MINHASH_SEED: Final = 20260814


def exact_key(text: str) -> bytes:
    return hashlib.blake2b(" ".join(words(text)).encode("utf-8"), digest_size=16).digest()


class NearDuplicates:
    """Texts seen so far; `seen` answers whether a new text is near one of them, and
    remembers it when it is not."""

    def __init__(self) -> None:
        from datasketch import MinHashLSH

        self.lsh: MinHashLSH = MinHashLSH(
            threshold=NEAR_DUPLICATE_THRESHOLD, num_perm=MINHASH_PERMUTATIONS
        )

    def seen(self, key: str, text: str) -> bool:
        from datasketch import MinHash

        tokens = words(text)
        shingles = {
            " ".join(tokens[i : i + SHINGLE_WORDS])
            for i in range(max(1, len(tokens) - SHINGLE_WORDS + 1))
        }
        signature = MinHash(num_perm=MINHASH_PERMUTATIONS, seed=MINHASH_SEED)
        for shingle in shingles:
            signature.update(shingle.encode("utf-8"))
        if self.lsh.query(signature):
            return True
        self.lsh.insert(key, signature)
        return False
