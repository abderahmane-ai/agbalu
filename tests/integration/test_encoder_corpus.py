"""Verify the real Mammeri conversion and batch accounting."""

import json
from pathlib import Path

import numpy as np
import pytest

from agbalu.encoder.batches import Batches
from agbalu.encoder.configuration import Locations
from agbalu.encoder.corpus import encode_groups, prepare, tokenizer, verify_data, write_json
from agbalu.llm.provenance import file_digest

pytestmark = pytest.mark.integration


def test_real_tokenizer_offsets_group_a_hyphen_chain() -> None:
    processor = tokenizer(Path("artifacts/tokenizer/agbalu-tok-base-16k.model"))
    ids, groups = encode_groups("yefka-yas-t-id", processor)
    assert ids == [964, 261, 528, 261, 268, 261, 314]
    assert groups == [0] * 7


def test_conversion_and_both_contexts_account_for_every_kept_token(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    rows = [f"Deg wexxam nni, yefka-yas-t-id umɣar {index}." for index in range(400)]
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in rows))
    locations = Locations(
        corpus=path,
        tokenizer=Path("artifacts/tokenizer/agbalu-tok-base-16k.model"),
        data=tmp_path / "data",
    )
    report = prepare(locations)
    assert report["documents_read"] == 400
    assert report["documents_kept"] == 400
    processor = tokenizer(locations.tokenizer)
    expected = sum(len(processor.encode(text, out_type=int)) for text in rows)
    assert report["text_tokens"] == expected
    assert prepare(locations) == report
    assert verify_data(locations.data) == report
    for context in (512, 1024):
        seen = 0
        for validation in (False, True):
            batches = Batches(locations.data, context, 1024, 4, validation=validation)
            for index in range(len(batches)):
                batch = batches[index]
                seen += int(((batch.ids >= 5) & batch.valid).sum())
                assert np.all(batch.positions.numpy()[batch.valid.numpy()] < context)
        assert seen == expected


def test_changed_token_bytes_fail_the_integrity_gate(tmp_path: Path) -> None:
    (tmp_path / "tokens.bin").write_bytes(b"00")
    (tmp_path / "groups.bin").write_bytes(b"0000")
    write_json(
        tmp_path / "corpus.stats.json",
        {"text_tokens": 1, "files": {"tokens.bin": file_digest(tmp_path / "tokens.bin")}},
    )
    (tmp_path / "tokens.bin").write_bytes(b"11")
    with pytest.raises(ValueError, match="checksum"):
        verify_data(tmp_path)
