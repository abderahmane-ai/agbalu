"""The stamp every Phase 11 artifact carries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agbalu.llm.provenance import (
    STAMP_FILE,
    ProvenanceError,
    code_stamp,
    config_hash,
    file_digest,
)


class TestCodeStamp:
    def test_the_checkout_reports_a_sha(self) -> None:
        stamp = code_stamp(Path())
        assert len(str(stamp["git_sha"])) == 40
        assert isinstance(stamp["dirty"], bool)

    def test_a_synced_tree_reads_the_stamp_file(self, tmp_path: Path) -> None:
        (tmp_path / STAMP_FILE).write_text(json.dumps({"git_sha": "a" * 40, "dirty": True}))
        assert code_stamp(tmp_path) == {"git_sha": "a" * 40, "dirty": True}

    def test_neither_git_nor_a_stamp_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ProvenanceError, match=r"tools\.launch"):
            code_stamp(tmp_path)


class TestHashes:
    def test_config_hash_ignores_key_order(self) -> None:
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_file_digest_is_sha256_across_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "x.bin"
        payload = bytes(range(256)) * 50
        path.write_bytes(payload)
        assert file_digest(path, chunk=100) == hashlib.sha256(payload).hexdigest()
