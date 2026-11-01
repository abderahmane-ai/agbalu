"""Verify real-tokenizer restoration artifacts and exact source accounting."""

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from agbalu.encoder.configuration import Geometry
from agbalu.encoder.previews import SENTENCES
from agbalu.restoration.case_views import revise_view
from agbalu.restoration.configuration import Locations, Settings
from agbalu.restoration.corpus import prepare, verify
from agbalu.restoration.dataset import Windows

pytestmark = pytest.mark.integration
TOKENIZER = Path("artifacts/encoder/measurements/agbalu-tok-base-16k.model")


class Admit:
    def accepts(self, texts: Sequence[str]) -> list[bool]:
        return [True] * len(texts)


def build(directory: Path) -> tuple[Locations, Settings]:
    settings = Settings(
        geometry=Geometry(
            hidden_size=16,
            intermediate_size=32,
            layers=2,
            heads=2,
            block_layers=1,
            buckets=8,
            dropout=0,
            backend="reference",
        ),
        workers=0,
        micro_windows=1,
        batch_windows=2,
        compile=False,
        require_smoke=False,
        minimum_words=3,
        minimum_template_support=1,
    )
    rows = [
        {"source": "taqbaylit.bejaia", "text": SENTENCES[2] + "\n\n" + SENTENCES[3]},
        {"source": "hf.imsidag.kabyle-corpus-hca", "text": SENTENCES[0]},
        {"source": "hf.hplt2-cleaned-kab", "text": SENTENCES[1]},
        {"source": "opus.nllb-kab", "text": "This is foreign text."},
    ]
    corpus = directory / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(
                {**row, "licence": "cc0-1.0", "redistribution": "permissive"}, ensure_ascii=False
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    speech = directory / "speech"
    speech.mkdir()
    (speech / "train.jsonl").write_text(
        json.dumps({"text": "Anida i tedduḍ a yelli?"}) + "\n", encoding="utf-8"
    )
    locations = Locations(
        corpus=corpus,
        tokenizer=TOKENIZER,
        speech=speech,
        data=directory / "encoded",
        encoder=directory / "encoder",
        runs=directory / "runs",
    )
    prepare(locations, settings, Admit())
    return locations, settings


@pytest.mark.skipif(not TOKENIZER.exists(), reason="real Mammeri artifact is absent")
def test_real_tokenizer_build_conserves_input_and_keeps_source_splits(tmp_path: Path) -> None:
    locations, settings = build(tmp_path)
    report = verify(locations.data)
    totals = report["totals"]
    assert isinstance(totals, dict)
    assert totals["input_records"] == 4
    assert totals["input_characters"] == (
        totals["accepted_characters"]
        + totals["rejected_characters"]
        + totals["separator_characters"]
    )
    assert report["split_documents"] == {"0": 3, "1": 1, "2": 1}
    table = np.load(locations.data / "documents.npy")
    assert int(table[:, 2].sum()) == report["words"]
    assert table[-1, 5:].tolist() == [1, 1]
    dataset = Windows(locations.data, settings, 0)
    assert len(dataset) == 3
    speech = dataset[-1]
    assert speech.edges == (True, True)
    assert speech.marks[-1] == 3
    with sqlite3.connect(locations.data / "records.sqlite3") as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM rejects WHERE reason='source_policy'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.skipif(not TOKENIZER.exists(), reason="real Mammeri artifact is absent")
def test_corrupt_generated_labels_are_detected(tmp_path: Path) -> None:
    locations, _ = build(tmp_path)
    with (locations.data / "marks.bin").open("ab") as handle:
        handle.write(b"lost-conservation")
    with pytest.raises(ValueError, match="checksum"):
        verify(locations.data)


@pytest.mark.skipif(not TOKENIZER.exists(), reason="real Mammeri artifact is absent")
def test_component_view_preserves_every_original_token_and_gap(tmp_path: Path) -> None:
    locations, settings = build(tmp_path)
    revised = locations.model_copy(
        update={"source_data": locations.data, "data": tmp_path / "revised"}
    )
    parent = verify(locations.data)
    report = revise_view(revised, settings)
    assert report["words"] == parent["words"]
    assert report["tokens"] == parent["tokens"]
    for name in ("tokens.bin", "offsets.npy", "documents.npy", "marks.bin", "structure.bin"):
        assert (revised.data / name).read_bytes() == (locations.data / name).read_bytes()
    assert revise_view(revised, settings) == report
