from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from agbalu.acquire.manifest import Manifest
from agbalu.acquire.models import ManifestEntry
from agbalu.bench.flores import Sentence
from agbalu.parallel.columns import choose_pair
from agbalu.parallel.nllb import pair_key
from agbalu.parallel.pipeline import ParallelBuilder, summary
from agbalu.parallel.readers import PairReadError, opus_members, read_opus_zip
from agbalu.registry.models import Registry, Source, SourceSize

KAB = "Aql-i deg wexxam, ur ttruḥuɣ ara ɣer temdint ass-a."
KAB2 = "Ɣef waya i d-nusa ɣer da, imi nezmer ad nemmeslay akken ilaq."
ENG = "I am at home and I will not go to the city today."
ENG2 = "This is why we came here, because we are able to speak."
FRA = "Je suis a la maison et je n irai pas en ville aujourd hui."


def make_source(source_id: str = "test.src", **kwargs: object) -> Source:
    defaults: dict[str, object] = {
        "id": source_id,
        "name": "Test source",
        "modality": "parallel",
        "tier": "core",
        "access": "hf",
        "uri": "https://huggingface.co/datasets/x/y",
        "licence": "cc0-1.0",
        "languages": ("kab", "eng"),
        "size": SourceSize(rows=2),
        "retrieved": date(2026, 8, 6),
    }
    return Source.model_validate(defaults | kwargs)


def registry(*sources: Source) -> Registry:
    return Registry(version="1.0.0", surveyed=date(2026, 8, 6), sources=sources)


def land(root: Path, source: Source, name: str, text: str) -> None:
    directory = root / source.id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text, encoding="utf-8")
    Manifest(root).append(
        ManifestEntry(
            source_id=source.id,
            path=name,
            bytes=len(text.encode()),
            sha256="0" * 64,
            kind="text",
            target="local",
            uri=source.uri,
            fetched_at=datetime.now(UTC),
            revision=None,
        )
    )


@dataclass(frozen=True, slots=True)
class Row:
    """One built pair, typed so assertions do not fight `object`."""

    kab: str
    foreign: str
    source: str
    licence: str
    redistribution: str
    defects: list[str]


def build(
    tmp_path: Path, reg: Registry, benchmark: list[Sentence] | None = None
) -> dict[str, list[Row]]:
    out = tmp_path / "out"
    ParallelBuilder(reg, tmp_path / "raw", benchmark).build(out)
    result: dict[str, list[Row]] = {}
    for lang in ("eng", "fra"):
        path = out / f"agbalu-parallel-v1.kab-{lang}.jsonl"
        rows: list[Row] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            rows.append(
                Row(
                    kab=raw["kab"],
                    foreign=raw[lang],
                    source=raw["source"],
                    licence=raw["licence"],
                    redistribution=raw["redistribution"],
                    defects=list(raw["defects"]),
                )
            )
        result[lang] = rows
    return result


def test_pairs_carry_both_sides_and_provenance(tmp_path: Path) -> None:
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\t{ENG}\n")
    rows = build(tmp_path, registry(source))["eng"]
    assert rows[0].kab == KAB
    assert rows[0].foreign == ENG
    assert rows[0].source == "test.src"
    assert rows[0].redistribution == "permissive"
    assert rows[0].defects == []


def test_the_kabyle_side_is_normalised(tmp_path: Path) -> None:
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\nteεyiḍ tafsut\t{ENG}\n")
    assert build(tmp_path, registry(source))["eng"][0].kab == "teɛyiḍ tafsut"


def test_french_pairs_land_in_their_own_file(tmp_path: Path) -> None:
    source = make_source(languages=("kab", "fra"))
    land(tmp_path / "raw", source, "a.tsv", f"kab\tfra\n{KAB}\t{FRA}\n")
    rows = build(tmp_path, registry(source))
    assert rows["fra"][0].foreign == FRA
    assert rows["eng"] == []


def test_defective_pairs_are_labelled_not_dropped(tmp_path: Path) -> None:
    """The defect rate is the measurement; filtering would destroy it."""
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\t{KAB}\n{KAB2}\t{ENG2}\n")
    rows = build(tmp_path, registry(source))["eng"]
    assert len(rows) == 2
    assert "untranslated-copy" in rows[0].defects


def test_duplicate_pairs_are_dropped_once(tmp_path: Path) -> None:
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\t{ENG}\n{KAB}\t{ENG}\n")
    assert len(build(tmp_path, registry(source))["eng"]) == 1


def test_same_kabyle_with_a_different_translation_is_kept(tmp_path: Path) -> None:
    """Dedup is on the pair. One Kabyle sentence has many valid translations."""
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\t{ENG}\n{KAB}\t{ENG2}\n")
    assert len(build(tmp_path, registry(source))["eng"]) == 2


def test_permissive_sources_claim_a_shared_pair_first(tmp_path: Path) -> None:
    mined = make_source("a.mined", licence="cc-by-nc-4.0", tier="supplementary")
    human = make_source("z.human", licence="cc-by-4.0")
    for source in (mined, human):
        land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\t{ENG}\n")
    rows = build(tmp_path, registry(mined, human))["eng"]
    assert len(rows) == 1
    assert rows[0].source == "z.human"


def test_benchmark_pairs_are_decontaminated(tmp_path: Path) -> None:
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\t{ENG}\n{KAB2}\t{ENG2}\n")
    blocked = [
        Sentence(
            id=1,
            text=KAB,
            split="dev",
            domain="",
            topic="",
            url="",
            last_updated="1.0",
            iso_639_3="kab",
            iso_15924="Latn",
        )
    ]
    rows = build(tmp_path, registry(source), blocked)["eng"]
    assert [r.kab for r in rows] == [KAB2]


def test_decontamination_survives_the_homoglyph_difference(tmp_path: Path) -> None:
    """FLORES+ is unnormalised; the corpus is not. Both forms must block."""
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\nteɛyiḍ tafsut\t{ENG}\n")
    blocked = [
        Sentence(
            id=1,
            text="teεyiḍ tafsut",
            split="dev",
            domain="",
            topic="",
            url="",
            last_updated="1.0",
            iso_639_3="kab",
            iso_15924="Latn",
        )
    ]
    assert build(tmp_path, registry(source), blocked)["eng"] == []


class TestOpusBundles:
    def bundle(self, tmp_path: Path, members: dict[str, str], name: str = "en-kab.txt.zip") -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, content in members.items():
                archive.writestr(member, content)
        return path

    def test_the_pair_is_read_from_the_bundle_stem(self, tmp_path: Path) -> None:
        """`.xml` is an alignment file, not a language side.

        Pairing by extension matched `bible-uedin.en-kab.xml`, whose line count
        differs, and the resulting error discarded every pair in the source.
        """
        path = self.bundle(
            tmp_path,
            {
                "x.en-kab.kab": f"{KAB}\n{KAB2}\n",
                "x.en-kab.en": f"{ENG}\n{ENG2}\n",
                "x.en-kab.xml": "<aligned/>\n",
                "README": "ignore",
            },
        )
        with zipfile.ZipFile(path) as archive:
            assert opus_members(archive) == [("x.en-kab.kab", "x.en-kab.en", "en")]
        assert [(k, f) for k, f, _ in read_opus_zip(path)] == [(KAB, ENG), (KAB2, ENG2)]

    def test_a_truncated_side_raises_rather_than_misaligning(self, tmp_path: Path) -> None:
        path = self.bundle(
            tmp_path, {"x.en-kab.kab": f"{KAB}\n{KAB2}\n", "x.en-kab.en": f"{ENG}\n"}
        )
        with pytest.raises(PairReadError, match="line count"):
            list(read_opus_zip(path))

    def test_blank_lines_are_skipped_without_shifting_alignment(self, tmp_path: Path) -> None:
        path = self.bundle(
            tmp_path, {"x.en-kab.kab": f"{KAB}\n\n{KAB2}\n", "x.en-kab.en": f"{ENG}\n\n{ENG2}\n"}
        )
        assert [(k, f) for k, f, _ in read_opus_zip(path)] == [(KAB, ENG), (KAB2, ENG2)]


class TestColumns:
    def test_named_sides_are_paired(self) -> None:
        records = [{"kab": KAB, "en": ENG}, {"kab": KAB2, "en": ENG2}]
        assert choose_pair(records) == ("kab", "en", "eng")

    def test_french_is_identified(self) -> None:
        records = [{"kab": KAB, "text": FRA}, {"kab": KAB2, "text": FRA}]
        assert choose_pair(records)[2] == "fra"

    def test_metadata_is_never_the_translation(self) -> None:
        records = [{"kab": KAB, "id": "1", "en": ENG}, {"kab": KAB2, "id": "2", "en": ENG2}]
        assert choose_pair(records)[1] == "en"

    def test_a_source_with_no_foreign_side_yields_none(self) -> None:
        records = [{"kab": KAB}, {"kab": KAB2}]
        assert choose_pair(records)[1] is None

    def test_a_sparse_column_cannot_be_the_translation(self) -> None:
        records = [{"kab": KAB, "note": ENG}, {"kab": KAB2}, {"kab": KAB2}, {"kab": KAB}]
        assert choose_pair(records)[1] is None


def test_pair_key_distinguishes_translations() -> None:
    assert pair_key(KAB, ENG) != pair_key(KAB, ENG2)
    assert pair_key(KAB, ENG) == pair_key(KAB, ENG)


def test_summary_separates_hard_from_soft(tmp_path: Path) -> None:
    source = make_source()
    land(
        tmp_path / "raw",
        source,
        "a.tsv",
        f"kab\teng\n{KAB}\t{KAB}\nYella 12 n wussan\tThere were 21 days here\n",
    )
    builder = ParallelBuilder(registry(source), tmp_path / "raw")
    totals = summary(builder.build(tmp_path / "out"))
    assert totals.hard_defective == 1
    assert totals.defective == 2
    assert totals.hard_defect_rate < totals.defect_rate


def test_a_failed_build_does_not_overwrite_the_previous_corpus(tmp_path: Path) -> None:
    """Writing in place over 11.7M records means a crash leaves a truncated corpus
    no consumer can tell from a complete one."""
    out = tmp_path / "out"
    out.mkdir()
    previous = out / "agbalu-parallel-v1.kab-eng.jsonl"
    previous.write_text('{"kab": "azul", "eng": "hello"}\n', encoding="utf-8")

    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\tThere were days\n")
    builder = ParallelBuilder(registry(source), tmp_path / "raw")

    died = "source enumeration died mid-build"

    def explode() -> Iterator[Source]:
        raise RuntimeError(died)

    builder.sources = explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        builder.build(out)

    assert previous.read_text(encoding="utf-8") == '{"kab": "azul", "eng": "hello"}\n'


def test_a_successful_build_leaves_no_partial_files(tmp_path: Path) -> None:
    source = make_source()
    land(tmp_path / "raw", source, "a.tsv", f"kab\teng\n{KAB}\tThere were days\n")
    out = tmp_path / "out"
    ParallelBuilder(registry(source), tmp_path / "raw").build(out)
    assert list(out.glob("*.part")) == []
    assert (out / "agbalu-parallel-v1.kab-eng.jsonl").is_file()
