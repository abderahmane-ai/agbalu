"""Unit tests for the HPC corpus expansion and multi-format replay pipelines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from agbalu.llm import expand, replay
from agbalu.llm.documents import SentenceRow
from agbalu.normalise import Normaliser


class FakeScorer:
    """Marks text as Kabyle unless it contains `NON_KABYLE`."""

    def probabilities(self, texts: Sequence[str]) -> list[dict[str, float]]:
        return [
            {"kab_Latn": 0.05, "NOT_AMAZIGH": 0.95}
            if "NON_KABYLE" in text
            else {"kab_Latn": 0.95, "NOT_AMAZIGH": 0.05}
            for text in texts
        ]


class TestLegacyFontRepair:
    def test_repairs_legacy_8bit_berber_glyphs_inside_words(self) -> None:
        raw = "Tamazi$t d tutlayt n Leqbayel, amek i d-yenna £ef tmusni."
        repaired = expand.repair_legacy_font(raw)
        assert "Tamaziɣt" in repaired
        assert "ɣef" in repaired

    def test_leaves_currency_before_digits_intact(self) -> None:
        assert expand.repair_legacy_font("Price is $100 today") == "Price is $100 today"


class TestAudioChunking:
    def test_splits_long_broadcast_at_energy_minima(self) -> None:
        sr = 16000
        # 30 seconds of synthetic signal with a silent gap at 15s
        signal = np.ones(30 * sr, dtype=np.float32) * 0.5
        signal[15 * sr - 800 : 15 * sr + 800] = 0.0
        chunks = expand.split_audio_chunks(signal, rate=sr, min_sec=10.0, max_sec=22.0)
        assert len(chunks) == 2
        assert abs(len(chunks[0]) - 15 * sr) <= sr


class TestThesisReconstruction:
    def test_reconstructs_continuous_chapters_sorted_by_folder_and_line_no(
        self, tmp_path: Path
    ) -> None:
        folder = tmp_path / "hf.imsidag.kabyle-corpus-ummto"
        folder.mkdir(parents=True)
        para = (
            "Tamurt n Leqbayel tella deg ugafa n Lezzayer gar yidurar d yill. "
            "Imezdaɣ-is ttmeslayen taqbaylit s yisekkilen ilatiniyen."
        )
        table = pa.table(
            {
                "id": ["2", "1", "3"],
                "text": [para + " (2)", para + " (1)", para + " (3)"],
                "folder": ["docA", "docA", "docA"],
                "line_no": [2, 1, 3],
            }
        )
        pq.write_table(table, folder / "0.parquet")
        docs = list(expand._reconstruct_imsidag_parquets(folder, Normaliser()))
        assert len(docs) == 1
        assert docs[0].index("(1)") < docs[0].index("(2)") < docs[0].index("(3)")


class TestTmxAndAligned:
    def test_parses_tmx_and_filters_expanded_aligned(self, tmp_path: Path) -> None:
        tmx_dir = tmp_path / "taqbaylit.weblate-kabyle"
        tmx_dir.mkdir(parents=True)
        tmx_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<tmx><body><tu>"
            '<tuv xml:lang="en"><seg>Open the settings window</seg></tuv>'
            '<tuv xml:lang="kab"><seg>Ldi asfaylu n yiɣewwaren</seg></tuv>'
            "</tu></body></tmx>"
        )
        (tmx_dir / "en-kab.tmx").write_text(tmx_xml, encoding="utf-8")
        out = tmp_path / "expanded-aligned.jsonl"
        counts = expand.extract_aligned_records(
            FakeScorer(), raw_root=tmp_path, raw_v1_root=tmp_path, out_path=out
        )
        assert counts["taqbaylit.weblate-kabyle"] == 1
        records = list(expand.read_expanded_aligned(out))
        assert len(records) == 1
        assert records[0].kabyle == "Ldi asfaylu n yiɣewwaren"
        assert records[0].other == "Open the settings window"


class TestMergeExpandedSentences:
    def test_reconstructed_imsidag_supersedes_fragmented_base_rows(self, tmp_path: Path) -> None:
        exp_docs = tmp_path / "expanded-docs.jsonl"
        exp_docs.write_text(
            json.dumps(
                {
                    "text": "Full reconstructed thesis chapter.\n\nSecond paragraph.",
                    "source": "hf.imsidag.kabyle-corpus-ummto",
                    "licence": "cc-by-sa-4.0",
                    "redistribution": "share-alike",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        base = iter(
            [
                SentenceRow(
                    text="Fragmented line 1",
                    source_id="hf.imsidag.kabyle-corpus-ummto",
                    licence="cc-by-sa-4.0",
                    redistribution="share-alike",
                    line=1,
                ),
                SentenceRow(
                    text="Kept sentence from another source",
                    source_id="other.source",
                    licence="cc0-1.0",
                    redistribution="permissive",
                    line=2,
                ),
            ]
        )
        merged = list(
            expand.merge_expanded_sentences(
                base,
                expanded_docs_path=exp_docs,
                expanded_audio_path=tmp_path / "missing.jsonl",
                raw_root=tmp_path,
            )
        )
        assert [r.source_id for r in merged] == [
            "hf.imsidag.kabyle-corpus-ummto",
            "other.source",
        ]
        assert "Full reconstructed thesis chapter." in merged[0].text


class TestResume:
    """A restart keeps what an earlier run wrote whole and repeats the source left over."""

    @staticmethod
    def row(source: str, text: str) -> str:
        return (
            json.dumps(
                {
                    "text": text,
                    "source": source,
                    "licence": "mit",
                    "redistribution": "permissive",
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    def test_reads_the_source_order_with_document_and_character_counts(
        self, tmp_path: Path
    ) -> None:
        corpus = tmp_path / "expanded.jsonl"
        corpus.write_text(
            self.row("bejaia", "tamurt")
            + self.row("bejaia", "azul")
            + self.row("bouira", "tamurt"),
            encoding="utf-8",
        )
        order, counts = expand.sources_in_file(corpus)
        assert order == ["bejaia", "bouira"]
        assert counts == {"bejaia": (2, 10), "bouira": (1, 6)}

    def test_without_a_sidecar_the_last_source_is_rebuilt_and_the_rest_kept(
        self, tmp_path: Path
    ) -> None:
        corpus = tmp_path / "expanded.jsonl"
        corpus.write_text(
            self.row("bejaia", "tamurt") + self.row("bouira", "azul") + '{"text": "torn',
            encoding="utf-8",
        )
        assert expand.resume_from_file(corpus) == {"bejaia": (1, 6)}
        assert corpus.read_text(encoding="utf-8") == self.row("bejaia", "tamurt")
        assert expand._read_progress(corpus) == ["bejaia"]

    def test_a_sidecar_source_is_kept_even_when_the_file_ends_there(self, tmp_path: Path) -> None:
        corpus = tmp_path / "expanded.jsonl"
        corpus.write_text(
            self.row("a", "x") + self.row("b", "y") + self.row("c", "z"), encoding="utf-8"
        )
        expand._write_progress(corpus, ["a", "b"])
        assert list(expand.resume_from_file(corpus)) == ["a", "b"]
        assert corpus.read_text(encoding="utf-8") == self.row("a", "x") + self.row("b", "y")

    def test_a_sidecar_source_the_file_no_longer_holds_is_ignored(self, tmp_path: Path) -> None:
        corpus = tmp_path / "expanded.jsonl"
        corpus.write_text(self.row("a", "x"), encoding="utf-8")
        expand._write_progress(corpus, ["a", "gone"])
        assert list(expand.resume_from_file(corpus)) == ["a"]

    def test_a_lone_source_without_a_sidecar_is_rebuilt_whole(self, tmp_path: Path) -> None:
        corpus = tmp_path / "expanded.jsonl"
        corpus.write_text(self.row("a", "x") + '{"text": "torn', encoding="utf-8")
        assert expand.resume_from_file(corpus) == {}
        assert corpus.read_text(encoding="utf-8") == ""

    def test_a_missing_file_reports_nothing(self, tmp_path: Path) -> None:
        assert expand.resume_from_file(tmp_path / "absent.jsonl") == {}

    @staticmethod
    def corpus_tree(root: Path) -> None:
        """The two text sources the extractor reads without OCR: Quran surahs and kabyle-corpus."""
        verse = "Tamurt n Leqbayel tella deg ugafa n Lezzayer, gar yidurar d yill."
        quran = root / "abdelhaqueidali.quran-kab"
        quran.mkdir(parents=True)
        verses = "".join(f"1\t{verse}\n" for _ in range(4))
        (quran / "quran_ascii_latin_amazigh_arabic_scripts.tsv").write_text(
            f"surah_number\tlatin_script\n{verses}", encoding="utf-8"
        )
        corpus = root / "taqbaylit.kabyle-corpus"
        corpus.mkdir(parents=True)
        (corpus / "kabyle_corpus_clean.jsonl").write_text(
            json.dumps({"text": " ".join([verse] * 5)}) + "\n", encoding="utf-8"
        )

    @staticmethod
    def extract(raw_root: Path, out: Path, *, resume: bool = False) -> list[expand.ExpandTally]:
        return expand.extract_pdfs_and_text(
            FakeScorer(),
            raw_root=raw_root,
            raw_v1_root=raw_root.parent / "raw_v1",
            models_root=raw_root.parent / "models",
            out_path=out,
            resume=resume,
        )

    def test_a_later_resume_leaves_a_finished_extraction_untouched(self, tmp_path: Path) -> None:
        raw_root = tmp_path / "raw"
        self.corpus_tree(raw_root)
        out = tmp_path / "expanded.jsonl"
        first = self.extract(raw_root, out)
        assert sum(tally.kept for tally in first) == 2
        written = out.read_text(encoding="utf-8")
        again = self.extract(raw_root, out, resume=True)
        assert out.read_text(encoding="utf-8") == written
        assert [tally.resumed for tally in again] == [True, True]
        assert sum(tally.kept for tally in again) == 2

    def test_a_crash_boundary_rebuilds_only_the_source_it_cut_off(self, tmp_path: Path) -> None:
        raw_root = tmp_path / "raw"
        self.corpus_tree(raw_root)
        out = tmp_path / "expanded.jsonl"
        self.extract(raw_root, out)
        kept = [
            line
            for line in out.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["source"] == "abdelhaqueidali.quran-kab"
        ]
        out.write_text("\n".join(kept) + "\n", encoding="utf-8")
        expand._progress_path(out).unlink()
        self.extract(raw_root, out, resume=True)
        sources = [
            json.loads(line)["source"] for line in out.read_text(encoding="utf-8").splitlines()
        ]
        assert sources.count("abdelhaqueidali.quran-kab") == len(kept)
        assert sources.count("taqbaylit.kabyle-corpus") == 1


class TestMultiFormatReplayParquet:
    def test_reads_problem_solution_and_messages_parquet_schemas(self, tmp_path: Path) -> None:
        ps_path = tmp_path / "math.parquet"
        pq.write_table(
            pa.table({"problem": ["Solve x + 2 = 5."], "solution": ["x = 3."]}),
            ps_path,
        )
        rows = list(replay._parquet_rows(ps_path, "problem+solution"))
        assert rows == ["Problem:\nSolve x + 2 = 5.\n\nSolution:\nx = 3."]

        msg_path = tmp_path / "ifeval.parquet"
        pq.write_table(
            pa.table(
                {
                    "messages": [
                        [
                            {"role": "user", "content": "Write 3 bullet points."},
                            {"role": "assistant", "content": "- One\n- Two\n- Three"},
                        ]
                    ]
                }
            ),
            msg_path,
        )
        msg_rows = list(replay._parquet_rows(msg_path, "messages"))
        assert msg_rows == ["User:\nWrite 3 bullet points.\n\nAssistant:\n- One\n- Two\n- Three"]
