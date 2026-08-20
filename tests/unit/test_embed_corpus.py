"""Unit tests for sentence embedding pair corpus extraction, leak exclusion, and
cluster splitting.
"""

from __future__ import annotations

import json
from pathlib import Path

from agbalu.embed.corpus import (
    Pair,
    build_parallel_pairs,
    build_tapaco_pairs,
    collect_benchmark_fingerprints,
    split_clusters,
    write_split,
)
from agbalu.extract import fingerprint
from agbalu.normalise import Normaliser


class TestBenchmarkFingerprints:
    def test_collects_both_raw_and_normalised_forms(self, tmp_path: Path) -> None:
        bench_file = tmp_path / "bench.jsonl"
        # Greek epsilon in raw
        bench_file.write_text(json.dumps({"sentence": "Azul fell-awεn"}) + "\n", encoding="utf-8")

        normaliser = Normaliser()
        blocked = collect_benchmark_fingerprints([bench_file], normaliser=normaliser)

        assert fingerprint("Azul fell-awεn") in blocked  # Greek epsilon U+03B5
        assert fingerprint("Azul fell-awɛn") in blocked  # Latin epsilon U+025B

    def test_handles_plain_text_files(self, tmp_path: Path) -> None:
        plain_file = tmp_path / "bench.txt"
        plain_file.write_text("Axxam amellal\n", encoding="utf-8")

        blocked = collect_benchmark_fingerprints([plain_file])
        assert fingerprint("Axxam amellal") in blocked


class TestTaPaCoPairs:
    def test_extracts_all_cluster_pairs(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "tapaco.tsv"
        tsv_file.write_text(
            "101\tkab\tYebɣa ad yeǧǧ axxam.\n"
            "101\tkab\tIra ad yeǧǧ axxam.\n"
            "101\tkab\tYebɣa ad iffeɣ seg wexxam.\n"
            "102\tkab\tAzul fell-awen.\n"
            "102\tkab\tAnsuf yis-wen.\n",
            encoding="utf-8",
        )

        pairs = build_tapaco_pairs(tsv_file, blocked=set())
        # Cluster 101 has 3 items -> 3 pairs; Cluster 102 has 2 items -> 1 pair -> 4 pairs total
        assert len(pairs) == 4
        assert all(p.pair_type == "paraphrase" for p in pairs)
        assert sum(1 for p in pairs if p.cluster_id == "tapaco_101") == 3
        assert sum(1 for p in pairs if p.cluster_id == "tapaco_102") == 1

    def test_excludes_clusters_touching_benchmark(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "tapaco.tsv"
        tsv_file.write_text(
            "101\tkab\tYebɣa ad yeǧǧ axxam.\n"
            "101\tkab\tIra ad yeǧǧ axxam.\n"
            "102\tkab\tAzul fell-awen.\n"
            "102\tkab\tAnsuf yis-wen.\n",
            encoding="utf-8",
        )

        blocked = {fingerprint("Yebɣa ad yeǧǧ axxam.")}
        pairs = build_tapaco_pairs(tsv_file, blocked=blocked)

        assert len(pairs) == 1
        assert pairs[0].cluster_id == "tapaco_102"


class TestParallelPairs:
    def test_excludes_mined_nllb_by_default(self, tmp_path: Path) -> None:
        par_dir = tmp_path / "parallel"
        par_dir.mkdir()

        nllb_file = par_dir / "nllb_mined_kab_eng.jsonl"
        row1 = {"src_lang": "kab", "tgt_lang": "eng", "src_text": "Azul", "tgt_text": "Hello"}
        nllb_file.write_text(json.dumps(row1) + "\n", encoding="utf-8")

        tatoeba_file = par_dir / "tatoeba_kab_eng.jsonl"
        row2 = {"src_lang": "kab", "tgt_lang": "eng", "src_text": "Amek", "tgt_text": "How"}
        tatoeba_file.write_text(json.dumps(row2) + "\n", encoding="utf-8")

        pairs = build_parallel_pairs(par_dir, blocked=set(), include_mined=False)
        assert len(pairs) == 1
        assert pairs[0].query == "Amek"
        assert pairs[0].passage == "How"
        assert pairs[0].pair_type == "translation_eng"

    def test_excludes_leaks_on_either_side(self, tmp_path: Path) -> None:
        par_dir = tmp_path / "parallel"
        par_dir.mkdir()

        par_file = par_dir / "tatoeba_kab_fra.jsonl"
        row1 = {"src_lang": "kab", "tgt_lang": "fra", "src_text": "Axxam", "tgt_text": "La maison"}
        row2 = {"src_lang": "kab", "tgt_lang": "fra", "src_text": "Aman", "tgt_text": "L'eau"}
        par_file.write_text(
            json.dumps(row1) + "\n" + json.dumps(row2) + "\n",
            encoding="utf-8",
        )

        blocked = {fingerprint("La maison")}
        pairs = build_parallel_pairs(par_dir, blocked=blocked)
        assert len(pairs) == 1
        assert pairs[0].query == "Aman"
        assert pairs[0].passage == "L'eau"


class TestClusterSplit:
    def test_no_cluster_crosses_train_and_dev_split(self) -> None:
        pairs = [
            Pair(
                query=f"q_{i}",
                passage=f"p_{i}",
                pair_type="paraphrase",
                cluster_id=f"c_{i // 3}",
                source_id="test",
            )
            for i in range(30)
        ]
        # 10 distinct clusters (c_0 to c_9)
        train, dev = split_clusters(pairs, dev_clusters=3, seed=42)

        train_clusters = {p.cluster_id for p in train}
        dev_clusters = {p.cluster_id for p in dev}

        assert len(dev_clusters) == 3
        assert len(train_clusters) == 7
        assert train_clusters.isdisjoint(dev_clusters)
        assert len(train) + len(dev) == 30


class TestWriteSplit:
    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        pairs = [
            Pair(
                query="Azul",
                passage="Hello",
                pair_type="translation_eng",
                cluster_id="c1",
                source_id="s1",
            ),
        ]
        out_file = tmp_path / "out.jsonl"
        count = write_split(out_file, pairs)

        assert count == 1
        lines = out_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["query"] == "Azul"
        assert data["passage"] == "Hello"
        assert data["cluster_id"] == "c1"
