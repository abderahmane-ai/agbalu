"""The sentence embedding fine-tuning corpus: pairs, clusters, and leak exclusion.

The objective is contrastive representation learning (InfoNCE / Matryoshka).
Training pairs originate from two curated sources:
1. **Parallel translations**: Curated human bitext (`kab-eng` and `kab-fra`) excluding
   mined NLLB output, where each sentence translates the other.
2. **TaPaCo Paraphrase Clusters**: Monolingual Kabyle paraphrase groups (`kab-kab`) where
   sentences belonging to the same cluster express identical semantic intent.

### Decontamination & Leak Exclusion
Sentence transformers are sensitive to evaluation benchmark memorisation.
Before any pair is accepted:
- All benchmark sentences (from FLORES+ dev/test and STS evaluation sets) are indexed
  under both raw and NFKD-normalised fingerprints (`agbalu.extract.fingerprint`).
- Any pair containing a query or passage matching a benchmark sentence is excluded.

### Cluster Integrity
In contrastive learning, in-batch negative sampling assumes distinct rows are semantically
different. Multiple sentences from the same paraphrase cluster in the same batch act as
false negatives. Every pair is assigned a `cluster_id`, and train/dev splitting is performed
strictly at the cluster level so that no cluster spans across splits.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypedDict

from agbalu.extract import fingerprint
from agbalu.extract.readers import read_parquet
from agbalu.normalise import Normaliser

PairType = Literal["paraphrase", "translation_eng", "translation_fra"]

NLLB_MARKER: Final[str] = "nllb"
"""Source ids carrying it are NLLB's mined output, excluded by default."""

DEV_CLUSTERS: Final[int] = 500
"""Held-out semantic clusters for validation and hyperparameter tuning."""
MIN_TSV_PARTS: Final[int] = 2


class CorpusStats(TypedDict):
    """What `build` wrote, and the counts that explain the filtering pipeline."""

    read_pairs: int
    leaks_excluded: int
    kept_pairs: int
    unique_clusters: int
    train_pairs: int
    dev_pairs: int
    by_pair_type: dict[str, int]
    by_source: dict[str, int]
    seed: int


@dataclass(frozen=True, slots=True)
class Pair:
    """One aligned sentence pair for contrastive embedding fine-tuning."""

    query: str
    passage: str
    pair_type: PairType
    cluster_id: str
    source_id: str

    def as_row(self) -> dict[str, str]:
        return {
            "query": self.query,
            "passage": self.passage,
            "pair_type": self.pair_type,
            "cluster_id": self.cluster_id,
            "source_id": self.source_id,
        }


def collect_benchmark_fingerprints(
    benchmark_paths: Sequence[Path], normaliser: Normaliser | None = None
) -> set[bytes]:
    """Fingerprints of all evaluation sentences that must not enter training."""
    norm = normaliser or Normaliser()
    blocked: set[bytes] = set()
    for path in benchmark_paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_raw in handle:
                line = line_raw.strip()
                if not line:
                    continue
                text = line
                if line.startswith("{"):
                    try:
                        payload = json.loads(line)
                        text = str(
                            payload.get(
                                "sentence",
                                payload.get("text", payload.get("query", "")),
                            )
                        )
                    except Exception:  # noqa: S112
                        continue
                if text:
                    blocked.add(fingerprint(text))
                    blocked.add(fingerprint(norm.normalise(text)))
    return blocked


def _read_cluster_file(candidate: Path) -> dict[str, list[str]]:
    """Read cluster map from parquet, tsv, or jsonl."""
    clusters: dict[str, list[str]] = defaultdict(list)
    if candidate.suffix == ".parquet":
        for rec in read_parquet(candidate):
            cid = str(rec.get("cluster_id") or rec.get("paraphrase_id") or rec.get("id") or "")
            txt = str(rec.get("text") or rec.get("sentence") or rec.get("paraphrase") or "")
            if cid and txt:
                clusters[cid].append(txt)
    elif candidate.suffix == ".tsv":
        with candidate.open("r", encoding="utf-8", errors="replace") as handle:
            for line_raw in handle:
                parts = line_raw.strip().split("\t")
                if len(parts) >= MIN_TSV_PARTS:
                    clusters[parts[0]].append(parts[-1])
    elif candidate.suffix in (".jsonl", ".ndjson"):
        with candidate.open("r", encoding="utf-8", errors="replace") as handle:
            for line_raw in handle:
                line = line_raw.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    cid = str(p.get("cluster_id") or p.get("id") or "")
                    txt = str(p.get("text") or p.get("sentence") or "")
                    if cid and txt:
                        clusters[cid].append(txt)
                except Exception:  # noqa: S112
                    continue
    return clusters


def build_tapaco_pairs(
    path: Path,
    blocked: set[bytes],
    normaliser: Normaliser | None = None,
) -> list[Pair]:
    """Extract paraphrase pairs from TaPaCo cluster parquet, TSV, or JSONL files."""
    candidate = path
    if not candidate.is_file():
        alt = Path("data/raw/hf.tapaco-kab/kab/train-00000-of-00001.parquet")
        if alt.is_file():
            candidate = alt
        else:
            return []

    norm = normaliser or Normaliser()
    clusters = _read_cluster_file(candidate)

    pairs: list[Pair] = []
    for cid, sentences in clusters.items():
        clean_sentences = [norm.normalise(s) for s in sentences if s.strip()]
        if any(fingerprint(s) in blocked for s in clean_sentences):
            continue
        for i in range(len(clean_sentences)):
            for j in range(i + 1, len(clean_sentences)):
                s1, s2 = clean_sentences[i], clean_sentences[j]
                if s1 != s2:
                    pairs.append(
                        Pair(
                            query=s1,
                            passage=s2,
                            pair_type="paraphrase",
                            cluster_id=f"tapaco_{cid}",
                            source_id="tapaco",
                        )
                    )
    return pairs


def split_clusters(
    pairs: Sequence[Pair],
    *,
    dev_clusters: int = DEV_CLUSTERS,
    seed: int = 42,
) -> tuple[list[Pair], list[Pair]]:
    """Split pairs into train and dev sets by cluster id, preserving cluster integrity."""
    cluster_groups: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        cluster_groups[pair.cluster_id].append(pair)

    cluster_ids = sorted(cluster_groups.keys())
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(cluster_ids)

    dev_set = set(cluster_ids[:dev_clusters])
    train: list[Pair] = []
    dev: list[Pair] = []

    for cid, cluster_pairs in cluster_groups.items():
        if cid in dev_set:
            dev.extend(cluster_pairs)
        else:
            train.extend(cluster_pairs)

    return train, dev


def _parse_pair_row(payload: dict[str, Any], path_name: str) -> tuple[str, str, str] | None:
    """Extract (kab_text, other_text, other_lang) from arbitrary schema dict."""
    if "kab" in payload:
        kab = str(payload.get("kab") or "").strip()
        other = str(payload.get("eng") or payload.get("fra") or "").strip()
        lang = "eng" if "eng" in payload else "fra"
        return (kab, other, lang) if (kab and other) else None

    src_lang = str(payload.get("src_lang") or payload.get("source_lang") or "")
    tgt_lang = str(payload.get("tgt_lang") or payload.get("target_lang") or "")
    src_text = str(payload.get("src_text") or payload.get("source") or "").strip()
    tgt_text = str(payload.get("tgt_text") or payload.get("target") or "").strip()

    if not src_text or not tgt_text:
        return None

    if src_lang == "kab" or "kab" in path_name:
        return (src_text, tgt_text, tgt_lang or ("eng" if "eng" in path_name else "fra"))
    return (tgt_text, src_text, src_lang or ("eng" if "eng" in path_name else "fra"))


def build_parallel_pairs(
    parallel_dir: Path,
    blocked: set[bytes],
    normaliser: Normaliser | None = None,
    *,
    include_mined: bool = False,
) -> list[Pair]:
    """Read parallel sentence pairs (kab-eng, kab-fra) excluding mined bitext and defects."""
    if not parallel_dir.is_dir():
        return []
    norm = normaliser or Normaliser()
    pairs: list[Pair] = []
    seen: set[tuple[bytes, bytes]] = set()

    for path in sorted(parallel_dir.glob("*.jsonl")) + sorted(parallel_dir.glob("*.ndjson")):
        source_stem = path.stem
        if not include_mined and NLLB_MARKER in source_stem.lower():
            continue

        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for idx, line_raw in enumerate(handle):
                line = line_raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:  # noqa: S112
                    continue
                if not isinstance(payload, dict) or payload.get("defects"):
                    continue

                source_id = str(payload.get("source") or source_stem)
                if not include_mined and NLLB_MARKER in source_id.lower():
                    continue

                extracted = _parse_pair_row(payload, path.name)
                if extracted is None:
                    continue
                kab_text, other_text, other_lang = extracted

                kab_clean = norm.normalise(kab_text)
                other_clean = other_text.strip()

                fp_kab = fingerprint(kab_clean)
                fp_other = fingerprint(other_clean)
                if fp_kab in blocked or fp_other in blocked:
                    continue

                pair_key = (fp_kab, fp_other)
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                pair_type: PairType = (
                    "translation_eng" if "eng" in other_lang else "translation_fra"
                )
                pairs.append(
                    Pair(
                        query=kab_clean,
                        passage=other_clean,
                        pair_type=pair_type,
                        cluster_id=f"par_{source_id}_{idx}",
                        source_id=source_id,
                    )
                )
    return pairs


def write_split(path: Path, pairs: Iterable[Pair]) -> int:
    """Write a sequence of pairs to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.as_row(), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_embed_corpus(
    parallel_dir: Path = Path("data/interim/parallel"),
    tapaco_path: Path = Path("data/raw/tatoeba/tapaco_kab_2026-08-05.tsv"),
    output_dir: Path = Path("data/processed/embed"),
    dev_clusters: int = DEV_CLUSTERS,
    seed: int = 42,
) -> CorpusStats:
    """End-to-end dataset builder for contrastive sentence embedding fine-tuning."""
    normaliser = Normaliser()
    benchmarks = (
        Path("data/processed/benchmarks/flores_plus_dev.jsonl"),
        Path("data/processed/benchmarks/flores_plus_test.jsonl"),
    )
    blocked = collect_benchmark_fingerprints(benchmarks, normaliser=normaliser)

    tapaco_pairs = build_tapaco_pairs(tapaco_path, blocked, normaliser=normaliser)
    parallel_pairs = build_parallel_pairs(parallel_dir, blocked, normaliser=normaliser)

    all_pairs = tapaco_pairs + parallel_pairs
    train_pairs, dev_pairs = split_clusters(all_pairs, dev_clusters=dev_clusters, seed=seed)

    write_split(output_dir / "train.jsonl", train_pairs)
    write_split(output_dir / "dev.jsonl", dev_pairs)

    by_type: dict[str, int] = Counter(pair.pair_type for pair in all_pairs)
    by_src: dict[str, int] = Counter(pair.source_id for pair in all_pairs)
    clusters = {pair.cluster_id for pair in all_pairs}

    stats: CorpusStats = {
        "read_pairs": len(all_pairs),
        "leaks_excluded": len(blocked),
        "kept_pairs": len(all_pairs),
        "unique_clusters": len(clusters),
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "by_pair_type": dict(by_type),
        "by_source": dict(by_src),
        "seed": seed,
    }

    stats_path = output_dir / "corpus.stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats
