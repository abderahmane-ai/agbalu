"""Build the Pan-Amazigh multi-script training, dev, and test datasets.

Extracts, normalizes, and stratifies sentences across 13 Amazigh varieties and scripts,
and 9 negative distractor classes (audited by NUMALIA and gold regional corpora).
"""

from __future__ import annotations

import bz2
import csv
import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from agbalu.dihya.artifacts import atomic_file, write_json
from agbalu.dihya.config import DEFAULT_DATA_DIR
from agbalu.dihya.corpus import (
    NORMALIZATION_VERSION,
    RawRow,
    class_counts,
    evaluation_key,
    select_rows,
    split_rows,
)
from agbalu.dihya.dataset import Example
from agbalu.dihya.taxonomy import COMPOSITE_CLASSES, detect_dominant_script
from agbalu.dihya.train import emit_event
from agbalu.llm.provenance import code_stamp, config_hash, file_digest

DEFAULT_RAW_DIR: Final = Path("data/raw")
DEFAULT_SEED: Final = 20261101
DEFAULT_CAP: Final = 8_000
MIN_WORDS: Final = 3
MAX_WORDS: Final = 120
MAX_BYTES: Final = 512


def read_records(path: Path) -> list[dict[str, object]]:
    """Read Parquet at the typed serialization boundary."""
    rows = pq.read_table(path).to_pylist()
    return [{str(key): value for key, value in row.items()} for row in rows]


def evaluation_exclusions(raw_dir: Path) -> tuple[set[str], dict[str, str]]:
    """Exclude LID, FLORES and SIB evaluation texts and their sentence fragments."""
    paths = [
        Path("data/processed/bench/lid-eval.jsonl"),
        Path("data/processed/bench/flores-plus-kab_Latn-corrected.jsonl"),
    ]
    for directory in ("hf.flores-plus-kab", "hf.flores-plus-taq"):
        paths.extend(sorted((raw_dir / directory).rglob("*.jsonl")))
    parquet_paths = sorted((raw_dir / "hf.sib200-kab").rglob("*.parquet"))
    excluded: set[str] = set()
    digests: dict[str, str] = {}
    for path in [*paths, *parquet_paths]:
        digests[str(path)] = file_digest(path)
        if path.suffix == ".parquet":
            texts = [str(row["text"]) for row in read_records(path)]
        else:
            with path.open(encoding="utf-8") as stream:
                texts = [json.loads(line)["text"] for line in stream if line.strip()]
        for text in texts:
            excluded.add(evaluation_key(text))
            excluded.update(
                evaluation_key(chunk) for chunk in PUNCT_SPLIT_RE.split(text) if chunk.strip()
            )
    excluded.discard("")
    if not excluded:
        msg = "No external evaluation exclusions were loaded"
        raise ValueError(msg)
    return excluded, digests


def read_split(directory: Path, split: str) -> list[Example]:
    """Read a split only when its hash matches the committed dataset manifest."""
    manifest = json.loads((directory / "dataset.stats.json").read_text(encoding="utf-8"))
    path = directory / f"{split}.parquet"
    expected = manifest["splits"][split]["sha256"]
    if file_digest(path) != expected:
        msg = f"{path}: SHA256 does not match dataset.stats.json {expected}"
        raise ValueError(msg)
    rows = pq.read_table(path).to_pylist()
    examples: list[Example] = []
    for index, row in enumerate(rows):
        try:
            examples.append(
                Example(row["text"], row["label"], row["domain"], row["source"], row["script"])
            )
        except (ValueError, TypeError, KeyError) as error:
            msg = f"{path}:row:{index}: {error}"
            raise ValueError(msg) from error
    return examples


WS_RE: Final = re.compile(r"\s+")
PUNCT_SPLIT_RE: Final = re.compile(r"[.!?؟؛\n]+")
CLEAN_OCR_RE: Final = re.compile(r"[\^~_■©►•*§@&]+")
FRENCH_STOPS: Final = frozenset(
    {
        "le",
        "la",
        "les",
        "des",
        "du",
        "de",
        "un",
        "une",
        "est",
        "sont",
        "dans",
        "pour",
        "avec",
        "par",
        "sur",
        "qui",
        "que",
        "ce",
        "cette",
        "ces",
        "il",
        "elle",
        "ils",
        "elles",
        "nous",
        "vous",
    }
)
GRAMMAR_TERMS: Final = frozenset(
    {
        "ad.",
        "adj.",
        "adv.",
        "part.",
        "pron.",
        "prép.",
        "loc.",
        "ex.",
        "lit.",
        "syn.",
        "tome",
        "page",
        "nom",
        "verbe",
        "masculin",
        "féminin",
        "pluriel",
        "singulier",
    }
)


def clean_line(text: str) -> str:
    """Normalize whitespace and strip external artifacts."""
    return WS_RE.sub(" ", text).strip()


def is_valid_line(text: str) -> bool:
    """Check whether a line meets word count and byte budget invariants."""
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    return len(text.encode("utf-8")) <= MAX_BYTES


def read_numalia_negatives(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Algerian, Moroccan, and Standard Arabic negatives from NUMALIA export."""
    path = raw_dir / "numalia_negatives.jsonl"
    if not path.is_file():
        return

    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            lbl = row["label"]
            if counts[lbl] >= cap:
                continue

            text = clean_line(row["text"])
            if is_valid_line(text):
                counts[lbl] += 1
                yield RawRow(
                    text=text,
                    label=lbl,
                    domain=row.get("domain", "negative_dialect"),
                    source=row.get("source", "dzair.numalia"),
                    script=row.get("script", detect_dominant_script(text)),
                )


def read_toda_negatives(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Moroccan Darija in Arabizi and Arabic script from TODa parallel corpus."""
    path = raw_dir / "toda_sentences.csv"
    if not path.is_file():
        return

    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, 2):
            if counts["ary_Arabizi"] < cap:
                darija = clean_line(row.get("darija", ""))
                if is_valid_line(darija):
                    counts["ary_Arabizi"] += 1
                    yield RawRow(
                        text=darija,
                        label="ary_Arabizi",
                        domain="social",
                        source="toda.darija_arabizi",
                        script="Arabizi",
                        origin=f"{path}:{row_number}:darija",
                        family=f"toda:{row_number}",
                    )

            if counts["ary_Arab"] < cap:
                darija_ar = clean_line(row.get("darija_ar", ""))
                if is_valid_line(darija_ar):
                    counts["ary_Arab"] += 1
                    yield RawRow(
                        text=darija_ar,
                        label="ary_Arab",
                        domain="social",
                        source="toda.darija_arab",
                        script="Arab",
                        origin=f"{path}:{row_number}:darija_ar",
                        family=f"toda:{row_number}",
                    )

            if counts["ary_Arabizi"] >= cap and counts["ary_Arab"] >= cap:
                break


def read_tatoeba_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read multilingual sentences from Tatoeba parquet export (both source and target)."""
    pq_path = raw_dir / "hf.tamazight-nlp.tatoeba/data/train-00000-of-00001.parquet"
    if not pq_path.is_file():
        return

    table = pq.read_table(pq_path)
    target_langs = table["target_lang"].to_pylist()
    target_strings = table["target_string"].to_pylist()
    source_langs = table["source_lang"].to_pylist()
    source_strings = table["source_string"].to_pylist()

    mapping: dict[str, tuple[str, str]] = {
        "zgh": ("zgh_Tfng", "everyday"),
        "shy": ("shy_Latn", "everyday"),
        "rif": ("rif_Latn", "everyday"),
        "shi": ("shi_Latn", "everyday"),
        "kab": ("kab_Latn", "everyday"),
        "fra": ("fra_Latn", "negative_foreign"),
        "eng": ("eng_Latn", "negative_foreign"),
        "spa": ("spa_Latn", "negative_foreign"),
    }

    counts: Counter[str] = Counter()

    def process_pair(lang: str, string: object, role: str, sentence_id: int) -> Iterator[RawRow]:
        if lang not in mapping or not isinstance(string, str):
            return
        label, domain = mapping[lang]
        if counts[label] >= cap:
            return

        text = clean_line(string)
        if is_valid_line(text):
            detected_script = detect_dominant_script(text)
            if detected_script == "Arabizi":
                detected_script = "Latn"
            label = f"{lang}_{detected_script}"
            counts[label] += 1
            yield RawRow(
                text=text,
                label=label,
                domain=domain,
                source=f"tatoeba.{lang}.{role}",
                script=detected_script,
                origin=f"{pq_path}:{role}:{sentence_id}",
                family=f"tatoeba:{sentence_id}",
            )

    for l_tgt, s_tgt, l_src, s_src, id_tgt, id_src in zip(
        target_langs,
        target_strings,
        source_langs,
        source_strings,
        table["id_tgt"].to_pylist(),
        table["id_src"].to_pylist(),
        strict=True,
    ):
        yield from process_pair(l_tgt, s_tgt, "tgt", id_tgt)
        yield from process_pair(l_src, s_src, "src", id_src)


def _stream_tzm_candidates(raw_dir: Path) -> Iterator[RawRow]:
    # 1. Ali Amaniss Dictionnaire Tamazight-Français (guillemets phrases, French stopword filtered)
    amaniss_path = raw_dir / "dihya_remote/aliamanissdictionnairefrancaistamazight.txt"
    if amaniss_path.is_file():
        full_text = amaniss_path.read_text(encoding="utf-8")
        start_idx = full_text.find("B I R M")
        body = full_text[start_idx:] if start_idx != -1 else full_text[10000:]
        for g in re.findall(r"«([^»\n]+)»", body):
            s = clean_line(CLEAN_OCR_RE.sub(" ", g))
            if is_valid_line(s) and len(set(s.lower().split()).intersection(FRENCH_STOPS)) <= 1:
                yield RawRow(
                    text=s,
                    label="tzm_Latn",
                    domain="lexicon",
                    source="ali_amaniss",
                    script="Latn",
                )

    # 2. Weblate tzm translations
    weblate_path = raw_dir / "hf.tamazight-nlp.weblate-tzm/train.parquet"
    if weblate_path.is_file():
        table = pq.read_table(weblate_path)
        for val in table["target"]:
            s = clean_line(val.as_py())
            if is_valid_line(s):
                yield RawRow(
                    text=s,
                    label="tzm_Latn",
                    domain="interface",
                    source="weblate_tzm",
                    script="Latn",
                )

    # 3. Beni Mellal (tzm_Latn)
    bm_path = raw_dir / "hf.tamazight-nlp.beni-mellal/data/train-00000-of-00001.parquet"
    if bm_path.is_file():
        table = pq.read_table(bm_path, columns=["Tamazight"])
        col_name = "Tamazight" if "Tamazight" in table.column_names else "sentence"
        for val in table[col_name]:
            text = clean_line(val.as_py())
            if is_valid_line(text):
                yield RawRow(
                    text=text,
                    label="tzm_Latn",
                    domain="everyday",
                    source="beni_mellal",
                    script="Latn",
                )

    # 4. Translatewiki tzm
    tw_path = raw_dir / "hf.tamazight-nlp.translatewiki-siblings/en-tzm.csv"
    if tw_path.is_file():
        with tw_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for parts in reader:
                if len(parts) >= 2:
                    text = clean_line(parts[1])
                    if is_valid_line(text) and detect_dominant_script(text) == "Latn":
                        yield RawRow(
                            text=text,
                            label="tzm_Latn",
                            domain="interface",
                            source="translatewiki_tzm",
                            script="Latn",
                        )

    # 5. Pontoon tzm
    pontoon_path = raw_dir / "hf.tamazight-nlp.pontoon-siblings/en-tzm.tsv"
    if pontoon_path.is_file():
        with pontoon_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    text = clean_line(parts[1])
                    if is_valid_line(text) and detect_dominant_script(text) == "Latn":
                        yield RawRow(
                            text=text,
                            label="tzm_Latn",
                            domain="interface",
                            source="pontoon_tzm",
                            script="Latn",
                        )

    # 6. Dihya-5M bench dataset gold rows
    for split_name in ("train.jsonl", "dev.jsonl"):
        split_path = Path("data/processed/bench/lid-dataset") / split_name
        if split_path.is_file():
            with split_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("language") == "tzm_Latn":
                        text = clean_line(row["text"])
                        if is_valid_line(text):
                            yield RawRow(
                                text=text,
                                label="tzm_Latn",
                                domain="bench_gold",
                                source="lid_dataset_tzm",
                                script="Latn",
                            )

    # 7. GlotCC tzm-Latn
    glot_dir = raw_dir / "hf.glotcc-v1-tzm/v1.0/tzm-Latn"
    if glot_dir.is_dir():
        for pq_file in sorted(glot_dir.rglob("*.parquet")):
            table = pq.read_table(pq_file, columns=["content"])
            for val in table.column("content"):
                doc = val.as_py()
                if isinstance(doc, str):
                    for chunk in PUNCT_SPLIT_RE.split(doc):
                        text = clean_line(chunk)
                        if is_valid_line(text):
                            yield RawRow(
                                text=text,
                                label="tzm_Latn",
                                domain="web",
                                source="glotcc.tzm",
                                script="Latn",
                            )


def read_tzm_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Central Atlas Tamazight (tzm_Latn) from all available gold corpora."""
    for count, row in enumerate(_stream_tzm_candidates(raw_dir)):
        if count >= cap:
            break
        yield row


def read_tzm_tifinagh_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Central Atlas / Standard Moroccan Tamazight in Neo-Tifinagh script."""
    count = 0

    # 1. NLLB seed tzm_Tfng
    seed_path = raw_dir / "hf.tamazight-nlp.nllb-seed-tzm/eng_Latn-tzm_Tfng.csv"
    if seed_path.is_file():
        with seed_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for parts in reader:
                if len(parts) >= 2:
                    for chunk in PUNCT_SPLIT_RE.split(parts[1]):
                        text = clean_line(chunk)
                        if is_valid_line(text) and detect_dominant_script(text) == "Tfng":
                            count += 1
                            yield RawRow(
                                text=text,
                                label="tzm_Tfng",
                                domain="encyclopaedia",
                                source="nllb_seed_tzm",
                                script="Tfng",
                            )
                            if count >= cap:
                                return

    # 2. Weblate zgh/tzm in Tifinagh
    weblate_zgh = raw_dir / "hf.tamazight-nlp.weblate-zgh/train.parquet"
    if weblate_zgh.is_file():
        table = pq.read_table(weblate_zgh)
        for val in table["target"]:
            text = clean_line(val.as_py())
            if is_valid_line(text) and detect_dominant_script(text) == "Tfng":
                count += 1
                yield RawRow(
                    text=text,
                    label="zgh_Tfng",
                    domain="interface",
                    source="weblate_zgh_tfng",
                    script="Tfng",
                )
                if count >= cap:
                    return

    # 3. GlotCC tzm-Tfng
    glot_dir = raw_dir / "hf.glotcc-v1-tzm/v1.0/tzm-Tfng"
    if glot_dir.is_dir():
        for pq_file in sorted(glot_dir.rglob("*.parquet")):
            table = pq.read_table(pq_file, columns=["content"])
            for val in table.column("content"):
                doc = val.as_py()
                if isinstance(doc, str):
                    for chunk in PUNCT_SPLIT_RE.split(doc):
                        text = clean_line(chunk)
                        if is_valid_line(text) and detect_dominant_script(text) == "Tfng":
                            count += 1
                            yield RawRow(
                                text=text,
                                label="tzm_Tfng",
                                domain="web",
                                source="glotcc.tzm_tfng",
                                script="Tfng",
                            )
                            if count >= cap:
                                return


def read_tashelhit_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Tashelhit in both Latin (shi_Latn) and Neo-Tifinagh (shi_Tfng)."""
    counts: Counter[str] = Counter()

    # 1. Parallel Tatoeba Tashelhit TSV
    tsv_path = raw_dir / "hf.abdelhaqueidali.shi-tifinagh/shi_tifinagh.tsv"
    if tsv_path.is_file():
        with tsv_path.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    tfng, latn = clean_line(parts[1]), clean_line(parts[2])
                    if counts["shi_Tfng"] < cap and is_valid_line(tfng):
                        counts["shi_Tfng"] += 1
                        yield RawRow(
                            text=tfng,
                            label="shi_Tfng",
                            domain="everyday",
                            source="idali.shi_tfng",
                            script="Tfng",
                            family=f"tatoeba:{parts[0]}",
                        )
                    if counts["shi_Latn"] < cap and is_valid_line(latn):
                        counts["shi_Latn"] += 1
                        yield RawRow(
                            text=latn,
                            label="shi_Latn",
                            domain="everyday",
                            source="idali.shi_latn",
                            script="Latn",
                            family=f"tatoeba:{parts[0]}",
                        )

    # 2. OPUS Tashelhit scripture
    bible_path = raw_dir / "opus.bible-uedin-shi/shi.txt.gz"
    if bible_path.is_file() and counts["shi_Latn"] < cap:
        with gzip.open(bible_path, "rt", encoding="utf-8") as f:
            for line in f:
                text = clean_line(line)
                if is_valid_line(text):
                    counts["shi_Latn"] += 1
                    yield RawRow(
                        text=text,
                        label="shi_Latn",
                        domain="scripture",
                        source="opus.bible.shi",
                        script="Latn",
                    )
                    if counts["shi_Latn"] >= cap:
                        break

    # 3. Dihya-5M bench dataset gold rows
    split_path = Path("data/processed/bench/lid-dataset/train.jsonl")
    if split_path.is_file() and counts["shi_Latn"] < cap:
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("language") == "shi_Latn":
                    text = clean_line(row["text"])
                    if is_valid_line(text):
                        counts["shi_Latn"] += 1
                        yield RawRow(
                            text=text,
                            label="shi_Latn",
                            domain="bench_gold",
                            source="lid_dataset_shi",
                            script="Latn",
                        )
                        if counts["shi_Latn"] >= cap:
                            break


def read_tarifit_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Tarifit in Latin (rif_Latn) and Neo-Tifinagh (rif_Tfng)."""
    counts: Counter[str] = Counter()

    # 1. NordinB Tarifit Bible JSONL (Latin)
    bible_jsonl = raw_dir / "hf.nordinb.tarifit-bible/tarifit_bible_dataset.jsonl"
    if bible_jsonl.is_file():
        with bible_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = clean_line(row.get("text", ""))
                if is_valid_line(text):
                    counts["rif_Latn"] += 1
                    yield RawRow(
                        text=text,
                        label="rif_Latn",
                        domain="scripture",
                        source="nordinb.tarifit_bible",
                        script="Latn",
                    )
                    if counts["rif_Latn"] >= cap:
                        break

    # 2. Historic Tarifit New Testament (Latin)
    rif_nt = raw_dir / "archive.tarifit-nt/Latin-MERGED_djvu.txt"
    if rif_nt.is_file() and counts["rif_Latn"] < cap:
        with rif_nt.open("r", encoding="utf-8") as f:
            for line in f:
                text = clean_line(line)
                if is_valid_line(text):
                    counts["rif_Latn"] += 1
                    yield RawRow(
                        text=text,
                        label="rif_Latn",
                        domain="scripture",
                        source="tarifit_nt",
                        script="Latn",
                    )
                    if counts["rif_Latn"] >= cap:
                        break

    # 3. IRCAM Tarifit stories in Neo-Tifinagh
    ircam_dir = raw_dir / "hf.tamazight-nlp.ircam-corpus"
    if ircam_dir.is_dir() and counts["rif_Tfng"] < cap:
        for txt_file in sorted(ircam_dir.glob("*.txt")):
            content = txt_file.read_text(encoding="utf-8")
            for chunk in PUNCT_SPLIT_RE.split(content):
                text = clean_line(chunk)
                if is_valid_line(text) and detect_dominant_script(text) == "Tfng":
                    counts["rif_Tfng"] += 1
                    yield RawRow(
                        text=text,
                        label="rif_Tfng",
                        domain="oral_heritage",
                        source="ircam.tarifit_stories",
                        script="Tfng",
                    )
                    if counts["rif_Tfng"] >= cap:
                        break
            if counts["rif_Tfng"] >= cap:
                break

    # 4. 1002-Amazigh-Proverbs (rif_Tfng)
    proverbs_path = raw_dir / "hf.tamazight-nlp.1002-amazigh-proverbs/data.json"
    if proverbs_path.is_file() and counts["rif_Tfng"] < cap:
        data = json.loads(proverbs_path.read_text(encoding="utf-8"))
        for item in data:
            if item.get("dialect") == "rif":
                for key in ("amazigh", "amazigh_explanation"):
                    raw_s = item.get(key, "")
                    for chunk in PUNCT_SPLIT_RE.split(raw_s):
                        text = clean_line(chunk)
                        if is_valid_line(text) and detect_dominant_script(text) == "Tfng":
                            counts["rif_Tfng"] += 1
                            yield RawRow(
                                text=text,
                                label="rif_Tfng",
                                domain="oral_heritage",
                                source="proverbs.rif_tfng",
                                script="Tfng",
                            )
                            if counts["rif_Tfng"] >= cap:
                                break
                if counts["rif_Tfng"] >= cap:
                    break


def read_shawiya_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Shawiya in Latin script (shy_Latn)."""
    count = 0

    # 1. Tatoeba Shawiya export
    p_tatoeba = raw_dir / "tatoeba_shy.tsv.bz2"
    if p_tatoeba.is_file():
        with bz2.open(p_tatoeba, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    text = clean_line(parts[2])
                    if is_valid_line(text) and detect_dominant_script(text) == "Latn":
                        count += 1
                        yield RawRow(
                            text=text,
                            label="shy_Latn",
                            domain="everyday",
                            source="tatoeba.shy",
                            script="Latn",
                        )
                        if count >= cap:
                            return

    # 2. Shawiya Wiktionary dump (* and #* and #: entry lines)
    shywikt_path = raw_dir / "wikimedia.shywiktionary/shywiktionary-20260801-pages-articles.xml.bz2"
    if shywikt_path.is_file() and count < cap:
        ns = "{http://www.mediawiki.org/xml/export-0.11/}"
        with bz2.open(shywikt_path, "rb") as f:
            for _, elem in ET.iterparse(f, events=("end",)):
                if not elem.tag.endswith("}page"):
                    continue
                rev_elem = elem.find(f"{ns}revision")
                text_elem = rev_elem.find(f"{ns}text") if rev_elem is not None else None
                if text_elem is not None and text_elem.text:
                    for line in text_elem.text.splitlines():
                        line_s = line.strip()
                        if line_s.startswith(("*", "#*", "#:")) and not line_s.startswith(
                            ("*#", "#* {{", "* {{")
                        ):
                            stripped = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", line_s)
                            stripped = re.sub(r"\{\{[^}]+\}\}", " ", stripped)
                            stripped = re.sub(r"[*#:]+", " ", stripped)
                            clean_t = clean_line(stripped)
                            if is_valid_line(clean_t) and not any(
                                w in clean_t.lower()
                                for w in ["http", "www", "wiki", "tome", "page", "category", "user"]
                            ):
                                count += 1
                                yield RawRow(
                                    text=clean_t,
                                    label="shy_Latn",
                                    domain="lexicon",
                                    source="wikimedia.shywiktionary",
                                    script="Latn",
                                )
                                if count >= cap:
                                    break
                elem.clear()
                if count >= cap:
                    break

    # 3. Shawiya translatewiki
    tw_shy = raw_dir / "hf.tamazight-nlp.translatewiki-siblings/en-shy.csv"
    if tw_shy.is_file() and count < cap:
        with tw_shy.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for parts in reader:
                if len(parts) >= 2:
                    text = clean_line(parts[1])
                    if is_valid_line(text):
                        count += 1
                        yield RawRow(
                            text=text,
                            label="shy_Latn",
                            domain="interface",
                            source="translatewiki.shy",
                            script="Latn",
                        )
                        if count >= cap:
                            break

    # 4. Dihya-5M bench dataset gold rows
    for split_name in ("train.jsonl", "dev.jsonl"):
        split_path = Path("data/processed/bench/lid-dataset") / split_name
        if split_path.is_file() and count < cap:
            with split_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("language") == "shy_Latn":
                        text = clean_line(row["text"])
                        if is_valid_line(text):
                            count += 1
                            yield RawRow(
                                text=text,
                                label="shy_Latn",
                                domain="bench_gold",
                                source="lid_dataset_shy",
                                script="Latn",
                            )
                            if count >= cap:
                                break


def read_tuareg_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Tamasheq Tuareg in Latin (taq_Latn) and Tifinagh (taq_Tfng)."""
    counts: Counter[str] = Counter()

    # 1. NLLB seed Tamasheq Latin
    taq_latn_path = raw_dir / "hf.tamazight-nlp.nllb-seed-taq/eng_Latn-taq_Latn.csv"
    if taq_latn_path.is_file():
        with taq_latn_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for parts in reader:
                if len(parts) >= 2:
                    text = clean_line(parts[1])
                    if is_valid_line(text):
                        counts["taq_Latn"] += 1
                        yield RawRow(
                            text=text,
                            label="taq_Latn",
                            domain="encyclopaedia",
                            source="nllb_seed_taq_latn",
                            script="Latn",
                        )
                        if counts["taq_Latn"] >= cap:
                            break

    # 2. NLLB seed Tamasheq Tifinagh
    taq_tfng_path = raw_dir / "hf.tamazight-nlp.nllb-seed-taq-tfng/eng_Latn-taq_Tfng.csv"
    if taq_tfng_path.is_file():
        with taq_tfng_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for parts in reader:
                if len(parts) >= 2:
                    for chunk in PUNCT_SPLIT_RE.split(parts[1]):
                        text = clean_line(chunk)
                        if is_valid_line(text) and detect_dominant_script(text) == "Tfng":
                            counts["taq_Tfng"] += 1
                            yield RawRow(
                                text=text,
                                label="taq_Tfng",
                                domain="encyclopaedia",
                                source="nllb_seed_taq_tfng",
                                script="Tfng",
                            )
                            if counts["taq_Tfng"] >= cap:
                                break
                if counts["taq_Tfng"] >= cap:
                    break

    # 3. Dihya-5M bench dataset gold rows (taq_Latn)
    split_path = Path("data/processed/bench/lid-dataset/train.jsonl")
    if split_path.is_file() and counts["taq_Latn"] < cap:
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("language") == "taq_Latn":
                    text = clean_line(row["text"])
                    if is_valid_line(text):
                        counts["taq_Latn"] += 1
                        yield RawRow(
                            text=text,
                            label="taq_Latn",
                            domain="bench_gold",
                            source="lid_dataset_taq",
                            script="Latn",
                        )
                        if counts["taq_Latn"] >= cap:
                            break


def read_mozabite_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Tumẓabt / Mozabite (mzb_Latn) from official ASPT Dictionnaire de Tumzabt."""
    path = raw_dir / "dihya_remote/aspt-2015-dictionnaire-de-tum-zabt-lexique-fra-mzb-v-15.10.txt"
    if not path.is_file():
        return

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = 0
    for line_entry in lines[1000:]:
        parts = line_entry.split(",")
        cand = clean_line(parts[0])
        words = cand.split()
        lower_words = set(cand.lower().split())
        if (
            3 <= len(words) <= 25
            and len(cand.encode("utf-8")) <= MAX_BYTES
            and not lower_words.intersection(FRENCH_STOPS)
            and not lower_words.intersection(GRAMMAR_TERMS)
        ):
            count += 1
            yield RawRow(
                text=cand,
                label="mzb_Latn",
                domain="lexicon",
                source="aspt_tumzabt",
                script="Latn",
            )
            if count >= cap:
                break


def read_chinese_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read Chinese negative sentences from cached Tatoeba export."""
    cache_path = raw_dir / "cmn_sentences.tsv.bz2"
    if not cache_path.is_file():
        return

    count = 0
    with bz2.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                text = clean_line(parts[2])
                if len(text) >= 4 and len(text.encode("utf-8")) <= MAX_BYTES:
                    count += 1
                    yield RawRow(
                        text=text,
                        label="zho_Hani",
                        domain="negative_foreign",
                        source="tatoeba.cmn",
                        script="Hani",
                    )
                    if count >= cap:
                        break


def read_kabyle_sources(raw_dir: Path, cap: int = DEFAULT_CAP) -> Iterator[RawRow]:
    """Read gold Kabyle text corpus in Latin (kab_Latn) and Tifinagh (kab_Tfng)."""
    # 1. Agbalu Kabyle gold corpus (kab_Latn)
    agbalu_text = Path("data/processed/text/agbalu-text-v1.jsonl")
    if agbalu_text.is_file():
        count = 0
        with agbalu_text.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = clean_line(row.get("text", ""))
                if is_valid_line(text):
                    count += 1
                    yield RawRow(
                        text=text,
                        label="kab_Latn",
                        domain="everyday",
                        source="agbalu.text",
                        script="Latn",
                    )
                    if count >= cap:
                        break

    # 2. Neo-Tifinagh Kabyle corpus (kab_Tfng)
    tfng_path = raw_dir / "hf.abdelhaqueidali.kab-latn-tfng/kab_tifinagh.tsv"
    if tfng_path.is_file():
        count = 0
        with tfng_path.open("r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    tfng_text = clean_line(parts[1])
                    if is_valid_line(tfng_text):
                        count += 1
                        yield RawRow(
                            text=tfng_text,
                            label="kab_Tfng",
                            domain="everyday",
                            source="kab_tifinagh_idali",
                            script="Tfng",
                            family=f"tatoeba:{parts[0]}",
                        )
                        if count >= cap:
                            break


def build_dihya_dataset(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_DATA_DIR,
    cap: int = DEFAULT_CAP,
    seed: int = DEFAULT_SEED,
) -> None:
    """Build the full stratified train, dev, and test datasets for Dihya-22M."""
    out_dir.mkdir(parents=True, exist_ok=True)
    emit_event("prepare", status="extracting_sources", raw_dir=str(raw_dir))
    readers = (
        read_tatoeba_sources,
        read_tzm_sources,
        read_tzm_tifinagh_sources,
        read_tashelhit_sources,
        read_tarifit_sources,
        read_shawiya_sources,
        read_tuareg_sources,
        read_mozabite_sources,
        read_numalia_negatives,
        read_toda_negatives,
        read_chinese_sources,
        read_kabyle_sources,
    )

    def candidates() -> Iterator[RawRow]:
        for reader in readers:
            count = 0
            for row in reader(raw_dir, cap=sys.maxsize):
                count += 1
                yield row
            emit_event("source", reader=reader.__name__, candidate_rows=count)

    excluded, exclusion_inputs = evaluation_exclusions(raw_dir)
    selected, rejects, conservation = select_rows(candidates(), cap, seed, excluded)
    splits = split_rows(selected, seed)
    write_json(out_dir / "exclusions.json", sorted(excluded))
    manifest: dict[str, object] = {}
    for name, split in splits.items():
        path = out_dir / f"{name}.parquet"
        with atomic_file(path) as stream:
            pq.write_table(pa.Table.from_pylist([asdict(row) for row in split]), stream)
        manifest[name] = {
            "rows": len(split),
            "sha256": file_digest(path),
            "classes": class_counts(split),
            "cells": dict(Counter(f"{row.label}/{row.domain}" for row in split)),
        }
    with atomic_file(out_dir / "rejects.jsonl") as stream:
        for row, reason in rejects:
            stream.write(
                (json.dumps({**asdict(row), "reason": reason}, ensure_ascii=False) + "\n").encode()
            )
    stats = {
        "format_version": 2,
        "normalization_version": NORMALIZATION_VERSION,
        "code": code_stamp(),
        "version_tag": None,
        "config_hash": config_hash({"cap": cap, "seed": seed}),
        "seed": seed,
        "conservation": conservation,
        "classes": class_counts(selected),
        "splits": manifest,
        "reject_reasons": dict(Counter(reason for _, reason in rejects)),
        "rejects_sha256": file_digest(out_dir / "rejects.jsonl"),
        "external_exclusion_inputs": exclusion_inputs,
        "external_exclusion_keys": len(excluded),
        "exclusions_sha256": file_digest(out_dir / "exclusions.json"),
        "missing_licence_rows": sum(row.licence is None for row in selected),
        "missing_retrieval_date_rows": sum(row.retrieved is None for row in selected),
        "release_ready": False,
    }
    write_json(out_dir / "dataset.stats.json", stats)
    emit_event("prepare", status="completed", **conservation, classes=len(COMPOSITE_CLASSES))
