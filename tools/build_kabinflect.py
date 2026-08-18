"""Build `agbalu/KabInflect` from the raw verb tables.

`hf.boffire.kabyle-verbs` is a French-language conjugation resource: its tense and person
labels are `aoriste intensif` and `3s_f`, which are not comparable to anything. They are
mapped here onto UD `FEATS`, so a Kabyle inflection result can be read beside every other
language's.

**The split is by lemma, not by form**, and that is the whole design. Kabyle verb
morphology is templatic, so two forms of one verb share almost every character: a split
by row puts `nɛedder` in train and `tɛeddreḍ` in test and measures memorisation. Splitting
by lemma means a test verb's paradigm has been seen in no cell at all.

Writes `data/tasks/inflection/` only. Staging for the Hub is `tools.export_datasets`,
which is where the card and the metadata are validated.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from agbalu.normalise import Normaliser

log: Final = logging.getLogger("agbalu.tools.build_kabinflect")

SEED: Final = 42
TRAIN_RATIO: Final = 0.80
DEV_RATIO: Final = 0.10

RAW_DIR: Final = Path("data/raw/hf.boffire.kabyle-verbs")
OUT_DIR: Final = Path("data/tasks/inflection")

TENSE_MAP: Final[dict[str, dict[str, str]]] = {
    "aoriste": {"Aspect": "Prosp", "Mood": "Ind"},
    "aoriste intensif": {"Aspect": "Imp", "Mood": "Ind"},
    "prétérit": {"Aspect": "Perf", "Mood": "Ind"},
    "prétérit négatif": {"Aspect": "Perf", "Mood": "Ind", "Polarity": "Neg"},
    "impératif": {"Aspect": "Prosp", "Mood": "Imp"},
    "impératif intensif": {"Aspect": "Imp", "Mood": "Imp"},
    "participe aoriste": {"Aspect": "Prosp", "VerbForm": "Part"},
    "participe aoriste intensif": {"Aspect": "Imp", "VerbForm": "Part"},
    "participe prétérit": {"Aspect": "Perf", "VerbForm": "Part"},
    "participe aoriste intensif négatif": {"Aspect": "Imp", "VerbForm": "Part", "Polarity": "Neg"},
    "participe prétérit négatif": {"Aspect": "Perf", "VerbForm": "Part", "Polarity": "Neg"},
}

PERSON_MAP: Final[dict[str, dict[str, str]]] = {
    "1s": {"Person": "1", "Number": "Sing"},
    "2s": {"Person": "2", "Number": "Sing"},
    "3s_m": {"Person": "3", "Number": "Sing", "Gender": "Masc"},
    "3s_f": {"Person": "3", "Number": "Sing", "Gender": "Fem"},
    "1p": {"Person": "1", "Number": "Plur"},
    "2p_m": {"Person": "2", "Number": "Plur", "Gender": "Masc"},
    "2p_f": {"Person": "2", "Number": "Plur", "Gender": "Fem"},
    "3p_m": {"Person": "3", "Number": "Plur", "Gender": "Masc"},
    "3p_f": {"Person": "3", "Number": "Plur", "Gender": "Fem"},
    "participe": {"VerbForm": "Part"},
}


class SplitError(Exception):
    """A partition that would leak a paradigm across splits."""


def build_feats_str(tense: str, person: str) -> str:
    """Build canonical UD FEATS string from tense and person descriptors."""
    feats: dict[str, str] = {}
    feats.update(TENSE_MAP.get(tense, {}))
    feats.update(PERSON_MAP.get(person, {}))
    if not feats:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


def _field(norm: Normaliser, row: dict[str, object], key: str) -> str:
    """One raw column, normalised. Absent and empty both read as the empty string."""
    return norm.normalise(str(row.get(key) or "").strip())


def read_forms(norm: Normaliser) -> dict[str, list[dict[str, str]]]:
    """Every inflected form the raw shards carry, grouped by its lemma."""
    log.info("reading verb shards from %s", RAW_DIR)

    lemmatizer_shards = sorted((RAW_DIR / "lemmatizer").glob("*.parquet"))
    if not lemmatizer_shards:
        message = f"no parquet shards under {RAW_DIR / 'lemmatizer'}; run `make acquire` first"
        raise FileNotFoundError(message)

    entries_by_lemma: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, ...]] = set()
    total_raw_rows = 0
    duplicates = 0

    for shard in lemmatizer_shards:
        table = pq.read_table(shard)
        for row in table.to_pylist():
            total_raw_rows += 1
            form = norm.normalise(str(row.get("form") or "").strip())
            lemma = norm.normalise(str(row.get("infinitif") or "").strip())
            tense = str(row.get("tense") or "").strip()
            person = str(row.get("person") or "").strip()

            if not form or not lemma:
                continue

            feats = build_feats_str(tense, person)

            record = {
                "form": form,
                "lemma": lemma,
                "tense_raw": tense,
                "person_raw": person,
                "feats": feats,
            }
            # The raw shards repeat rows verbatim — one verb's intensive stems can generate
            # a cell twice. An identical row carries no information and only reweights the
            # form during training, so it is dropped here rather than published.
            key = (form, lemma, tense, person, feats)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            entries_by_lemma.setdefault(lemma, []).append(record)

    log.info(
        "%d raw rows, %d duplicates dropped, %d kept over %d lemmas",
        total_raw_rows,
        duplicates,
        total_raw_rows - duplicates,
        len(entries_by_lemma),
    )
    return entries_by_lemma


def partition(entries_by_lemma: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    """Split by lemma, so no test verb's paradigm was seen in any cell."""
    unique_lemmas = sorted(entries_by_lemma)
    shuffled_lemmas = list(unique_lemmas)
    random.Random(SEED).shuffle(shuffled_lemmas)  # noqa: S311 — a seeded split

    n_lemmas = len(shuffled_lemmas)
    dev_start = int(n_lemmas * TRAIN_RATIO)
    test_start = int(n_lemmas * (TRAIN_RATIO + DEV_RATIO))

    train_lemmas = set(shuffled_lemmas[:dev_start])
    dev_lemmas = set(shuffled_lemmas[dev_start:test_start])
    test_lemmas = set(shuffled_lemmas[test_start:])

    log.info("lemmas train=%d dev=%d test=%d", len(train_lemmas), len(dev_lemmas), len(test_lemmas))
    # Checked on the built sets rather than trusted from the slice arithmetic: an
    # invariant asserted on the artifact is what caught the held-out split that leaked
    # through a partner in Phase 11.
    leaked = (train_lemmas & dev_lemmas) | (train_lemmas & test_lemmas) | (dev_lemmas & test_lemmas)
    if leaked:
        message = f"{len(leaked)} lemmas appear in more than one split, e.g. {sorted(leaked)[:3]}"
        raise SplitError(message)

    # Build split records
    splits: dict[str, list[dict[str, str]]] = {"train": [], "dev": [], "test": []}
    for lemma, records in entries_by_lemma.items():
        if lemma in train_lemmas:
            split_name = "train"
        elif lemma in dev_lemmas:
            split_name = "dev"
        else:
            split_name = "test"

        splits[split_name].extend(records)

    for name, records in splits.items():
        log.info("%s: %d forms", name, len(records))
    return splits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    norm = Normaliser()
    entries_by_lemma = read_forms(norm)
    splits = partition(entries_by_lemma)
    train_lemmas = {r["lemma"] for r in splits["train"]}
    dev_lemmas = {r["lemma"] for r in splits["dev"]}
    test_lemmas = {r["lemma"] for r in splits["test"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Export Inflection Config Parquet
    inflection_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("lemma", pa.string()),
            pa.field("feats", pa.string()),
            pa.field("tense_raw", pa.string()),
            pa.field("person_raw", pa.string()),
            pa.field("form", pa.string()),
        ]
    )

    for split_name, records in splits.items():
        rows = []
        for idx, r in enumerate(records):
            rows.append(
                {
                    "id": f"kab_inflect_{split_name}_{idx:06d}",
                    "lemma": r["lemma"],
                    "feats": r["feats"],
                    "tense_raw": r["tense_raw"],
                    "person_raw": r["person_raw"],
                    "form": r["form"],
                }
            )
        table = pa.Table.from_pylist(rows, schema=inflection_schema)

        # Write to task dir and release dir
        for base_path in (OUT_DIR,):
            config_dir = base_path / "inflection"
            config_dir.mkdir(parents=True, exist_ok=True)
            out_file = config_dir / f"{split_name}.parquet"
            pq.write_table(table, out_file, compression="snappy")

    # 2. Export Analysis Config Parquet
    analysis_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("form", pa.string()),
            pa.field("lemma", pa.string()),
            pa.field("feats", pa.string()),
            pa.field("tense_raw", pa.string()),
            pa.field("person_raw", pa.string()),
        ]
    )

    for split_name, records in splits.items():
        rows = []
        for idx, r in enumerate(records):
            rows.append(
                {
                    "id": f"kab_analysis_{split_name}_{idx:06d}",
                    "form": r["form"],
                    "lemma": r["lemma"],
                    "feats": r["feats"],
                    "tense_raw": r["tense_raw"],
                    "person_raw": r["person_raw"],
                }
            )
        table = pa.Table.from_pylist(rows, schema=analysis_schema)

        for base_path in (OUT_DIR,):
            config_dir = base_path / "analysis"
            config_dir.mkdir(parents=True, exist_ok=True)
            out_file = config_dir / f"{split_name}.parquet"
            pq.write_table(table, out_file, compression="snappy")

    # 3. Export Conjugation Table Paradigms Config Parquet
    log.info("exporting conjugation tables")
    conj_shards = sorted((RAW_DIR / "conjugation-tables").glob("*.parquet"))
    conj_rows = []

    paradigm_schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("translation", pa.string()),
            pa.field("is_irregular", pa.bool_()),
            pa.field("is_derived", pa.bool_()),
            pa.field("pattern_verb", pa.string()),
            pa.field("imperative", pa.string()),
            pa.field("aorist", pa.string()),
            pa.field("preterite", pa.string()),
            pa.field("negative_preterite", pa.string()),
            pa.field("aorist_participle", pa.string()),
            pa.field("preterite_participle", pa.string()),
            pa.field("negative_preterite_participle", pa.string()),
            pa.field("intensive_forms", pa.string()),
        ]
    )

    p_idx = 0
    for shard in conj_shards:
        table = pq.read_table(shard)
        for row in table.to_pylist():
            name = norm.normalise(str(row.get("name") or "").strip())
            if not name:
                continue
            conj_rows.append(
                {
                    "id": f"kab_paradigm_{p_idx:05d}",
                    "name": name,
                    "translation": norm.normalise(str(row.get("translation") or "").strip()),
                    "is_irregular": bool(row.get("isIrregular")),
                    "is_derived": bool(row.get("isDerived")),
                    "pattern_verb": norm.normalise(str(row.get("pattern_verb") or "").strip()),
                    "imperative": norm.normalise(str(row.get("imperative") or "").strip()),
                    "aorist": _field(norm, row, "aorist"),
                    "preterite": _field(norm, row, "preterite"),
                    "negative_preterite": _field(norm, row, "negativePreterite"),
                    "aorist_participle": _field(norm, row, "aoristParticiple"),
                    "preterite_participle": _field(norm, row, "preteriteParticiple"),
                    "negative_preterite_participle": _field(
                        norm, row, "negativePreteriteParticiple"
                    ),
                    "intensive_forms": _field(norm, row, "intensiveForms"),
                }
            )
            p_idx += 1

    p_table = pa.Table.from_pylist(conj_rows, schema=paradigm_schema)
    for base_path in (OUT_DIR,):
        config_dir = base_path / "paradigms"
        config_dir.mkdir(parents=True, exist_ok=True)
        out_file = config_dir / "train.parquet"
        pq.write_table(p_table, out_file, compression="snappy")

    # 4. Generate dataset_info.json
    dataset_info = {
        "dataset_name": "agbalu/KabInflect",
        "description": "Kabyle verb inflection, analysis and conjugation tables",
        "language": "kab_Latn",
        "license": "CC-BY-4.0",
        "total_inflected_forms": sum(len(rows) for rows in splits.values()),
        "unique_lemmas": len(train_lemmas | dev_lemmas | test_lemmas),
        "unique_paradigms": len(conj_rows),
        "splits": {
            "train": {"lemmas": len(train_lemmas), "forms": len(splits["train"])},
            "dev": {"lemmas": len(dev_lemmas), "forms": len(splits["dev"])},
            "test": {"lemmas": len(test_lemmas), "forms": len(splits["test"])},
        },
        "source": "hf.boffire.kabyle-verbs (CC-BY-4.0)",
    }

    for base_path in (OUT_DIR,):
        info_file = base_path / "dataset_info.json"
        info_file.write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")

    log.info("wrote %s", OUT_DIR)


if __name__ == "__main__":
    main()
