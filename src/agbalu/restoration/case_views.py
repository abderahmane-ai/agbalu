"""Rebuild component casing from the accepted restoration references."""

import shutil
import sqlite3
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from agbalu.encoder.corpus import tokenizer, write_json
from agbalu.llm.progress import Heartbeat, Tracker, stage
from agbalu.llm.provenance import code_stamp, config_hash, file_digest
from agbalu.restoration.annotations import annotate
from agbalu.restoration.configuration import Locations, Settings
from agbalu.restoration.corpus import FORMAT_VERSION, Writer, verify


def revise_view(locations: Locations, settings: Settings) -> dict[str, object]:
    source = locations.source_data
    parent = verify(source)
    identity = {
        "parent_stats_sha256": file_digest(source / "corpus.stats.json"),
        "tokenizer_sha256": file_digest(locations.tokenizer),
        "case_source": {
            name: file_digest(Path(__file__).with_name(name))
            for name in ("casing.py", "alignment.py", "case_views.py", "corpus.py")
        },
    }
    policy = config_hash(identity)
    if (locations.data / "corpus.stats.json").exists():
        existing = verify(locations.data)
        if existing.get("case_policy") != policy:
            message = f"{locations.data}: changed component policy requires a new view"
            raise ValueError(message)
        return existing
    processor = tokenizer(locations.tokenizer)
    staging = locations.data.with_name(f".{locations.data.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    connection = sqlite3.connect(f"file:{source}/records.sqlite3?mode=ro", uri=True)
    try:
        with (
            ExitStack() as stack,
            stage("align restoration case components"),
            Heartbeat("case view"),
        ):
            writer = Writer(staging, stack)
            tracker = Tracker(
                "case-view records",
                int(str(parent["totals"]["accepted_paragraphs"]))
                if isinstance(parent["totals"], Mapping)
                else None,
            )
            for count, (record_id, text, split, source_id) in enumerate(
                connection.execute("SELECT id,text,split,source FROM records ORDER BY id")
            ):
                document = annotate(str(text))
                tokens = writer.append(
                    document, processor, int(split), known_edges=source_id == "speech.train"
                )
                if not tokens or int(record_id) != count:
                    message = (
                        f"{source}/records.sqlite3: record {record_id} "
                        "failed exact view conservation"
                    )
                    raise ValueError(message)
                tracker.advance()
            templates = writer.finish(staging, settings)
            report = {
                **parent,
                "format_version": FORMAT_VERSION,
                "case_policy": policy,
                "case_identity": identity,
                "code": code_stamp(),
                "case_units": writer.case_count,
                "words": writer.word_count,
                "tokens": writer.token_count,
                "masked": dict(writer.masked),
                "templates": list(templates),
            }
    finally:
        connection.close()
    for name in ("tokens.bin", "offsets.npy", "documents.npy", "marks.bin", "structure.bin"):
        if file_digest(staging / name) != file_digest(source / name):
            message = (
                f"{staging / name}: component conversion changed an immutable word/gap artifact"
            )
            raise ValueError(message)
    shutil.copy2(source / "records.sqlite3", staging / "records.sqlite3")
    report["files"] = {path.name: file_digest(path) for path in sorted(staging.iterdir())}
    write_json(staging / "corpus.stats.json", report)
    verify(staging)
    locations.data.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(locations.data)
    return verify(locations.data)
