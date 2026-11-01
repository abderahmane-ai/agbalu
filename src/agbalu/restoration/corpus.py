"""Build audited restoration arrays from the existing HPC document corpus."""

import json
import os
import re
import shutil
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO, Protocol

import numpy as np
import sentencepiece as spm

from agbalu.encoder.corpus import read_json, tokenizer, write_json
from agbalu.lid.infer import Identifier
from agbalu.llm.decontaminate import Decontaminator
from agbalu.llm.progress import Heartbeat, Tracker, stage
from agbalu.llm.provenance import code_stamp, config_hash, file_digest
from agbalu.normalise.normaliser import normalise
from agbalu.restoration.alignment import EncodedWord, align_cases, encode_word
from agbalu.restoration.annotations import IGNORE, PRIMARY, Document, annotate
from agbalu.restoration.configuration import Locations, Settings

TRAIN_SOURCES = frozenset(
    {
        "taqbaylit.bejaia",
        "taqbaylit.bouira",
        "taqbaylit.adlis-pdfs",
        "taqbaylit.ayamun-pdfs",
        "hf.imsidag.kabyle-corpus-ummto",
        "hf.imsidag.kabyle-corpus-ubouira",
        "hf.imsidag.kabyle-datasets-mono",
        "hf.dcad2000-kab",
        "hf.finepdfs-kab",
    }
)
DEVELOPMENT_SOURCES = frozenset({"hf.imsidag.kabyle-corpus-hca"})
TEST_SOURCES = frozenset(
    {
        "hf.hplt2-cleaned-kab",
        "hf.fineweb2-kab",
        "hf.glotcc-v1-kab",
        "wikimedia.kabwiki",
        "sifal.kabyle-wikipedia",
    }
)
SOURCES = TRAIN_SOURCES | DEVELOPMENT_SOURCES | TEST_SOURCES
NON_PROSE = re.compile(r"https?://|</?[A-Za-z][^>]*>|\|[^\n]*\||%[0-9$sdfl]|\{\d")
MINIMUM_ALPHABETIC_SHARE = 0.65
FORMAT_VERSION = 2
MINIMUM_LID_CHARACTERS = 32
MINIMUM_LID_CONFIDENCE = 0.65
MAXIMUM_WORD_TOKENS = 256
SPECIAL_TOKENS = 5
PREPARATION_BATCH = 64


class Gate(Protocol):
    def accepts(self, texts: Sequence[str]) -> list[bool]: ...


class LanguageGate:
    def __init__(self, identifier: Identifier) -> None:
        self.identifier = identifier

    def accepts(self, texts: Sequence[str]) -> list[bool]:
        fragments: list[str] = []
        owners: list[int] = []
        for index, text in enumerate(texts):
            for start in range(0, len(text), 600):
                fragment = text[start : start + 600]
                if len(fragment.strip()) >= MINIMUM_LID_CHARACTERS:
                    fragments.append(fragment)
                    owners.append(index)
        accepted = [True] * len(texts)
        scores = self.identifier.predict(fragments, batch_size=512)
        for owner, prediction in zip(owners, scores, strict=True):
            if prediction.language != "kab_Latn" or prediction.confidence < MINIMUM_LID_CONFIDENCE:
                accepted[owner] = False
        return accepted


def split_for(source: str) -> int:
    if source in DEVELOPMENT_SOURCES:
        return 1
    return 2 if source in TEST_SOURCES else 0


def candidates(path: Path) -> Iterator[tuple[int, int, dict[str, str]]]:
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            try:
                payload: object = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                msg = f"{path}:{number}: malformed JSON: {error}"
                raise ValueError(msg) from error
            if not isinstance(payload, dict):
                msg = f"{path}:{number}: expected object, got {payload!r}"
                raise TypeError(msg)
            row: dict[str, str] = {}
            for field in ("text", "source", "licence", "redistribution"):
                value = payload.get(field)
                if not isinstance(value, str) or not value:
                    msg = f"{path}:{number}: invalid {field}={value!r}"
                    raise ValueError(msg)
                row[field] = value
            yield number, len(raw), row


def rejection(text: str, settings: Settings) -> str | None:
    if NON_PROSE.search(text):
        return "markup_or_table"
    if len(text.split()) < settings.minimum_words:
        return "too_short"
    visible = [char for char in text if not char.isspace()]
    if sum(char.isalpha() for char in visible) / max(len(visible), 1) < MINIMUM_ALPHABETIC_SHARE:
        return "non_prose_character_share"
    if not any(char in ".?!,;:…" for char in text) or not any(char.isupper() for char in text):
        return "missing_written_annotations"
    if any(unicodedata.name(char, "").startswith(("GREEK", "CYRILLIC", "ARABIC")) for char in text):
        return "foreign_script"
    return None


def paragraphs(text: str) -> list[str]:
    # Removed paragraph boundaries remain separate records; no synthetic adjacency is created.
    return [part for part in re.split(r"\n\s*\n", text) if part.strip()]


def exclusion(path: Path, settings: Settings) -> Decontaminator:
    held: list[str] = []
    with stage("index source-disjoint restoration references"):
        for _, _, row in candidates(path):
            if row["source"] in DEVELOPMENT_SOURCES | TEST_SOURCES:
                held.extend(
                    part
                    for part in paragraphs(normalise(row["text"]))
                    if rejection(part, settings) is None
                )
    if not held:
        msg = f"{path}: no held-out reference text"
        raise ValueError(msg)
    result = Decontaminator.build(held, whole=held)
    result.require_positive(max(held, key=len))
    return result


class Writer:
    def __init__(self, directory: Path, stack: ExitStack) -> None:
        self.handles: dict[str, BinaryIO] = {
            name: stack.enter_context((directory / f"{name}.bin").open("wb"))
            for name in (
                "tokens",
                "offsets",
                "cases",
                "marks",
                "structure",
                "case_offsets",
                "case_spans",
            )
        }
        self.documents: list[tuple[int, ...]] = []
        self.word_count = self.token_count = self.gap_count = 0
        self.case_count = 0
        self.templates: dict[str, int] = {PRIMARY: 0}
        self.template_support: Counter[str] = Counter()
        self.cache: dict[str, EncodedWord] = {}
        self.masked: Counter[str] = Counter()

    def append(
        self,
        document: Document,
        processor: spm.SentencePieceProcessor,
        split: int,
        *,
        known_edges: bool = False,
    ) -> int:
        unknown = list(dict.fromkeys(word for word in document.words if word not in self.cache))
        self.cache.update((word, encode_word(word, processor)) for word in unknown)
        rows = [self.cache[word].ids for word in document.words]
        if any(
            not ids
            or len(ids) > MAXIMUM_WORD_TOKENS
            or any(value < SPECIAL_TOKENS for value in ids)
            for ids in rows
        ):
            return 0
        ids = [value for row in rows for value in row]
        offsets = np.cumsum([self.token_count, *(len(row) for row in rows)], dtype=np.int64)
        if max(int(offsets[-1]), self.word_count + len(rows)) > np.iinfo(np.int32).max:
            message = (
                f"case alignment exceeds int32 capacity: tokens={offsets[-1]}, "
                f"words={self.word_count + len(rows)}"
            )
            raise ValueError(message)
        labels: list[int] = []
        case_rows: list[tuple[int, ...]] = []
        for index, (start, end) in enumerate(document.spans):
            self.handles["case_offsets"].write(np.asarray([self.case_count], np.int64).tobytes())
            word = document.reference[start:end]
            for first, last, char_start, char_end, length, label in align_cases(
                word, self.cache[document.words[index]]
            ):
                case_rows.append(
                    (
                        int(offsets[index]) + first,
                        int(offsets[index]) + last,
                        self.word_count + index,
                        char_start,
                        char_end,
                        length,
                    )
                )
                labels.append(label)
                self.case_count += 1
        self.handles["tokens"].write(np.asarray(ids, dtype=np.uint16).tobytes())
        self.handles["offsets"].write(offsets[:-1].tobytes())
        self.handles["cases"].write(np.asarray(labels, dtype=np.int16).tobytes())
        self.handles["case_spans"].write(np.asarray(case_rows, dtype=np.int32).tobytes())
        marks: list[int] = []
        structures: list[int] = []
        for index, gap in enumerate(document.gaps):
            is_edge = index in (0, len(document.words))
            valid = gap.valid and (known_edges or not is_edge)
            marks.append(gap.mark if valid else IGNORE)
            self.masked["unsupported_gaps"] += int(not gap.valid)
            self.masked["unknown_record_edges"] += int(is_edge and not known_edges)
            if gap.template not in self.templates:
                self.templates[gap.template] = len(self.templates)
            structures.append(self.templates[gap.template] if valid else IGNORE)
            if split == 0 and valid:
                self.template_support[gap.template] += 1
        self.masked["unsupported_whole_word_casing"] += sum(
            value == IGNORE for value in document.casing
        )
        self.masked["unsupported_case_units"] += sum(value == IGNORE for value in labels)
        self.handles["marks"].write(np.asarray(marks, dtype=np.int16).tobytes())
        self.handles["structure"].write(np.asarray(structures, dtype=np.int32).tobytes())
        self.documents.append(
            (
                self.token_count,
                self.word_count,
                len(rows),
                self.gap_count,
                split,
                int(known_edges),
                int(known_edges),
            )
        )
        self.token_count += len(ids)
        self.word_count += len(rows)
        self.gap_count += len(marks)
        return len(ids)

    def finish(self, directory: Path, settings: Settings) -> tuple[str, ...]:
        self.handles["offsets"].write(np.asarray([self.token_count], dtype=np.int64).tobytes())
        self.handles["case_offsets"].write(np.asarray([self.case_count], dtype=np.int64).tobytes())
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        chosen = [
            PRIMARY,
            *(
                name
                for name, count in self.template_support.most_common()
                if name != PRIMARY and count >= settings.minimum_template_support
            ),
        ]
        templates = tuple(chosen[: settings.maximum_templates])
        remap = np.full(len(self.templates), IGNORE, np.int16)
        for index, name in enumerate(templates):
            remap[self.templates[name]] = index
        original: np.memmap = np.memmap(directory / "structure.bin", np.int32, "r")
        staging = directory / "structure.compact"
        with staging.open("wb") as handle:
            for start in range(0, original.size, 1_000_000):
                chunk = np.asarray(original[start : start + 1_000_000])
                mapped = remap[np.maximum(chunk, 0)]
                mapped = np.where(chunk == IGNORE, IGNORE, mapped).astype(np.int16)
                self.masked["rare_structural_templates"] += int(
                    ((mapped == IGNORE) & (chunk != IGNORE)).sum()
                )
                handle.write(mapped.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        del original
        staging.replace(directory / "structure.bin")
        for name in ("offsets", "case_offsets"):
            offsets = np.fromfile(directory / f"{name}.bin", np.int64)
            np.save(directory / f"{name}.npy", offsets, allow_pickle=False)
            (directory / f"{name}.bin").unlink()
        np.save(
            directory / "documents.npy",
            np.asarray(self.documents, dtype=np.int64),
            allow_pickle=False,
        )
        return templates


def verify(directory: Path) -> dict[str, object]:
    report = read_json(directory / "corpus.stats.json")
    files = report["files"]
    if not isinstance(files, dict):
        msg = f"{directory}: missing file manifest"
        raise TypeError(msg)
    for name, digest in files.items():
        path = directory / str(name)
        if file_digest(path) != digest:
            msg = f"{path}: checksum mismatch"
            raise ValueError(msg)
    documents = np.load(directory / "documents.npy", mmap_mode="r")
    version = report["format_version"]
    if version not in (1, 2):
        message = f"{directory}: unsupported format_version={version!r}"
        raise ValueError(message)
    words = int(documents[:, 2].sum())
    if words != int(str(report["words"])):
        msg = f"{directory}: document word conservation failed"
        raise ValueError(msg)
    for name, count in (
        ("cases", words if version == 1 else int(str(report["case_units"]))),
        ("marks", words + len(documents)),
        ("structure", words + len(documents)),
    ):
        if (directory / f"{name}.bin").stat().st_size != count * 2:
            msg = f"{directory}/{name}.bin: invalid byte count for {count} labels"
            raise ValueError(msg)
    if version == FORMAT_VERSION:
        cases = int(str(report["case_units"]))
        case_offsets = np.load(directory / "case_offsets.npy", mmap_mode="r")
        if len(case_offsets) != words + 1 or int(case_offsets[-1]) != cases:
            message = (
                f"{directory}: case component conservation failed: "
                f"words={words}, components={cases}"
            )
            raise ValueError(message)
        if (directory / "case_spans.bin").stat().st_size != cases * 6 * 4:
            message = f"{directory}/case_spans.bin: invalid span bytes for {cases} components"
            raise ValueError(message)
    return report


def prepare(locations: Locations, settings: Settings, gate: Gate) -> dict[str, object]:
    processor = tokenizer(locations.tokenizer)
    identity = {
        "input_sha256": file_digest(locations.corpus),
        "tokenizer_sha256": file_digest(locations.tokenizer),
        "speech_sha256": file_digest(locations.speech / "train.jsonl"),
        "policy": config_hash(
            {
                "train": sorted(TRAIN_SOURCES),
                "development": sorted(DEVELOPMENT_SOURCES),
                "test": sorted(TEST_SOURCES),
                "minimum_words": settings.minimum_words,
                "minimum_template_support": settings.minimum_template_support,
                "maximum_templates": settings.maximum_templates,
                "source": {
                    name: file_digest(Path(__file__).with_name(name))
                    for name in ("corpus.py", "annotations.py")
                },
            }
        ),
    }
    if (locations.data / "corpus.stats.json").is_file():
        existing = verify(locations.data)
        if any(existing.get(key) != value for key, value in identity.items()):
            msg = f"{locations.data}: changed input/policy requires a new data directory"
            raise ValueError(msg)
        return existing
    staging = locations.data.with_name(f".{locations.data.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    decontaminator = exclusion(locations.corpus, settings)
    totals: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    split_tokens: Counter[int] = Counter()
    splits: Counter[int] = Counter()
    pending: list[tuple[int, int, dict[str, str]]] = []
    with (
        ExitStack() as stack,
        stage("prepare restoration supervision"),
        Heartbeat("restoration data"),
    ):
        writer = Writer(staging, stack)
        database = sqlite3.connect(staging / "records.sqlite3")
        stack.callback(database.close)
        database.execute(
            "CREATE TABLE records (id INTEGER PRIMARY KEY, line INTEGER, source TEXT, "
            "licence TEXT, redistribution TEXT, text TEXT, split INTEGER)"
        )
        database.execute("CREATE TABLE rejects (line INTEGER, reason TEXT, characters INTEGER)")
        tracker = Tracker("restoration input records", None)

        def reject(line: int, reason: str, text: str) -> None:
            rejected[reason] += 1
            database.execute("INSERT INTO rejects VALUES (?,?,?)", (line, reason, len(text)))

        def consume() -> None:
            texts = [row["text"] for _, _, row in pending]
            admitted = gate.accepts(texts)
            for (line, _, row), accepted in zip(pending, admitted, strict=True):
                text, source = row["text"], row["source"]
                if not accepted:
                    reject(line, "language_gate", text)
                    continue
                split = split_for(source)
                if split == 0 and decontaminator.matches(text):
                    reject(line, "heldout_overlap", text)
                    continue
                document = annotate(text)
                count = writer.append(
                    document, processor, split, known_edges=source == "speech.train"
                )
                if not count:
                    reject(line, "unencodable_or_oversized_word", text)
                    continue
                document_id = len(writer.documents) - 1
                database.execute(
                    "INSERT INTO records VALUES (?,?,?,?,?,?,?)",
                    (
                        document_id,
                        line,
                        source,
                        row["licence"],
                        row["redistribution"],
                        document.reference,
                        split,
                    ),
                )
                by_source[source] += 1
                splits[split] += 1
                split_tokens[split] += count
                totals["accepted_paragraphs"] += 1
                totals["accepted_characters"] += len(text)
            pending.clear()
            database.commit()

        for line, size_bytes, row in candidates(locations.corpus):
            totals["input_records"] += 1
            totals["input_bytes"] += size_bytes
            text = normalise(row["text"])
            totals["input_characters"] += len(text)
            if row["source"] not in SOURCES:
                reject(line, "source_policy", text)
            else:
                accounted = 0
                for part in paragraphs(text):
                    clean_part = normalise(part)
                    accounted += len(part)
                    reason = rejection(clean_part, settings)
                    if reason:
                        reject(line, reason, clean_part)
                    else:
                        pending.append((line, size_bytes, {**row, "text": clean_part}))
                    if len(pending) >= PREPARATION_BATCH:
                        consume()
                totals["separator_characters"] += len(text) - accounted
            tracker.advance()
        if pending:
            consume()
        speech_settings = settings.model_copy(update={"minimum_words": 3})
        with (locations.speech / "train.jsonl").open(encoding="utf-8") as handle:
            for line, raw in enumerate(handle, 1):
                payload = json.loads(raw)
                raw_text = payload.get("text")
                if not isinstance(raw_text, str) or not raw_text:
                    message = f"{locations.speech}/train.jsonl:{line}: invalid text={raw_text!r}"
                    raise ValueError(message)
                text = normalise(raw_text)
                totals["speech_records"] += 1
                totals["speech_characters"] += len(text)
                totals["input_characters"] += len(text)
                reason = rejection(text, speech_settings)
                if reason or text.rstrip()[-1:] not in ".?!":
                    reject(-line, reason or "speech_missing_terminal_mark", text)
                    continue
                pending.append(
                    (
                        -line,
                        len(raw.encode("utf-8")),
                        {
                            "text": text,
                            "source": "speech.train",
                            "licence": "cc0-1.0",
                            "redistribution": "permissive",
                        },
                    )
                )
                if len(pending) >= PREPARATION_BATCH:
                    consume()
        if pending:
            consume()
        if any(not splits[split] for split in (0, 1, 2)):
            msg = f"{locations.corpus}: empty train/dev/test split: {dict(splits)}"
            raise ValueError(msg)
        templates = writer.finish(staging, settings)
        database.execute("CREATE INDEX source_lines ON records(source,line)")
        database.commit()
        other_rejected = database.execute(
            "SELECT COALESCE(SUM(characters),0) FROM rejects"
        ).fetchone()
        totals["rejected_characters"] += int(other_rejected[0])
        if totals["input_characters"] != (
            totals["accepted_characters"]
            + totals["rejected_characters"]
            + totals["separator_characters"]
        ):
            msg = f"character conservation failed: {dict(totals)}"
            raise ValueError(msg)
        report: dict[str, object] = {
            **identity,
            "format_version": FORMAT_VERSION,
            "code": code_stamp(),
            "version_tag": None,
            "seed": settings.seed,
            "totals": dict(totals),
            "words": writer.word_count,
            "case_units": writer.case_count,
            "tokens": writer.token_count,
            "split_documents": {str(key): value for key, value in splits.items()},
            "split_tokens": {str(key): value for key, value in split_tokens.items()},
            "by_source": dict(by_source),
            "rejected": dict(rejected),
            "masked": dict(writer.masked),
            "templates": list(templates),
            "reference_quality": "writer-derived silver; source-disjoint, pretrained-seen",
            "retrieval_date": None,
            "retrieval_date_missing": len(writer.documents),
        }
    for path, key in (
        (locations.corpus, "input_sha256"),
        (locations.speech / "train.jsonl", "speech_sha256"),
        (locations.tokenizer, "tokenizer_sha256"),
    ):
        if file_digest(path) != identity[key]:
            msg = f"{path}: input changed during preparation"
            raise ValueError(msg)
    for path in staging.iterdir():
        with path.open("rb") as durable_file:
            os.fsync(durable_file.fileno())
    report["files"] = {path.name: file_digest(path) for path in sorted(staging.iterdir())}
    write_json(staging / "corpus.stats.json", report)
    if locations.data.exists():
        msg = f"refusing to replace {locations.data}"
        raise FileExistsError(msg)
    staging.rename(locations.data)
    return verify(locations.data)
