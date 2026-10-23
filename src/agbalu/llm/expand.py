"""HPC corpus expansion: fetch, OCR/ASR/Boulifa/Dihya extraction, and full compilation.

Downloads new PDF, audio, text, and parallel repositories directly on the HPC pod, extracts
full-document text via PyMuPDF + Feraoun-Pro OCR (for scanned pages) and Fadhma-300M ASR +
Belaid-31M punctuation + Boulifa-48M orthography standardisation (for Radio Chaine 2 and
audiobooks), repairs legacy 8-bit Amazigh font encodings, filters every document and aligned
pair with Normaliser 1.3.0 and Dihya-5M, and compiles the full CPT dataset (`kab-docs.jsonl`,
`aligned.jsonl`, 10-source `replay.jsonl`, and packed `cpt-v2.npy`).
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from collections import Counter as Tally
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from defusedxml.ElementTree import iterparse

from agbalu.llm.corpus import LANGUAGE_TAG, Record
from agbalu.llm.documents import (
    KABYLE,
    LID_WINDOW_CHARS,
    LID_WINDOWS,
    MIN_DOCUMENT_CHARS,
    LanguageScorer,
    SentenceRow,
    clean,
    verdict,
    windows,
)
from agbalu.llm.progress import Tracker, progress, stage
from agbalu.normalise import Normaliser

if TYPE_CHECKING:
    from numpy.typing import NDArray

RAW_V2_ROOT: Final = Path("data/raw_v2")
HUB_MODELS_ROOT: Final = Path("artifacts/hub_models")
EXPANDED_DIR: Final = Path("data/interim/llm")
EXPANDED_DOCS_OUT: Final = EXPANDED_DIR / "expanded-kab-docs.jsonl"
EXPANDED_AUDIO_OUT: Final = EXPANDED_DIR / "expanded-audio-docs.jsonl"
EXPANDED_ALIGNED_OUT: Final = EXPANDED_DIR / "expanded-aligned.jsonl"

PDF_REPOS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("taqbaylit.bejaia", "taqbaylit/bejaia", "apache-2.0", "permissive"),
    ("taqbaylit.bouira", "taqbaylit/bouira", "apache-2.0", "permissive"),
    ("taqbaylit.ayamun-pdfs", "taqbaylit/ayamun-pdfs", "mit", "permissive"),
    ("taqbaylit.adlis-pdfs", "taqbaylit/adlis-pdfs", "mit", "permissive"),
)

AUDIO_REPOS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("taqbaylit.rradyu-tis-snat", "taqbaylit/rradyu-tis-snat", "mit", "permissive"),
    (
        "taqbaylit.tughalin",
        "taqbaylit/Tughalin_n_Weqcic_Ijahen",
        "cc-by-4.0",
        "permissive",
    ),
)

TEXT_AND_PARALLEL_REPOS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    (
        "abdelhaqueidali.quran-kab",
        "abdelhaqueidali/Quran-kabyle-ayt-mensour",
        ("quran_ascii_latin_amazigh_arabic_scripts.tsv",),
    ),
    (
        "taqbaylit.kabyle-corpus",
        "taqbaylit/kabyle-corpus",
        ("kabyle_corpus_clean.jsonl", "kabyle_corpus_clean.txt"),
    ),
    (
        "taqbaylit.tatoeba-mono-cleaned",
        "taqbaylit/tatoeba-kabyle-mono-cleaned",
        ("data/train-00000-of-00001.parquet",),
    ),
    (
        "hillal.tamazight-kabyle-text",
        "Hillal-titouh/tamazight-kabyle-text",
        ("data.csv",),
    ),
    (
        "sifal.kabyle-wikipedia",
        "Sifal/KabyleWikipedia",
        ("WIKABV1.zip",),
    ),
    (
        "taqbaylit.tatoeba-en-kab",
        "taqbaylit/tatoeba-en-kab",
        (
            "data/train-00000-of-00001.parquet",
            "data/dev-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ),
    ),
    (
        "taqbaylit.weblate-kabyle",
        "taqbaylit/weblate-kabyle",
        ("en-kab.tmx",),
    ),
    (
        "tutlaytai.multilang-berber",
        "TutlaytAI/Multilang-to-Berber-Translation",
        (
            "data/en-00000-of-00001.parquet",
            "data/fr-00000-of-00001.parquet",
            "data/spa-00000-of-00001.parquet",
            "data/ara-00000-of-00001.parquet",
            "data/epo-00000-of-00001.parquet",
        ),
    ),
    (
        "ad019el.dataset-kabyle",
        "ad019el/dataset_kabyle",
        (
            "data/train-00000-of-00001-efe2bb5df33b641e.parquet",
            "data/test-00000-of-00001-cf16b7259d8822bd.parquet",
        ),
    ),
    (
        "taqbaylit.nllb-en-kab",
        "taqbaylit/nllb_en_kab",
        ("nllb_en_kab.parquet",),
    ),
)

HUB_MODEL_REPOS: Final[tuple[tuple[str, str], ...]] = (
    ("fadhma", "agbalu/Fadhma-300M"),
    ("belaid", "agbalu/Belaid-31M"),
    ("boulifa", "agbalu/Boulifa-48M"),
    ("feraoun_pro", "agbalu/Feraoun-Pro"),
    ("mammeri_tok", "agbalu/Mammeri-Tok"),
)

LEGACY_FONT_MAP: Final[dict[str, str]] = {
    "Ö": "Ṛ",
    "ô": "ṛ",
    "Ê": "Ḥ",
    "ê": "ḥ",
    "Î": "Ṭ",
    "î": "ṭ",
    "Û": "Ẓ",
    "û": "ẓ",
    "Ë": "Ɛ",
    "ë": "ɛ",
    "Ï": "Ɣ",
    "ï": "ɣ",
    "Ñ": "Ǧ",
    "ñ": "ǧ",
}
"""8-bit MCA / Timezrit / Tamazight legacy font codepoints mapped to standard INALCO Latin."""

LEGACY_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"(?<=[a-zA-Z])\$(?=[a-zA-Z])"), "ɣ"),
    (re.compile(r"£(?=[a-zA-Z])|(?<=[a-zA-Z])£"), "ɣ"),
    (re.compile(r"\b([tT])a\s+uri\b"), r"\1aɣuri"),
    (re.compile(r"\b([iI])mezda\s+en\b"), r"\1mezdaɣen"),
    (re.compile(r"\b([yY])eooen\b"), r"\1eǧǧen"),
    (re.compile(r"\b([tT])aooalt\b"), r"\1aǧǧalt"),
    (re.compile(r"\b([aA])m\s+ar\b"), r"\1mɣar"),
    (re.compile(r"\b([tT])am\s+art\b"), r"\1amɣart"),
    (re.compile(r"\b([aA])\s+erbaz\b"), r"\1ɣerbaz"),
    (re.compile(r"\b([uU])\s+erbaz\b"), r"\1ɣerbaz"),
    (re.compile(r"\b([tT])i\s+remt\b"), r"\1iɣremt"),
    (re.compile(r"\b([yY])iḍrisen\b"), r"\1iḍrisen"),
    (re.compile(r"\b([yY])ivrisen\b"), r"\1iḍrisen"),
    (re.compile(r"\b([aA])vris\b"), r"\1ḍris"),
    (re.compile(r"\b([uU])vris\b"), r"\1ḍris"),
)

KABYLE_DIACRITICS: Final[frozenset[str]] = frozenset("ɣɛḥḍṣṭẓṛčǧţƔƐḤḌṢṬẒṚČǦŢ")
MAX_DOC_CHARS: Final = 16_000
MIN_DIGITAL_PAGE_CHARS: Final = 80
SAMPLE_RATE: Final = 16_000
MIN_CHUNK_SEC: Final = 10.0
MAX_CHUNK_SEC: Final = 22.0


class ExpandError(Exception):
    """A download, extraction, or compilation stage failed."""


@dataclass(slots=True)
class ExpandTally:
    source_id: str
    licence: str
    redistribution: str
    read: int = 0
    kept: int = 0
    characters: int = 0
    rejected: Tally[str] = field(default_factory=Tally)
    # True when an earlier run wrote this source's rows: `kept` and `characters` are read back
    # from the file, while `read` and `rejected` count this invocation alone.
    resumed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source_id,
            "licence": self.licence,
            "redistribution": self.redistribution,
            "read": self.read,
            "kept": self.kept,
            "characters": self.characters,
            "rejected": dict(sorted(self.rejected.items())),
            "resumed": self.resumed,
        }


def repair_legacy_font(text: str) -> str:
    """Repair 8-bit Amazigh/Kabyle font mojibake when legacy markers dominate."""
    if not text:
        return ""
    legacy_hits = sum(text.count(ch) for ch in LEGACY_FONT_MAP)
    standard_hits = sum(text.count(ch) for ch in KABYLE_DIACRITICS)
    out = text
    if legacy_hits >= 3 and legacy_hits > standard_hits:
        out = "".join(LEGACY_FONT_MAP.get(ch, ch) for ch in out)
    for pattern, repl in LEGACY_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def is_kabyle_window(
    text: str,
    scorer: LanguageScorer,
    *,
    min_prob: float = 0.0,
) -> bool:
    """Check whether `Dihya-5M` classifies the text as `kab_Latn`."""
    cleaned = text.strip()
    if len(cleaned) < 20:
        return False
    slices = windows(cleaned, size=LID_WINDOW_CHARS, count=LID_WINDOWS)
    probs = scorer.probabilities(slices)
    if verdict(probs) != KABYLE:
        return False
    mean_kab = sum(row.get(KABYLE, 0.0) for row in probs) / len(probs)
    return mean_kab >= min_prob


def fetch_all(
    raw_root: Path = RAW_V2_ROOT,
    models_root: Path = HUB_MODELS_ROOT,
) -> dict[str, object]:
    """Download all expansion datasets and Hub model checkpoints onto the HPC."""
    from huggingface_hub import hf_hub_download, snapshot_download

    raw_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    fetched_datasets: list[str] = []
    fetched_models: list[str] = []

    for name, repo, _, _ in (*PDF_REPOS, *AUDIO_REPOS):
        target = raw_root / name
        with stage("fetch dataset", repo=repo, target=str(target)):
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=target,
                max_workers=8,
            )
            fetched_datasets.append(repo)

    for name, repo, filenames in TEXT_AND_PARALLEL_REPOS:
        target = raw_root / name
        target.mkdir(parents=True, exist_ok=True)
        with stage("fetch files", repo=repo, count=len(filenames)):
            for fname in filenames:
                hf_hub_download(
                    repo_id=repo,
                    filename=fname,
                    repo_type="dataset",
                    local_dir=target,
                )
            fetched_datasets.append(repo)

    for name, repo in HUB_MODEL_REPOS:
        target = models_root / name
        with stage("fetch model", repo=repo, target=str(target)):
            snapshot_download(
                repo_id=repo,
                repo_type="model",
                local_dir=target,
                max_workers=4,
            )
            fetched_models.append(repo)

    return {
        "raw_root": str(raw_root),
        "models_root": str(models_root),
        "datasets": fetched_datasets,
        "models": fetched_models,
    }


def _needs_boulifa(text: str) -> bool:
    """True when a Latin text has very few Kabyle diacritics and likely needs Boulifa.

    Only used in the audio pipeline.  PDF extraction disables Boulifa entirely
    (modern Unicode theses already carry correct diacritics; legacy-font text is
    handled by repair_legacy_font).  For ASR output this always returns True
    because Fadhma produces bare Latin with no diacritics at all — the function
    is kept so the caller remains explicit rather than unconditionally applying
    Boulifa to every hypothetical future caller.
    """
    alpha = sum(1 for c in text if c.isalpha())
    if alpha < 60:
        return False
    diac = sum(1 for c in text if c in KABYLE_DIACRITICS)
    return (diac / alpha) < 0.008


class HubOcrEngine:
    """Feraoun-Pro loaded from `artifacts/hub_models/feraoun_pro` on CUDA."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch

        from agbalu.hub.feraoun_pro.configuration_feraoun_pro import FeraounProConfig
        from agbalu.hub.feraoun_pro.modeling_feraoun_pro import FeraounProForCTC

        self.device = torch.device(device)
        config = FeraounProConfig.from_pretrained(model_dir)
        placed: torch.nn.Module = FeraounProForCTC.from_pretrained(model_dir, config=config)
        placed.eval()
        self.model: FeraounProForCTC = placed.to(self.device)  # type: ignore[assignment]

    def transcribe_page(self, image: object) -> str:
        from PIL import Image as PILImage

        from agbalu.hub.feraoun_pro.modeling_feraoun_pro import segment_page_into_lines

        if not isinstance(image, PILImage.Image):
            return ""
        lines = segment_page_into_lines(image)
        if not lines:
            return ""
        out: list[str] = []
        for crop in lines:
            text = self.model.transcribe(crop).strip()
            if text:
                out.append(text)
        return "\n".join(out)


class HubBoulifaEngine:
    """Boulifa-48M loaded from `artifacts/hub_models/boulifa` or local checkpoint on CUDA."""

    def __init__(self, model_dir: Path, device: str) -> None:
        import torch

        from agbalu.hub.boulifa.configuration_boulifa import BoulifaConfig
        from agbalu.hub.boulifa.modeling_boulifa import BoulifaForSeq2SeqLM
        from agbalu.standardise.tokenizer import Tokenizer as StandardTokenizer

        self.device = torch.device(device)
        self.tokenizer = StandardTokenizer.build()
        config = BoulifaConfig.from_pretrained(model_dir)
        placed: torch.nn.Module = BoulifaForSeq2SeqLM.from_pretrained(model_dir, config=config)
        placed.eval()
        self.model: BoulifaForSeq2SeqLM = placed.to(self.device)  # type: ignore[assignment]

    def _standardise_line(self, line: str) -> str:
        import torch

        ids = self.tokenizer.encode(line, max_length=384)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            context = self.model.encode(input_ids)
            dec = torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long, device=self.device)
            for _ in range(min(448, len(ids) + 32)):
                next_id = int(self.model.decode(dec, context)[0, -1].argmax().item())
                if next_id == self.tokenizer.eos_id:
                    break
                step = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                dec = torch.cat([dec, step], dim=1)
        return self.tokenizer.decode(dec[0, 1:].tolist())

    def standardise_lines(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        out: list[str] = []
        for line in lines:
            if len(line) > 380:
                out.append(line)
            else:
                out.append(self._standardise_line(line))
        return "\n".join(out)


def _extract_single_pdf(
    pdf_path: Path,
    ocr: HubOcrEngine | None,
    boulifa: HubBoulifaEngine | None,
    scorer: LanguageScorer,
    normaliser: Normaliser,
) -> list[str]:
    """Extract Kabyle pages from one PDF and group consecutive pages into documents."""
    import fitz
    from PIL import Image

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []

    kept_pages: list[str] = []
    for page_index in range(len(doc)):
        try:
            page = doc[page_index]
            raw_text = str(page.get_text("text") or "").strip()
        except Exception:
            continue

        if len(raw_text) < MIN_DIGITAL_PAGE_CHARS and ocr is not None:
            try:
                pix = page.get_pixmap(dpi=180, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                raw_text = ocr.transcribe_page(img)
            except Exception:
                raw_text = ""

        if len(raw_text.strip()) < MIN_DIGITAL_PAGE_CHARS:
            continue

        repaired = repair_legacy_font(raw_text)
        # Boulifa is disabled for PDFs (use_boulifa=False default); this guard
        # keeps _extract_single_pdf safe if it is ever re-enabled for a specific
        # source where diacritic restoration is needed.
        if boulifa is not None and _needs_boulifa(repaired):
            repaired = boulifa.standardise_lines(repaired)

        cleaned = clean(repaired, normaliser)
        if len(cleaned) < MIN_DIGITAL_PAGE_CHARS:
            continue
        if is_kabyle_window(cleaned, scorer):
            kept_pages.append(cleaned)

    doc.close()
    if not kept_pages:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for page_text in kept_pages:
        if current_len + len(page_text) > MAX_DOC_CHARS and current:
            joined = "\n\n".join(current)
            if len(joined) >= MIN_DOCUMENT_CHARS:
                chunks.append(joined)
            current = [page_text]
            current_len = len(page_text)
        else:
            current.append(page_text)
            current_len += len(page_text) + 2
    if current:
        joined = "\n\n".join(current)
        if len(joined) >= MIN_DOCUMENT_CHARS:
            chunks.append(joined)
    return chunks


def _reconstruct_imsidag_parquets(
    folder_path: Path,
    normaliser: Normaliser,
) -> Iterator[str]:
    """Reconstruct continuous multi-page documents from shuffled `(folder, line_no)` rows."""
    import pyarrow.parquet as pq

    files = sorted(folder_path.glob("*.parquet"))
    if not files:
        return
    rows: list[tuple[str, int, str]] = []
    for path in files:
        table = pq.read_table(path, columns=["folder", "line_no", "text"])
        for fld, lno, txt in zip(
            table["folder"].to_pylist(),
            table["line_no"].to_pylist(),
            table["text"].to_pylist(),
            strict=True,
        ):
            if isinstance(txt, str) and txt.strip():
                rows.append((str(fld or ""), int(lno or 0), txt.strip()))

    rows.sort(key=lambda item: (item[0], item[1]))
    current: list[str] = []
    current_len = 0
    prev_folder = ""
    prev_line = -100

    for fld, lno, txt in rows:
        gap = lno - prev_line
        if (fld != prev_folder or gap > 15 or current_len + len(txt) > MAX_DOC_CHARS) and current:
            doc_text = clean(repair_legacy_font("\n".join(current)), normaliser)
            if len(doc_text) >= MIN_DOCUMENT_CHARS:
                yield doc_text
            current = []
            current_len = 0
        current.append(txt)
        current_len += len(txt) + 1
        prev_folder = fld
        prev_line = lno

    if current:
        doc_text = clean(repair_legacy_font("\n".join(current)), normaliser)
        if len(doc_text) >= MIN_DOCUMENT_CHARS:
            yield doc_text


def _read_quran_surahs(tsv_path: Path, normaliser: Normaliser) -> Iterator[str]:
    """Group the 6,236 Quran verses in `latin_script` into 114 full-Surah documents."""
    if not tsv_path.is_file():
        return
    by_surah: dict[int, list[str]] = {}
    with tsv_path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            surah_raw = row.get("surah_number", "")
            verse = str(row.get("latin_script", "") or "").strip()
            if surah_raw and verse:
                try:
                    s_num = int(surah_raw)
                except ValueError:
                    continue
                by_surah.setdefault(s_num, []).append(verse)
    for s_num in sorted(by_surah):
        text = clean("\n".join(by_surah[s_num]), normaliser)
        if len(text) >= MIN_DOCUMENT_CHARS:
            yield text


def _progress_path(out_path: Path) -> Path:
    """The sidecar naming the sources a run has written whole, beside the corpus itself."""
    return out_path.with_name(f"{out_path.name}.progress.json")


def _write_progress(out_path: Path, completed: list[str]) -> None:
    """Record the finished sources, atomically, so a crash cannot tear the sidecar."""
    path = _progress_path(out_path)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps({"completed": completed}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_progress(out_path: Path) -> list[str]:
    """The sources an earlier run finished, or an empty list when there is no sidecar."""
    path = _progress_path(out_path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    completed = payload.get("completed") if isinstance(payload, dict) else None
    if not isinstance(completed, list):
        return []
    return [source for source in completed if isinstance(source, str)]


def sources_in_file(out_path: Path) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """A corpus file's source ids in order, each with its `(documents, characters)`.

    A torn last line — what a crash leaves — is skipped rather than raised on.
    """
    order: list[str] = []
    counts: dict[str, tuple[int, int]] = {}
    if not out_path.is_file():
        return order, counts
    with out_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            source = row.get("source")
            text = row.get("text")
            if not isinstance(source, str) or not source:
                continue
            length = len(text) if isinstance(text, str) else 0
            documents, characters = counts.get(source, (0, 0))
            if source not in counts:
                order.append(source)
            counts[source] = (documents + 1, characters + length)
    return order, counts


def _truncate_to_sources(out_path: Path, keep: set[str]) -> None:
    """Rewrite *out_path* with only the rows of *keep*, dropping every other and torn line."""
    temporary = out_path.with_name(f"{out_path.name}.tmp")
    with (
        out_path.open(encoding="utf-8", errors="replace") as source,
        temporary.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("source") in keep:
                target.write(line if line.endswith("\n") else f"{line}\n")
    temporary.replace(out_path)


def resume_from_file(out_path: Path) -> dict[str, tuple[int, int]]:
    """Keep the sources an earlier run finished, and report each one's document count.

    The sidecar `<name>.progress.json` is authoritative when it exists. Without it — a file
    written before the sidecar existed — every source but the last is trusted, because the
    last is the one an interrupted run was writing. Rows of every other source are dropped,
    which takes a torn final line with them, and the file is left ready to append to.
    """
    order, counts = sources_in_file(out_path)
    if not order:
        return {}
    recorded = set(_read_progress(out_path))
    finished = [source for source in order if source in recorded] if recorded else order[:-1]
    _truncate_to_sources(out_path, set(finished))
    _write_progress(out_path, finished)
    progress(
        "resume",
        source=str(out_path),
        finished=len(finished),
        remaining=len(order) - len(finished),
    )
    return {source: counts[source] for source in finished}


def _source_tally(
    name: str,
    licence: str,
    redistribution: str,
    earlier: tuple[int, int] | None,
) -> ExpandTally:
    """A source's tally: the earlier run's counts when it is finished, a fresh one otherwise."""
    if earlier is None:
        return ExpandTally(name, licence, redistribution)
    documents, characters = earlier
    return ExpandTally(
        name, licence, redistribution, kept=documents, characters=characters, resumed=True
    )


def extract_pdfs_and_text(
    scorer: LanguageScorer,
    *,
    raw_root: Path = RAW_V2_ROOT,
    raw_v1_root: Path = Path("data/raw"),
    models_root: Path = HUB_MODELS_ROOT,
    out_path: Path = EXPANDED_DOCS_OUT,
    device: str = "cuda",
    use_ocr: bool = True,
    use_boulifa: bool = False,
    resume: bool = False,
) -> list[ExpandTally]:
    """Extract PDFs, reconstructed theses, Quran Surahs, and new text corpora on the HPC.

    Boulifa is disabled by default for PDF extraction: modern Unicode theses from
    taqbaylit repos already carry correct Kabyle diacritics; legacy-font text is
    handled upstream by repair_legacy_font().  Boulifa is reserved for the audio
    pipeline (expand-audio) where Fadhma ASR output is always diacritic-free.

    With *resume* the sources an earlier run wrote whole are read back, the file is
    truncated to them — a source cut off mid-write is dropped, so no row is ever written
    twice — and only the sources left over are extracted. Their progress sidecar is
    rewritten as each one lands, so a later restart resumes from there too. Without it, or
    without an existing *out_path*, the file is rebuilt from the first pdf.
    """
    normaliser = Normaliser()
    ocr = (
        HubOcrEngine(models_root / "feraoun_pro", device)
        if use_ocr and (models_root / "feraoun_pro" / "config.json").is_file()
        else None
    )
    boulifa = (
        HubBoulifaEngine(models_root / "boulifa", device)
        if use_boulifa and (models_root / "boulifa" / "config.json").is_file()
        else None
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tallies: list[ExpandTally] = []

    # Resume: the sources an earlier run wrote whole, each with its document count. Without
    # them — no --resume, or no file — every source is extracted again from the first pdf.
    finished: dict[str, tuple[int, int]] = {}
    if resume and out_path.is_file():
        finished = resume_from_file(out_path)
    if not finished:
        # Nothing is reused, so a sidecar left by an earlier run would be stale.
        _progress_path(out_path).unlink(missing_ok=True)

    completed = list(finished)
    with out_path.open("a" if finished else "w", encoding="utf-8") as sink:

        def mark(source: str) -> None:
            """A source is whole once its rows are flushed, so a restart never repeats it."""
            completed.append(source)
            sink.flush()
            _write_progress(out_path, completed)

        # 1. All 4 taqbaylit PDF repositories (1,560 PDFs)
        for source_id, _, licence, redist in PDF_REPOS:
            folder = raw_root / source_id
            if not folder.is_dir():
                continue
            if source_id in finished:
                tallies.append(_source_tally(source_id, licence, redist, finished[source_id]))
                continue
            pdfs = sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".pdf")
            tally = ExpandTally(source_id, licence, redist)
            tallies.append(tally)
            tracker = Tracker(f"pdf {source_id}", len(pdfs))
            for pdf_path in pdfs:
                tally.read += 1
                chunks = _extract_single_pdf(pdf_path, ocr, boulifa, scorer, normaliser)
                if not chunks:
                    tally.rejected["no_kabyle_pages"] += 1
                for chunk in chunks:
                    tally.kept += 1
                    tally.characters += len(chunk)
                    sink.write(
                        json.dumps(
                            {
                                "text": chunk,
                                "source": source_id,
                                "licence": licence,
                                "redistribution": redist,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                tracker.advance()
            mark(source_id)

        # 2. Reconstructed continuous documents from UMMTO, UBouira, HCA, and adlis-hca.txt
        imsidag_sources = (
            ("hf.imsidag.kabyle-corpus-ummto", "cc-by-sa-4.0", "share-alike"),
            ("hf.imsidag.kabyle-corpus-ubouira", "cc-by-sa-4.0", "share-alike"),
            ("hf.imsidag.kabyle-corpus-hca", "cc-by-sa-4.0", "share-alike"),
        )
        for source_id, licence, redist in imsidag_sources:
            folder = raw_v1_root / source_id
            if not folder.is_dir():
                continue
            if source_id in finished:
                tallies.append(
                    _source_tally(f"{source_id}.docs", licence, redist, finished[source_id])
                )
                continue
            tally = ExpandTally(f"{source_id}.docs", licence, redist)
            tallies.append(tally)
            for doc_text in _reconstruct_imsidag_parquets(folder, normaliser):
                tally.read += 1
                if not is_kabyle_window(doc_text, scorer):
                    tally.rejected["language"] += 1
                    continue
                tally.kept += 1
                tally.characters += len(doc_text)
                sink.write(
                    json.dumps(
                        {
                            "text": doc_text,
                            "source": source_id,
                            "licence": licence,
                            "redistribution": redist,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            mark(source_id)

        # 3. Quran 114 Surahs (`abdelhaqueidali/Quran-kabyle-ayt-mensour`)
        quran_source = "abdelhaqueidali.quran-kab"
        quran_tsv = (
            raw_root / "abdelhaqueidali.quran-kab" / "quran_ascii_latin_amazigh_arabic_scripts.tsv"
        )
        if quran_tsv.is_file():
            tally = _source_tally(
                quran_source, "cc-by-4.0", "permissive", finished.get(quran_source)
            )
            tallies.append(tally)
            if not tally.resumed:
                for surah_text in _read_quran_surahs(quran_tsv, normaliser):
                    tally.read += 1
                    if not is_kabyle_window(surah_text, scorer):
                        tally.rejected["language"] += 1
                        continue
                    tally.kept += 1
                    tally.characters += len(surah_text)
                    sink.write(
                        json.dumps(
                            {
                                "text": surah_text,
                                "source": tally.source_id,
                                "licence": tally.licence,
                                "redistribution": tally.redistribution,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                mark(quran_source)

        # 4. `taqbaylit/kabyle-corpus` JSONL
        kab_source = "taqbaylit.kabyle-corpus"
        kab_corpus_jsonl = raw_root / "taqbaylit.kabyle-corpus" / "kabyle_corpus_clean.jsonl"
        if kab_corpus_jsonl.is_file():
            tally = _source_tally(kab_source, "mit", "permissive", finished.get(kab_source))
            tallies.append(tally)
            if not tally.resumed:
                with kab_corpus_jsonl.open(encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        tally.read += 1
                        try:
                            raw_txt = str(json.loads(line).get("text", "") or "")
                        except json.JSONDecodeError:
                            tally.rejected["invalid_json"] += 1
                            continue
                        cleaned = clean(repair_legacy_font(raw_txt), normaliser)
                        if len(cleaned) < MIN_DOCUMENT_CHARS:
                            tally.rejected["too_short"] += 1
                            continue
                        if not is_kabyle_window(cleaned, scorer):
                            tally.rejected["language"] += 1
                            continue
                        tally.kept += 1
                        tally.characters += len(cleaned)
                        sink.write(
                            json.dumps(
                                {
                                    "text": cleaned,
                                    "source": tally.source_id,
                                    "licence": tally.licence,
                                    "redistribution": tally.redistribution,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                mark(kab_source)

        # 5. `Sifal/KabyleWikipedia` (`WIKABV1.zip`)
        wiki_source = "sifal.kabyle-wikipedia"
        wikab_zip = raw_root / "sifal.kabyle-wikipedia" / "WIKABV1.zip"
        if wikab_zip.is_file():
            tally = _source_tally(
                wiki_source, "cc-by-sa-3.0", "share-alike", finished.get(wiki_source)
            )
            tallies.append(tally)
            if not tally.resumed:
                with zipfile.ZipFile(wikab_zip) as archive:
                    for member in sorted(archive.namelist()):
                        if not member.endswith(".txt"):
                            continue
                        tally.read += 1
                        raw_txt = archive.read(member).decode("utf-8", errors="replace")
                        cleaned = clean(repair_legacy_font(raw_txt), normaliser)
                        if len(cleaned) < MIN_DOCUMENT_CHARS:
                            tally.rejected["too_short"] += 1
                            continue
                        if not is_kabyle_window(cleaned, scorer):
                            tally.rejected["language"] += 1
                            continue
                        tally.kept += 1
                        tally.characters += len(cleaned)
                        sink.write(
                            json.dumps(
                                {
                                    "text": cleaned,
                                    "source": tally.source_id,
                                    "licence": tally.licence,
                                    "redistribution": tally.redistribution,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                mark(wiki_source)

    return tallies


def split_audio_chunks(
    audio: NDArray[np.float32],
    rate: int = SAMPLE_RATE,
    min_sec: float = MIN_CHUNK_SEC,
    max_sec: float = MAX_CHUNK_SEC,
) -> list[NDArray[np.float32]]:
    """Split a long broadcast into 10-22s chunks at local energy minima."""
    total = len(audio)
    max_samples = int(max_sec * rate)
    min_samples = int(min_sec * rate)
    if total <= max_samples:
        return [audio] if total >= int(1.5 * rate) else []

    chunks: list[NDArray[np.float32]] = []
    cursor = 0
    window = int(0.1 * rate)  # 100ms energy window
    while cursor < total:
        remaining = total - cursor
        if remaining <= max_samples:
            if remaining >= int(1.5 * rate):
                chunks.append(audio[cursor:])
            break
        search_start = cursor + min_samples
        search_end = min(total, cursor + max_samples)
        segment = audio[search_start:search_end]
        if len(segment) < window * 2:
            chunks.append(audio[cursor:search_end])
            cursor = search_end
            continue
        n_frames = len(segment) // window
        framed = segment[: n_frames * window].reshape(n_frames, window)
        energies = np.mean(framed * framed, axis=1)
        best_frame = int(np.argmin(energies))
        cut = search_start + best_frame * window + window // 2
        chunks.append(audio[cursor:cut])
        cursor = cut
    return chunks


def _load_audio_16k(path: Path) -> NDArray[np.float32]:
    """Read MP3 or WAV via soundfile and resample to 16 kHz mono float32."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        gcd = math.gcd(int(sr), SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // gcd, int(sr) // gcd).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def extract_audio_corpora(
    scorer: LanguageScorer,
    *,
    raw_root: Path = RAW_V2_ROOT,
    models_root: Path = HUB_MODELS_ROOT,
    out_path: Path = EXPANDED_AUDIO_OUT,
    device: str = "cuda",
) -> list[ExpandTally]:
    """Transcribe Chaine 2 (`rradyu-tis-snat`) and `Tughalin` with Fadhma + Belaid + Boulifa."""
    import sentencepiece as spm
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    from agbalu.hub.belaid.configuration_belaid import BelaidConfig
    from agbalu.hub.belaid.modeling_belaid import BelaidForTokenClassification
    from agbalu.speech.lm import build_ctc_decoder
    from agbalu.speech.transcribe import AUDIO_SUFFIXES, LM_FILE, VOCAB_FILE
    from agbalu.speech.vocabulary import decode

    fadhma_dir = models_root / "fadhma"
    belaid_dir = models_root / "belaid"
    boulifa_dir = models_root / "boulifa"
    tok_dir = models_root / "mammeri_tok"

    if not (fadhma_dir / "config.json").is_file():
        message = f"Fadhma-300M not found at {fadhma_dir}; run `expand-fetch` first"
        raise ExpandError(message)

    vocab: dict[str, int] = json.loads((fadhma_dir / VOCAB_FILE).read_text(encoding="utf-8"))
    kenlm_path = fadhma_dir / LM_FILE
    decoder = (
        build_ctc_decoder(vocab, unigrams=None, kenlm_path=kenlm_path)
        if kenlm_path.is_file()
        else None
    )

    torch_device = torch.device(device)
    placed_asr: torch.nn.Module = Wav2Vec2ForCTC.from_pretrained(fadhma_dir)
    asr_model = placed_asr.to(torch_device).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(fadhma_dir)

    belaid_model: BelaidForTokenClassification | None = None
    belaid_tok: object | None = None
    spm_candidates = list(tok_dir.rglob("*16k*.model")) or list(
        Path("artifacts/tokenizer").glob("*16k*.model")
    )
    if (belaid_dir / "config.json").is_file() and spm_candidates:
        b_cfg = BelaidConfig.from_pretrained(belaid_dir)
        placed_belaid: torch.nn.Module = BelaidForTokenClassification.from_pretrained(
            belaid_dir, config=b_cfg
        )
        belaid_model = placed_belaid.to(torch_device).eval()  # type: ignore[assignment]
        proc = spm.SentencePieceProcessor()
        proc.load(str(spm_candidates[0]))

        class _SpmAdapter:
            cls_token_id = 2
            sep_token_id = 3

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                _ = add_special_tokens
                return [int(i) for i in proc.encode(text)]

        belaid_tok = _SpmAdapter()

    boulifa = (
        HubBoulifaEngine(boulifa_dir, device) if (boulifa_dir / "config.json").is_file() else None
    )
    normaliser = Normaliser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tallies: list[ExpandTally] = []

    with out_path.open("w", encoding="utf-8") as sink:
        for source_id, _, licence, redist in AUDIO_REPOS:
            folder = raw_root / source_id
            if not folder.is_dir():
                continue
            clips = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
            tally = ExpandTally(source_id, licence, redist)
            tallies.append(tally)
            tracker = Tracker(f"audio {source_id}", len(clips))

            for clip_path in clips:
                tally.read += 1
                try:
                    waveform = _load_audio_16k(clip_path)
                except Exception:
                    tally.rejected["unreadable_audio"] += 1
                    tracker.advance()
                    continue

                chunks = split_audio_chunks(waveform)
                if not chunks:
                    tally.rejected["too_short"] += 1
                    tracker.advance()
                    continue

                utterances: list[str] = []
                with torch.inference_mode():
                    for chunk in chunks:
                        batch = extractor(
                            chunk,
                            sampling_rate=SAMPLE_RATE,
                            return_tensors="pt",
                            return_attention_mask=True,
                        )
                        logits = asr_model(
                            batch.input_values.to(torch_device),
                            attention_mask=batch.attention_mask.to(torch_device),
                        ).logits[0]
                        raw_hyp = (
                            str(decoder.decode(logits.cpu().numpy()))
                            if decoder is not None
                            else decode(logits.argmax(dim=-1).tolist(), vocab)
                        ).strip()
                        if len(raw_hyp) < 15:
                            continue
                        if belaid_model is not None and belaid_tok is not None:
                            restored = belaid_model.restore(
                                [raw_hyp],
                                belaid_tok,  # type: ignore[arg-type]
                            )
                            if restored and restored[0]:
                                raw_hyp = restored[0]
                        # ASR output is always bare Latin (no diacritics); always
                        # apply Boulifa unconditionally rather than gating on a
                        # threshold that is unreliable for zero-diacritic text.
                        if boulifa is not None:
                            raw_hyp = boulifa.standardise_lines(raw_hyp)
                        normed = clean(raw_hyp, normaliser)
                        if is_kabyle_window(normed, scorer):
                            utterances.append(normed)

                if not utterances:
                    tally.rejected["no_kabyle_speech"] += 1
                    tracker.advance()
                    continue

                full_doc = clean("\n\n".join(utterances), normaliser)
                if len(full_doc) < MIN_DOCUMENT_CHARS:
                    tally.rejected["too_short"] += 1
                    tracker.advance()
                    continue

                tally.kept += 1
                tally.characters += len(full_doc)
                sink.write(
                    json.dumps(
                        {
                            "text": full_doc,
                            "source": source_id,
                            "licence": licence,
                            "redistribution": redist,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                tracker.advance()

    return tallies


def _read_tmx_pairs(tmx_path: Path) -> Iterator[tuple[str, str]]:
    """Read `(kab, eng)` pairs from a TMX translation memory file."""
    kab_text = ""
    eng_text = ""
    current_lang = ""
    with tmx_path.open("rb") as stream:
        for _, elem in iterparse(stream, events=("start", "end")):
            tag = elem.tag.rsplit("}", 1)[-1].lower()
            if tag == "tuv":
                for k, v in elem.attrib.items():
                    if k.lower().endswith("lang"):
                        current_lang = v.lower()
            elif tag == "seg" and elem.text:
                if current_lang.startswith("kab"):
                    kab_text = elem.text.strip()
                elif current_lang.startswith("en"):
                    eng_text = elem.text.strip()
            elif tag == "tu":
                if kab_text and eng_text:
                    yield kab_text, eng_text
                kab_text, eng_text, current_lang = "", "", ""
                elem.clear()


def extract_aligned_records(
    scorer: LanguageScorer,
    *,
    raw_root: Path = RAW_V2_ROOT,
    raw_v1_root: Path = Path("data/raw"),
    out_path: Path = EXPANDED_ALIGNED_OUT,
) -> dict[str, int]:
    """Extract and Dihya-filter all new parallel datasets into `expanded-aligned.jsonl`."""
    import pyarrow.parquet as pq

    normaliser = Normaliser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    index = 0

    with out_path.open("w", encoding="utf-8") as sink:

        def write_record(kab: str, other: str, code: str, source_name: str) -> None:
            nonlocal index
            sink.write(
                json.dumps(
                    {
                        "kabyle": kab,
                        "other": other,
                        "code": code,
                        "index": index,
                        "source": source_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            index += 1
            counts[source_name] = counts.get(source_name, 0) + 1

        # 1. `taqbaylit/tatoeba-en-kab` (244,954 en-kab pairs)
        tatoeba_dir = raw_root / "taqbaylit.tatoeba-en-kab"
        for pq_file in sorted(tatoeba_dir.rglob("*.parquet")):
            table = pq.read_table(pq_file, columns=["translation"])
            for item in table["translation"].to_pylist():
                if isinstance(item, dict):
                    kab = normaliser.normalise(str(item.get("kab", "") or "")).strip()
                    eng = str(item.get("en", "") or "").strip()
                    if kab and eng:
                        write_record(kab, eng, LANGUAGE_TAG["eng"], "taqbaylit.tatoeba-en-kab")

        # 2. `taqbaylit/weblate-kabyle` (`en-kab.tmx`)
        tmx_path = raw_root / "taqbaylit.weblate-kabyle" / "en-kab.tmx"
        if tmx_path.is_file():
            for kab_raw, eng_raw in _read_tmx_pairs(tmx_path):
                kab = normaliser.normalise(kab_raw).strip()
                eng = eng_raw.strip()
                if len(kab) >= 10 and len(eng) >= 10 and kab != eng:
                    write_record(kab, eng, LANGUAGE_TAG["eng"], "taqbaylit.weblate-kabyle")

        # 3. `TutlaytAI/Multilang-to-Berber-Translation` (filtered by Dihya-5M!)
        tut_dir = raw_root / "tutlaytai.multilang-berber" / "data"
        split_map = {
            "en": LANGUAGE_TAG["eng"],
            "fr": LANGUAGE_TAG["fra"],
            "ara": LANGUAGE_TAG.get("ara", "arb_Arab"),
        }
        for split_prefix, lang_code in split_map.items():
            for pq_file in sorted(tut_dir.glob(f"{split_prefix}-*.parquet")):
                table = pq.read_table(pq_file, columns=["Ber Sentence", "Translated Sentence"])
                ber_list = [
                    normaliser.normalise(str(x or "")).strip()
                    for x in table["Ber Sentence"].to_pylist()
                ]
                oth_list = [str(x or "").strip() for x in table["Translated Sentence"].to_pylist()]
                for start in range(0, len(ber_list), 1024):
                    b_chunk = ber_list[start : start + 1024]
                    o_chunk = oth_list[start : start + 1024]
                    valid_idx = [
                        i
                        for i, (b, o) in enumerate(zip(b_chunk, o_chunk, strict=True))
                        if len(b) >= 12 and len(o) >= 10 and b != o
                    ]
                    if not valid_idx:
                        continue
                    probs = scorer.probabilities([b_chunk[i][:LID_WINDOW_CHARS] for i in valid_idx])
                    for idx_in_chunk, row_prob in zip(valid_idx, probs, strict=True):
                        if row_prob.get(KABYLE, 0.0) >= 0.80:
                            write_record(
                                b_chunk[idx_in_chunk],
                                o_chunk[idx_in_chunk],
                                lang_code,
                                f"tutlaytai.multilang-{split_prefix}",
                            )

        # 4. `taqbaylit/nllb_en_kab` (high-precision Dihya-5M >= 0.90 + length/copy filter)
        nllb_pq = raw_root / "taqbaylit.nllb-en-kab" / "nllb_en_kab.parquet"
        if not nllb_pq.is_file():
            alt = raw_v1_root / "hf.boffire.nllb-en-kab" / "nllb_en_kab.parquet"
            if alt.is_file():
                nllb_pq = alt
        if nllb_pq.is_file():
            pf = pq.ParquetFile(nllb_pq)
            cols = pf.schema_arrow.names
            kab_col = next((c for c in ("kabyle", "kab", "kab_Latn") if c in cols), cols[-1])
            eng_col = next((c for c in ("english", "en", "eng_Latn") if c in cols), cols[0])
            for batch in pf.iter_batches(columns=[kab_col, eng_col], batch_size=2048):
                k_raw = batch.column(kab_col).to_pylist()
                e_raw = batch.column(eng_col).to_pylist()
                candidates: list[tuple[str, str]] = []
                for k_val, e_val in zip(k_raw, e_raw, strict=True):
                    k_str = normaliser.normalise(str(k_val or "")).strip()
                    e_str = str(e_val or "").strip()
                    if len(k_str) < 30 or len(e_str) < 25 or k_str == e_str:
                        continue
                    ratio = len(k_str) / max(1, len(e_str))
                    if not (0.45 <= ratio <= 2.2):
                        continue
                    if not any(ch in KABYLE_DIACRITICS for ch in k_str):
                        continue
                    candidates.append((k_str, e_str))
                if not candidates:
                    continue
                probs = scorer.probabilities([c[0][:LID_WINDOW_CHARS] for c in candidates])
                for (k_str, e_str), row_prob in zip(candidates, probs, strict=True):
                    if row_prob.get(KABYLE, 0.0) >= 0.90:
                        write_record(k_str, e_str, LANGUAGE_TAG["eng"], "taqbaylit.nllb-en-kab-hq")

    return counts


def read_expanded_aligned(path: Path = EXPANDED_ALIGNED_OUT) -> Iterator[Record]:
    """Stream `Record` objects from `expanded-aligned.jsonl` into `mixture.build`."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield Record(
                kabyle=str(row["kabyle"]),
                index=int(row["index"]),
                other=str(row["other"]),
                code=str(row["code"]),
            )


def merge_expanded_sentences(
    base_sentences: Iterator[SentenceRow],
    expanded_docs_path: Path = EXPANDED_DOCS_OUT,
    expanded_audio_path: Path = EXPANDED_AUDIO_OUT,
    raw_root: Path = RAW_V2_ROOT,
) -> Iterator[SentenceRow]:
    """Chain expanded PDFs, reconstructed theses, ASR transcripts, and `agbalu-text-v1.jsonl`."""
    line_no = 0
    reconstructed_imsidag: set[str] = set()
    for extra_path in (expanded_docs_path, expanded_audio_path):
        if not extra_path.is_file():
            continue
        with extra_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                line_no += 1
                row = json.loads(line)
                src = str(row["source"])
                if src.startswith("hf.imsidag.kabyle-corpus-"):
                    reconstructed_imsidag.add(src)
                yield SentenceRow(
                    text=str(row["text"]),
                    source_id=src,
                    licence=str(row["licence"]),
                    redistribution=str(row["redistribution"]),
                    line=line_no,
                )

    for item in base_sentences:
        if item.source_id in reconstructed_imsidag:
            continue
        line_no += 1
        yield item

    # Also stream new sentence-level sources (`tatoeba-mono-cleaned`, `hillal`, `ad019el`)
    import pyarrow.parquet as pq

    mono_pq = raw_root / "taqbaylit.tatoeba-mono-cleaned" / "data" / "train-00000-of-00001.parquet"
    if mono_pq.is_file():
        pf = pq.ParquetFile(mono_pq)
        col = "text" if "text" in pf.schema_arrow.names else pf.schema_arrow.names[0]
        for batch in pf.iter_batches(columns=[col], batch_size=2048):
            for val in batch.column(col).to_pylist():
                if isinstance(val, str) and len(val.strip()) >= 20:
                    line_no += 1
                    yield SentenceRow(
                        text=val.strip(),
                        source_id="taqbaylit.tatoeba-mono-cleaned",
                        licence="mit",
                        redistribution="permissive",
                        line=line_no,
                    )

    hillal_csv = raw_root / "hillal.tamazight-kabyle-text" / "data.csv"
    if hillal_csv.is_file():
        with hillal_csv.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                cleaned = repair_legacy_font(raw_line.strip().strip('"'))
                if len(cleaned) >= 25 and cleaned != "text":
                    line_no += 1
                    yield SentenceRow(
                        text=cleaned,
                        source_id="hillal.tamazight-kabyle-text",
                        licence="mit",
                        redistribution="permissive",
                        line=line_no,
                    )

    ad_dir = raw_root / "ad019el.dataset-kabyle" / "data"
    for pq_file in sorted(ad_dir.glob("*.parquet")) if ad_dir.is_dir() else []:
        table = pq.read_table(pq_file, columns=["sentence"])
        for val in table["sentence"].to_pylist():
            if isinstance(val, str) and len(val.strip()) >= 15:
                line_no += 1
                yield SentenceRow(
                    text=val.strip(),
                    source_id="ad019el.dataset-kabyle",
                    licence="cc-by-4.0",
                    redistribution="permissive",
                    line=line_no,
                )


FRENCH_WORDS: Final[frozenset[str]] = frozenset(
    [
        "le",
        "la",
        "les",
        "des",
        "du",
        "de",
        "que",
        "qui",
        "est",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "ne",
        "pas",
        "pour",
        "dans",
        "avec",
        "sur",
        "une",
        "un",
        "et",
        "mais",
        "ce",
        "cette",
        "sont",
        "être",
        "avoir",
        "fait",
        "comme",
        "plus",
        "très",
        "aussi",
    ]
)
"""French function words; Kabyle uses none of these as a word, so a line made of them is French."""

FRENCH_LINE_SHARE: Final = 0.25
"""A line whose share of French function words exceeds this is dropped."""

GLUED_TOKEN_CHARS: Final = 25
"""A token this long is two or more words glued together by the PDF extractor."""

GLUED_DOCUMENT_SHARE: Final = 0.01
"""A document whose share of glued tokens exceeds this is dropped."""

_WORD_RE: Final = re.compile(r"[^\W\d_]+")


def french_share(line: str) -> float:
    """Fraction of a line's words that are French function words."""
    words = _WORD_RE.findall(line.lower())
    if not words:
        return 0.0
    return sum(word in FRENCH_WORDS for word in words) / len(words)


def clean_document(text: str) -> str | None:
    """Drop French lines; return None if the document is glued or too short afterwards."""
    words = _WORD_RE.findall(text)
    if words and sum(len(w) > GLUED_TOKEN_CHARS for w in words) / len(words) > GLUED_DOCUMENT_SHARE:
        return None
    lines = [ln for ln in text.split("\n") if french_share(ln) <= FRENCH_LINE_SHARE]
    cleaned = "\n".join(lines).strip()
    return cleaned if len(cleaned) >= MIN_DOCUMENT_CHARS else None


def clean_jsonl(path: Path, scorer: LanguageScorer) -> dict[str, dict[str, int]]:
    """Rewrite `path` in place: French lines out, then the 4B document rule re-applied.

    The original is kept as `<name>.raw`. Returns per-source counts of read, kept and the
    reason each rejected document was dropped.
    """
    raw = path.with_name(path.name + ".raw")
    if not raw.exists():
        path.rename(raw)
    counts: dict[str, dict[str, int]] = {}
    with raw.open(encoding="utf-8") as src, path.open("w", encoding="utf-8") as sink:
        for line in src:
            row = json.loads(line)
            tally = counts.setdefault(
                row["source"], {"read": 0, "kept": 0, "glued_or_short": 0, "language": 0}
            )
            tally["read"] += 1
            cleaned = clean_document(row["text"])
            if cleaned is None:
                tally["glued_or_short"] += 1
                continue
            if not is_kabyle_window(cleaned, scorer):
                tally["language"] += 1
                continue
            row["text"] = cleaned
            tally["kept"] += 1
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts
