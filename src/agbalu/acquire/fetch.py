"""Fetch a registered source to disk and describe what landed.

One function per access type, dispatched exhaustively over `Source.access`.
Downloading itself is delegated: `huggingface_hub` already does resumable,
retrying, parallel transfer for the 49 HF sources, and `git` does it for the 3
git ones. Only plain HTTP needs our own loop.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import requests
from huggingface_hub import repo_info, snapshot_download

from agbalu._about import __version__
from agbalu.acquire.models import Artifact
from agbalu.acquire.placement import artifact_kind, classify
from agbalu.acquire.storage import PART_SUFFIX, finalize, sha256_file
from agbalu.registry.models import SourceBase

log: Final = logging.getLogger("agbalu.acquire")

HF_PREFIX: Final = "https://huggingface.co/datasets/"
HF_MODEL_PREFIX: Final = "https://huggingface.co/"

FETCH_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    # Sources that are a single-language slice of a multilingual mega-dataset.
    # Without a filter, HPLT alone is 112,335 files; the Kabyle share is one.
    "hf.fineweb2-kab": ("data/kab_Latn/**",),
    "hf.hplt2-cleaned-kab": ("kab_Latn/**",),
    "hf.glotcc-v1-kab": ("v1.0/kab-Latn/**",),
    "hf.cohere.xp3x-kab": ("data/kab_Latn/**",),
    "hf.ayymen.pontoon-translations": ("*kab*",),
    "hf.ayymen.weblate-translations": ("*kab*",),
    # `finetranslations` is 1.39 TB whole.
    "hf.finetranslations-kab": ("data/kab_Latn/**",),
    "hf.finepdfs-kab": ("data/kab_Latn/**",),
    "hf.dcad2000-kab": ("kab_Latn/**",),
    "hf.glot500-kab": ("kab_Latn/**",),
    "hf.hplt2-edu-annotation-kab": ("kab-Latn/**",),
    "hf.goldfish.fish-food-kab": ("*kab*",),
    "hf.tapaco-kab": ("kab/**",),
    "hf.tamazight-nlp.weblate": ("*kab*",),
    "hf.tamazight-nlp.pontoon": ("*kab*",),
    "hf.tamazight-nlp.translatewiki": ("*kab*",),
    "hf.tamazight-nlp.anki": ("*kab*",),
    # Common Voice ships 103 languages and 18.6 GB. Only the Kabyle transcripts
    # belong on this machine: the audio is 7.89 GiB that nothing local reads, and
    # it goes straight to the Modal volume instead (docs/asr_design.md §2).
    "hf.fsicoli.common-voice-22-kab": ("transcript/kab/**",),
    # `sib200` ships 200 languages in sibling directories; `panlex-meanings` has no
    # kab-named file at all, so filtering it yields nothing and it is taken whole.
    "hf.sib200-kab": ("kab_Latn/**",),
    # FLORES+ is 485 files over ~200 languages. `eng_Latn` comes too: it is the
    # source side of every en-kab MT evaluation, and task 7.0 audits the pair.
    # French is not optional: `fr-kab` is the larger human-authored Tatoeba pair
    # (93,908 against 58,674) and no published Kabyle MT model evaluates it.
    # Arabic, Spanish and German are the pivot-synthesis targets: each needs a
    # source side to score its X-kab direction against the untuned baseline.
    # Unfiltered this repo is 485 files / 223.6 MB.
    "hf.flores-plus-kab": (
        "*/kab_Latn.jsonl",
        "*/eng_Latn.jsonl",
        "*/fra_Latn.jsonl",
        "*/arb_Arab.jsonl",
        "*/spa_Latn.jsonl",
        "*/deu_Latn.jsonl",
        "*.md",
    ),
    # The repo ships model.bin plus v1/v2/v3 beside it; model.bin is v3 byte for
    # byte. Unfiltered this is 6.45 GB for 1.69 GB of distinct weights.
    "hf.glotlid-model": ("model.bin", "*.md", "LICENSE"),
    "hf.nllb-lid218e": ("model.bin", "*.md"),
    # Task 7.4. GlotCC is 1,252 language-script partitions; each sibling is one.
    # Latin only: a Tifinagh partition is separable from Kabyle on script alone,
    # which measures the encoding, not the language.
    "hf.glotcc-v1-shi": ("v1.0/shi-Latn/**",),
    "hf.glotcc-v1-rif": ("v1.0/rif-Latn/**",),
    "hf.glotcc-v1-taq": ("v1.0/taq-Latn/**",),
    "hf.glotcc-v1-tzm": ("v1.0/tzm-Latn/**",),
    # The one sibling FLORES+ carries in Latin, and the only controlled contrast
    # available: identical source sentences, so topic cannot leak into the label.
    "hf.flores-plus-taq": ("*/taq_Latn.jsonl", "*.md"),
    # Only the tokenizer is assessed; the weights are 2.5 GB and CC-BY-NC-4.0.
    "hf.facebook.nllb-200-distilled-600m": (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "*.md",
    ),
}

GIT_REFS: Final[dict[str, str]] = {
    # The only Kabyle treebank has never been in a UD release; it lives on `dev`.
    "git.ud-kabyle-adpt": "dev",
}

EXCLUDED_DIRS: Final = frozenset({".git", ".cache", "__pycache__"})

OPUS_API: Final = "https://opus.nlpl.eu/opusapi/"
WIKIMEDIA_DUMPS: Final = "https://dumps.wikimedia.org"

USER_AGENT: Final = (
    f"AGBALU/{__version__} (Kabyle NLP corpus; +https://github.com/abdoumagico/agbalu)"
)
"""Wikimedia's User-Agent policy rejects generic agents with HTTP 403, and it is
the courteous thing to send to every provider regardless."""

HEADERS: Final[dict[str, str]] = {"User-Agent": USER_AGENT}

ISO3_TO_ISO2: Final[dict[str, str]] = {
    # OPUS indexes by 2-letter code; the registry stores ISO 639-3.
    "eng": "en", "fra": "fr", "ara": "ar", "spa": "es", "deu": "de",
    "ita": "it", "rus": "ru", "por": "pt", "tur": "tr", "kab": "kab",
}  # fmt: skip

_DUMP_DATE_RE: Final = re.compile(r"\b(20\d{6})/")

CHUNK_BYTES: Final = 1 << 20
HTTP_TIMEOUT_S: Final = 60.0

_OK: Final = 200
_PARTIAL_CONTENT: Final = 206
_RANGE_NOT_SATISFIABLE: Final = 416


class FetchError(Exception):
    """A source could not be retrieved."""


@dataclass(frozen=True)
class FetchResult:
    """What landed, and the upstream revision it came from.

    `revision` is the whole point of the manifest: without it a re-fetch cannot
    be told from a silent upstream edit.
    """

    revision: str | None
    artifacts: tuple[Artifact, ...]


def repo_id(source: SourceBase) -> str:
    """Extract the HuggingFace repo id from a dataset *or* model URL."""
    return hf_repo(source)[0]


def hf_repo(source: SourceBase) -> tuple[str, Literal["dataset", "model"]]:
    """Return `(repo_id, repo_type)` for a HuggingFace URL.

    Models live at `huggingface.co/<owner>/<name>`, datasets at
    `huggingface.co/datasets/<owner>/<name>`. Assuming every source is a dataset
    is what made the whole `reference` tier unfetchable: six of its sources are
    the published Kabyle tokenizers, the POS model and GlotLID, all model repos.
    """
    if source.uri.startswith(HF_PREFIX):
        return source.uri.removeprefix(HF_PREFIX).strip("/"), "dataset"
    if source.uri.startswith(HF_MODEL_PREFIX):
        repo = source.uri.removeprefix(HF_MODEL_PREFIX).strip("/")
        if repo and repo.count("/") == 1:
            return repo, "model"
    msg = f"{source.id}: not a HuggingFace dataset or model URL: {source.uri}"
    raise FetchError(msg)


def walk_artifacts(root: Path, source: SourceBase) -> Iterator[Artifact]:
    """Hash every file under `root` and describe it."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if EXCLUDED_DIRS & set(path.relative_to(root).parts):
            continue
        if path.name.endswith(PART_SUFFIX):
            continue
        relative = path.relative_to(root).as_posix()
        digest, size = sha256_file(path)
        yield Artifact(
            path=relative,
            bytes=size,
            sha256=digest,
            kind=artifact_kind(relative, source.modality),
            target=classify(
                tier=source.tier, modality=source.modality, path=relative, size_bytes=size
            ),
        )


def fetch_hf(source: SourceBase, dest: Path) -> str | None:
    repo, repo_type = hf_repo(source)
    patterns = FETCH_PATTERNS.get(source.id)
    revision = repo_info(repo, repo_type=repo_type).sha
    log.info(
        "fetch source=%s access=hf type=%s repo=%s rev=%s",
        source.id,
        repo_type,
        repo,
        (revision or "?")[:12],
    )
    snapshot_download(
        repo,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=list(patterns) if patterns else None,
    )
    return revision


def git_binary() -> str:
    """Absolute path to `git`, or a clear error naming the missing dependency."""
    found = shutil.which("git")
    if found is None:
        msg = "git is not on PATH; the git fetcher cannot run"
        raise FetchError(msg)
    return found


def _git(dest: Path, *args: str) -> str:
    result = subprocess.run(
        [git_binary(), "-C", str(dest), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        msg = f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        raise FetchError(msg)
    return result.stdout.strip()


def fetch_git(source: SourceBase, dest: Path) -> str | None:
    ref = GIT_REFS.get(source.id)
    log.info("fetch source=%s access=git uri=%s ref=%s", source.id, source.uri, ref or "default")
    if dest.exists():
        shutil.rmtree(dest)
    command = [git_binary(), "clone", "--depth", "1", "--quiet"]
    if ref:
        command += ["--branch", ref]
    command += [source.uri, str(dest)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = f"{source.id}: git clone failed ({result.returncode}): {result.stderr.strip()}"
        raise FetchError(msg)
    return _git(dest, "rev-parse", "HEAD")


def download_file(url: str, final: Path) -> None:
    """Stream `url` to `final`, resuming a previous `.part` if one survives."""
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + PART_SUFFIX)
    offset = part.stat().st_size if part.exists() else 0
    headers = dict(HEADERS)
    if offset:
        headers["Range"] = f"bytes={offset}-"

    with requests.get(
        url, headers=headers, stream=True, timeout=HTTP_TIMEOUT_S, allow_redirects=True
    ) as response:
        if response.status_code not in (_OK, _PARTIAL_CONTENT, _RANGE_NOT_SATISFIABLE):
            msg = f"{url}: HTTP {response.status_code}"
            raise FetchError(msg)
        if response.status_code == _RANGE_NOT_SATISFIABLE:
            finalize(part, final)
            return
        mode = "ab" if response.status_code == _PARTIAL_CONTENT and offset else "wb"
        with part.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                handle.write(chunk)
    finalize(part, final)


def fetch_http(source: SourceBase, dest: Path) -> str | None:
    log.info("fetch source=%s access=http uri=%s", source.id, source.uri)
    name = source.uri.rstrip("/").rsplit("/", maxsplit=1)[-1] or f"{source.id}.bin"
    download_file(source.uri, dest / name)
    return None


def opus_corpus(source: SourceBase) -> str:
    """Corpus name from an OPUS URI (`https://opus.nlpl.eu/<Corpus>/corpus/version/...`)."""
    tail = source.uri.removeprefix("https://opus.nlpl.eu/").strip("/")
    name = tail.split("/", maxsplit=1)[0]
    if not name:
        msg = f"{source.id}: cannot read a corpus name from {source.uri}"
        raise FetchError(msg)
    return name


def fetch_opus(source: SourceBase, dest: Path) -> str | None:
    """Resolve every declared kab-* pair through the OPUS API, then download each bundle.

    The registry's `languages` is the contract for which pairs belong to a source;
    OPUS itself carries many more (Tatoeba alone has 93).
    """
    corpus = opus_corpus(source)
    wanted = {ISO3_TO_ISO2.get(code, code) for code in source.languages if code != "kab"}
    response = requests.get(
        OPUS_API,
        params={"corpus": corpus, "source": "kab", "preprocessing": "moses", "version": "latest"},
        timeout=HTTP_TIMEOUT_S,
        headers=HEADERS,
    )
    if response.status_code != _OK:
        msg = f"{source.id}: OPUS API returned HTTP {response.status_code}"
        raise FetchError(msg)

    version: str | None = None
    found = 0
    for entry in response.json().get("corpora", []):
        pair = {entry["source"], entry["target"]}
        if not (pair - {"kab"}) & wanted:
            continue
        url = entry["url"]
        name = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        log.info(
            "fetch source=%s access=opus pair=%s-%s version=%s",
            source.id,
            entry["source"],
            entry["target"],
            entry["version"],
        )
        download_file(url, dest / name)
        version = entry["version"]
        found += 1

    if not found:
        msg = f"{source.id}: OPUS has no {corpus} bundle for kab-{sorted(wanted)}"
        raise FetchError(msg)
    return version


def wikimedia_wiki(source: SourceBase) -> str:
    """Wiki name from a dumps URI (`https://dumps.wikimedia.org/<wiki>/`)."""
    name = source.uri.removeprefix(WIKIMEDIA_DUMPS).strip("/").split("/", maxsplit=1)[0]
    if not name:
        msg = f"{source.id}: cannot read a wiki name from {source.uri}"
        raise FetchError(msg)
    return name


def fetch_wikimedia(source: SourceBase, dest: Path) -> str | None:
    """Download the newest dated `pages-articles` dump.

    The dated directory is used rather than `latest/` so the manifest pins a real
    dump, not a symlink whose target changes weekly.
    """
    wiki = wikimedia_wiki(source)
    index = requests.get(f"{WIKIMEDIA_DUMPS}/{wiki}/", timeout=HTTP_TIMEOUT_S, headers=HEADERS)
    if index.status_code != _OK:
        msg = f"{source.id}: dump index returned HTTP {index.status_code}"
        raise FetchError(msg)
    dates: list[str] = sorted(set(_DUMP_DATE_RE.findall(index.text)))
    if not dates:
        msg = f"{source.id}: no dated dump directory at {WIKIMEDIA_DUMPS}/{wiki}/"
        raise FetchError(msg)

    stamp = dates[-1]
    name = f"{wiki}-{stamp}-pages-articles.xml.bz2"
    log.info("fetch source=%s access=wikimedia dump=%s", source.id, stamp)
    download_file(f"{WIKIMEDIA_DUMPS}/{wiki}/{stamp}/{name}", dest / name)
    return stamp


def fetch_manual(source: SourceBase, dest: Path) -> None:
    """A source already in the tree. Nothing is fetched; the bytes are verified in place."""
    origin = Path(source.uri)
    if not origin.is_file():
        msg = f"{source.id}: declared local file is missing: {origin}"
        raise FetchError(msg)
    log.info("verify source=%s access=manual path=%s", source.id, origin)
    if origin.resolve() == (dest / origin.name).resolve():
        return
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin, dest / origin.name)


def fetch(source: SourceBase, dest: Path) -> FetchResult:
    """Retrieve `source` into `dest` and describe what landed."""
    dest.mkdir(parents=True, exist_ok=True)
    revision: str | None = None
    match source.access:
        case "hf":
            revision = fetch_hf(source, dest)
        case "git":
            revision = fetch_git(source, dest)
        case "http":
            revision = fetch_http(source, dest)
        case "manual":
            fetch_manual(source, dest)
        case "opus":
            revision = fetch_opus(source, dest)
        case "wikimedia":
            revision = fetch_wikimedia(source, dest)
    artifacts = tuple(walk_artifacts(dest, source))
    if not artifacts:
        # Silence here is how 720 MB landed with no provenance in Phase 1: an
        # allow_patterns that matches nothing, or an interrupted walk, both left the
        # manifest empty while `fetch` reported success.
        msg = f"{source.id}: fetch produced no artifacts under {dest}"
        raise FetchError(msg)
    return FetchResult(revision=revision, artifacts=artifacts)
