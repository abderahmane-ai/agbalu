"""Baseline the untouched base model on Kabyle (task 11.2).

Stage 0 of `docs/llm_adaptation_design.md` §8: the number every later Phase 11 claim must
beat, and the English and French numbers the retention half of the exit criterion is
measured against. Nothing downstream is falsifiable until this artifact exists, which is why
it runs before a single block is copied.

Two measurements, one model load. Perplexity and bits per character on four held-out sets,
and AƔBALU-Bench translation driven as a prompt and scored by the same
`agbalu.bench.mt.score` as NLLB-1.3B and Amrouche — so the base model's Kabyle lands in the
existing table rather than in a new one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from agbalu.llm.bases import BASE
from modal_app.bench import (
    REMOTE_CORRECTED,
    SPLIT,
    check_corpora,
    sources_and_references,
)
from modal_app.common import (
    DATA_PATH,
    LLM_TIMEOUT,
    LLM_VOLUMES,
    REMOTE_FLORES,
    RETRIES,
    app,
    data_volume,
    llm_image,
    models_volume,
)

if TYPE_CHECKING:
    from agbalu.bench.mt import Direction
    from agbalu.llm.baseline import CausalModel, Scored
    from agbalu.llm.prompting import ChatTokenizer, Shot, TextGenerator

GPU: Final = "A10"

BASE_MODEL: Final = BASE
"""One definition with the corpus counter and the fertility table, so the model this scores
is the model the token budget was measured in."""

LOCAL_HELDOUT: Final = Path("data/processed/llm")
REMOTE_HELDOUT: Final = "llm"
RESULTS: Final = "llm"

HELDOUT_LANGUAGES: Final[tuple[tuple[str, str], ...]] = (
    ("kab", "kab_Latn"),
    ("eng", "eng_Latn"),
    ("fra", "fra_Latn"),
)
"""Kabyle is the objective; English and French are the retention control. All three are
drawn from the CPT corpus itself by `agbalu.llm.holdout`, so a later comparison is
before-and-after over one distribution."""

BASELINE_DIRECTIONS: Final[tuple[str, ...]] = ("kab-eng", "eng-kab", "kab-fra", "fra-kab")
"""The four directions the MT fine-tune was trained and scored on. The three into-Kabyle
extras exist in the harness and can be requested, but they are not part of the baseline."""

SHOTS: Final = 5
SAMPLES: Final = 5
"""Translations recorded verbatim per direction. A chrF++ figure says a system moved; it
does not show what the Kabyle looks like, and a generative model has to be read."""

BATCH_SIZE: Final = 8

PERPLEXITY_FILE: Final = "baseline-perplexity.json"
TRANSLATION_FILE: Final = "baseline-mt-devtest.json"
CORPUS_FILE: Final = "cpt.jsonl"

log: Final = logging.getLogger("agbalu.llm")

Encode = Callable[[list[str]], list[list[int]]]


class PerplexityRow(TypedDict):
    """One evaluation set. A TypedDict because the value crosses the Modal boundary and
    lands in a JSON artifact, staying a plain dict on both sides."""

    model: str
    name: str
    language: str
    documents: int
    tokens: int
    characters: int
    nll: float
    truncated: int
    nll_per_token: float
    perplexity: float
    bits_per_character: float


class Sample(TypedDict):
    source: str
    hypothesis: str
    reference: str


class TranslationRow(TypedDict):
    """One direction, in the shape `modal_app.bench` writes so the two tables merge."""

    model: str
    family: str
    direction: str
    split: str
    sentences: int
    shots: int
    reference: str
    normaliser_version: str
    raw: dict[str, float]
    normalised: dict[str, float]
    chrf_gap: float
    bleu_gap: float
    samples: list[Sample]


class BaselineResult(TypedDict):
    model: str
    perplexity: list[PerplexityRow]
    translation: list[TranslationRow]


def perplexity_row(model_id: str, scored: Scored) -> PerplexityRow:
    """One `Scored` as the artifact's row.

    Written field by field rather than by spreading `as_dict()`: the spread type-checks only
    under an ignore, and the ignore would also hide a key this TypedDict stops declaring.
    """
    return {
        "model": model_id,
        "name": scored.name,
        "language": scored.language,
        "documents": scored.documents,
        "tokens": scored.tokens,
        "characters": scored.characters,
        "nll": round(scored.nll, 4),
        "truncated": scored.truncated,
        "nll_per_token": round(scored.nll_per_token, 6),
        "perplexity": round(scored.perplexity, 4),
        "bits_per_character": round(scored.bits_per_character, 6),
    }


def corrected_devtest() -> list[str]:
    """The corrected FLORES+ Kabyle reference, in `(split, id)` order."""
    path = Path(DATA_PATH) / REMOTE_CORRECTED
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        str(row["text"])
        for row in sorted((r for r in rows if r["split"] == SPLIT), key=lambda r: int(r["id"]))
    ]


def check_inputs(directions: Sequence[str]) -> None:
    """Every file this run reads, before the checkpoint is downloaded.

    The evaluation sets and FLORES+ are pushed by different targets, so either can be the
    one that was forgotten; a stat call here is what keeps that from being discovered after
    the weights have been paid for, times `RETRIES`.
    """
    from agbalu.bench.mt import DIRECTIONS

    unknown = [d for d in directions if d not in DIRECTIONS]
    if unknown:
        message = f"unknown direction(s) {unknown}; have {list(DIRECTIONS)}"
        raise ValueError(message)

    root = Path(DATA_PATH) / REMOTE_HELDOUT
    missing = [
        str(root / f"heldout-{short}.jsonl")
        for short, _ in HELDOUT_LANGUAGES
        if not (root / f"heldout-{short}.jsonl").is_file()
    ]
    if missing:
        message = (
            f"{len(missing)} evaluation set(s) absent from the volume: {missing}. "
            f"Run `make llm TASK=holdout` then `make modal-upload TASK=llm`."
        )
        raise FileNotFoundError(message)
    check_corpora([d for d in DIRECTIONS if d in set(directions)], corrected=True)


def populations() -> list[tuple[str, str, list[str]]]:
    """Every set that gets a perplexity, with the language each one measures.

    FLORES+ joins the three held-out sets because its decontamination against the corpus is
    measured and positive-controlled (task 7.0), and because it is the one population every
    other system in the project has already been scored on.
    """
    from agbalu.llm.holdout import read

    root = Path(DATA_PATH) / REMOTE_HELDOUT
    sets = [
        (f"heldout-{short}", code, read(root / f"heldout-{short}.jsonl"))
        for short, code in HELDOUT_LANGUAGES
    ]
    sets.append(("flores-devtest-kab", "kab_Latn", corrected_devtest()))
    return sets


def score_perplexity(
    model: CausalModel,
    encode: Encode,
    *,
    model_id: str,
    bos_id: int,
    pad_id: int,
    device: str,
    batch_size: int,
    out: Path,
) -> list[PerplexityRow]:
    """Every population, written after each one rather than at the end."""
    from agbalu.llm.baseline import likelihoods, measure

    rows: list[PerplexityRow] = []
    for name, language, texts in populations():
        values = likelihoods(
            texts,
            encode,
            model,
            bos_id=bos_id,
            pad_id=pad_id,
            device=device,
            batch_size=batch_size,
        )
        scored = measure(name, language, texts, values)
        rows.append(perplexity_row(model_id, scored))
        log.info(
            "perplexity %s | %s | %d docs | %d tokens | ppl %.3f | bpc %.4f | truncated %d",
            name,
            language,
            scored.documents,
            scored.tokens,
            scored.perplexity,
            scored.bits_per_character,
            scored.truncated,
        )
        (out / PERPLEXITY_FILE).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        data_volume.commit()
    return rows


def demonstrations(direction: Direction, count: int) -> tuple[Shot, ...]:
    """`count` shots from FLORES+ `dev`, which exists so prompts never touch `devtest`."""
    from agbalu.bench.flores import read_split
    from agbalu.bench.mt import LANGUAGE_CODE, source_language, target_language
    from agbalu.llm.prompting import shots

    root = Path(DATA_PATH) / REMOTE_FLORES

    def side(language: str) -> list[str]:
        rows = read_split(root, "dev", LANGUAGE_CODE[language])
        return [sentence.text for sentence in sorted(rows, key=lambda s: s.id)]

    return shots(side(source_language(direction)), side(target_language(direction)), count)


def translate(
    sources: Sequence[str],
    direction: Direction,
    model: TextGenerator,
    tokenizer: ChatTokenizer,
    encode: Encode,
    *,
    examples: Sequence[Shot],
    device: str,
    batch_size: int,
) -> list[str]:
    """Hypotheses in input order.

    Batched longest-first and restored by index: the prompt is five shots plus a sentence,
    so a corpus-order batch pads every row up to its longest neighbour for no reason, and
    scoring pairs hypothesis *i* with reference *i*.
    """
    import torch

    from agbalu.llm.prompting import (
        hypothesis,
        length_ordered_batches,
        messages,
        new_tokens,
        require_str,
        require_tensor,
    )

    prompts = [
        require_str(
            tokenizer.apply_chat_template(
                messages(direction, text, examples), tokenize=False, add_generation_prompt=True
            ),
            "apply_chat_template",
        )
        for text in sources
    ]
    lengths = [len(ids) for ids in encode(list(sources))]

    hypotheses = [""] * len(sources)
    for indices in length_ordered_batches(lengths, batch_size):
        # The template writes `<bos>` into the text itself, so letting the tokenizer add
        # another would put two on every prompt.
        encoded = tokenizer(
            [prompts[i] for i in indices],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)
        input_ids = require_tensor(encoded["input_ids"], "input_ids")
        with torch.inference_mode():
            produced = require_tensor(
                model.generate(
                    input_ids=input_ids,
                    attention_mask=require_tensor(encoded["attention_mask"], "attention_mask"),
                    max_new_tokens=new_tokens([lengths[i] for i in indices]),
                    do_sample=False,
                ),
                "generate",
            )
        width = int(input_ids.shape[1])
        for position, index in enumerate(indices):
            reply = tokenizer.decode(produced[position, width:], skip_special_tokens=True)
            hypotheses[index] = hypothesis(require_str(reply, "decode"))
    return hypotheses


def score_translation(
    directions: Sequence[str],
    model: TextGenerator,
    tokenizer: ChatTokenizer,
    encode: Encode,
    *,
    model_id: str,
    shot_count: int,
    sentences: int,
    device: str,
    batch_size: int,
    out: Path,
) -> list[TranslationRow]:
    """Every requested direction, written after each one rather than at the end."""
    from agbalu.bench.mt import as_direction, score

    rows: list[TranslationRow] = []
    for direction in directions:
        typed = as_direction(direction)
        sources, references = sources_and_references(typed, corrected=True)
        if sentences:
            sources, references = sources[:sentences], references[:sentences]
        examples = demonstrations(typed, shot_count)
        hypotheses = translate(
            sources,
            typed,
            model,
            tokenizer,
            encode,
            examples=examples,
            device=device,
            batch_size=batch_size,
        )
        result = score(hypotheses, references, typed, SPLIT)
        row: TranslationRow = {
            "model": model_id,
            "family": "base-prompted",
            "direction": direction,
            "split": result.split,
            "sentences": result.sentences,
            "shots": len(examples),
            "reference": "corrected",
            "normaliser_version": result.normaliser_version,
            "raw": {m.name: m.score for m in result.raw.metrics},
            "normalised": {m.name: m.score for m in result.normalised.metrics},
            "chrf_gap": round(result.gap("chrf++"), 4),
            "bleu_gap": round(result.gap("bleu"), 4),
            "samples": [
                {"source": s, "hypothesis": h, "reference": r}
                for s, h, r in zip(
                    sources[:SAMPLES], hypotheses[:SAMPLES], references[:SAMPLES], strict=True
                )
            ],
        }
        rows.append(row)
        log.info(
            "translate %s | %d sentences | %d shots | chrf++ %.2f | bleu %.2f | first: %s",
            direction,
            result.sentences,
            len(examples),
            row["normalised"]["chrf++"],
            row["normalised"]["bleu"],
            hypotheses[0][:80],
        )
        (out / TRANSLATION_FILE).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        data_volume.commit()
    return rows


@app.function(image=llm_image, gpu=GPU, volumes=LLM_VOLUMES, timeout=LLM_TIMEOUT, retries=RETRIES)
def baseline(
    model_id: str = BASE_MODEL,
    directions: list[str] | None = None,
    shot_count: int = SHOTS,
    sentences: int = 0,
    batch_size: int = BATCH_SIZE,
) -> BaselineResult:
    """Perplexity on the held-out sets, then translation as a prompt, on one model load.

    `sentences` caps each direction for a cheap end-to-end check; 0 scores the whole split.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from agbalu.llm.baseline import leading_and_pad

    wanted = list(directions) if directions else list(BASELINE_DIRECTIONS)
    check_inputs(wanted)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # `device_map` rather than `.to(device)`: the weights land on the GPU as they are read
    # instead of being materialised in host memory and copied, which is one traversal
    # rather than two. `from_pretrained` documents that it also
    # leaves the model in evaluation mode, so nothing here sets it.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", device_map=device)
    bos_id, pad_id = leading_and_pad(
        tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    )
    log.info("loaded %s | %s | bos=%d pad=%d", model_id, device, bos_id, pad_id)

    def encode(texts: list[str]) -> list[list[int]]:
        return [list(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]

    out = Path(DATA_PATH) / RESULTS
    out.mkdir(parents=True, exist_ok=True)

    perplexity = score_perplexity(
        model,
        encode,
        model_id=model_id,
        bos_id=int(bos_id),
        pad_id=int(pad_id),
        device=device,
        batch_size=batch_size,
        out=out,
    )
    translation = score_translation(
        wanted,
        model,
        tokenizer,
        encode,
        model_id=model_id,
        shot_count=shot_count,
        sentences=sentences,
        device=device,
        batch_size=batch_size,
        out=out,
    )

    models_volume.commit()
    log.info("wrote %d perplexity rows and %d directions", len(perplexity), len(translation))
    return {"model": model_id, "perplexity": perplexity, "translation": translation}


@app.local_entrypoint()
def upload_llm() -> None:
    """Push the evaluation sets and the CPT corpus to the data volume.

    The corpus is ~630 MB and only the training run reads it, so it is pushed
    here rather than by `modal-upload`'s general target — but it is pushed by the *same*
    command, because a target that uploads half of what a run needs is how a job discovers
    the other half after the GPU has been paid for.
    """
    wanted = [LOCAL_HELDOUT / f"heldout-{short}.jsonl" for short, _ in HELDOUT_LANGUAGES]
    wanted.append(LOCAL_HELDOUT / CORPUS_FILE)
    missing = [str(path) for path in wanted if not path.is_file()]
    if missing:
        message = f"{missing} absent; run `make llm TASK=holdout` and `make llm TASK=mixture` first"
        raise SystemExit(message)
    with data_volume.batch_upload(force=True) as batch:
        for path in wanted:
            batch.put_file(path, f"/{REMOTE_HELDOUT}/{path.name}")
    total = sum(path.stat().st_size for path in wanted)
    print(f"uploaded {len(wanted)} files, {total / 1e6:.1f} MB")


@app.local_entrypoint()
def run_baseline(
    model: str = BASE_MODEL, directions: str = "", shot_count: int = SHOTS, sentences: int = 0
) -> None:
    """Score the base model and write both artifacts into the repository.

    Written here as well as onto the volume: a results file only a container can see is one
    the next session cannot cite.
    """
    wanted = [d.strip() for d in directions.split(",") if d.strip()] or None
    result = baseline.remote(
        model_id=model, directions=wanted, shot_count=shot_count, sentences=sentences
    )

    LOCAL_HELDOUT.mkdir(parents=True, exist_ok=True)
    artifacts = (
        (PERPLEXITY_FILE, result["perplexity"]),
        (TRANSLATION_FILE, result["translation"]),
    )
    for name, rows in artifacts:
        (LOCAL_HELDOUT / name).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(f"\n{'set':<24}{'lang':<10}{'docs':>8}{'ppl':>10}{'bits/char':>12}")
    for row in result["perplexity"]:
        print(
            f"{row['name']:<24}{row['language']:<10}{row['documents']:>8,}"
            f"{row['perplexity']:>10.3f}{row['bits_per_character']:>12.4f}"
        )
    print(f"\n{'direction':<12}{'chrf++':>9}{'bleu':>8}{'spbleu':>8}")
    for translated in result["translation"]:
        print(
            f"{translated['direction']:<12}{translated['normalised']['chrf++']:>9.2f}"
            f"{translated['normalised']['bleu']:>8.2f}{translated['normalised']['spbleu']:>8.2f}"
        )
    print(f"\n-> {LOCAL_HELDOUT / PERPLEXITY_FILE}  {LOCAL_HELDOUT / TRANSLATION_FILE}")
