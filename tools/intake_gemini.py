"""Intake and bulletproof validation for Gemini-generated SFT batches.

Takes Gemini chat output (from stdin, a file, or string), validates turn structure,
card IDs, and thinking mode, passes Kabyle text through the versioned normaliser,
filters instruction leaks and base model names, runs Dihya LID verification,
and writes clean JSONL to `data/raw/gemini-sft/<batch>.txt`, then runs ingest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter as Tally
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from agbalu.lid.infer import Identifier
from agbalu.llm import sft_files, teacher
from agbalu.llm.chat import MARKERS
from agbalu.llm.seeds import CONVERSATIONS_PER_CARD, Card, Passage
from agbalu.normalise import Normaliser

DIHYA_DIR: Final = Path("artifacts/runs/dihya")
BATCH_DIR: Final = Path("data/interim/llm/sft/batches")
REPLY_DIR: Final = Path("data/raw/gemini-sft")

ARABIC_SCRIPT: Final = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
LEAK_TERMS: Final = re.compile(
    r"\b(?:ilmend\s+n\s+uḍris|deg\s+uḍris|deg\s+uɣbalu|deg\s+umagrad\s+n\s+wikipedya|according\s+to\s+the\s+text)\b",
    re.IGNORECASE,
)
BASE_MODELS: Final = re.compile(r"\b(?:qwen|alibaba|chatgpt|openai|meta\s+llama|claude)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BatchValidation:
    batch: str
    total_received: int
    valid_rows: list[dict[str, object]]
    fixes: list[str]
    warnings: list[str]
    errors: list[str]


def clean_text_field(
    text: str,
    normaliser: Normaliser,
    is_kabyle: bool = True,
) -> tuple[str, list[str]]:
    """Clean a text field: normalise orthography, remove accidental Arabic glosses, fix leaks."""
    notes: list[str] = []
    if not text.strip():
        return text, notes

    cleaned = text

    # Remove accidental Arabic glosses in Kabyle text (e.g. "يوض (yewweḍ)" -> "yewweḍ")
    if is_kabyle and ARABIC_SCRIPT.search(cleaned):
        cleaned = re.sub(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+\s*", "", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        notes.append("stripped_arabic_gloss")

    # Replace forbidden base model names with sovereign creator identity
    if BASE_MODELS.search(cleaned):
        cleaned = BASE_MODELS.sub("AƔBALU", cleaned)
        notes.append("fixed_base_model_leak")

    # Strip instruction leaks if present
    if LEAK_TERMS.search(cleaned):
        cleaned = LEAK_TERMS.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        notes.append("stripped_instruction_leak")

    # Pass Kabyle text through the versioned normaliser
    if is_kabyle:
        before_norm = cleaned
        cleaned = normaliser.normalise(cleaned)
        if cleaned != before_norm:
            notes.append("normalised_orthography")

    return cleaned.strip(), notes


def load_card_map() -> dict[str, Card]:
    """Map of all seed cards by ID."""
    cards: dict[str, Card] = {}
    for row in sft_files.read_jsonl(sft_files.SEEDS):
        passage_data = row.get("passage")
        passage = Passage(**passage_data) if isinstance(passage_data, dict) else None
        c = Card(
            id=str(row["id"]),
            task=str(row["task"]),
            domain=str(row["domain"]),
            think=bool(row["think"]),
            input_language=str(row["input_language"]),
            system=str(row["system"]),  # type: ignore[arg-type]
            user_turns=int(str(row["user_turns"])),
            split=str(row["split"]),  # type: ignore[arg-type]
            passage=passage,
        )
        cards[c.id] = c
    return cards


def validate_batch_text(
    batch_name: str,
    raw_content: str,
    cards_map: dict[str, Card] | None = None,
    dihya: Identifier | None = None,
) -> BatchValidation:
    """Validate and clean raw Gemini output for a specific batch."""
    if cards_map is None:
        cards_map = load_card_map()

    manifest_path = BATCH_DIR / f"{batch_name}.json"
    if not manifest_path.is_file():
        return BatchValidation(batch_name, 0, [], [], [], [f"Manifest not found: {manifest_path}"])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cards: set[str] = set(manifest["cards"])

    normaliser = Normaliser()
    valid_rows: list[dict[str, object]] = []
    fixes: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    lines = raw_content.splitlines()
    total_received = 0
    seen_keys: set[tuple[str, int]] = set()

    for line_num, line in enumerate(lines, start=1):
        cleaned_line = line.strip()
        if not cleaned_line or cleaned_line.startswith("```") or cleaned_line.startswith("#"):
            continue

        try:
            row = json.loads(cleaned_line)
        except json.JSONDecodeError as err:
            errors.append(f"Line {line_num}: JSON decode error: {err}")
            continue

        if not isinstance(row, dict):
            errors.append(f"Line {line_num}: Not a JSON object")
            continue

        card_id = row.get("card")
        n = row.get("n")
        turns = row.get("turns")

        if not isinstance(card_id, str) or card_id not in expected_cards:
            errors.append(f"Line {line_num}: Unknown or unexpected card {card_id!r} (expected {sorted(expected_cards)})")
            continue

        if not isinstance(n, int) or isinstance(n, bool) or not (1 <= n <= CONVERSATIONS_PER_CARD):
            errors.append(f"Line {line_num}: Invalid n {n!r} for card {card_id}")
            continue

        if (card_id, n) in seen_keys:
            errors.append(f"Line {line_num}: Duplicate conversation for ({card_id}, {n})")
            continue

        seen_keys.add((card_id, n))
        total_received += 1

        card = cards_map.get(card_id)
        if card is None:
            errors.append(f"Line {line_num}: Card {card_id} not found in seeds")
            continue

        if not isinstance(turns, list) or not turns:
            errors.append(f"Line {line_num}: Turns must be a non-empty list")
            continue

        cleaned_turns: list[dict[str, str]] = []
        turn_error = False

        for t_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                errors.append(f"Line {line_num}: Turn {t_idx} is not a dictionary")
                turn_error = True
                break

            role = turn.get("role")
            content = turn.get("content")
            reasoning = turn.get("reasoning")

            if role not in ("system", "user", "assistant"):
                errors.append(f"Line {line_num}: Invalid role {role!r} in turn {t_idx}")
                turn_error = True
                break

            if not isinstance(content, str) or not content.strip():
                errors.append(f"Line {line_num}: Turn {t_idx} ({role}) has empty content")
                turn_error = True
                break

            # Handle thinking mode constraints
            if role == "assistant":
                if card.think and (reasoning is None or not str(reasoning).strip()):
                    errors.append(f"Line {line_num}: Card {card_id} requires think: yes, but turn {t_idx} has no reasoning")
                    turn_error = True
                    break
                if not card.think and reasoning is not None:
                    # Strip unexpected reasoning on non-think card
                    turn = {k: v for k, v in turn.items() if k != "reasoning"}
                    reasoning = None
                    fixes.append(f"{card_id} n={n}: stripped unexpected reasoning for non-think card")

            # Normalise and clean Kabyle content
            is_kabyle_turn = role == "assistant" or (role == "user" and card.input_language == "kab")
            clean_content, c_notes = clean_text_field(content, normaliser, is_kabyle=is_kabyle_turn)
            if c_notes:
                fixes.append(f"{card_id} n={n} turn {t_idx} content: {', '.join(c_notes)}")

            cleaned_turn = {"role": role, "content": clean_content}

            if role == "assistant" and card.think and isinstance(reasoning, str):
                clean_reasoning, r_notes = clean_text_field(reasoning, normaliser, is_kabyle=True)
                if r_notes:
                    fixes.append(f"{card_id} n={n} turn {t_idx} reasoning: {', '.join(r_notes)}")
                cleaned_turn["reasoning"] = clean_reasoning

            cleaned_turns.append(cleaned_turn)

        if turn_error:
            continue

        # Optional Dihya LID check on the assistant output
        if dihya is not None:
            asst_text = " ".join(t["content"] for t in cleaned_turns if t["role"] == "assistant")
            if asst_text:
                pred = dihya.predict([asst_text])[0]
                if pred.language != "kab_Latn":
                    warnings.append(
                        f"{card_id} n={n}: Dihya identified assistant turn as {pred.language} "
                        f"({pred.confidence:.2f}), not kab_Latn"
                    )

        valid_rows.append({"card": card_id, "n": n, "turns": cleaned_turns})

    return BatchValidation(batch_name, total_received, valid_rows, fixes, warnings, errors)


def save_batch_reply(
    batch_name: str,
    valid_rows: Sequence[dict[str, object]],
    model_name: str = "Gemini 2.5 Pro",
    date_str: str | None = None,
) -> Path:
    """Save cleaned and validated batch reply to `data/raw/gemini-sft/<batch>.txt`."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    REPLY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPLY_DIR / f"{batch_name}.txt"

    lines = [f"# model: {model_name} | date: {date_str}"]
    for row in valid_rows:
        lines.append(json.dumps(row, ensure_ascii=False))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def run_ingest() -> dict[str, object]:
    """Run `agbalu.llm.cli ingest` programmatically and return statistics."""
    from agbalu.llm import cli, seeds

    args = argparse.Namespace()
    cards = {card.id: card for card in seeds.read(sft_files.SEEDS)}
    candidates: list[teacher.Candidate] = []
    rejects: list[dict[str, object]] = []
    batches: list[dict[str, object]] = []

    for name, ids, version in sft_files.manifests(teacher.BATCH_DIR, "batch"):
        path = teacher.REPLY_DIR / f"{name}.txt"
        if not path.is_file():
            batches.append({"batch": name, "waiting": True})
            continue
        asked = {card_id: cards[card_id] for card_id in ids}
        found, dropped, tally = teacher.parse_reply(teacher.read_reply(path), name, asked, version)
        candidates.extend(found)
        rejects.extend(dropped)
        batches.append(
            {
                "batch": name,
                "requested": tally.requested,
                "returned": tally.returned,
                "missing": tally.missing,
                "rejected": dict(tally.rejected),
                "ignored": tally.ignored,
            }
        )

    sft_files.write_jsonl(sft_files.CANDIDATES, sft_files.candidate_rows(candidates))
    sft_files.write_jsonl(sft_files.INTERIM / "rejects.jsonl", rejects)
    covered = teacher.coverage(candidates)

    stats: dict[str, object] = {
        "candidates": len(candidates),
        "rejected": len(rejects),
        "cards_complete": sum(n >= CONVERSATIONS_PER_CARD for n in covered.values()),
        "think_candidates": sum(cards[c.card].think for c in candidates if c.card in cards),
        "batches_completed": sum(1 for b in batches if not b.get("waiting")),
    }
    return stats


def print_scorecard(stats: dict[str, object]) -> None:
    """Print an inspiring, detailed scorecard towards the 3,500 target."""
    candidates = int(str(stats.get("candidates", 0)))
    think_candidates = int(str(stats.get("think_candidates", 0)))
    think_pct = (think_candidates / candidates * 100) if candidates > 0 else 0.0
    completed_cards = int(str(stats.get("cards_complete", 0)))

    print("\n" + "=" * 60)
    print("🏆  AƔBALU SFT SCORECARD (Jugurtha-9B Expansion)")
    print("=" * 60)
    print(f"• Total Candidate Conversations: {candidates:,} / 3,500 target ({(candidates / 3500) * 100:.1f}%)")
    print(f"• Thinking Mode Traces:          {think_candidates:,} ({think_pct:.1f}% vs 25.0% target)")
    print(f"• Fully Completed Cards:         {completed_cards:,}")
    print(f"• Batches Ingested:              {stats.get('batches_completed')}")
    print("=" * 60 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and ingest Gemini SFT batch output")
    parser.add_argument("--batch", required=True, help="Batch name (e.g. batch-0054)")
    parser.add_argument("--file", type=Path, help="Path to raw reply file")
    parser.add_argument("--model", default="Gemini 2.5 Pro", help="Model name as shown in interface")
    parser.add_argument("--skip-dihya", action="store_true", help="Skip Dihya LID verification")
    args = parser.parse_args()

    if args.file:
        raw_text = args.file.read_text(encoding="utf-8")
    else:
        print(f"Reading from stdin for {args.batch}... (Paste and press Ctrl-D):")
        raw_text = sys.stdin.read()

    dihya: Identifier | None = None
    if not args.skip_dihya and DIHYA_DIR.is_dir():
        dihya = Identifier.load(DIHYA_DIR, device="cpu")

    cards_map = load_card_map()
    validation = validate_batch_text(args.batch, raw_text, cards_map=cards_map, dihya=dihya)

    print(f"\n--- Validation Results for {args.batch} ---")
    print(f"Total rows parsed: {validation.total_received} (valid: {len(validation.valid_rows)})")

    if validation.fixes:
        print(f"Fixes applied ({len(validation.fixes)}):")
        for fix in validation.fixes[:10]:
            print(f"  ✓ {fix}")
        if len(validation.fixes) > 10:
            print(f"  ... and {len(validation.fixes) - 10} more fixes")

    if validation.warnings:
        print(f"Warnings ({len(validation.warnings)}):")
        for w in validation.warnings:
            print(f"  ⚠️  {w}")

    if validation.errors:
        print(f"Errors ({len(validation.errors)}):")
        for err in validation.errors:
            print(f"  ❌ {err}")
        return 1

    saved_path = save_batch_reply(args.batch, validation.valid_rows, model_name=args.model)
    print(f"Saved verified reply to: {saved_path}")

    stats = run_ingest()
    print_scorecard(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
