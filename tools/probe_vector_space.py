"""Probe encoder vector space geometry: nearest neighbours, analogies, and isotropy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from torch.nn import functional as F  # noqa: N812

DEFAULT_TOKENIZER = Path(
    "/home/jovyan/agbalu/artifacts/encoder/measurements/mammeri-tok-base-16k.model"
)
DEFAULT_RUN_DIR = Path("/home/jovyan/agbalu/artifacts/runs/encoder-pretrain")


def get_token_id(word: str, sp: spm.SentencePieceProcessor) -> tuple[str | None, int | None]:
    p = sp.encode_as_pieces(word)
    ids = sp.encode_as_ids(word)
    if not ids:
        return None, None
    return p[0], ids[0]


def find_neighbours(
    target_vec: torch.Tensor,
    table: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    top_k: int = 10,
    exclude_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    dropped = set() if exclude_ids is None else set(exclude_ids)
    unit_table = F.normalize(table, dim=-1)
    unit_target = F.normalize(target_vec, dim=-1)
    scores = torch.mv(unit_table, unit_target)

    wanted = min(top_k + len(dropped), scores.shape[0])
    values, indices = torch.topk(scores, k=wanted)

    results: list[dict[str, Any]] = []
    for val, idx in zip(values.tolist(), indices.tolist(), strict=True):
        if idx in dropped:
            continue
        results.append(
            {
                "piece": sp.id_to_piece(idx),
                "id": idx,
                "similarity": round(float(val), 4),
            }
        )
        if len(results) >= top_k:
            break
    return results


def run_analogy(  # noqa: PLR0913
    a_w: str,
    b_w: str,
    c_w: str,
    table: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    top_k: int = 10,
) -> dict[str, Any]:
    p_a, id_a = get_token_id(a_w, sp)
    p_b, id_b = get_token_id(b_w, sp)
    p_c, id_c = get_token_id(c_w, sp)

    if id_a is None or id_b is None or id_c is None:
        return {"error": f"Could not tokenize one of: {a_w}, {b_w}, {c_w}"}

    vec_a = table[id_a]
    vec_b = table[id_b]
    vec_c = table[id_c]

    target_vec = vec_a - vec_b + vec_c
    nbrs = find_neighbours(target_vec, table, sp, top_k=top_k, exclude_ids={id_a, id_b, id_c})
    return {
        "expression": f"{a_w} ({p_a}) - {b_w} ({p_b}) + {c_w} ({p_c})",
        "top_matches": nbrs,
    }


def run_probes(snapshot_path: Path, tokenizer_path: Path) -> None:
    ckpt_path = snapshot_path / "training.pt"
    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path}")
        return

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))

    print("=" * 72)
    print(f"PROBING VECTOR SPACE: {snapshot_path.name}")
    print("=" * 72)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    table = ckpt["weights"]["embedding.weight"].float()

    norms = torch.norm(table, dim=-1)
    print(f"Embedding Table: shape {list(table.shape)}")
    print(
        f"Norms: mean = {norms.mean().item():.4f}, std = {norms.std().item():.4f}, "
        f"min = {norms.min().item():.4f}, max = {norms.max().item():.4f}"
    )

    unit_table = F.normalize(table, dim=-1)
    g = torch.Generator().manual_seed(42)
    idx1 = torch.randint(0, table.size(0), (5000,), generator=g)
    idx2 = torch.randint(0, table.size(0), (5000,), generator=g)
    mask = idx1 != idx2
    cos_sims = (unit_table[idx1[mask]] * unit_table[idx2[mask]]).sum(dim=-1)
    print(
        f"Random pairs (isotropic probe, N=5000): "
        f"mean cosine = {cos_sims.mean().item():.4f}, std = {cos_sims.std().item():.4f}"
    )
    print()

    analogies = [
        # Gender
        ("agellid", "argaz", "tameṭṭut", "tagellidt (king - man + woman = queen)"),
        ("amɣar", "argaz", "tameṭṭut", "tamɣart (old man - man + woman = old woman)"),
        ("aqcic", "argaz", "tameṭṭut", "taqcict (boy - man + woman = girl)"),
        ("ayaziḍ", "argaz", "tameṭṭut", "tayaziḍt (rooster - man + woman = hen)"),
        ("afellaḥ", "argaz", "tameṭṭut", "tafellaḥt (farmer - man + woman = female farmer)"),
        # Plural
        ("taddart", "argaz", "irgazen", "tuddar (village - man + men = villages)"),
        ("tamurt", "argaz", "irgazen", "timura (land - man + men = lands)"),
        ("afus", "argaz", "irgazen", "ifassen (hand - man + men = hands)"),
        ("awal", "argaz", "irgazen", "awalen (word - man + men = words)"),
        ("argaz", "tameṭṭut", "tilawin", "irgazen (man - woman + women = men)"),
        # Verb aspect / person
        ("yura", "argaz", "tameṭṭut", "tura (he wrote - man + woman = she wrote)"),
        ("yečča", "argaz", "tameṭṭut", "tečča (he ate - man + woman = she ate)"),
        ("yedda", "argaz", "tameṭṭut", "tedda (he went - man + woman = she went)"),
        # Profession / Agentive
        ("ccna", "tira", "amyaru", "acennay (singing - writing + writer = singer)"),
        # Antonyms / Polarity
        ("tafat", "lferḥ", "leḥzen", "ṭṭlam (light - joy + sorrow = darkness)"),
        ("amellal", "lferḥ", "leḥzen", "aberkan (white - joy + sorrow = black)"),
    ]

    print("--- SEMANTIC & MORPHOLOGICAL ANALOGIES (a - b + c) ---")
    for a, b, c, desc in analogies:
        res = run_analogy(a, b, c, table, sp, top_k=5)
        print(f"Target concept: {desc}")
        print(f"  Expr: {res['expression']}")
        for rank, match in enumerate(res["top_matches"][:5], 1):
            print(f"    [{rank}] {match['piece']:15s} (cos: {match['similarity']:.4f})")
        print()

    probe_words = [
        "tamaziɣt",
        "taqbaylit",
        "agellid",
        "tilelli",
        "tayri",
        "idurar",
        "asif",
        "taddart",
        "awal",
        "amennuɣ",
    ]
    print("--- NEAREST NEIGHBOURS (Top-5 Cosine) ---")
    for w in probe_words:
        p, idx = get_token_id(w, sp)
        if idx is not None:
            nbrs = find_neighbours(table[idx], table, sp, top_k=5, exclude_ids={idx})
            print(f"Word: {w} (piece: {p}, id: {idx})")
            for rank, match in enumerate(nbrs, 1):
                print(f"    [{rank}] {match['piece']:15s} (cos: {match['similarity']:.4f})")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Snapshot directory path")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    run_probes(args.snapshot, args.tokenizer)


if __name__ == "__main__":
    main()
