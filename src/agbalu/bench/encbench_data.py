"""Deterministic zero-annotation dataset compiler for KabEncBench (`data/tasks/encbench/`).

Builds all six core supervised tasks (`syntax`, `morph`, `ner`, `sent`, `topic`, `punct`)
and three diagnostic suites (`diag_clitic`, `diag_para`, `diag_spell`) reproducibly from
local artifacts on disk, emitting `data/tasks/encbench/encbench.stats.json`.
"""

from __future__ import annotations

import bz2
import csv
import hashlib
import io
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from defusedxml import ElementTree

from agbalu.treebank import Sentence as TreebankSentence
from agbalu.treebank import Token as TreebankToken
from agbalu.treebank import read_split as read_treebank_split

DEFAULT_OUT_DIR: Final = Path("data/tasks/encbench")
DEFAULT_RAW_DIR: Final = Path("data/raw")
DEFAULT_PROCESSED_DIR: Final = Path("data/processed")
DEFAULT_TASKS_DIR: Final = Path("data/tasks")

CORE_TASKS: Final[tuple[str, ...]] = ("syntax", "morph", "ner", "sent", "topic", "punct")
DIAG_TASKS: Final[tuple[str, ...]] = ("diag_clitic", "diag_para", "diag_spell")

COMMON_FUNCTION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "deg",
        "di",
        "seg",
        "si",
        "ɣer",
        "ar",
        "ɣef",
        "af",
        "s",
        "d",
        "n",
        "i",
        "u",
        "neɣ",
        "maca",
        "walakin",
        "imi",
        "akken",
        "almi",
        "alammi",
        "asmi",
        "wasmi",
        "mi",
        "ara",
        "ad",
        "ur",
        "ulac",
        "yella",
        "tella",
        "llan",
        "llant",
        "acku",
        "axater",
        "syin",
        "din",
        "dagi",
        "da",
        "anki",
        "anda",
        "amek",
        "anwa",
        "anta",
        "acu",
        "ayen",
        "win",
        "tin",
        "wid",
        "tid",
        "yagi",
        "agi",
        "nni",
        "ihin",
        "tura",
        "ass",
        "useggas",
        "aseggas",
        "tallit",
        "tamurt",
        "tmurt",
        "timura",
        "tamdint",
        "temdint",
        "timdinin",
        "taddart",
        "tuddar",
        "taɣiwant",
        "tiɣiwanin",
        "adrar",
        "idurar",
        "asif",
        "isaffen",
        "akal",
        "awal",
        "awalen",
        "adlis",
        "idlisen",
    }
)

ORG_HEADS: Final[tuple[str, ...]] = (
    "tasdawit",
    "tesdawit",
    "tseddawit",
    "aɣlif",
    "uɣlif",
    "aseqqamu",
    "useqqamu",
    "usqamu",
    "akabar",
    "ukabar",
    "agraw",
    "ugraw",
    "tirni",
    "terni",
    "tuddsa",
    "tiddukla",
    "tdukli",
    "anegmam",
    "unegmam",
    "asira",
    "usira",
    "anabaḍ",
    "unabaḍ",
    "ilemẓiyen inaddalen",
    "yilemẓiyen inaddalen",
    "aɣerbaz",
    "uɣerbaz",
    "anzaḍ",
)

MISC_HEADS: Final[tuple[str, ...]] = (
    "tutlayt",
    "tutlayin",
    "amussu",
    "umussu",
    "tafsut",
    "tefsut",
    "tfaska",
    "tafaska",
    "amtawa",
    "umtawa",
)

CANONICAL_ORG_ACRONYMS: Final[tuple[str, ...]] = (
    "HCA",
    "FFS",
    "RCD",
    "MAK",
    "FLN",
    "RND",
    "MSP",
    "PT",
    "UGTA",
    "JSK",
    "JSMB",
    "MOB",
    "USMA",
    "MCA",
    "UNESCO",
    "ONU",
    "OMS",
    "FIFA",
    "CAF",
    "NATO",
    "UE",
    "APN",
    "ENTV",
    "BRTV",
    "MCB",
)

UD_PER_LEMMAS: Final[frozenset[str]] = frozenset(
    {
        "ṭiṭem",
        "azwaw",
        "uzwaw",
        "muḥemmed",
        "dihya",
        "xira",
        "yugurten",
        "malḥa",
        "kateb",
        "yasin",
        "aksil",
        "adem",
        "hewwa",
        "ɛebdelqader",
        "uḥemmu",
        "kamel",
        "tayep",
    }
)

UD_LOC_LEMMAS: Final[frozenset[str]] = frozenset(
    {
        "wejda",
        "lzzayer",
        "lezzayer",
        "zzayer",
        "dzayer",
        "fransa",
        "lpari",
        "rrus",
        "lɛiraq",
        "lhend",
        "maser",
        "kuba",
        "lmerruk",
        "merruk",
        "spenyul",
        "ssaɛudeyya",
        "merrakec",
        "werzazat",
        "legniz",
        "rrif",
        "werdin",
        "swiss",
        "bgayet",
        "vgayet",
        "tubiret",
        "bumerdas",
        "wehran",
        "qsemṭina",
        "tunes",
        "turukt",
        "turuft",
        "tafriqt",
        "temriqt",
        "lalman",
        "ṭṭelyan",
        "lizyipt",
    }
)

SYNTHETIC_TOPONYM_TEMPLATES: Final = re.compile(
    r"^(Aql-iyi (di|deg) .+ tura\.|"
    r"Zzedɣeɣ (di|deg) .+ acḥal aya\.|"
    r"Axxam-iw yella (di|deg) .+ acḥal-aya\.|"
    r"Aṭas aya seg wasmi i zedɣeɣ (di|deg) .+\.|"
    r"Yewweḍ-d (si|seg) .+ zik taṣebḥit-a\.|"
    r"Yusa-d (si|seg) .+ taṣebḥit-a\.|"
    r"Yeṭṭef-d tabraṭ (si|seg) .+\.|"
    r"Ilul (di|deg) .+ deg useggas-nni\.)$"
)

FRENCH_NOISE_WORDS: Final[frozenset[str]] = frozenset(
    {
        "le",
        "la",
        "les",
        "des",
        "du",
        "une",
        "dans",
        "avec",
        "pour",
        "sur",
        "par",
        "plus",
        "janvier",
        "février",
        "mars",
        "avril",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
        "manifestation",
        "texte",
        "pétition",
        "syndicat",
    }
)

POS_EN_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "love",
        "loves",
        "loved",
        "like",
        "liked",
        "happy",
        "glad",
        "great",
        "best",
        "better",
        "wonderful",
        "beautiful",
        "good",
        "well",
        "thank",
        "thanks",
        "enjoy",
        "enjoyed",
        "nice",
        "excellent",
        "proud",
        "pleased",
        "delicious",
        "helpful",
        "kind",
        "fantastic",
        "amazing",
        "lucky",
        "smile",
        "smiled",
        "success",
        "successful",
        "perfect",
        "smart",
        "clever",
        "brave",
        "peace",
        "friend",
        "friends",
        "welcome",
        "win",
        "won",
        "easy",
        "safe",
        "healthy",
        "sweet",
        "fun",
        "funny",
        "clean",
        "bright",
    }
)

POS_KAB_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "ḥemmleɣ",
        "iḥemmel",
        "tḥemmel",
        "ḥemmlen",
        "yelha",
        "telha",
        "lhan",
        "igerrez",
        "tgerrez",
        "gerrzen",
        "ifaz",
        "tfaz",
        "ifazen",
        "yefreḥ",
        "tefreḥ",
        "ferḥeɣ",
        "lferḥ",
        "tanemmirt",
        "tayri",
        "tumert",
        "ucebbiḥ",
        "tacebbiḥt",
        "acemmaḥ",
        "yeɛǧeb",
        "teɛǧeb",
        "yesselha",
        "tafukt",
        "talwit",
        "imeddukal",
        "ameddakel",
        "rrbeḥ",
        "yerbeḥ",
        "terbeḥ",
    }
)

NEG_EN_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "hate",
        "hates",
        "hated",
        "bad",
        "worse",
        "worst",
        "wrong",
        "terrible",
        "awful",
        "horrible",
        "sad",
        "angry",
        "afraid",
        "scared",
        "pain",
        "sick",
        "ill",
        "hurt",
        "lie",
        "lied",
        "lying",
        "stole",
        "steal",
        "dead",
        "died",
        "die",
        "kill",
        "killed",
        "problem",
        "problems",
        "mistake",
        "mistakes",
        "fail",
        "failed",
        "failure",
        "useless",
        "unfounded",
        "stupid",
        "fool",
        "boring",
        "bored",
        "tired",
        "sorry",
        "regret",
        "unfortunately",
        "disappointed",
        "refuse",
        "refused",
        "annoying",
        "dangerous",
        "poor",
        "cry",
        "cried",
        "crying",
        "damage",
        "destroyed",
        "blame",
        "guilty",
        "ugly",
        "dirty",
        "hard",
        "difficult",
        "impossible",
        "fear",
        "worry",
        "worried",
        "lost",
        "lose",
        "alone",
        "lonely",
        "broken",
        "expensive",
        "late",
        "noise",
        "threw",
        "stone",
        "stones",
        "prison",
        "war",
        "attack",
    }
)

NEG_KAB_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "kerheɣ",
        "ikreh",
        "tekreh",
        "yir",
        "diri",
        "yuɛer",
        "tuɛer",
        "yexṣer",
        "texṣer",
        "ugur",
        "wugur",
        "iɣeblan",
        "aɣbel",
        "ukren",
        "yuker",
        "tukerḍa",
        "yemmut",
        "temmut",
        "lmut",
        "tugdi",
        "ugadeɣ",
        "yugad",
        "tugad",
        "aɣilif",
        "leḥzen",
        "yeḥzen",
        "teḥzen",
        "yugi",
        "tugi",
        "ugiɣ",
        "leḥbas",
        "lḥebs",
        "imenɣi",
        "ṭṭrad",
        "agufsu",
        "lekdeb",
        "yeskiddib",
        "yessers",
        "yiẓra",
        "aḍon",
        "yuḍen",
        "tuḍen",
        "llaẓ",
        "ulac",
    }
)

SLANG_FALSE_POSITIVES: Final[frozenset[str]] = frozenset(
    {"wizard", "witch", "ghost", "magic", "spell", "harry", "genius", "demon", "demons"}
)

SIB200_LABELS: Final[tuple[str, ...]] = (
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
)


@dataclass(frozen=True, slots=True)
class BuildLocations:
    """Input and output directories for compiling KabEncBench."""

    out: Path = DEFAULT_OUT_DIR
    raw: Path = DEFAULT_RAW_DIR
    processed: Path = DEFAULT_PROCESSED_DIR
    tasks: Path = DEFAULT_TASKS_DIR


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _parse_feats(feats_str: str | None) -> dict[str, str]:
    if not feats_str or feats_str == "_":
        return {}
    result: dict[str, str] = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def _surface_syntax_item(sentence: TreebankSentence, idx: int, split: str) -> dict[str, object]:
    """Map a UD sentence (with possible MWTs) onto surface tokens and their governors."""
    word_by_id = {w.id: w for w in sentence.words}
    surface_forms: list[str] = []
    rep_word_ids: list[int] = []
    word_id_to_surf_idx: dict[int, int] = {}

    for s_pos, tok in enumerate(sentence.tokens):
        span_words = tok.words
        span_ids = {sw.id for sw in span_words}
        governor = span_words[0]
        for sw in span_words:
            if sw.head not in span_ids or sw.upos in {
                "VERB",
                "NOUN",
                "PROPN",
                "ADJ",
                "ADV",
                "ADP",
            }:
                governor = sw
                break
        surface_forms.append(tok.form)
        rep_word_ids.append(governor.id)
        for sw in span_words:
            word_id_to_surf_idx[sw.id] = s_pos

    upos_list: list[str] = []
    case_state: list[str] = []
    feats_list: list[str] = []
    heads: list[int] = []
    deprels: list[str] = []

    for s_pos, rep_id in enumerate(rep_word_ids):
        gw = word_by_id[rep_id]
        upos_list.append(gw.upos)
        f_map = _parse_feats(gw.feats)
        case_val = f_map.get("Case", "")
        if case_val == "Nom":
            c_state = "FREE"
        elif case_val in {"Acc", "Gen", "Loc", "Dat"}:
            c_state = "ANNEXED"
        else:
            c_state = "NONE"
        case_state.append(c_state)
        core_f = "|".join(
            f"{k}={f_map[k]}" for k in ("Case", "Gender", "Number", "Person") if k in f_map
        )
        feats_list.append(core_f or "_")
        if gw.head is None or gw.head == 0 or gw.head not in word_id_to_surf_idx:
            heads.append(0)
        else:
            target_surf = word_id_to_surf_idx[gw.head]
            heads.append(0 if target_surf == s_pos else target_surf + 1)
        deprels.append(gw.deprel.split(":")[0])

    return {
        "id": f"kabsyntax_{split}_{idx:04d}",
        "sent_id": sentence.sent_id,
        "text": sentence.text,
        "tokens": surface_forms,
        "upos": upos_list,
        "case_state": case_state,
        "core_feats": feats_list,
        "head": heads,
        "deprel": deprels,
    }


SPELL_CATEGORY_CAP: Final[int] = 250
MIN_FORM_LEN: Final[int] = 4
MIN_NAME_LEN: Final[int] = 3
MIN_CONLLU_COLS: Final[int] = 4
MIN_SPAN_LEN: Final[int] = 2
MIN_NER_TOKENS: Final[int] = 6
MAX_NER_TOKENS: Final[int] = 38
MIN_SENT_WORDS: Final[int] = 3
MAX_SENT_WORDS: Final[int] = 25
MIN_PUNCT_WORDS: Final[int] = 3
MAX_PUNCT_WORDS: Final[int] = 45
MIN_CLUSTER_SENTS: Final[int] = 2

NER_CATEGORY_MAP: Final[dict[str, str]] = {
    "Location": "LOC",
    "Person": "PER",
    "Institution": "ORG",
    "Organization": "ORG",
    "Political Party": "ORG",
    "Movement": "MISC",
    "Location / Person": "LOC",
}

WIKI_PER_CATS: Final[tuple[str, ...]] = (
    "tameddurt",
    "amaru",
    "imaru",
    "acennay",
    "icennayen",
    "amedyaz",
    "imedyazen",
    "asertay",
    "isertayen",
    "amusnaw",
    "imusnawen",
    "ilul deg",
    "yemmut deg",
)
WIKI_LOC_CATS: Final[tuple[str, ...]] = (
    "timdinin",
    "tuddar",
    "tiɣiwanin",
    "taɣiwant",
    "timura",
    "tamurt",
    "twilayt",
    "idurar",
    "isaffen",
    "awanak",
    "iwunak",
    "amenẓaw",
)
WIKI_ORG_CATS: Final[tuple[str, ...]] = (
    "ikabaren",
    "akabar",
    "tuddsiwin",
    "tuddsa",
    "tisdawiyin",
    "tasdawit",
    "aneɣlaf",
    "aɣlif",
    "agraw",
    "irebbaɛ",
    "irbaɛen",
    "tkebbaniyin",
)
WIKI_MISC_CATS: Final[tuple[str, ...]] = (
    "tutlayin",
    "tutlayt",
    "idlisen",
    "isura",
    "amezruy n",
    "tira",
    "igrawen",
    "timucuha",
    "tafsut",
)
WIKI_EXCLUDE_CATS: Final[tuple[str, ...]] = (
    "imɣan",
    "iɣersiwen",
    "igṭaṭ",
    "timsuṭad",
    "tiwsitin",
    "ussan n useggas",
    "agguren",
    "tusnakt",
    "tasengama",
    "takrura",
    "tujjya",
    "irmad",
)


def _clitic_valency_label(tok: TreebankToken) -> str:
    low_form = tok.form.lower()
    upos_seq = [p.upos for p in tok.words]
    deprel_seq = {p.deprel for p in tok.words}
    has_dir_d = low_form.endswith("-d") or "-d-" in low_form or "-id" in low_form
    has_dir_n = low_form.endswith("-n") or "-n-" in low_form or "-in" in low_form
    has_iobj = "iobj" in deprel_seq
    has_obj = "obj" in deprel_seq
    if upos_seq[0] == "ADP":
        return "PREP_PRON"
    if "nmod:poss" in deprel_seq or "det" in deprel_seq:
        return "NOM_POSS"
    if has_iobj and has_obj:
        return "DAT_ACC"
    if has_iobj:
        return "DAT_DIR" if (has_dir_d or has_dir_n) else "DAT"
    if has_obj:
        return "ACC_DIR" if (has_dir_d or has_dir_n) else "ACC"
    return "DIR_D" if has_dir_d else ("DIR_NN" if has_dir_n else "OTHER")


def build_syntax_and_clitic(locations: BuildLocations) -> dict[str, object]:
    """Compile Task 1 (`KabSyntax`) and Diagnostic 1 (`Diag-Clitic`) from UD_Kabyle-ADPT."""
    ud_root = locations.raw / "git.ud-kabyle-adpt"
    raw_train = read_treebank_split(ud_root, "train")
    raw_test = read_treebank_split(ud_root, "test")

    train_sents: list[TreebankSentence] = []
    dev_sents: list[TreebankSentence] = []
    for sent in raw_train:
        digest = hashlib.sha256(sent.text.encode("utf-8")).digest()
        if int.from_bytes(digest[:4], "big") % 5 == 0:
            dev_sents.append(sent)
        else:
            train_sents.append(sent)

    syntax_counts: dict[str, int] = {}
    clitic_counts: dict[str, int] = {}
    for split_name, sents in (("train", train_sents), ("dev", dev_sents), ("test", raw_test)):
        syn_rows = [_surface_syntax_item(s, i, split_name) for i, s in enumerate(sents)]
        syntax_counts[split_name] = _write_jsonl(
            locations.out / "syntax" / f"{split_name}.jsonl", syn_rows
        )

        clitic_rows: list[dict[str, object]] = []
        for s in sents:
            surf_tokens = [t.form for t in s.tokens]
            for t_idx, tok in enumerate(s.tokens):
                if tok.is_multiword:
                    clitic_rows.append(
                        {
                            "id": f"diag_clitic_{split_name}_{len(clitic_rows):04d}",
                            "sent_id": s.sent_id,
                            "tokens": surf_tokens,
                            "target_index": t_idx,
                            "mwt_form": tok.form,
                            "label": _clitic_valency_label(tok),
                        }
                    )
        clitic_counts[split_name] = _write_jsonl(
            locations.out / "diag_clitic" / f"{split_name}.jsonl", clitic_rows
        )

    return {"syntax": syntax_counts, "diag_clitic": clitic_counts}


def _build_spell_pairs(test_forms: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    spell_rows: list[dict[str, object]] = []
    spirant_map = str.maketrans({"t": "ṯ", "d": "ḏ", "b": "ḇ", "k": "ḵ", "g": "ḡ"})
    emph_map = str.maketrans({"ḍ": "d", "ṭ": "t", "ẓ": "z", "ṣ": "s", "ḥ": "h", "ɛ": "e", "ɣ": "g"})
    counts: Counter[str] = Counter()

    for form, lemma, _ in test_forms:
        if counts["SPIRANT_TH"] < SPELL_CATEGORY_CAP:
            corrupted = form.translate(spirant_map)
            if corrupted != form:
                counts["SPIRANT_TH"] += 1
                spell_rows.append(
                    {
                        "id": f"diag_spell_{len(spell_rows):04d}",
                        "category": "SPIRANT_TH",
                        "good": f"Yenna-yas belli {form} ass-a.",
                        "bad": f"Yenna-yas belli {corrupted} ass-a.",
                        "lemma": lemma,
                    }
                )
        if counts["EMPHATIC_DROP"] < SPELL_CATEGORY_CAP:
            corrupted_e = form.translate(emph_map)
            if corrupted_e != form:
                counts["EMPHATIC_DROP"] += 1
                spell_rows.append(
                    {
                        "id": f"diag_spell_{len(spell_rows):04d}",
                        "category": "EMPHATIC_DROP",
                        "good": f"Iban-d dakken {form} deg taddart.",
                        "bad": f"Iban-d dakken {corrupted_e} deg taddart.",
                        "lemma": lemma,
                    }
                )
        if (
            counts["CONSTRUCT_ANNEX"] < SPELL_CATEGORY_CAP
            and len(lemma) >= MIN_FORM_LEN
            and lemma[0].lower() not in "aeiouɛɣ"
        ):
            counts["CONSTRUCT_ANNEX"] += 1
            spell_rows.append(
                {
                    "id": f"diag_spell_{len(spell_rows):04d}",
                    "category": "CONSTRUCT_ANNEX",
                    "good": f"Yewweḍ-d seg tmurt n Leqbayel, dɣa {form}.",
                    "bad": f"Yewweḍ-d seg tamurt n Leqbayel, dɣa {form}.",
                    "lemma": lemma,
                }
            )
        if (
            counts["CLITIC_ORDER"] < SPELL_CATEGORY_CAP
            and "-" not in form
            and len(form) >= MIN_FORM_LEN
        ):
            counts["CLITIC_ORDER"] += 1
            spell_rows.append(
                {
                    "id": f"diag_spell_{len(spell_rows):04d}",
                    "category": "CLITIC_ORDER",
                    "good": f"Mi d-yusa ɣer taddart, {form}-yas-tt-id.",
                    "bad": f"Mi d-yusa ɣer taddart, {form}-tt-yas-id.",
                    "lemma": lemma,
                }
            )
    return spell_rows


def _read_parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, object]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    tbl = pq.read_table(path, columns=columns)  # type: ignore[no-untyped-call]
    rows: list[dict[str, object]] = tbl.to_pylist()
    return rows


def build_morph_and_spell(locations: BuildLocations) -> dict[str, object]:
    """Compile Task 2 (`KabMorph`) and Diagnostic 3 (`Diag-Spell`) from KabInflect and UD."""
    morph_dir = locations.tasks / "inflection" / "analysis"
    rng = random.Random(42)  # noqa: S311
    morph_counts: dict[str, int] = {}
    test_forms_for_spell: list[tuple[str, str, str]] = []

    for split_name, per_lemma in (("train", 2), ("dev", 1), ("test", 2)):
        by_lemma: dict[str, list[dict[str, object]]] = defaultdict(list)
        for r in _read_parquet_rows(morph_dir / f"{split_name}.parquet"):
            by_lemma[str(r["lemma"])].append(r)

        sampled: list[dict[str, object]] = []
        for lemma in sorted(by_lemma.keys()):
            pool = by_lemma[lemma]
            for item in rng.sample(pool, min(per_lemma, len(pool))):
                f_str = str(item["feats"])
                f_map = _parse_feats(f_str)
                form_str = str(item["form"])
                sampled.append(
                    {
                        "id": f"kabmorph_{split_name}_{len(sampled):05d}",
                        "form": form_str,
                        "lemma": lemma,
                        "tense_raw": str(item.get("tense_raw", "")),
                        "feats_bundle": f_str,
                        "attributes": {
                            "Aspect": f_map.get("Aspect", "None"),
                            "Mood": f_map.get("Mood", "None"),
                            "Person": f_map.get("Person", "None"),
                            "Number": f_map.get("Number", "None"),
                            "Gender": f_map.get("Gender", "None"),
                            "Polarity": f_map.get("Polarity", "None"),
                            "VerbForm": f_map.get("VerbForm", "Fin"),
                        },
                    }
                )
                if split_name == "test":
                    test_forms_for_spell.append((form_str, lemma, f_str))
        morph_counts[split_name] = _write_jsonl(
            locations.out / "morph" / f"{split_name}.jsonl", sampled
        )

    spell_rows = _build_spell_pairs(test_forms_for_spell)
    spell_count = _write_jsonl(locations.out / "diag_spell" / "test.jsonl", spell_rows)
    return {"morph": morph_counts, "diag_spell": {"test": spell_count}}


def _load_common_kabyle_words(locations: BuildLocations) -> set[str]:
    words = set(COMMON_FUNCTION_WORDS)
    lex_path = locations.processed / "lexicon" / "agbalu-lexicon-v1.jsonl"
    if lex_path.is_file():
        with lex_path.open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                for k in ("lemma", "form", "word"):
                    words.update(re.findall(r"[\w]+", str(r.get(k) or "").lower()))
    for sp in ("train", "dev", "test"):
        pq_path = locations.tasks / "inflection" / "analysis" / f"{sp}.parquet"
        if pq_path.is_file():
            for row in _read_parquet_rows(pq_path, columns=["form"]):
                words.update(re.findall(r"[\w]+", str(row["form"]).lower()))
    for sp in ("train", "test"):
        ud_path = locations.raw / "git.ud-kabyle-adpt" / f"kab_adpt-ud-{sp}.conllu"
        if ud_path.is_file():
            for line in ud_path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t") if line and not line.startswith("#") else []
                if len(parts) >= MIN_CONLLU_COLS and parts[3] != "PROPN":
                    words.add(parts[1].lower())
                    words.add(parts[2].lower())
    return words


def _register_ner_entry(
    kb: dict[tuple[str, ...], tuple[str, str, bool]],
    common_words: set[str],
    name: str,
    tag: str,
    *,
    force: bool = False,
) -> None:
    clean = re.sub(r"\s*\([^)]*\)$", "", name).strip()
    if len(clean) < MIN_NAME_LEN:
        return
    low = clean.lower()
    if any(low.startswith(h + " ") for h in ORG_HEADS):
        tag = "ORG"
    elif any(low.startswith(h + " ") for h in MISC_HEADS):
        tag = "MISC"
    if not force and " " not in clean and low in common_words:
        return
    parts = re.findall(r"[\w\-]+", low)
    if not parts:
        return
    canonical = " ".join(parts)
    if force or tuple(parts) not in kb:
        kb[tuple(parts)] = (tag, canonical, False)
    w0 = parts[0]
    if w0.startswith("a") and len(w0) > MIN_SPAN_LEN and not w0.startswith("at"):
        kb[("u" + w0[1:], *parts[1:])] = (tag, canonical, True)
    elif w0.startswith("ta") and len(w0) > MIN_NAME_LEN:
        kb[("te" + w0[2:], *parts[1:])] = (tag, canonical, True)
    elif w0.startswith("i") and len(w0) > MIN_SPAN_LEN and w0[1] not in "iy":
        kb[("yi" + w0[1:], *parts[1:])] = (tag, canonical, True)


def _classify_wiki_page(title: str, cats: list[str], infos: list[str]) -> str | None:
    cat_str, info_str = " | ".join(cats), " | ".join(infos)
    if any(ex in cat_str for ex in WIKI_EXCLUDE_CATS) or re.match(r"^\d+$", title):
        return None
    per_infos = (
        "biography",
        "person",
        "musical artist",
        "writer",
        "officeholder",
        "scientist",
        "philosopher",
    )
    if any(k in info_str for k in per_infos) or any(k in cat_str for k in WIKI_PER_CATS):
        return "PER"
    org_infos = ("university", "football club", "organization", "company", "party")
    if any(k in info_str for k in org_infos) or any(c.startswith(WIKI_ORG_CATS) for c in cats):
        return "ORG"
    loc_infos = ("city", "country", "tamnaḍt", "settlement", "river", "mountain")
    if any(k in info_str for k in loc_infos) or any(k in cat_str for k in WIKI_LOC_CATS):
        return "LOC"
    misc_infos = ("language", "ethnic group", "intellectual work", "book", "film", "music")
    if any(k in info_str for k in misc_infos) or any(k in cat_str for k in WIKI_MISC_CATS):
        return "MISC"
    return None


def _build_ner_kb_and_pages(
    locations: BuildLocations, common_words: set[str]
) -> tuple[dict[tuple[str, ...], tuple[str, str, bool]], list[tuple[str, str]]]:
    kb: dict[tuple[str, ...], tuple[str, str, bool]] = {}
    ne_tsv = (
        locations.raw / "hf.boffire.kabyle-named-entities" / "data" / "kabyle_named_entities.tsv"
    )
    with ne_tsv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            tag = NER_CATEGORY_MAP[row["entity_category"].strip()]
            surf = row["kabyle_standardized"].strip()
            _register_ner_entry(kb, common_words, surf, tag, force=(" " in surf))

    for acr in CANONICAL_ORG_ACRONYMS:
        kb[(acr.lower(),)] = ("ORG", acr.lower(), False)
    for p in UD_PER_LEMMAS:
        kb[(p,)] = ("PER", p, p == "uzwaw")
    for loc in UD_LOC_LEMMAS:
        kb[(loc,)] = ("LOC", loc, False)
    kb[("kateb", "yasin")] = ("PER", "kateb yasin", False)

    pages: list[tuple[str, str]] = []
    wiki_bz2 = locations.raw / "wikimedia.kabwiki" / "kabwiki-20260801-pages-articles.xml.bz2"
    with bz2.open(wiki_bz2, "rt", encoding="utf-8") as handle:
        for _, elem in ElementTree.iterparse(handle, events=("end",)):
            if elem.tag.endswith("page"):
                ns = elem.find("{*}ns")
                title_el = elem.find("{*}title")
                text_el = elem.find(".//{*}text")
                title = (title_el.text or "").strip() if title_el is not None else ""
                text = text_el.text if text_el is not None and text_el.text else ""
                if (
                    ns is not None
                    and ns.text == "0"
                    and title
                    and not text.lower().startswith("#redirect")
                ):
                    cats = [
                        m.group(1).strip().lower()
                        for m in re.finditer(
                            r"\[\[(?:Taggayt|Category|Catégorie):([^\]|]+)", text, re.I
                        )
                    ]
                    infos = [
                        m.group(1).strip().lower()
                        for m in re.finditer(
                            r"\{\{(?:Infobox|Taxobox|Tafelwit)\s*([^|\n\}]+)", text, re.I
                        )
                    ]
                    w_tag = _classify_wiki_page(title, cats, infos)
                    if w_tag is not None:
                        _register_ner_entry(kb, common_words, title, w_tag)
                    pages.append((title, text))
                elem.clear()

    cities = (
        "Tizi Wezzu",
        "Bgayet",
        "Lezzayer",
        "Tubiret",
        "Bumerdas",
        "Wehran",
        "Qsemṭina",
        "Lpari",
    )
    for city in cities:
        for head in ("Tasdawit n", "Tesdawit n", "Tasdawit", "Tesdawit"):
            _register_ner_entry(kb, common_words, f"{head} {city}", "ORG", force=True)
    return kb, pages


def _match_ner_spans(
    tokens: list[str],
    low: list[str],
    by_first: Mapping[str, list[tuple[tuple[str, ...], str, str, bool]]],
) -> tuple[list[str], list[dict[str, object]]]:
    n = len(tokens)
    labels = ["O"] * n
    spans: list[dict[str, object]] = []
    i = 0
    while i < n:
        matched = False
        for cand, tag, canon, is_annexed in by_first.get(low[i], ()):
            k = len(cand)
            surf_slice = tokens[i : i + k]
            if (
                i + k <= n
                and tuple(low[i : i + k]) == cand
                and (any(t[0].isupper() for t in surf_slice) or k >= MIN_SPAN_LEN)
            ):
                labels[i] = f"B-{tag}"
                for j in range(1, k):
                    labels[i + j] = f"I-{tag}"
                prev_tok = low[i - 1] if i > 0 else ""
                trig = is_annexed or prev_tok in {"n", "deg", "di", "seg", "si", "ɣer", "ɣef"}
                spans.append(
                    {
                        "start": i,
                        "end": i + k,
                        "surface": " ".join(surf_slice),
                        "canonical": canon,
                        "label": tag,
                        "annexed": trig,
                    }
                )
                i += k
                matched = True
                break
        if not matched:
            i += 1
    return labels, spans


def _tag_ner_sentence(
    text: str,
    source: str,
    by_first: Mapping[str, list[tuple[tuple[str, ...], str, str, bool]]],
    common_words: set[str],
) -> dict[str, object] | None:
    if any(ch in text for ch in ("(", ")", '"', "«", "»", "[", "]", "<", ">")):
        return None
    if SYNTHETIC_TOPONYM_TEMPLATES.match(text):
        return None
    tokens = re.findall(r"[\w\-]+|[.,!?;:]", text)
    low = [t.lower() for t in tokens]
    if not (MIN_NER_TOKENS <= len(tokens) <= MAX_NER_TOKENS) or any(
        w in FRENCH_NOISE_WORDS for w in low
    ):
        return None
    labels, spans = _match_ner_spans(tokens, low, by_first)
    if not spans:
        return None
    first_unknown = (
        labels[0] == "O" and low[0].split("-")[0] not in common_words and low[0] not in common_words
    )
    mid_cap_o = any(
        labels[idx] == "O" and tokens[idx][0].isupper() for idx in range(1, len(tokens))
    )
    if first_unknown or mid_cap_o:
        return None
    return {"source": source, "text": text, "tokens": tokens, "labels": labels, "spans": spans}


def _clean_wiki_paragraph_lines(pages: list[tuple[str, str]]) -> list[str]:
    skip_prefixes = (
        "{",
        "|",
        "!",
        "*",
        "#",
        "=",
        "[[Taggayt",
        "[[Category",
        "[[File",
        "[[Image",
    )
    lines: list[str] = []
    for _, text in pages:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(skip_prefixes):
                continue
            if any(sub in line for sub in ("{{", "}}", "<ref", "http")):
                continue
            plain = re.sub(r"\[\[([^\]|:#]+)\|([^\]]+)\]\]", r"\2", line)
            plain = re.sub(
                r"\[\[([^\]|:#]+)\]\]",
                lambda m: re.sub(r"\s*\([^)]*\)$", "", m.group(1)),
                plain,
            )
            lines.append(re.sub(r"\x27{2,}", "", plain).strip())
    return lines


def _collect_ner_candidates(
    locations: BuildLocations,
    pages: list[tuple[str, str]],
    by_first: Mapping[str, list[tuple[tuple[str, ...], str, str, bool]]],
    common_words: set[str],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen_texts: set[str] = set()

    for plain in _clean_wiki_paragraph_lines(pages):
        for raw_sent in re.split(r"(?<=[.!?])\s+", plain):
            sent = raw_sent.strip()
            if sent and sent not in seen_texts:
                tagged = _tag_ner_sentence(sent, "kabwiki", by_first, common_words)
                if tagged is not None:
                    seen_texts.add(sent)
                    candidates.append(tagged)

    for sp_name in ("dev", "test", "ood", "train"):
        p_file = locations.processed / "punctuation" / f"{sp_name}.jsonl"
        if not p_file.is_file():
            continue
        with p_file.open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                for raw_sent in re.split(r"(?<=[.!?])\s+", str(r["text"]).strip()):
                    sent = raw_sent.strip()
                    if sent and sent not in seen_texts:
                        tagged = _tag_ner_sentence(sent, f"punct.{sp_name}", by_first, common_words)
                        if tagged is not None:
                            seen_texts.add(sent)
                            candidates.append(tagged)
    return candidates


def _spans_of(item: Mapping[str, object]) -> list[dict[str, object]]:
    sp = item.get("spans")
    return sp if isinstance(sp, list) else []


def build_ner(locations: BuildLocations) -> dict[str, object]:
    """Compile Task 3 (`KabNER`) via the 6-gate deterministic zero-annotation pipeline."""
    common_words = _load_common_kabyle_words(locations)
    kb, pages = _build_ner_kb_and_pages(locations, common_words)
    by_first: dict[str, list[tuple[tuple[str, ...], str, str, bool]]] = {}
    for toks, (tag, canon, annexed) in kb.items():
        if toks:
            by_first.setdefault(toks[0], []).append((toks, tag, canon, annexed))
    for entries in by_first.values():
        entries.sort(key=lambda x: len(x[0]), reverse=True)

    candidates = _collect_ner_candidates(locations, pages, by_first, common_words)
    priority = {"ORG": 0, "PER": 1, "MISC": 2, "LOC": 3}
    by_dom: dict[str, list[dict[str, object]]] = {"ORG": [], "PER": [], "MISC": [], "LOC": []}
    for cand in candidates:
        tags = [str(s["label"]) for s in _spans_of(cand)]
        dom = min(tags, key=lambda t: priority.get(t, 9))
        by_dom[dom].append(cand)

    rng = random.Random(42)  # noqa: S311
    selected_pool: list[dict[str, object]] = []
    entity_counts: Counter[str] = Counter()
    quotas = (("ORG", 350, 15), ("PER", 1000, 12), ("MISC", 850, 12), ("LOC", 1800, 10))
    for tag, target_count, max_per_entity in quotas:
        pool = list(by_dom[tag])
        rng.shuffle(pool)
        added = 0
        for item in pool:
            if added >= target_count:
                break
            canons = [str(s["canonical"]) for s in _spans_of(item)]
            if any(entity_counts[c] >= max_per_entity for c in canons):
                continue
            for c in canons:
                entity_counts[c] += 1
            selected_pool.append(item)
            added += 1

    rng.shuffle(selected_pool)
    splits = (
        ("train", selected_pool[900:2900]),
        ("dev", selected_pool[600:900]),
        ("test", selected_pool[:600]),
    )
    train_entities = {str(s["canonical"]) for item in splits[0][1] for s in _spans_of(item)}

    ner_counts: dict[str, int] = {}
    for split_name, split_list in splits:
        formatted = [
            {
                "id": f"kabner_{split_name}_{idx:04d}",
                "source": item["source"],
                "text": item["text"],
                "tokens": item["tokens"],
                "labels": item["labels"],
                "spans": [
                    {**s, "unseen_in_train": str(s["canonical"]) not in train_entities}
                    for s in _spans_of(item)
                ],
            }
            for idx, item in enumerate(split_list)
        ]
        ner_counts[split_name] = _write_jsonl(
            locations.out / "ner" / f"{split_name}.jsonl", formatted
        )
    return {"ner": ner_counts, "kb_entries": len(kb)}


def _score_sentiment_row(row: Mapping[str, object]) -> float:
    kab = str(row["text_kab"]).strip()
    en = str(row["text_en"]).strip()
    kab_w = set(re.findall(r"[\w]+", kab.lower()))
    en_w = set(re.findall(r"[\w]+", en.lower()))
    if not (MIN_SENT_WORDS <= len(kab.split()) <= MAX_SENT_WORDS) or (en_w & SLANG_FALSE_POSITIVES):
        return -1.0
    cls = str(row["label_name"])
    conf = float(str(row["confidence_score"]))
    pos_hit = bool((en_w & POS_EN_MARKERS) or (kab_w & POS_KAB_MARKERS))
    neg_hit = bool((en_w & NEG_EN_MARKERS) or (kab_w & NEG_KAB_MARKERS))
    has_neg = "ur " in kab.lower() or "not " in en.lower() or "n't" in en.lower()
    if cls == "positive":
        return -1.0 if (neg_hit or has_neg) else conf + (0.15 if pos_hit else 0.0)
    if cls == "negative":
        return -1.0 if (pos_hit and not has_neg) else conf + (0.15 if neg_hit else 0.0)
    return -1.0 if (pos_hit or neg_hit or "!" in kab or "!" in en) else conf


def build_sent(locations: BuildLocations) -> dict[str, object]:
    """Compile Task 4 (`KabSent`) by selecting top-confidence bilingual-consistent items."""
    sent_counts: dict[str, int] = {}
    mean_confs: dict[str, float] = {}
    for split_name, per_class_cap in (("train", 2000), ("dev", 150), ("test", 200)):
        in_path = locations.tasks / "sentiment" / f"{split_name}.jsonl"
        by_cls: dict[str, list[tuple[float, dict[str, object]]]] = {
            "negative": [],
            "neutral": [],
            "positive": [],
        }
        with in_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    sc = _score_sentiment_row(row)
                    if sc > 0.0:
                        by_cls[str(row["label_name"])].append((sc, row))

        selected: list[dict[str, object]] = []
        for cls_name in ("negative", "neutral", "positive"):
            ranked = sorted(by_cls[cls_name], key=lambda x: x[0], reverse=True)
            for _, item in ranked[:per_class_cap]:
                selected.append(
                    {
                        "id": f"kabsent_{split_name}_{len(selected):04d}",
                        "orig_id": item["id"],
                        "text": item["text_kab"],
                        "text_en": item["text_en"],
                        "label": int(str(item["label"])),
                        "label_name": item["label_name"],
                        "confidence_score": float(str(item["confidence_score"])),
                    }
                )
        rng = random.Random(42)  # noqa: S311
        rng.shuffle(selected)
        sent_counts[split_name] = _write_jsonl(
            locations.out / "sent" / f"{split_name}.jsonl", selected
        )
        mean_confs[split_name] = round(
            sum(float(str(x["confidence_score"])) for x in selected) / max(1, len(selected)), 4
        )

    return {"sent": sent_counts, "sent_mean_confidence": mean_confs}


def build_topic(locations: BuildLocations) -> dict[str, object]:
    """Compile Task 5 (`KabTopic`) fixing the 99-row SIB-200 test duplicate & linking FLORES+."""
    flores_by_idx: dict[int, dict[str, object]] = {}
    flores_path = locations.processed / "bench" / "flores-plus-kab_Latn-corrected.jsonl"
    if flores_path.is_file():
        with flores_path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if line.strip():
                    flores_by_idx[idx] = json.loads(line)

    label_to_id = {name: i for i, name in enumerate(SIB200_LABELS)}
    sib_raw_dir = locations.raw / "hf.sib200-kab" / "kab_Latn"
    topic_counts: dict[str, int] = {}
    corrected_counts: dict[str, int] = {}

    for split_name, pq_name in (("train", "train"), ("dev", "validation"), ("test", "test")):
        raw_rows: list[dict[str, object]] = []
        try:
            url = f"https://huggingface.co/datasets/Davlan/sib200/resolve/main/data/kab_Latn/{split_name}.tsv"
            tsv_bytes = urllib.request.urlopen(url, timeout=15).read().decode("utf-8")  # noqa: S310
            raw_rows = [
                {
                    "index_id": int(r["index_id"]),
                    "category": r["category"].strip(),
                    "text": r["text"].strip(),
                }
                for r in csv.DictReader(io.StringIO(tsv_bytes), delimiter="\t")
            ]
        except OSError:
            raw_rows = []

        if not raw_rows:
            pq_files = sorted(sib_raw_dir.glob(f"{pq_name}-*.parquet"))
            if pq_files:
                for row_i, r in enumerate(_read_parquet_rows(pq_files[0])):
                    cat_val = r.get("category", r.get("label", 0))
                    cat_str = (
                        SIB200_LABELS[int(cat_val)] if isinstance(cat_val, int) else str(cat_val)
                    )
                    raw_rows.append(
                        {
                            "index_id": int(str(r.get("index_id", -1 - row_i))),
                            "category": cat_str,
                            "text": str(r["text"]).strip(),
                        }
                    )

        out_rows: list[dict[str, object]] = []
        corr_n = 0
        for i, r in enumerate(raw_rows):
            idx_id = int(str(r["index_id"]))
            raw_txt = str(r["text"])
            f_entry = flores_by_idx.get(idx_id)
            corr_txt = str(f_entry["text"]) if f_entry else raw_txt
            is_corr = bool(f_entry.get("corrected", False)) if f_entry else False
            if is_corr:
                corr_n += 1
            cat_str = str(r["category"])
            out_rows.append(
                {
                    "id": f"kabtopic_{split_name}_{i:04d}",
                    "index_id": idx_id,
                    "text": corr_txt,
                    "text_raw_sib200": raw_txt,
                    "flores_corrected": is_corr,
                    "label": label_to_id[cat_str],
                    "label_name": cat_str,
                }
            )
        topic_counts[split_name] = _write_jsonl(
            locations.out / "topic" / f"{split_name}.jsonl", out_rows
        )
        corrected_counts[split_name] = corr_n

    return {"topic": topic_counts, "topic_flores_corrected": corrected_counts}


def _punct_tag_for(trail: str) -> str:
    if "?" in trail:
        return "QUESTION"
    if ":" in trail or ";" in trail:
        return "COLON"
    if "," in trail:
        return "COMMA"
    return "PERIOD" if ("." in trail or "!" in trail) else "NONE"


def _extract_punct_case_sequence(text: str) -> tuple[list[str], list[str], list[str]] | None:
    raw_tokens = text.strip().split()
    if not (MIN_PUNCT_WORDS <= len(raw_tokens) <= MAX_PUNCT_WORDS):
        return None
    words: list[str] = []
    punct_labels: list[str] = []
    case_labels: list[str] = []

    for tok in raw_tokens:
        m = re.match(r"^([^\w]*)([\w\-]+)([.,!?:;]*)$", tok)
        if not m:
            continue
        stem = m.group(2)
        trail = m.group(3)
        words.append(stem.lower())
        if stem.isupper() and len(stem) > 1:
            case_labels.append("ALL_CAP")
        elif stem[0].isupper():
            case_labels.append("CAP")
        else:
            case_labels.append("LOWER")
        punct_labels.append(_punct_tag_for(trail))

    if len(words) < MIN_PUNCT_WORDS:
        return None
    return words, punct_labels, case_labels


def build_punct(locations: BuildLocations) -> dict[str, object]:
    """Compile Task 6 (`KabPunct`) from Common Voice test/dev (`0%` overlap) and HCA OOD."""
    punct_dir = locations.processed / "punctuation"
    punct_counts: dict[str, int] = {}
    rng = random.Random(42)  # noqa: S311

    for split_name, cap in (("train", 30000), ("dev", None), ("test", None), ("ood", 2500)):
        in_path = punct_dir / f"{split_name}.jsonl"
        pool: list[dict[str, object]] = []
        with in_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                parsed = _extract_punct_case_sequence(str(r["text"]))
                if parsed is None:
                    continue
                words, p_tags, c_tags = parsed
                pool.append(
                    {
                        "text": r["text"],
                        "source": r.get("source", split_name),
                        "words": words,
                        "punct": p_tags,
                        "case": c_tags,
                    }
                )
        if cap is not None and len(pool) > cap:
            rare = [x for x in pool if "COLON" in x["punct"]]  # type: ignore[operator]
            rest = [x for x in pool if "COLON" not in x["punct"]]  # type: ignore[operator]
            rng.shuffle(rest)
            pool = (rare + rest)[:cap]
            rng.shuffle(pool)

        formatted = [
            {"id": f"kabpunct_{split_name}_{i:05d}", **item} for i, item in enumerate(pool)
        ]
        punct_counts[split_name] = _write_jsonl(
            locations.out / "punct" / f"{split_name}.jsonl", formatted
        )

    return {"punct": punct_counts}


def build_diag_para(locations: BuildLocations) -> dict[str, object]:
    """Compile Diagnostic 2 (`Diag-Para`) from TaPaCo-kab multi-sentence paraphrase clusters."""
    tapaco_pq = locations.raw / "hf.tapaco-kab" / "kab" / "train-00000-of-00001.parquet"
    by_cluster: dict[str, list[str]] = defaultdict(list)
    if tapaco_pq.is_file():
        for r in _read_parquet_rows(tapaco_pq, columns=["paraphrase_set_id", "paraphrase"]):
            cid = str(r["paraphrase_set_id"]).strip()
            txt = str(r["paraphrase"]).strip()
            if cid and cid != "0" and txt:
                by_cluster[cid].append(txt)

    rows: list[dict[str, object]] = []
    multi_clusters = 0
    for cid in sorted(by_cluster.keys(), key=int):
        sents = list(dict.fromkeys(by_cluster[cid]))
        if len(sents) >= MIN_CLUSTER_SENTS:
            multi_clusters += 1
            for s in sents:
                rows.append(
                    {
                        "id": f"diag_para_{len(rows):04d}",
                        "cluster_id": int(cid),
                        "text": s,
                    }
                )
    count = _write_jsonl(locations.out / "diag_para" / "test.jsonl", rows)
    return {"diag_para": {"test": count, "clusters": multi_clusters}}


def compile_all(locations: BuildLocations | None = None) -> dict[str, object]:
    """Build all 6 core tasks + 3 diagnostic tasks and write `encbench.stats.json`."""
    locs = locations or BuildLocations()
    locs.out.mkdir(parents=True, exist_ok=True)

    stats: dict[str, object] = {
        "benchmark": "KabEncBench",
        "version": "1.0.0",
        "core_tasks": list(CORE_TASKS),
        "diagnostic_tasks": list(DIAG_TASKS),
    }
    stats.update(build_syntax_and_clitic(locs))
    stats.update(build_morph_and_spell(locs))
    stats.update(build_ner(locs))
    stats.update(build_sent(locs))
    stats.update(build_topic(locs))
    stats.update(build_punct(locs))
    stats.update(build_diag_para(locs))

    stats_path = locs.out / "encbench.stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats
