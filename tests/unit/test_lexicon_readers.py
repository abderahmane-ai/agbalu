from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agbalu.lexicon.models import UPOS_TAGS, LexiconError, features_of
from agbalu.lexicon.pos import DEFAULT_CONFIDENCE, HUNSPELL_UPOS, pos_confidence, upos_for
from agbalu.lexicon.readers import (
    MIN_TOPONYM_LENGTH,
    read_amawal,
    read_g2p,
    read_hunspell,
    read_tafsut,
    read_toponyms,
    read_verb_forms,
    read_verb_lemmas,
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestHunspell:
    def test_the_first_line_is_a_count_not_an_entry(self, tmp_path: Path) -> None:
        path = write(tmp_path / "kab.dic", "2\nimseɣti/y po:isem\nd \n")
        entries = list(read_hunspell(path))
        assert [e.form for e in entries] == ["imseɣti", "d"]

    def test_flags_are_stripped_from_the_form(self, tmp_path: Path) -> None:
        path = write(tmp_path / "kab.dic", "1\nimseɣtiyen/y st:imseɣti is:asget\n")
        entry = next(iter(read_hunspell(path)))
        assert entry.form == "imseɣtiyen"
        assert entry.lemma == "imseɣti"
        assert entry.features == (("Number", "Plur"),)

    def test_an_entry_with_no_morphology_is_kept_unlabelled(self, tmp_path: Path) -> None:
        """`d`, `ur`, `ara` carry no annotation and are still real words."""
        path = write(tmp_path / "kab.dic", "1\nara \n")
        entry = next(iter(read_hunspell(path)))
        assert entry.form == "ara"
        assert entry.lemma is None
        assert entry.upos is None

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(LexiconError, match="hunspell dictionary not found"):
            list(read_hunspell(tmp_path / "absent.dic"))

    def test_an_empty_file_is_an_error(self, tmp_path: Path) -> None:
        path = write(tmp_path / "kab.dic", "")
        with pytest.raises(LexiconError, match="hunspell dictionary is empty"):
            list(read_hunspell(path))

    def test_a_file_with_only_a_count_yields_nothing(self, tmp_path: Path) -> None:
        assert list(read_hunspell(write(tmp_path / "kab.dic", "0\n"))) == []


class TestPos:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [("isem", "NOUN"), ("amyag", "VERB"), ("tanzeɣt", "ADP"), ("isem_amḍan", "NUM")],
    )
    def test_labels_map_to_the_tag_their_exemplars_support(self, label: str, expected: str) -> None:
        assert upos_for(label) == expected

    def test_an_unseen_label_becomes_x_not_a_default_tag(self) -> None:
        """Hunspell is a living file; a new label must not inherit a meaning."""
        assert upos_for("a_label_that_does_not_exist") == "X"

    def test_every_mapping_is_a_real_upos_tag(self) -> None:
        assert set(HUNSPELL_UPOS.values()) <= UPOS_TAGS

    def test_the_curated_dictionary_outranks_the_gazetteer(self) -> None:
        assert pos_confidence("hf.boffire.hunspell-kab") < pos_confidence(
            "hf.boffire.kabyle-toponyms"
        )

    def test_an_unknown_source_is_least_trusted(self) -> None:
        assert pos_confidence("some.new.source") == DEFAULT_CONFIDENCE


class TestVerbs:
    def forms(self, tmp_path: Path) -> Path:
        directory = tmp_path / "lemmatizer"
        directory.mkdir(parents=True)
        table = pa.table(
            {
                "form": ["teḥḍimemt", "yiwsiɛen"],
                "target": ["x", "y"],
                "infinitif": ["ḥḍem", "iwsiɛ"],
                "tense": ["prétérit négatif", "participe aoriste"],
                "person": ["2p_f", "participe"],
            }
        )
        pq.write_table(table, directory / "train-00000.parquet")
        return directory

    def test_french_tense_names_become_aspect_features(self, tmp_path: Path) -> None:
        entries = list(read_verb_forms(self.forms(tmp_path)))
        assert entries[0].features == (
            ("Aspect", "PerfNeg"),
            ("Gender", "Fem"),
            ("Number", "Plur"),
            ("Person", "2"),
        )

    def test_a_participle_carries_no_person(self, tmp_path: Path) -> None:
        entries = list(read_verb_forms(self.forms(tmp_path)))
        assert entries[1].features == (("Aspect", "PartAor"),)

    def test_every_form_is_tagged_verb(self, tmp_path: Path) -> None:
        assert all(e.upos == "VERB" for e in read_verb_forms(self.forms(tmp_path)))

    def test_an_empty_directory_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(LexiconError, match="no verb parquet shards"):
            list(read_verb_forms(tmp_path / "empty"))

    def test_conjugation_tables_carry_the_gloss_and_irregularity(self, tmp_path: Path) -> None:
        directory = tmp_path / "conjugation-tables"
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "name": ["addi"],
                    "translation": ["tendre un piège"],
                    "isIrregular": ["True"],
                    "isDerived": ["False"],
                }
            ),
            directory / "train-00000.parquet",
        )
        entry = next(iter(read_verb_lemmas(directory)))
        assert entry.lemma == "addi"
        assert entry.glosses[0].text == "tendre un piège"
        assert ("Irregular", "Yes") in entry.features
        assert ("Derived", "Yes") not in entry.features


class TestToponyms:
    def land(self, tmp_path: Path, names: list[str]) -> Path:
        directory = tmp_path / "data"
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "kabyle": names,
                    "french": ["x"] * len(names),
                    "category": ["city"] * len(names),
                }
            ),
            directory / "train-00000.parquet",
        )
        return directory

    def test_road_reference_codes_are_not_place_names(self, tmp_path: Path) -> None:
        """Two OSM service roads named `D` made `d` analyse as a proper noun 801 times."""
        directory = self.land(tmp_path, ["D", "2a", "Qsemṭina"])
        assert [e.form for e in read_toponyms(directory)] == ["Qsemṭina"]

    def test_a_three_character_name_is_kept(self, tmp_path: Path) -> None:
        """`Sig`, `Tiṭ`, `Ɛuf` are genuine, so the threshold cannot go higher."""
        directory = self.land(tmp_path, ["Sig", "Tiṭ", "Ɛuf"])
        assert len(list(read_toponyms(directory))) == MIN_TOPONYM_LENGTH

    def test_a_name_with_digits_is_kept(self, tmp_path: Path) -> None:
        """`20 Ɣuct 1955` is a street name, so digits cannot be the discriminator."""
        directory = self.land(tmp_path, ["20 Ɣuct 1955"])
        assert [e.form for e in read_toponyms(directory)] == ["20 Ɣuct 1955"]


class TestTafsut:
    def test_capitalised_main_entries_are_lowercased(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "t.jsonl",
            json.dumps({"fr": "ABAISSER", "kab": "SIDER", "type": "main_entry"}) + "\n",
        )
        entry = next(iter(read_tafsut(path)))
        assert entry.form == "sider"
        assert entry.glosses[0].text == "abaisser"

    def test_a_subentry_phrase_is_kept_whole(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "t.jsonl",
            json.dumps({"fr": "abaisser la hauteur", "kab": "sider tiddi", "type": "subentry"})
            + "\n",
        )
        entry = next(iter(read_tafsut(path)))
        assert entry.form == "sider tiddi"
        assert entry.features == (("Domain", "Math"),)


class TestAmawal:
    def row(self, title: str, content: str) -> str:
        return "ID,post_content,post_title\n" + f'"1","{content}","{title}"\n'

    def test_the_gloss_triple_is_read(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "a.csv",
            self.row("adlis", "<strong>Adlis</strong> (idlisen), livre - libro - كتاب"),
        )
        entry = next(iter(read_amawal(path)))
        assert [(g.language, g.text) for g in entry.glosses] == [
            ("fra", "livre"),
            ("spa", "libro"),
            ("ara", "كتاب"),
        ]

    def test_the_plural_becomes_its_own_entry(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "a.csv",
            self.row("adlis", "<strong>Adlis</strong> (idlisen), livre - libro - x"),
        )
        entries = list(read_amawal(path))
        plural = next(e for e in entries if e.form == "idlisen")
        assert plural.lemma == "adlis"
        assert plural.features == (("Number", "Plur"),)

    def test_the_annexed_state_becomes_its_own_entry(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "a.csv",
            self.row("adlis", "x - y <strong>Addad amaruz:</strong> wedlis"),
        )
        annexed = next(e for e in read_amawal(path) if e.form == "wedlis")
        assert annexed.lemma == "adlis"
        assert annexed.features == (("State", "Cons"),)

    def test_a_trailing_stop_is_stripped_from_the_annexed_form(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "a.csv", self.row("iman", "x - y <strong>Addad amaruz:</strong> yiman.")
        )
        assert any(e.form == "yiman" for e in read_amawal(path))

    def test_a_multiword_headword_is_skipped(self, tmp_path: Path) -> None:
        path = write(tmp_path / "a.csv", self.row("two words", "x - y"))
        assert list(read_amawal(path)) == []

    def test_prose_without_a_translation_triple_yields_no_gloss(self, tmp_path: Path) -> None:
        path = write(tmp_path / "a.csv", self.row("gn", "dormir"))
        entry = next(iter(read_amawal(path)))
        assert entry.glosses == ()


class TestG2PReader:
    def test_a_line_without_a_tab_is_skipped(self, tmp_path: Path) -> None:
        path = write(tmp_path / "g.tsv", "no tab here\nAzul\tæzul\n")
        assert list(read_g2p(path)) == [("Azul", "æzul")]

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(LexiconError, match="g2p training data not found"):
            list(read_g2p(tmp_path / "absent.tsv"))


class TestFeatures:
    def test_unset_values_are_dropped(self) -> None:
        assert features_of(Number="Plur", Gender=None) == (("Number", "Plur"),)

    def test_features_are_sorted_so_equal_entries_hash_equal(self) -> None:
        assert features_of(Number="Sing", Aspect="Aor") == features_of(Aspect="Aor", Number="Sing")

    def test_no_features_is_an_empty_tuple(self) -> None:
        assert features_of() == ()
