from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from agbalu.normalise import Normaliser
from agbalu.tokenizer.evaluate import (
    CLITIC_HOSTS,
    CLITICS,
    STATE_PAIRS,
    evaluate,
    load,
)
from agbalu.tokenizer.seed import build_pool, write_seed_file
from agbalu.tokenizer.spec import (
    CLS_PIECE,
    MASK_PIECE,
    PAD_ID,
    PAD_PIECE,
    SEP_PIECE,
    UNK_PIECE,
    TokenizerError,
    TokenizerSpec,
)
from agbalu.tokenizer.train import BuildResult, sha256, train

TINY_VOCAB = 1_000

_LETTERS = "abcdefgɣhḥijklmnqrstṭuwxyzẓṛṣɛčǧţ"


def _words(rng: random.Random, count: int) -> list[str]:
    words = {free for free, _ in STATE_PAIRS} | {annexed for _, annexed in STATE_PAIRS}
    words |= set(CLITIC_HOSTS) | set(CLITICS)
    words |= {f"{host}-{clitic}" for host in CLITIC_HOSTS for clitic in CLITICS}
    while len(words) < count:
        length = rng.randint(3, 9)
        words.add("".join(rng.choice(_LETTERS) for _ in range(length)))
    return sorted(words)


def synthetic_corpus(directory: Path) -> tuple[Path, list[str]]:
    """A corpus rich enough to fill a 1,000-piece vocabulary, with the real evaluation
    fixtures embedded so the Kabyle-specific criteria score against present material."""
    rng = random.Random(20260807)
    vocabulary = _words(rng, 4_000)
    sentences = [
        " ".join(rng.choice(vocabulary) for _ in range(rng.randint(4, 12))) for _ in range(6_000)
    ]
    plain = directory / "corpus.txt"
    plain.write_text("".join(s + "\n" for s in sentences), encoding="utf-8")
    return plain, sentences


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    directory = tmp_path_factory.mktemp("tokenizer")
    plain, sentences = synthetic_corpus(directory)
    return plain, sentences


@pytest.fixture(scope="module")
def trained(workspace: tuple[Path, list[str]]) -> BuildResult:
    plain, _ = workspace
    return train(TokenizerSpec(vocab_size=TINY_VOCAB, num_threads=2), plain, plain.parent)


@pytest.mark.slow
class TestTrain:
    def test_produces_a_model_of_the_requested_size(self, trained: BuildResult) -> None:
        assert trained.model.is_file()
        assert trained.pieces == TINY_VOCAB

    def test_reserves_the_special_token_ids(self, trained: BuildResult) -> None:
        processor = load(trained.model)
        assert processor.id_to_piece(PAD_ID) == PAD_PIECE
        for piece in (UNK_PIECE, CLS_PIECE, SEP_PIECE, MASK_PIECE):
            assert processor.piece_to_id(piece) >= 0

    def test_keeps_the_hyphen_atomic_so_clitics_can_split(self, trained: BuildResult) -> None:
        processor = load(trained.model)
        assert processor.piece_to_id("-") > 0

    def test_stamps_the_normaliser_version_in_full(self, trained: BuildResult) -> None:
        payload = json.loads(trained.metadata.read_text(encoding="utf-8"))
        assert payload["normaliser_version"] == Normaliser().version
        assert payload["normaliser_version"].count("+rules") == 1

    def test_records_the_hash_of_the_model_it_describes(self, trained: BuildResult) -> None:
        payload = json.loads(trained.metadata.read_text(encoding="utf-8"))
        assert payload["model_sha256"] == sha256(trained.model)
        assert payload["pieces"] == trained.pieces

    def test_records_that_it_was_not_seeded(self, trained: BuildResult) -> None:
        payload = json.loads(trained.metadata.read_text(encoding="utf-8"))
        assert payload["spec"]["seeded"] is False
        assert payload["spec"]["seed_file"] is None

    def test_missing_corpus_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(TokenizerError, match="training corpus not found"):
            train(TokenizerSpec(), tmp_path / "absent.txt", tmp_path)

    def test_missing_seed_file_is_named(self, tmp_path: Path) -> None:
        plain = tmp_path / "corpus.txt"
        plain.write_text("azul\n", encoding="utf-8")
        spec = TokenizerSpec(seed_file=tmp_path / "absent.tsv")
        with pytest.raises(TokenizerError, match="seed file not found"):
            train(spec, plain, tmp_path)


@pytest.mark.slow
class TestSeededTraining:
    def test_a_seeded_run_reaches_the_same_vocabulary_size(
        self, workspace: tuple[Path, list[str]], tmp_path: Path
    ) -> None:
        """The pool has to be complete. `seed_sentencepieces_file` replaces SentencePiece's
        own extraction, so an incomplete pool caps the vocabulary below the target."""
        plain, sentences = workspace
        freq: Counter[str] = Counter()
        for sentence in sentences:
            freq.update(sentence.split())
        lexicon = tmp_path / "lexicon.jsonl"
        lexicon.write_text(
            "".join(
                json.dumps({"form": form}, ensure_ascii=False) + "\n" for form, _ in STATE_PAIRS
            ),
            encoding="utf-8",
        )
        seed_file = tmp_path / "seed.tsv"
        write_seed_file(build_pool(freq, lexicon), seed_file)

        spec = TokenizerSpec(vocab_size=TINY_VOCAB, seed_file=seed_file, num_threads=2)
        result = train(spec, plain, tmp_path)
        assert result.pieces == TINY_VOCAB
        assert result.spec.seeded

        payload = json.loads(result.metadata.read_text(encoding="utf-8"))
        assert payload["spec"]["seeded"] is True
        assert payload["spec"]["seed_file"] == str(seed_file)


@pytest.mark.slow
class TestEvaluate:
    def test_refuses_an_empty_sample(self, trained: BuildResult) -> None:
        with pytest.raises(TokenizerError, match="empty sample"):
            evaluate(trained.model, [])

    def test_missing_model_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(TokenizerError, match="model not found"):
            evaluate(tmp_path / "absent.model", ["azul"])

    def test_round_trips_every_sentence(
        self, trained: BuildResult, workspace: tuple[Path, list[str]]
    ) -> None:
        _, sentences = workspace
        report = evaluate(trained.model, sentences[:400])
        assert report.roundtrip_failures == 0

    def test_byte_fallback_round_trips_text_outside_the_alphabet(
        self, trained: BuildResult
    ) -> None:
        report = evaluate(trained.model, ["日本語 текст 🙂"])
        assert report.roundtrip_failures == 0
        assert report.byte_pieces > 0

    def test_reports_fertility_above_one(self, trained: BuildResult) -> None:
        report = evaluate(trained.model, ["axxam wexxam tamurt"])
        assert report.fertility >= 1.0
        assert report.tokens_per_char > 0

    def test_scores_the_kabyle_criteria_within_their_trial_counts(
        self, trained: BuildResult
    ) -> None:
        report = evaluate(trained.model, ["azul"])
        assert report.state_trials == len(STATE_PAIRS)
        assert report.clitic_trials == len(CLITIC_HOSTS) * len(CLITICS)
        assert 0 <= report.state_share <= report.state_trials
        assert 0 <= report.clitic_atomic <= report.clitic_trials

    def test_reports_the_embedding_cost_of_the_vocabulary(self, trained: BuildResult) -> None:
        report = evaluate(trained.model, ["azul"])
        assert report.embedding_params[384] == report.vocab_size * 384
        assert report.embedding_params[768] == report.vocab_size * 768

    def test_serialises_every_field(self, trained: BuildResult) -> None:
        report = evaluate(trained.model, ["azul"])
        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["vocab_size"] == TINY_VOCAB
        assert payload["state_share"] == f"{report.state_share}/{report.state_trials}"
        assert payload["embedding_params"]["384"] == TINY_VOCAB * 384

    def test_a_sample_of_blank_sentences_does_not_divide_by_zero(
        self, trained: BuildResult
    ) -> None:
        report = evaluate(trained.model, [""])
        assert report.fertility == 0.0
        assert report.tokens_per_char == 0.0
