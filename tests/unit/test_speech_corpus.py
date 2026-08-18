"""Common Voice transcripts into a speech corpus (task 5.1-5.4).

Two assertions carry the phase. Speaker disjointness must be a *refusal*, because a
leaked speaker raises the test score and nothing downstream would contradict it. And
a clip whose transcript is longer than its frame sequence must be rejected here: CTC
returns infinite loss for that row, which surfaces as a NaN mid-run rather than as a
bad row at build time.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from agbalu.speech import corpus
from agbalu.speech.corpus import SpeechError

COLUMNS = ("client_id", "path", "sentence", "up_votes", "down_votes", "locale")
KAB = "Azul fell-awen ay imdanen n tmurt-nneɣ."

INVISIBLES = tuple(chr(cp) for cp in (0xFEFF, 0x200B, 0x00A0, 0x2060, 0x200D, 0x00AD))
"""BOM, zero-width space, no-break space, word joiner, ZWJ, soft hyphen — named by
codepoint rather than typed. A literal one here would be invisible to a reviewer,
which is the same property that makes it a defect in a transcript."""


def write_tsv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...] = COLUMNS) -> Path:
    """Written by hand, because Common Voice writes quotes that `csv.writer` refuses."""
    lines = ["\t".join(columns)]
    lines += ["\t".join(row.get(column, "") for column in columns) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    return path


def row(clip: str, sentence: str = KAB, speaker: str = "spk-a") -> dict[str, str]:
    return {
        "client_id": speaker,
        "path": clip,
        "sentence": sentence,
        "up_votes": "2",
        "down_votes": "0",
        "locale": "kab",
    }


def durations(path: Path, mapping: dict[str, int]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("clip\tduration[ms]\n")
        for clip, ms in mapping.items():
            handle.write(f"{clip}\t{ms}\n")
    return path


def corpus_root(
    tmp_path: Path, splits: dict[str, list[dict[str, str]]], ms: dict[str, int]
) -> Path:
    root = tmp_path / "kab"
    root.mkdir()
    for name, rows in splits.items():
        write_tsv(root / f"{name}.tsv", rows)
    durations(root / corpus.DURATIONS, ms)
    return root


def three_splits(tmp_path: Path) -> Path:
    return corpus_root(
        tmp_path,
        {
            "train": [row("a.mp3", speaker="spk-a"), row("b.mp3", speaker="spk-b")],
            "dev": [row("c.mp3", speaker="spk-c")],
            "test": [row("d.mp3", speaker="spk-d")],
        },
        {"a.mp3": 3000, "b.mp3": 3200, "c.mp3": 2800, "d.mp3": 3100},
    )


class TestDurations:
    def test_reads_milliseconds(self, tmp_path: Path) -> None:
        path = durations(tmp_path / "d.tsv", {"a.mp3": 1234})
        assert corpus.read_durations(path) == {"a.mp3": 1234}

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SpeechError, match="not found"):
            corpus.read_durations(tmp_path / "absent.tsv")

    def test_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = durations(tmp_path / "d.tsv", {})
        with pytest.raises(SpeechError, match="no durations"):
            corpus.read_durations(path)

    def test_missing_clip_column(self, tmp_path: Path) -> None:
        path = tmp_path / "d.tsv"
        path.write_text("name\tduration[ms]\na.mp3\t10\n", encoding="utf-8")
        with pytest.raises(SpeechError, match="no `clip` column"):
            corpus.read_durations(path)

    def test_missing_duration_column(self, tmp_path: Path) -> None:
        path = tmp_path / "d.tsv"
        path.write_text("clip\tlength\na.mp3\t10\n", encoding="utf-8")
        with pytest.raises(SpeechError, match="no duration column"):
            corpus.read_durations(path)

    def test_non_integer_duration(self, tmp_path: Path) -> None:
        path = tmp_path / "d.tsv"
        path.write_text("clip\tduration[ms]\na.mp3\t3.5s\n", encoding="utf-8")
        with pytest.raises(SpeechError, match="not an integer"):
            corpus.read_durations(path)

    def test_blank_rows_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "d.tsv"
        path.write_text("clip\tduration[ms]\na.mp3\t100\n\t\nb.mp3\t200\n", encoding="utf-8")
        assert corpus.read_durations(path) == {"a.mp3": 100, "b.mp3": 200}


class TestRows:
    def test_missing_required_column(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {}, {"a.mp3": 3000})
        write_tsv(root / "train.tsv", [{"path": "a.mp3", "sentence": KAB}], ("path", "sentence"))
        with pytest.raises(SpeechError, match="client_id"):
            corpus.build(root, tmp_path / "out", splits=("train",))

    def test_missing_split_file(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {}, {"a.mp3": 3000})
        with pytest.raises(SpeechError, match="split not found"):
            corpus.build(root, tmp_path / "out", splits=("train",))

    def test_short_row_is_dropped_not_fatal(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {"train": [row("a.mp3")]}, {"a.mp3": 3000, "b.mp3": 3000})
        with (root / "train.tsv").open("a", encoding="utf-8") as handle:
            handle.write("spk-b\tb.mp3\n")
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].kept == 1

    def test_long_row_is_dropped_not_fatal(self, tmp_path: Path) -> None:
        """An extra column means an unescaped tab, so `sentence` is a truncated prefix."""
        root = corpus_root(tmp_path, {"train": [row("a.mp3")]}, {"a.mp3": 3000, "b.mp3": 3000})
        with (root / "train.tsv").open("a", encoding="utf-8") as handle:
            handle.write("spk-b\tb.mp3\t" + KAB + "\t2\t0\tkab\textra\n")
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].kept == 1

    def test_quote_in_transcript_survives(self, tmp_path: Path) -> None:
        quoted = 'Yenna-yas: "Azul".'
        root = corpus_root(
            tmp_path,
            {"train": [row("a.mp3", quoted), row("b.mp3")]},
            {"a.mp3": 3000, "b.mp3": 3000},
        )
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].kept == 2

    def test_crlf_line_endings(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {"train": [row("a.mp3")]}, {"a.mp3": 3000})
        text = (root / "train.tsv").read_text(encoding="utf-8")
        (root / "train.tsv").write_text(text.replace("\n", "\r\n"), encoding="utf-8", newline="")
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].kept == 1


class TestFiltering:
    @pytest.mark.parametrize(
        ("ms", "reason"),
        [
            (corpus.MIN_DURATION_MS - 1, "too-short"),
            (corpus.MAX_DURATION_MS + 1, "too-long"),
            (0, "too-short"),
        ],
    )
    def test_duration_bounds(self, tmp_path: Path, ms: int, reason: str) -> None:
        root = corpus_root(
            tmp_path, {"train": [row("a.mp3"), row("b.mp3")]}, {"a.mp3": ms, "b.mp3": 3000}
        )
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].rejected == {reason: 1}
        assert report.splits[0].kept == 1

    def test_clip_without_a_duration(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {"train": [row("a.mp3"), row("b.mp3")]}, {"b.mp3": 3000})
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].rejected == {"no-duration": 1}

    def test_punctuation_only_transcript(self, tmp_path: Path) -> None:
        root = corpus_root(
            tmp_path,
            {"train": [row("a.mp3", "?! ..."), row("b.mp3")]},
            {"a.mp3": 3000, "b.mp3": 3000},
        )
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].rejected == {"empty-transcript": 1}

    def test_transcript_longer_than_the_frame_sequence(self, tmp_path: Path) -> None:
        """CTC loss is infinite when the target exceeds the input; reject, never train."""
        long_text = " ".join(["azul"] * 40)
        root = corpus_root(
            tmp_path,
            {"train": [row("a.mp3", long_text), row("b.mp3")]},
            {"a.mp3": corpus.MIN_DURATION_MS, "b.mp3": 3000},
        )
        assert len(long_text) > corpus.frames(corpus.MIN_DURATION_MS)
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].rejected == {"ctc-infeasible": 1}

    def test_a_feasible_clip_of_the_same_text_is_kept(self, tmp_path: Path) -> None:
        long_text = " ".join(["azul"] * 40)
        root = corpus_root(tmp_path, {"train": [row("a.mp3", long_text)]}, {"a.mp3": 8000})
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        assert report.splits[0].kept == 1

    def test_no_split_keeps_nothing(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {"train": [row("a.mp3", "!!!")]}, {"a.mp3": 3000})
        with pytest.raises(SpeechError, match="kept no clips"):
            corpus.build(root, tmp_path / "out", splits=("train",))

    def test_no_splits_requested(self, tmp_path: Path) -> None:
        root = three_splits(tmp_path)
        with pytest.raises(SpeechError, match="no splits requested"):
            corpus.build(root, tmp_path / "out", splits=())


class TestNormalisation:
    def test_greek_epsilon_is_repaired_and_flagged(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {"train": [row("a.mp3", "Aεdawen n tmurt.")]}, {"a.mp3": 3000})
        report = corpus.build(root, tmp_path / "out", splits=("train",))
        clip = corpus.read(tmp_path / "out" / "train.jsonl")[0]
        assert "ɛ" in clip.text
        assert "ε" not in clip.text
        assert clip.repaired
        assert report.splits[0].repaired == 1

    def test_clean_transcript_is_not_flagged(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {"train": [row("a.mp3", "azul fell-awen")]}, {"a.mp3": 3000})
        corpus.build(root, tmp_path / "out", splits=("train",))
        assert not corpus.read(tmp_path / "out" / "train.jsonl")[0].repaired

    @pytest.mark.parametrize("invisible", INVISIBLES)
    def test_invisibles_do_not_reach_the_target(self, tmp_path: Path, invisible: str) -> None:
        text = f"azul{invisible} fell-awen"
        root = corpus_root(tmp_path, {"train": [row("a.mp3", text)]}, {"a.mp3": 3000})
        corpus.build(root, tmp_path / "out", splits=("train",))
        clip = corpus.read(tmp_path / "out" / "train.jsonl")[0]
        assert clip.target == "azul fell-awen"

    def test_combining_mark_is_composed(self, tmp_path: Path) -> None:
        """NFD-built rather than typed, so the input is provably decomposed."""
        decomposed = unicodedata.normalize("NFD", "aṭas")
        assert len(decomposed) == 5
        root = corpus_root(tmp_path, {"train": [row("a.mp3", decomposed)]}, {"a.mp3": 3000})
        corpus.build(root, tmp_path / "out", splits=("train",))
        target = corpus.read(tmp_path / "out" / "train.jsonl")[0].target
        assert target == "aṭas"
        assert len(target) == 4

    def test_hyphen_survives_into_the_target(self, tmp_path: Path) -> None:
        root = corpus_root(tmp_path, {"train": [row("a.mp3", "Yenna-yas-d.")]}, {"a.mp3": 3000})
        corpus.build(root, tmp_path / "out", splits=("train",))
        assert corpus.read(tmp_path / "out" / "train.jsonl")[0].target == "yenna-yas-d"


class TestSpeakerDisjointness:
    def test_disjoint_splits_report_zero(self, tmp_path: Path) -> None:
        report = corpus.build(three_splits(tmp_path), tmp_path / "out")
        assert report.overlaps == {"dev-test": 0, "dev-train": 0, "test-train": 0}

    def test_a_shared_speaker_is_a_refusal(self, tmp_path: Path) -> None:
        root = corpus_root(
            tmp_path,
            {
                "train": [row("a.mp3", speaker="spk-a")],
                "dev": [row("c.mp3", speaker="spk-c")],
                "test": [row("d.mp3", speaker="spk-a")],
            },
            {"a.mp3": 3000, "c.mp3": 3000, "d.mp3": 3000},
        )
        with pytest.raises(SpeechError, match="share speakers"):
            corpus.build(root, tmp_path / "out")

    def test_the_refusal_names_the_pair_and_the_count(self, tmp_path: Path) -> None:
        root = corpus_root(
            tmp_path,
            {
                "train": [row("a.mp3", speaker="spk-a"), row("b.mp3", speaker="spk-b")],
                "dev": [row("c.mp3", speaker="spk-a")],
                "test": [row("d.mp3", speaker="spk-b")],
            },
            {"a.mp3": 3000, "b.mp3": 3000, "c.mp3": 3000, "d.mp3": 3000},
        )
        with pytest.raises(SpeechError, match=r"dev-train: 1.*test-train: 1"):
            corpus.build(root, tmp_path / "out")


class TestReport:
    def test_hours_and_counts(self, tmp_path: Path) -> None:
        report = corpus.build(three_splits(tmp_path), tmp_path / "out")
        assert [s.kept for s in report.splits] == [2, 1, 1]
        assert report.hours == pytest.approx(12100 / 3_600_000)

    def test_stats_carry_the_normaliser_version(self, tmp_path: Path) -> None:
        report = corpus.build(three_splits(tmp_path), tmp_path / "out")
        payload = report.as_dict()
        assert payload["normaliser_version"]
        assert payload["conv_stride"] == list(corpus.CONV_STRIDE)
        assert payload["clips"] == 4

    def test_written_records_round_trip(self, tmp_path: Path) -> None:
        corpus.build(three_splits(tmp_path), tmp_path / "out")
        clips = corpus.read(tmp_path / "out" / "train.jsonl")
        assert [c.clip for c in clips] == ["a.mp3", "b.mp3"]
        assert all(c.split == "train" for c in clips)

    def test_one_file_per_split(self, tmp_path: Path) -> None:
        report = corpus.build(three_splits(tmp_path), tmp_path / "out")
        assert [p.name for p in report.paths] == ["train.jsonl", "dev.jsonl", "test.jsonl"]
        assert all(p.is_file() for p in report.paths)


class TestRead:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SpeechError, match="not built"):
            corpus.read(tmp_path / "absent.jsonl")

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(SpeechError, match="empty"):
            corpus.read(path)

    def test_record_missing_a_field(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text(json.dumps({"clip": "a.mp3"}) + "\n", encoding="utf-8")
        with pytest.raises(SpeechError, match="not a clip record"):
            corpus.read(path)

    def test_record_with_an_unparseable_duration(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        path.write_text(
            json.dumps(
                {
                    "clip": "a.mp3",
                    "speaker": "s",
                    "split": "train",
                    "duration_ms": "soon",
                    "text": KAB,
                    "target": "azul",
                    "repaired": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(SpeechError, match="not a clip record"):
            corpus.read(path)


class TestFrames:
    """The CTC feasibility budget. Over-estimating admits rows whose loss is infinite."""

    @staticmethod
    def _oracle(ms: int) -> int:
        """`Wav2Vec2PreTrainedModel._get_feat_extract_output_lengths`, written out."""
        length = ms * corpus.SAMPLE_RATE // 1000
        for kernel, stride in zip(corpus.CONV_KERNEL, corpus.CONV_STRIDE, strict=True):
            length = (length - kernel) // stride + 1
        return max(length, 0)

    @pytest.mark.parametrize("ms", [0, 19, 20, 400, 1000, 3463, 5000, 20000])
    def test_agrees_with_the_convolution_chain(self, ms: int) -> None:
        assert corpus.frames(ms) == self._oracle(ms)

    def test_never_negative(self) -> None:
        assert all(corpus.frames(ms) >= 0 for ms in range(0, 200, 7))

    def test_monotonic_in_duration(self) -> None:
        counts = [corpus.frames(ms) for ms in range(0, 5000, 37)]
        assert counts == sorted(counts)

    def test_is_stricter_than_the_naive_total_stride(self) -> None:
        """320 samples of total stride suggests duration/20ms; the kernels cost one more
        frame, and that one frame is the difference between a rejected row and a NaN."""
        for ms in (400, 1000, 3463, 20000):
            assert corpus.frames(ms) == ms // 20 - 1
