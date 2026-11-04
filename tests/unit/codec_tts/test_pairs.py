"""Reject self-reference and preserve pair accounting."""

from agbalu.codec_tts.artifacts import Recording
from agbalu.codec_tts.pairs import pair_recordings
from agbalu.codec_tts.partition import Partition


def records() -> list[Recording]:
    values: list[Recording] = []
    for index, text in enumerate(("a", "b", "a")):
        values.append(
            Recording(
                record_id=str(index),
                shard="x",
                row=index,
                audio_sha256=str(index),
                speaker="s",
                text=text,
                text_key=text,
                stored_rate_hz=24000,
                samples=24000,
                prepared_samples=24000,
                duration_s=1,
                peak=0.5,
                rms_dbfs=-10,
                clipping_fraction=0,
                source_id="x",
                source_revision="a" * 40,
                license="CC0-1.0",
                retrieval_date="2026-11-03",
            )
        )
    return values


def test_pairs_use_different_scripts() -> None:
    items = records()
    result = pair_recordings(items, Partition(dict.fromkeys(("0", "1", "2"), "train"), {}), 4)
    texts = {item.record_id: item.text for item in items}
    assert len(result.pairs) == 3
    assert result.excluded == {}
    for pair in result.pairs:
        assert pair.target != pair.reference
        assert texts[pair.target] != texts[pair.reference]
        assert pair.split == "train"


def test_singleton_is_explicitly_excluded() -> None:
    result = pair_recordings(records()[:1], Partition({"0": "test"}, {}), 4)
    assert result.pairs == ()
    assert result.excluded == {"0": "no distinct same-speaker reference script/audio"}


def test_empty_pairing() -> None:
    result = pair_recordings([], Partition({}, {}), 4)
    assert result.pairs == ()
    assert result.excluded == {}
