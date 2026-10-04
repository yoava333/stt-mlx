from collections.abc import Sequence

from stt_mlx.cli import TranscriptSegment
from stt_mlx.parakeet_transcribe import build_segments

# Tiny synthetic vocabulary for the fake decoder.
VOCAB = {0: "<blank>", 1: "שלום", 2: ",", 3: "עולם", 4: ".", 5: "כן", 6: "לא"}
BLANK = 0
PUNCT = {",", ".", "?", "!"}


def decode(ids: Sequence[int]) -> str:
    return " ".join(VOCAB[i] for i in ids)


def test_splits_on_punctuation_tokens() -> None:
    # "שלום ," then "עולם ." → two clause-level segments.
    segments = build_segments(
        [1, 2, 3, 4],
        [5, 2, 5, 2],
        decode,
        blank_id=BLANK,
        punct=PUNCT,
        frame_s=0.1,
    )
    assert segments == [
        TranscriptSegment(0.0, 0.7, "שלום ,"),
        TranscriptSegment(0.7, 1.4, "עולם ."),
    ]


def test_splits_on_long_pause_between_words() -> None:
    # A blank with a long duration creates a >1s gap → a segment boundary.
    segments = build_segments(
        [1, 0, 5],
        [5, 20, 5],
        decode,
        blank_id=BLANK,
        punct=PUNCT,
        frame_s=0.1,
        pause_s=1.0,
    )
    assert segments == [
        TranscriptSegment(0.0, 0.5, "שלום"),
        TranscriptSegment(2.5, 3.0, "כן"),
    ]


def test_offset_shifts_all_timestamps() -> None:
    segments = build_segments(
        [1, 4],
        [5, 2],
        decode,
        blank_id=BLANK,
        punct=PUNCT,
        frame_s=0.1,
        offset=10.0,
    )
    assert segments == [TranscriptSegment(10.0, 10.7, "שלום .")]


def test_drops_punctuation_only_segments() -> None:
    # A span that decodes to punctuation only must not be emitted.
    assert (
        build_segments([4], [2], decode, blank_id=BLANK, punct=PUNCT, frame_s=0.1) == []
    )


def test_flushes_trailing_span_without_punctuation() -> None:
    segments = build_segments(
        [1, 3],
        [5, 5],
        decode,
        blank_id=BLANK,
        punct=PUNCT,
        frame_s=0.1,
    )
    assert segments == [TranscriptSegment(0.0, 1.0, "שלום עולם")]


def test_blank_tokens_are_skipped_but_advance_time() -> None:
    # Leading blank advances the clock; first word starts after it.
    segments = build_segments(
        [0, 1],
        [3, 5],
        decode,
        blank_id=BLANK,
        punct=PUNCT,
        frame_s=0.1,
    )
    assert segments == [TranscriptSegment(0.3, 0.8, "שלום")]
