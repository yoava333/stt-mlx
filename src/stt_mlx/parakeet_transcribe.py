"""Transcription backend using the Hebrew Parakeet-TDT model via transformers.

Parakeet is a Token-and-Duration Transducer: alongside each emitted token it
predicts a duration in encoder frames, so per-token timestamps come directly
from a cumulative sum of those durations. This module turns that raw output
into the same ``TranscriptSegment`` list the whisper backend produces, so the
rest of the pipeline (diarization, speaker assignment, output) is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

from stt_mlx.cli import TranscriptSegment

DEFAULT_MODEL = "thewh1teagle/parakeet-tdt-0.6b-v3-he"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30.0
# Clause- and sentence-final marks where a segment should break.
PUNCT = {".", ",", "?", "!", "…", ";", ":"}


def build_segments(
    token_ids: Sequence[int],
    durations: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    *,
    blank_id: int,
    punct: set[str],
    frame_s: float,
    offset: float = 0.0,
    pause_s: float = 1.0,
) -> list[TranscriptSegment]:
    """Group TDT tokens into timestamped segments.

    Args:
        token_ids: Emitted token ids (includes blanks).
        durations: Per-token frame advances, parallel to ``token_ids``.
        decode: Maps a list of token ids to text (same tokenizer as the model).
        blank_id: Id of the TDT blank token, which carries time but no text.
        punct: Characters that force a segment break after the token.
        frame_s: Seconds per encoder frame.
        offset: Seconds to add to every timestamp (chunk start in the file).
        pause_s: A silence gap larger than this between words starts a segment.

    Returns:
        Segments ordered by time; punctuation-only spans are dropped.
    """
    segments: list[TranscriptSegment] = []
    span: list[int] = []
    seg_start = 0.0
    seg_end = 0.0
    prev_end: float | None = None
    cum = 0

    def flush() -> None:
        nonlocal span
        if span:
            text = decode(span).strip()
            if any(ch.isalnum() for ch in text):
                segments.append(
                    TranscriptSegment(round(seg_start, 3), round(seg_end, 3), text)
                )
        span = []

    for tok, dur in zip(token_ids, durations):
        start = offset + cum * frame_s
        end = offset + (cum + dur) * frame_s
        cum += dur
        if tok == blank_id:
            continue
        if span and prev_end is not None and (start - prev_end) > pause_s:
            flush()
        if not span:
            seg_start = start
        span.append(tok)
        seg_end = end
        prev_end = end
        if decode([tok]).strip() in punct:
            flush()
    flush()
    return segments


def transcribe_parakeet(
    wav: Path,
    model: str,
    language: str,
    chunk_seconds: float = CHUNK_SECONDS,
) -> list[TranscriptSegment]:
    """Transcribe a 16 kHz mono WAV with Parakeet-TDT.

    ``language`` is accepted for a uniform backend signature but ignored: this
    model is Hebrew-only.
    """
    import librosa
    import torch
    from transformers import AutoModelForTDT, AutoProcessor

    processor = AutoProcessor.from_pretrained(model)
    net = AutoModelForTDT.from_pretrained(model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = net.to(device).eval()

    blank_id = net.config.blank_token_id
    frame_s = (
        processor.feature_extractor.hop_length
        / processor.feature_extractor.sampling_rate
        * net.config.encoder_config.subsampling_factor
    )

    def decode(ids: Sequence[int]) -> str:
        return processor.tokenizer.decode(list(ids), skip_special_tokens=True)

    audio, _ = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
    total = len(audio) / SAMPLE_RATE
    chunks = max(1, math.ceil(total / chunk_seconds))

    segments: list[TranscriptSegment] = []
    for i in range(chunks):
        start = int(i * chunk_seconds * SAMPLE_RATE)
        end = int(min((i + 1) * chunk_seconds, total) * SAMPLE_RATE)
        window = audio[start:end]
        if len(window) == 0:
            continue
        inputs = processor(window, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            output = net.generate(**inputs)
        segments.extend(
            build_segments(
                output.sequences[0].tolist(),
                output.durations[0].tolist(),
                decode,
                blank_id=blank_id,
                punct=PUNCT,
                frame_s=frame_s,
                offset=i * chunk_seconds,
            )
        )
    return segments
