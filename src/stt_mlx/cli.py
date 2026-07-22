"""stt — transcribe + diarize an audio file using mlx_whisper and senko."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-mlx"
OutputFormat = Literal["txt", "srt", "json"]
FORMAT_NAMES: dict[str, OutputFormat] = {
    "txt": "txt",
    "srt": "srt",
    "json": "json",
}


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class AttributedSegment(TranscriptSegment):
    speaker: str


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def to_16k_mono_wav(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required and not on PATH")
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", str(src),
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
            str(dst),
        ],
        check=True,
    )


def transcribe(
    wav: Path, model: str, language: str | None
) -> list[TranscriptSegment]:
    import mlx_whisper

    kwargs: dict[str, object] = {"path_or_hf_repo": model}
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(str(wav), **kwargs)
    return [
        TranscriptSegment(
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]).strip(),
        )
        for segment in result["segments"]
    ]


def diarize(wav: Path) -> list[SpeakerTurn]:
    import senko

    diarizer = senko.Diarizer(device="auto", warmup=True, quiet=False)
    result = diarizer.diarize(str(wav), generate_colors=False)
    if result is None:
        return []
    turns: list[SpeakerTurn] = []
    for segment in result["merged_segments"]:
        if not isinstance(segment, Mapping):
            raise TypeError("senko merged segments must be mappings")
        turns.append(
            SpeakerTurn(
                start=float(segment["start"]),
                end=float(segment["end"]),
                speaker=str(segment["speaker"]),
            )
        )
    return turns


def assign_speakers(
    segments: list[TranscriptSegment], turns: list[SpeakerTurn]
) -> list[AttributedSegment]:
    attributed: list[AttributedSegment] = []
    for segment in segments:
        best_overlap = 0.0
        best_speaker = "UNKNOWN"
        for turn in turns:
            overlap = max(
                0.0,
                min(segment.end, turn.end) - max(segment.start, turn.start),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        attributed.append(
            AttributedSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=best_speaker,
            )
        )
    return attributed


def fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_srt(t: float) -> str:
    total_ms = round(t * 1000)
    total_seconds, ms = divmod(total_ms, 1000)
    total_minutes, s = divmod(total_seconds, 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_txt(path: Path, segments: list[AttributedSegment]) -> None:
    lines: list[str] = []
    current_spk: str | None = None
    for seg in segments:
        if seg.speaker != current_spk:
            if current_spk is not None:
                lines.append("")
            lines.append(f"=== {seg.speaker} ===")
            current_spk = seg.speaker
        lines.append(f"[{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}] {seg.text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(path: Path, segments: list[AttributedSegment]) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}")
        lines.append(f"{seg.speaker}: {seg.text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    path: Path,
    segments: list[AttributedSegment],
    turns: list[SpeakerTurn],
) -> None:
    payload = {
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": segment.speaker,
            }
            for segment in segments
        ],
        "diarization": [
            {"start": turn.start, "end": turn.end, "speaker": turn.speaker}
            for turn in turns
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_formats(value: str) -> frozenset[OutputFormat]:
    names = {part.strip() for part in value.split(",") if part.strip()}
    if not names:
        raise argparse.ArgumentTypeError("at least one output format is required")
    unknown = names - FORMAT_NAMES.keys()
    if unknown:
        formatted = ", ".join(sorted(unknown))
        raise argparse.ArgumentTypeError(f"unknown output format(s): {formatted}")
    return frozenset(FORMAT_NAMES[name] for name in names)


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="stt",
        description="Transcribe and diarize an audio file (mlx_whisper + senko).",
    )
    p.add_argument("audio", type=Path, help="Input audio file (any format ffmpeg accepts)")
    p.add_argument(
        "-o", "--output-prefix", type=Path, default=None,
        help="Output file prefix (default: input file's path without extension)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Whisper model HF repo (default: {DEFAULT_MODEL})")
    p.add_argument("--language", default=None, help="Language code (e.g. he, en); auto-detect if omitted")
    p.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    p.add_argument("--no-transcribe", action="store_true", help="Skip transcription (diarization only)")
    p.add_argument(
        "--formats",
        type=parse_formats,
        default=parse_formats("txt,srt,json"),
        help="Comma-separated output formats: txt,srt,json (default: all three)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.no_diarize and args.no_transcribe:
        print("error: --no-diarize and --no-transcribe both set; nothing to do", file=sys.stderr)
        return 2

    audio_in = args.audio.expanduser().resolve()
    if not audio_in.exists():
        print(f"error: not found: {audio_in}", file=sys.stderr)
        return 2

    out_prefix = args.output_prefix if args.output_prefix else audio_in.with_suffix("")

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio_16k.wav"
        print(f"Converting → {wav}", file=sys.stderr, flush=True)
        to_16k_mono_wav(audio_in, wav)

        segments: list[TranscriptSegment] = []
        if not args.no_transcribe:
            print(f"Transcribing with {args.model}...", file=sys.stderr, flush=True)
            segments = transcribe(wav, args.model, args.language)
            print(f"  {len(segments)} segments", file=sys.stderr, flush=True)

        turns: list[SpeakerTurn] = []
        if not args.no_diarize:
            print("Diarizing with senko...", file=sys.stderr, flush=True)
            turns = diarize(wav)
            print(f"  {len(turns)} speaker turns", file=sys.stderr, flush=True)

    if args.no_transcribe:
        rttm_path = output_path(out_prefix, ".rttm")
        with rttm_path.open("w", encoding="utf-8") as f:
            audio_id = audio_in.stem
            for turn in turns:
                f.write(
                    f"SPEAKER {audio_id} 1 {turn.start:.3f} "
                    f"{turn.end - turn.start:.3f} <NA> <NA> "
                    f"{turn.speaker} <NA> <NA>\n"
                )
        print(f"Wrote {rttm_path}", file=sys.stderr)
        return 0

    if turns:
        attributed_segments = assign_speakers(segments, turns)
    else:
        attributed_segments = [
            AttributedSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker="SPEAKER_00",
            )
            for segment in segments
        ]

    if "txt" in args.formats:
        path = output_path(out_prefix, ".diarized.txt")
        write_txt(path, attributed_segments)
        print(f"Wrote {path}", file=sys.stderr)
    if "srt" in args.formats:
        path = output_path(out_prefix, ".diarized.srt")
        write_srt(path, attributed_segments)
        print(f"Wrote {path}", file=sys.stderr)
    if "json" in args.formats:
        path = output_path(out_prefix, ".diarized.json")
        write_json(path, attributed_segments, turns)
        print(f"Wrote {path}", file=sys.stderr)

    if turns:
        spk_dur: dict[str, float] = {}
        for turn in turns:
            spk_dur[turn.speaker] = spk_dur.get(turn.speaker, 0.0) + (
                turn.end - turn.start
            )
        print("\nSpeaker totals:", file=sys.stderr)
        for spk, dur in sorted(spk_dur.items(), key=lambda x: -x[1]):
            print(f"  {spk}: {dur / 60:.1f} min", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
