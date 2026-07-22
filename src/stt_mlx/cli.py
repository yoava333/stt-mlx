"""stt — transcribe + diarize an audio file using mlx_whisper and senko."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-mlx"


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


def transcribe(wav: Path, model: str, language: str | None) -> list[dict]:
    import mlx_whisper

    kwargs: dict[str, Any] = {"path_or_hf_repo": model}
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(str(wav), **kwargs)
    return [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
        for s in result["segments"]
    ]


def diarize(wav: Path) -> list[tuple[float, float, str]]:
    import senko

    diarizer = senko.Diarizer(device="auto", warmup=True, quiet=False)
    result = diarizer.diarize(str(wav), generate_colors=False)
    turns: list[tuple[float, float, str]] = []
    for s in result["merged_segments"]:
        if isinstance(s, dict):
            turns.append((float(s["start"]), float(s["end"]), str(s["speaker"])))
        else:
            turns.append((float(s.start), float(s.end), str(s.speaker)))
    return turns


def assign_speaker(seg: dict, turns: list[tuple[float, float, str]]) -> str:
    best_overlap = 0.0
    best_spk = "UNKNOWN"
    for ts, te, spk in turns:
        overlap = max(0.0, min(seg["end"], te) - max(seg["start"], ts))
        if overlap > best_overlap:
            best_overlap = overlap
            best_spk = spk
    return best_spk


def fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def fmt_srt(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_txt(path: Path, segments: list[dict]) -> None:
    lines: list[str] = []
    current_spk: str | None = None
    for seg in segments:
        if seg["speaker"] != current_spk:
            if current_spk is not None:
                lines.append("")
            lines.append(f"=== {seg['speaker']} ===")
            current_spk = seg["speaker"]
        lines.append(f"[{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}] {seg['text']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(path: Path, segments: list[dict]) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}")
        lines.append(f"{seg['speaker']}: {seg['text']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, segments: list[dict], turns: list[tuple[float, float, str]]) -> None:
    payload = {
        "segments": segments,
        "diarization": [{"start": s, "end": e, "speaker": spk} for s, e, spk in turns],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "--formats", default="txt,srt,json",
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
    formats = {f.strip() for f in args.formats.split(",") if f.strip()}

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio_16k.wav"
        print(f"Converting → {wav}", file=sys.stderr, flush=True)
        to_16k_mono_wav(audio_in, wav)

        segments: list[dict] = []
        if not args.no_transcribe:
            print(f"Transcribing with {args.model}...", file=sys.stderr, flush=True)
            segments = transcribe(wav, args.model, args.language)
            print(f"  {len(segments)} segments", file=sys.stderr, flush=True)

        turns: list[tuple[float, float, str]] = []
        if not args.no_diarize:
            print("Diarizing with senko...", file=sys.stderr, flush=True)
            turns = diarize(wav)
            print(f"  {len(turns)} speaker turns", file=sys.stderr, flush=True)

    if turns:
        for seg in segments:
            seg["speaker"] = assign_speaker(seg, turns)
    elif segments:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"

    if not segments:
        rttm_path = out_prefix.with_suffix(".rttm")
        with rttm_path.open("w", encoding="utf-8") as f:
            audio_id = audio_in.stem
            for s, e, spk in turns:
                f.write(f"SPEAKER {audio_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n")
        print(f"Wrote {rttm_path}", file=sys.stderr)
        return 0

    if "txt" in formats:
        path = out_prefix.with_suffix(".diarized.txt")
        write_txt(path, segments)
        print(f"Wrote {path}", file=sys.stderr)
    if "srt" in formats:
        path = out_prefix.with_suffix(".diarized.srt")
        write_srt(path, segments)
        print(f"Wrote {path}", file=sys.stderr)
    if "json" in formats:
        path = out_prefix.with_suffix(".diarized.json")
        write_json(path, segments, turns)
        print(f"Wrote {path}", file=sys.stderr)

    if turns:
        spk_dur: dict[str, float] = {}
        for s, e, spk in turns:
            spk_dur[spk] = spk_dur.get(spk, 0.0) + (e - s)
        print("\nSpeaker totals:", file=sys.stderr)
        for spk, dur in sorted(spk_dur.items(), key=lambda x: -x[1]):
            print(f"  {spk}: {dur / 60:.1f} min", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
