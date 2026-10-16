"""stt — transcribe + diarize an audio file using mlx_whisper and senko."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from stt_mlx.speaker_profiles import (
    CONFIDENT_MATCH_THRESHOLD,
    SpeakerProfile,
    combine_profiles,
    create_anonymous_profile,
    default_profiles_path,
    delete_profile,
    enroll_profile,
    load_profiles,
    match_speakers,
    profile_score,
    rename_profile,
)

DEFAULT_MODEL = "mlx-community/ivrit-ai-whisper-large-v3-mlx"
PARAKEET_DEFAULT_MODEL = "thewh1teagle/parakeet-tdt-0.6b-v3-he"
DEFAULT_LANGUAGE = "he"
ENGINES = ("whisper", "parakeet")
MIN_PROFILE_SPEECH_SECONDS = 180.0
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


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    turns: list[SpeakerTurn]
    centroids: dict[str, list[float]]


def to_16k_mono_wav(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required and not on PATH")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
    )


def transcribe(wav: Path, model: str, language: str) -> list[TranscriptSegment]:
    import mlx_whisper

    from stt_mlx.whisper_performance import enable_fast_attention

    enable_fast_attention()
    result = mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=model, language=language, verbose=False
    )
    return [
        TranscriptSegment(
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]).strip(),
        )
        for segment in result["segments"]
    ]


def transcribe_with_parakeet(
    wav: Path, model: str, language: str
) -> list[TranscriptSegment]:
    from stt_mlx.parakeet_transcribe import transcribe_parakeet

    return transcribe_parakeet(wav, model, language)


def resolve_model(engine: str, model: str | None) -> str:
    if model is not None:
        return model
    return PARAKEET_DEFAULT_MODEL if engine == "parakeet" else DEFAULT_MODEL


def _speaker_turns(segments: object, name: str) -> list[SpeakerTurn]:
    if not isinstance(segments, list):
        raise TypeError(f"senko {name} segments must be a list")
    turns: list[SpeakerTurn] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise TypeError(f"senko {name} segments must be mappings")
        turns.append(
            SpeakerTurn(
                start=float(segment["start"]),
                end=float(segment["end"]),
                speaker=str(segment["speaker"]),
            )
        )
    return turns


def diarize(
    wav: Path, num_speakers: int | None, profiles: Sequence[SpeakerProfile]
) -> DiarizationResult:
    import numpy as np
    import senko

    from stt_mlx.clustering import cluster_windows
    from stt_mlx.voiceprints import voiceprints, window_labels

    diarizer = senko.Diarizer(device="auto", warmup=True, quiet=False)
    spectral = diarizer.spectral_cluster
    if num_speakers is not None:
        spectral.cluster.k = num_speakers
        spectral.min_cluster_size = 0
        # The requested count is exact, so no cluster is dropped as a fragment.
        profiles = []

    # Senko density-clusters long recordings, which merges similar voices.
    # Spectral clustering separates them, so use it for every recording.
    def clusterer(embeddings: np.ndarray) -> np.ndarray:
        return cluster_windows(embeddings, spectral, profiles)

    diarizer.spectral_cluster = clusterer
    diarizer.umap_hdbscan_cluster = clusterer

    # Senko reports only a plain mean embedding per speaker. Capture the
    # per-window embeddings it clusters so voiceprints can skip noisy windows.
    windows: list[tuple[object, object]] = []
    cluster = diarizer._perform_clustering

    def capture_windows(embeddings: object, subsegments: object) -> object:
        windows.append((embeddings, subsegments))
        return cluster(embeddings, subsegments)

    diarizer._perform_clustering = capture_windows
    result = diarizer.diarize(str(wav), generate_colors=False)
    if result is None:
        return DiarizationResult([], {})
    turns = _speaker_turns(result["merged_segments"], "merged")
    raw_turns = _speaker_turns(result["raw_segments"], "raw")
    if not windows:
        raise RuntimeError("senko did not report window embeddings")
    embeddings, subsegments = windows[-1]
    labels = window_labels(
        [(float(start), float(end)) for start, end in subsegments],
        [(turn.start, turn.end, turn.speaker) for turn in raw_turns],
    )
    centroids = voiceprints(np.asarray(embeddings, dtype=np.float64), labels)
    return DiarizationResult(turns, centroids)


def rename_speakers(
    turns: list[SpeakerTurn], names: Mapping[str, str]
) -> list[SpeakerTurn]:
    return [
        SpeakerTurn(turn.start, turn.end, names.get(turn.speaker, turn.speaker))
        for turn in turns
    ]


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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def speaker_name(value: str) -> str:
    name = value.strip()
    if not name or "\n" in name or "\r" in name:
        raise argparse.ArgumentTypeError("must be a non-empty single-line name")
    return name


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="stt",
        description="Transcribe and diarize an audio file (mlx_whisper + senko).",
        epilog="Manage enrolled voices with: stt profiles --help",
    )
    p.add_argument(
        "audio", type=Path, help="Input audio file (any format ffmpeg accepts)"
    )
    p.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        default=None,
        help="Output file prefix (default: input file's path without extension)",
    )
    p.add_argument(
        "--engine",
        choices=ENGINES,
        default="parakeet",
        help="Transcription backend (default: parakeet)",
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "Transcription model HF repo. Defaults to "
            f"{DEFAULT_MODEL} for whisper and {PARAKEET_DEFAULT_MODEL} for parakeet."
        ),
    )
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )
    p.add_argument(
        "--num-speakers",
        type=positive_int,
        default=None,
        help="Exact number of speakers (default: detect automatically)",
    )
    p.add_argument(
        "--enroll-speaker",
        type=speaker_name,
        metavar="NAME",
        help="Save this single-speaker recording as NAME for future identification",
    )
    p.add_argument("--no-diarize", action="store_true", help="Skip diarization")
    p.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip transcription (diarization only)",
    )
    p.add_argument(
        "--formats",
        type=parse_formats,
        default=parse_formats("txt,srt,json"),
        help="Comma-separated output formats: txt,srt,json (default: all three)",
    )
    return p.parse_args(argv)


def parse_profiles_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="stt profiles", description="Manage enrolled speaker profiles."
    )
    commands = parser.add_subparsers(dest="profiles_command", required=True)
    commands.add_parser("list", help="List all enrolled speaker profiles")

    rename = commands.add_parser("rename", help="Change a profile's speaker tag")
    rename.add_argument("profile", help="Profile ID or exact speaker tag")
    rename.add_argument("new_tag", type=speaker_name, help="New unique speaker tag")

    delete = commands.add_parser("delete", help="Delete a speaker profile")
    delete.add_argument("profile", help="Profile ID or exact speaker tag")

    combine = commands.add_parser(
        "combine", help="Merge a source profile into a target profile"
    )
    combine.add_argument("source", help="Source profile ID or exact speaker tag")
    combine.add_argument("target", help="Target profile ID or exact speaker tag")
    return parser.parse_args(argv)


def manage_profiles(argv: list[str]) -> int:
    args = parse_profiles_args(argv)
    path = default_profiles_path()
    try:
        if args.profiles_command == "list":
            profiles = load_profiles(path)
            if not profiles:
                print("No speaker profiles enrolled.")
                return 0
            print("ID\tTAG\tSAMPLES")
            for profile in profiles:
                print(f"{profile.id}\t{profile.tag}\t{len(profile.embeddings)}")
            return 0
        if args.profiles_command == "rename":
            profile = rename_profile(path, args.profile, args.new_tag)
            print(f"Renamed profile {profile.id} to {profile.tag}")
            return 0
        if args.profiles_command == "delete":
            profile = delete_profile(path, args.profile)
            print(f"Deleted profile {profile.id} ({profile.tag})")
            return 0
        if args.profiles_command == "combine":
            profile = combine_profiles(path, args.source, args.target)
            print(
                f"Combined profiles as {profile.id} ({profile.tag}); "
                f"{len(profile.embeddings)} samples"
            )
            return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"unknown profiles command: {args.profiles_command}")


def main(argv: list[str] | None = None) -> int:
    raw_argv = argv if argv is not None else sys.argv[1:]
    if raw_argv and raw_argv[0] == "profiles":
        return manage_profiles(raw_argv[1:])
    args = parse_args(raw_argv)

    if args.no_diarize and args.no_transcribe:
        print(
            "error: --no-diarize and --no-transcribe both set; nothing to do",
            file=sys.stderr,
        )
        return 2
    if args.enroll_speaker and args.no_diarize:
        print(
            "error: --enroll-speaker cannot be used with --no-diarize", file=sys.stderr
        )
        return 2

    audio_in = args.audio.expanduser().resolve()
    if not audio_in.exists():
        print(f"error: not found: {audio_in}", file=sys.stderr)
        return 2

    out_prefix = args.output_prefix if args.output_prefix else audio_in.with_suffix("")

    profiles: list[SpeakerProfile] = []
    if not args.no_diarize and not args.enroll_speaker:
        try:
            profiles = load_profiles(default_profiles_path())
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"error: could not load speaker profiles: {error}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio_16k.wav"
        print(f"Converting → {wav}", file=sys.stderr, flush=True)
        to_16k_mono_wav(audio_in, wav)

        segments: list[TranscriptSegment] = []
        if not args.no_transcribe and not args.enroll_speaker:
            model = resolve_model(args.engine, args.model)
            print(
                f"Transcribing with {args.engine} ({model})...",
                file=sys.stderr,
                flush=True,
            )
            transcriber = (
                transcribe_with_parakeet if args.engine == "parakeet" else transcribe
            )
            segments = transcriber(wav, model, args.language)
            print(f"  {len(segments)} segments", file=sys.stderr, flush=True)

        diarization = DiarizationResult([], {})
        if not args.no_diarize:
            print("Diarizing with senko...", file=sys.stderr, flush=True)
            speaker_count = 1 if args.enroll_speaker else args.num_speakers
            diarization = diarize(wav, speaker_count, profiles)
            print(
                f"  {len(diarization.turns)} speaker turns",
                file=sys.stderr,
                flush=True,
            )

    if args.enroll_speaker:
        if len(diarization.centroids) != 1:
            print(
                "error: enrollment audio must contain detectable speech from one speaker",
                file=sys.stderr,
            )
            return 2
        embedding = next(iter(diarization.centroids.values()))
        profiles_path = default_profiles_path()
        try:
            profile = enroll_profile(profiles_path, args.enroll_speaker, embedding)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"error: could not save speaker profile: {error}", file=sys.stderr)
            return 2
        print(
            f"Enrolled {profile.tag} as profile {profile.id} "
            f"({len(profile.embeddings)} samples) in {profiles_path}",
            file=sys.stderr,
        )
        return 0

    turns = diarization.turns
    if turns and diarization.centroids:
        profiles_path = default_profiles_path()
        names = match_speakers(diarization.centroids, profiles)
        # A loose match names the speaker only tentatively, and learning from
        # it could teach the profile another person's voice.
        uncertain = {
            speaker
            for speaker, tag in names.items()
            if profile_score(
                diarization.centroids[speaker],
                next(profile for profile in profiles if profile.tag == tag),
            )
            < CONFIDENT_MATCH_THRESHOLD
        }
        speech: dict[str, float] = {}
        for turn in turns:
            speech[turn.speaker] = speech.get(turn.speaker, 0.0) + (
                turn.end - turn.start
            )
        for speaker, centroid in diarization.centroids.items():
            # Short clusters are usually crosstalk or a split-off fragment of
            # another speaker; their voiceprints would pollute the profiles.
            if speech.get(speaker, 0.0) < MIN_PROFILE_SPEECH_SECONDS:
                continue
            try:
                if speaker in names:
                    if speaker not in uncertain:
                        enroll_profile(profiles_path, names[speaker], centroid)
                    continue
                profile = create_anonymous_profile(profiles_path, centroid)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                print(
                    f"error: could not update speaker profiles: {error}",
                    file=sys.stderr,
                )
                return 2
            names[speaker] = profile.tag
            print(
                f"  Created profile {profile.id} ({profile.tag}) for {speaker}",
                file=sys.stderr,
            )
        for speaker in uncertain:
            names[speaker] = f"{names[speaker]}?"
        turns = rename_speakers(turns, names)
        for speaker, name in sorted(names.items()):
            print(f"  Identified {speaker} as {name}", file=sys.stderr)

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
