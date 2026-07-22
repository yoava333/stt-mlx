import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import stt_mlx.cli as cli
from stt_mlx.cli import fmt_srt


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (59.9996, "00:01:00,000"),
        (3599.9996, "01:00:00,000"),
    ],
)
def test_fmt_srt_carries_rounded_milliseconds(seconds: float, expected: str) -> None:
    assert fmt_srt(seconds) == expected


def test_diarize_treats_none_as_no_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_senko = SimpleNamespace(
        Diarizer=lambda **kwargs: SimpleNamespace(diarize=lambda *args, **kwargs: None)
    )
    monkeypatch.setitem(sys.modules, "senko", fake_senko)

    assert cli.diarize(Path("silent.wav")) == []


def test_transcribe_normalizes_whisper_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    whisper_result = {
        "text": " hello ",
        "language": "en",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0,
                "end": 1.25,
                "text": " hello ",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.1,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
        ],
    }
    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper",
        SimpleNamespace(transcribe=lambda *args, **kwargs: whisper_result),
    )

    assert cli.transcribe(Path("audio.wav"), "model", "en") == [
        cli.TranscriptSegment(0.0, 1.25, "hello")
    ]


def test_diarize_normalizes_mapping_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "raw_segments": [],
        "raw_speakers_detected": 1,
        "merged_speakers_detected": 1,
        "merged_segments": [{"start": 0, "end": 2.5, "speaker": "SPEAKER_01"}],
        "speaker_centroids": {},
        "timing_stats": {},
        "vad": [(0, 2.5)],
    }
    fake_senko = SimpleNamespace(
        Diarizer=lambda **kwargs: SimpleNamespace(
            diarize=lambda *args, **kwargs: result
        )
    )
    monkeypatch.setitem(sys.modules, "senko", fake_senko)

    assert cli.diarize(Path("audio.wav")) == [cli.SpeakerTurn(0.0, 2.5, "SPEAKER_01")]


def test_diarize_rejects_non_mapping_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "raw_segments": [],
        "raw_speakers_detected": 1,
        "merged_speakers_detected": 1,
        "merged_segments": [SimpleNamespace(start=0, end=2.5, speaker="SPEAKER_01")],
        "speaker_centroids": {},
        "timing_stats": {},
        "vad": [(0, 2.5)],
    }
    fake_senko = SimpleNamespace(
        Diarizer=lambda **kwargs: SimpleNamespace(
            diarize=lambda *args, **kwargs: result
        )
    )
    monkeypatch.setitem(sys.modules, "senko", fake_senko)

    with pytest.raises(TypeError, match="must be mappings"):
        cli.diarize(Path("audio.wav"))


def test_assign_speakers_returns_complete_values_without_mutation() -> None:
    segments = [cli.TranscriptSegment(0.0, 2.0, "hello")]

    result = cli.assign_speakers(segments, [cli.SpeakerTurn(0.0, 2.0, "SPEAKER_01")])

    assert result == [cli.AttributedSegment(0.0, 2.0, "hello", "SPEAKER_01")]
    assert segments == [cli.TranscriptSegment(0.0, 2.0, "hello")]


def test_assign_speakers_uses_unknown_without_positive_overlap() -> None:
    segments = [cli.TranscriptSegment(0.0, 1.0, "hello")]

    result = cli.assign_speakers(segments, [cli.SpeakerTurn(1.0, 2.0, "SPEAKER_01")])

    assert result == [cli.AttributedSegment(0.0, 1.0, "hello", "UNKNOWN")]


def test_write_json_preserves_output_schema(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    cli.write_json(
        path,
        [cli.AttributedSegment(0.0, 1.0, "hello", "SPEAKER_01")],
        [cli.SpeakerTurn(0.0, 1.0, "SPEAKER_01")],
    )

    assert json.loads(path.read_text()) == {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "speaker": "SPEAKER_01",
            }
        ],
        "diarization": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"}],
    }


def test_main_uses_normalized_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "speech.wav"
    audio.touch()
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *args: [cli.TranscriptSegment(0.0, 1.0, "hello")],
    )
    monkeypatch.setattr(
        cli,
        "diarize",
        lambda wav: [cli.SpeakerTurn(0.0, 1.0, "SPEAKER_01")],
    )

    assert cli.main([str(audio), "--formats", "json"]) == 0
    assert json.loads((tmp_path / "speech.diarized.json").read_text())["segments"] == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "speaker": "SPEAKER_01",
        }
    ]


def test_parse_formats_deduplicates_valid_values() -> None:
    assert cli.parse_formats("txt, json,txt") == frozenset({"txt", "json"})


@pytest.mark.parametrize("value", ["bogus", " , "])
def test_parse_formats_rejects_invalid_selections(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_formats(value)


def test_parse_args_rejects_unknown_format() -> None:
    with pytest.raises(SystemExit) as error:
        cli.parse_args(["audio.wav", "--formats", "bogus"])

    assert error.value.code == 2


def test_output_path_preserves_literal_prefix() -> None:
    assert cli.output_path(Path("report.v1"), ".diarized.txt") == Path(
        "report.v1.diarized.txt"
    )


def test_empty_transcription_writes_selected_transcript_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "silent.wav"
    audio.touch()
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(cli, "transcribe", lambda *args: [])

    assert cli.main([str(audio), "--no-diarize", "--formats", "txt"]) == 0
    assert (tmp_path / "silent.diarized.txt").exists()
    assert not (tmp_path / "silent.rttm").exists()


def test_main_preserves_literal_output_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "speech.wav"
    audio.touch()
    output_prefix = tmp_path / "report.v1"
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *args: [cli.TranscriptSegment(0.0, 1.0, "hello")],
    )

    assert (
        cli.main(
            [
                str(audio),
                "--no-diarize",
                "--formats",
                "txt",
                "-o",
                str(output_prefix),
            ]
        )
        == 0
    )
    assert (tmp_path / "report.v1.diarized.txt").exists()
    assert not (tmp_path / "report.diarized.txt").exists()


def test_diarization_only_writes_rttm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "speech.wav"
    audio.touch()
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(
        cli,
        "diarize",
        lambda wav: [cli.SpeakerTurn(0.0, 1.5, "SPEAKER_01")],
    )

    assert cli.main([str(audio), "--no-transcribe"]) == 0
    rttm = tmp_path / "speech.rttm"
    assert rttm.exists()
    assert rttm.read_text().startswith(
        "SPEAKER speech 1 0.000 1.500 <NA> <NA> SPEAKER_01"
    )


def test_main_rejects_disabling_both_operations(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    audio.touch()

    assert cli.main([str(audio), "--no-transcribe", "--no-diarize"]) == 2
