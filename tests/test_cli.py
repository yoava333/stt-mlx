import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stt_mlx import cli
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

    assert cli.diarize(Path("silent.wav"), None) == cli.DiarizationResult([], {})


def test_transcribe_normalizes_whisper_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stt_mlx.whisper_performance.enable_fast_attention", lambda: None
    )
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
    transcribe_kwargs: dict[str, object] = {}

    def fake_transcribe(*args: object, **kwargs: object) -> object:
        transcribe_kwargs.update(kwargs)
        return whisper_result

    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper",
        SimpleNamespace(transcribe=fake_transcribe),
    )

    assert cli.transcribe(Path("audio.wav"), "model", "en") == [
        cli.TranscriptSegment(0.0, 1.25, "hello")
    ]
    assert transcribe_kwargs["verbose"] is False


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

    assert cli.diarize(Path("audio.wav"), None) == cli.DiarizationResult(
        [cli.SpeakerTurn(0.0, 2.5, "SPEAKER_01")], {}
    )


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
        cli.diarize(Path("audio.wav"), None)


def test_diarize_configures_exact_speaker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spectral = SimpleNamespace(cluster=SimpleNamespace(k=None), min_cluster_size=4)
    diarizer = SimpleNamespace(
        spectral_cluster=spectral,
        umap_hdbscan_cluster=SimpleNamespace(),
        diarize=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "senko",
        SimpleNamespace(Diarizer=lambda **kwargs: diarizer),
    )

    cli.diarize(Path("audio.wav"), 3)

    assert spectral.cluster.k == 3
    assert spectral.min_cluster_size == 0
    assert diarizer.umap_hdbscan_cluster is spectral


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
        lambda wav, num_speakers: cli.DiarizationResult(
            [cli.SpeakerTurn(0.0, 1.0, "SPEAKER_01")], {}
        ),
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


def test_parse_args_uses_hebrew_for_default_hebrew_model() -> None:
    args = cli.parse_args(["audio.wav"])

    assert args.language == "he"


def test_parse_args_accepts_speaker_count() -> None:
    args = cli.parse_args(["audio.wav", "--num-speakers", "3"])

    assert args.num_speakers == 3


def test_parse_args_accepts_trimmed_speaker_name() -> None:
    args = cli.parse_args(["voice.wav", "--enroll-speaker", "  Yoav  "])

    assert args.enroll_speaker == "Yoav"


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_parse_args_rejects_invalid_speaker_count(value: str) -> None:
    with pytest.raises(SystemExit) as error:
        cli.parse_args(["audio.wav", "--num-speakers", value])

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
        lambda wav, num_speakers: cli.DiarizationResult(
            [cli.SpeakerTurn(0.0, 1.5, "SPEAKER_01")], {}
        ),
    )

    assert cli.main([str(audio), "--no-transcribe"]) == 0
    rttm = tmp_path / "speech.rttm"
    assert rttm.exists()
    assert rttm.read_text().startswith(
        "SPEAKER speech 1 0.000 1.500 <NA> <NA> SPEAKER_01"
    )


def test_enroll_speaker_saves_single_centroid_without_transcribing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "voice.wav"
    audio.touch()
    profiles = tmp_path / "profiles.json"
    requested_speaker_counts: list[int | None] = []
    monkeypatch.setattr(cli, "default_profiles_path", lambda: profiles)
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *args: pytest.fail("enrollment should not transcribe"),
    )

    def fake_diarize(wav: Path, num_speakers: int | None) -> cli.DiarizationResult:
        requested_speaker_counts.append(num_speakers)
        return cli.DiarizationResult(
            [cli.SpeakerTurn(0.0, 5.0, "SPEAKER_01")],
            {"SPEAKER_01": [1.0, 0.0]},
        )

    monkeypatch.setattr(cli, "diarize", fake_diarize)

    assert cli.main([str(audio), "--enroll-speaker", "Yoav"]) == 0
    assert requested_speaker_counts == [1]
    stored = json.loads(profiles.read_text())
    assert stored["profiles"][0]["tag"] == "Yoav"
    assert stored["profiles"][0]["embeddings"] == [[1.0, 0.0]]
    assert not (tmp_path / "voice.diarized.json").exists()


def test_main_replaces_matching_speaker_tags_with_enrolled_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "meeting.wav"
    audio.touch()
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {"profiles": [{"id": "yoav-id", "tag": "Yoav", "embeddings": [[1.0, 0.0]]}]}
        )
    )
    monkeypatch.setattr(cli, "default_profiles_path", lambda: profiles)
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *args: [cli.TranscriptSegment(0.0, 1.0, "hello")],
    )
    monkeypatch.setattr(
        cli,
        "diarize",
        lambda wav, num_speakers: cli.DiarizationResult(
            [cli.SpeakerTurn(0.0, 1.0, "SPEAKER_01")],
            {"SPEAKER_01": [0.99, 0.01]},
        ),
    )

    assert cli.main([str(audio), "--formats", "json"]) == 0
    output = json.loads((tmp_path / "meeting.diarized.json").read_text())
    assert output["segments"][0]["speaker"] == "Yoav"
    assert output["diarization"][0]["speaker"] == "Yoav"


def test_main_creates_and_uses_anonymous_profile_for_unmatched_speaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "meeting.wav"
    audio.touch()
    profiles = tmp_path / "profiles.json"
    monkeypatch.setattr(cli, "default_profiles_path", lambda: profiles)
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *args: [cli.TranscriptSegment(0.0, 1.0, "hello")],
    )
    monkeypatch.setattr(
        cli,
        "diarize",
        lambda wav, num_speakers: cli.DiarizationResult(
            [cli.SpeakerTurn(0.0, 1.0, "SPEAKER_01")],
            {"SPEAKER_01": [1.0, 0.0]},
        ),
    )

    assert cli.main([str(audio), "--formats", "json"]) == 0

    profile = json.loads(profiles.read_text())["profiles"][0]
    assert profile["tag"].startswith("speaker-")
    output = json.loads((tmp_path / "meeting.diarized.json").read_text())
    assert output["segments"][0]["speaker"] == profile["tag"]
    assert output["diarization"][0]["speaker"] == profile["tag"]

    assert cli.main([str(audio), "--formats", "json"]) == 0
    stored_profiles = json.loads(profiles.read_text())["profiles"]
    assert len(stored_profiles) == 1
    assert stored_profiles[0]["id"] == profile["id"]


def test_profiles_list_prints_ids_tags_and_sample_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "id": "yoav-id",
                        "tag": "Yoav",
                        "embeddings": [[1.0, 0.0], [0.9, 0.1]],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(cli, "default_profiles_path", lambda: profiles)

    assert cli.main(["profiles", "list"]) == 0
    assert capsys.readouterr().out == "ID\tTAG\tSAMPLES\nyoav-id\tYoav\t2\n"


def test_profiles_rename_delete_and_combine_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "profiles": [
                    {"id": "a", "tag": "Person A", "embeddings": [[1.0, 0.0]]},
                    {"id": "b", "tag": "Person B", "embeddings": [[0.9, 0.1]]},
                    {"id": "c", "tag": "Wrong", "embeddings": [[0.0, 1.0]]},
                ]
            }
        )
    )
    monkeypatch.setattr(cli, "default_profiles_path", lambda: profiles)

    assert cli.main(["profiles", "rename", "a", "Yoav"]) == 0
    assert cli.main(["profiles", "combine", "Person B", "Yoav"]) == 0
    assert cli.main(["profiles", "delete", "Wrong"]) == 0

    stored = json.loads(profiles.read_text())["profiles"]
    assert stored == [
        {
            "id": "a",
            "tag": "Yoav",
            "embeddings": [[1.0, 0.0], [0.9, 0.1]],
        }
    ]


def test_profiles_command_reports_missing_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "default_profiles_path", lambda: tmp_path / "missing.json")

    assert cli.main(["profiles", "delete", "missing"]) == 2
    assert "speaker profile not found" in capsys.readouterr().err


def test_main_rejects_disabling_both_operations(tmp_path: Path) -> None:
    audio = tmp_path / "speech.wav"
    audio.touch()

    assert cli.main([str(audio), "--no-transcribe", "--no-diarize"]) == 2
