# Code Quality Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ambiguous pipeline state and untyped dependency data with a tested, reproducible, explicit CLI contract.

**Architecture:** Keep the application in `src/stt_mlx/cli.py`. Normalize third-party results into immutable records at adapter boundaries, derive attributed transcript records without mutation, and select outputs from requested operations rather than result cardinality.

**Tech Stack:** Python 3.12–3.13, argparse, dataclasses, pytest, uv/hatchling

## Global Constraints

- Preserve the existing default model and transcription arguments.
- Preserve default output filenames and normal transcript contents.
- Treat empty transcription and no-speech diarization as successful empty results.
- Keep the concurrency design out of scope.
- Track `uv.lock` for reproducible dependency resolution.

---

### Task 1: Establish the test harness and timestamp contract

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_cli.py`
- Modify: `src/stt_mlx/cli.py:77-82`
- Track: `uv.lock`

**Interfaces:**
- Consumes: existing `fmt_srt(t: float) -> str`
- Produces: validated SRT formatting and a reusable pytest suite

- [ ] **Step 1: Add pytest as a development dependency**

Run: `uv add --dev pytest`

Expected: `pyproject.toml` gains a development dependency group and `uv.lock` is refreshed.

- [ ] **Step 2: Write failing boundary tests**

```python
import pytest

from stt_mlx.cli import fmt_srt


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(59.9996, "00:01:00,000"), (3599.9996, "01:00:00,000")],
)
def test_fmt_srt_carries_rounded_milliseconds(seconds: float, expected: str) -> None:
    assert fmt_srt(seconds) == expected
```

- [ ] **Step 3: Run the focused test and verify RED**

Run: `uv run pytest tests/test_cli.py -v`

Expected: FAIL because the current formatter emits a `,1000` millisecond field.

- [ ] **Step 4: Implement total-millisecond decomposition**

```python
def fmt_srt(t: float) -> str:
    total_ms = round(t * 1000)
    total_seconds, ms = divmod(total_ms, 1000)
    total_minutes, s = divmod(total_seconds, 60)
    h, m = divmod(total_minutes, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run: `uv run pytest tests/test_cli.py -v`

Expected: all timestamp tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/test_cli.py src/stt_mlx/cli.py
git commit -m "test: establish CLI regression coverage"
```

### Task 2: Normalize dependency boundaries and remove segment mutation

**Files:**
- Modify: `src/stt_mlx/cli.py:12-67,85-113`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `TranscriptSegment`, `AttributedSegment`, `SpeakerTurn`, `assign_speakers(segments, turns) -> list[AttributedSegment]`
- Consumes: MLX Whisper mappings and Senko mappings or `None`

- [ ] **Step 1: Write failing adapter and attribution tests**

```python
def test_diarize_treats_none_as_no_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(Diarizer=lambda **kwargs: SimpleNamespace(
        diarize=lambda *args, **kwargs: None
    ))
    monkeypatch.setitem(sys.modules, "senko", fake)
    assert diarize(Path("silent.wav")) == []


def test_assign_speakers_returns_complete_values_without_mutating_input() -> None:
    segments = [TranscriptSegment(0.0, 2.0, "hello")]
    result = assign_speakers(segments, [SpeakerTurn(0.0, 2.0, "SPEAKER_01")])
    assert result == [AttributedSegment(0.0, 2.0, "hello", "SPEAKER_01")]
    assert segments == [TranscriptSegment(0.0, 2.0, "hello")]
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_cli.py -k 'diarize or assign_speakers' -v`

Expected: FAIL because silence raises `TypeError` and the typed records/functions do not exist.

- [ ] **Step 3: Add immutable records and strict adapters**

```python
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


def transcribe(wav: Path, model: str, language: str | None) -> list[TranscriptSegment]:
    import mlx_whisper

    kwargs: dict[str, object] = {"path_or_hf_repo": model}
    if language:
        kwargs["language"] = language
    result = mlx_whisper.transcribe(str(wav), **kwargs)
    return [
        TranscriptSegment(float(raw["start"]), float(raw["end"]), str(raw["text"]).strip())
        for raw in result["segments"]
    ]


def diarize(wav: Path) -> list[SpeakerTurn]:
    import senko

    result = senko.Diarizer(device="auto", warmup=True, quiet=False).diarize(
        str(wav), generate_colors=False
    )
    if result is None:
        return []
    turns: list[SpeakerTurn] = []
    for raw in result["merged_segments"]:
        if not isinstance(raw, Mapping):
            raise TypeError("senko merged segments must be mappings")
        turns.append(SpeakerTurn(float(raw["start"]), float(raw["end"]), str(raw["speaker"])))
    return turns


def assign_speakers(
    segments: list[TranscriptSegment], turns: list[SpeakerTurn]
) -> list[AttributedSegment]:
    attributed: list[AttributedSegment] = []
    for segment in segments:
        speaker = max(
            turns,
            key=lambda turn: max(
                0.0, min(segment.end, turn.end) - max(segment.start, turn.start)
            ),
            default=None,
        )
        overlap = 0.0 if speaker is None else max(
            0.0, min(segment.end, speaker.end) - max(segment.start, speaker.start)
        )
        attributed.append(
            AttributedSegment(
                segment.start,
                segment.end,
                segment.text,
                speaker.speaker if overlap > 0.0 else "UNKNOWN",
            )
        )
    return attributed
```

Use `list[AttributedSegment]` in the TXT/SRT writer signatures and replace dictionary indexing with attribute access. Keep JSON stable with explicit serialization:

```python
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
```

- [ ] **Step 4: Run the suite and verify GREEN**

Run: `uv run pytest tests/test_cli.py -v`

Expected: all adapter, attribution, writer, and timestamp tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/stt_mlx/cli.py tests/test_cli.py
git commit -m "refactor: normalize transcription data boundaries"
```

### Task 3: Make CLI mode and output selection explicit

**Files:**
- Modify: `src/stt_mlx/cli.py:116-206`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `parse_formats(value: str) -> frozenset[OutputFormat]`, `output_path(prefix: Path, suffix: str) -> Path`
- Consumes: parsed `argparse.Namespace`, typed segments, typed turns

- [ ] **Step 1: Write failing validation, prefix, and orchestration tests**

```python
def test_parse_formats_rejects_unknown_and_empty_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_formats("bogus")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_formats(" , ")


def test_parse_formats_deduplicates_valid_values() -> None:
    assert parse_formats("txt,json,txt") == frozenset({"txt", "json"})


def test_output_path_preserves_literal_prefix() -> None:
    assert output_path(Path("report.v1"), ".diarized.txt") == Path("report.v1.diarized.txt")


def test_empty_transcription_still_writes_requested_transcript(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "silent.wav"
    audio.touch()
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(cli, "transcribe", lambda *args: [])
    assert cli.main([str(audio), "--no-diarize", "--formats", "txt"]) == 0
    assert (tmp_path / "silent.diarized.txt").exists()
    assert not (tmp_path / "silent.rttm").exists()


def test_diarization_only_writes_rttm(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "speech.wav"
    audio.touch()
    monkeypatch.setattr(cli, "to_16k_mono_wav", lambda src, dst: dst.touch())
    monkeypatch.setattr(
        cli,
        "diarize",
        lambda wav: [SpeakerTurn(0.0, 1.5, "SPEAKER_01")],
    )
    assert cli.main([str(audio), "--no-transcribe"]) == 0
    assert (tmp_path / "speech.rttm").read_text().startswith("SPEAKER speech")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_cli.py -k 'formats or prefix or empty_transcription' -v`

Expected: FAIL because formats are not validated, suffix replacement loses `.v1`, and empty transcription writes RTTM.

- [ ] **Step 3: Implement explicit parsing and mode-driven output**

```python
OutputFormat = Literal["txt", "srt", "json"]
FORMAT_NAMES: dict[str, OutputFormat] = {
    "txt": "txt",
    "srt": "srt",
    "json": "json",
}


def parse_formats(value: str) -> frozenset[OutputFormat]:
    names = {part.strip() for part in value.split(",") if part.strip()}
    unknown = names - FORMAT_NAMES.keys()
    if not names:
        raise argparse.ArgumentTypeError("at least one output format is required")
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown output format(s): {', '.join(sorted(unknown))}")
    return frozenset(FORMAT_NAMES[name] for name in names)


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}{suffix}"
```

Configure the argument with `type=parse_formats` and `default=parse_formats("txt,srt,json")`. In `main`, use this explicit structure:

```python
if args.no_transcribe:
    write_rttm(output_path(out_prefix, ".rttm"), audio_in.stem, turns)
    return 0

segments_with_speakers = assign_speakers(segments, turns)
if not turns:
    segments_with_speakers = [
        AttributedSegment(segment.start, segment.end, segment.text, "SPEAKER_00")
        for segment in segments
    ]

if "txt" in args.formats:
    write_txt(output_path(out_prefix, ".diarized.txt"), segments_with_speakers)
if "srt" in args.formats:
    write_srt(output_path(out_prefix, ".diarized.srt"), segments_with_speakers)
if "json" in args.formats:
    write_json(output_path(out_prefix, ".diarized.json"), segments_with_speakers, turns)
```

Extract the existing RTTM loop into `write_rttm(path: Path, audio_id: str, turns: list[SpeakerTurn]) -> None` so orchestration and rendering remain separate.

- [ ] **Step 4: Run all tests and verify GREEN**

Run: `uv run pytest -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/stt_mlx/cli.py tests/test_cli.py
git commit -m "fix: make CLI output state explicit"
```

### Task 4: Final reproducibility and quality verification

**Files:**
- Modify: `.gitignore`
- Verify: `uv.lock`, `pyproject.toml`, `src/stt_mlx/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: completed CLI and test suite
- Produces: tracked lockfile and verified repository state

- [ ] **Step 1: Stop ignoring the lockfile**

Remove the exact `uv.lock` line from `.gitignore` while preserving all other ignore rules.

- [ ] **Step 2: Verify lock reproducibility**

Run: `uv lock --check`

Expected: exit 0 and no lockfile changes.

- [ ] **Step 3: Run the full test suite and package build**

Run: `uv run pytest -v && uv build`

Expected: all tests pass and both source distribution and wheel build successfully.

- [ ] **Step 4: Check repository hygiene**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intended remediation files are modified or untracked.

- [ ] **Step 5: Commit**

```bash
git add .gitignore uv.lock
git commit -m "build: track reproducible dependency lock"
```
