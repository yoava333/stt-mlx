# Code Quality Remediation Design

## Goal

Resolve every issue identified by the repository-wide thermo-nuclear code quality review without changing the normal transcription and diarization behavior.

## Architecture

Keep the application in `src/stt_mlx/cli.py` while it remains small. Introduce explicit typed records for transcript segments and speaker turns, and normalize third-party results into those records inside the `transcribe` and `diarize` adapters. The rest of the application must not inspect alternate third-party shapes or depend on loosely typed dictionaries.

Drive output behavior from the operations requested by the user. An empty transcription is a valid transcription result and must not be interpreted as diarization-only mode. Diarization-only mode writes RTTM; every mode that requested transcription writes the selected transcript formats, even when the transcript is empty.

## CLI and Output Contract

- `--formats` accepts a comma-separated subset of `txt`, `srt`, and `json`.
- Unknown formats and an empty format selection are argument errors with exit status 2.
- Duplicate formats are harmless and produce each selected file once.
- `--output-prefix` is a literal prefix. For example, `-o report.v1` produces `report.v1.diarized.txt` rather than replacing `.v1`.
- When no output prefix is supplied, the input filename without its final extension remains the default prefix.
- Existing default output names remain unchanged.
- SRT timestamps are derived from rounded total milliseconds so carries across millisecond, second, and minute boundaries are valid.

## Dependency Boundaries

`transcribe` converts MLX Whisper segment mappings into typed transcript segments. `diarize` converts Senko's merged segment mappings into typed speaker turns. Senko's documented no-speech result, `None`, becomes an empty list. Unexpected non-mapping segment shapes are rejected rather than accepted through speculative attribute access.

The repository will track `uv.lock`. This pins the Git-backed Senko dependency and all transitive dependencies for reproducible application installs while leaving the declared dependency source unchanged.

## Pipeline State

The requested flags are the source of truth:

- Both operations enabled: transcribe and diarize, attribute speakers, then write selected transcript formats.
- `--no-diarize`: transcribe, assign the existing `SPEAKER_00` fallback, then write selected transcript formats.
- `--no-transcribe`: diarize and write RTTM, ignoring transcript format selection because no transcript is requested.
- Both disabled: reject the invocation as today.

Speaker attribution returns new complete segment values rather than adding a `speaker` key to partially shaped dictionaries in place.

## Error Handling

Argument validation occurs in `argparse`, before conversion or model loading. Empty Senko results and empty transcription results are successful domain outcomes. Third-party exceptions and FFmpeg failures continue to propagate; changing their presentation is outside this remediation.

## Testing

Add pytest as a development dependency and create focused unit and orchestration tests covering:

- SRT carry behavior.
- Valid, duplicate, unknown, and empty format selections.
- Literal output-prefix handling.
- Senko `None` normalization and mapping conversion.
- MLX Whisper mapping conversion.
- Speaker attribution without mutation.
- Empty transcription in transcription modes.
- Diarization-only RTTM output.
- Rejection when both operations are disabled.

Expensive model calls and FFmpeg conversion are replaced at the process boundary in orchestration tests. Tests assert observable files and exit codes rather than internal call counts where possible.

## Scope

The prospective concurrent transcription/diarization design is not implemented here. Module splitting, plugin systems, generalized dependency injection, and unrelated CLI changes are intentionally excluded.
