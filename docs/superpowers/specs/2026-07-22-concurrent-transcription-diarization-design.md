# Concurrent Transcription and Diarization Design

## Goal

Reduce end-to-end runtime without changing the default Whisper model or its decoding behavior.

## Baseline

The benchmark input is `/Users/yoav/Downloads/radio-show/kan_bet_20260702_slice_1340-2446.mp3`, an 11:06 MP3.

- Existing sequential CLI: 89.27 seconds
- Concurrent prototype using the same large model: 67.08 seconds
- Improvement: 22.19 seconds, or 24.9%
- Result shape in both runs: 151 transcription segments and 27 diarization turns

## Design

Keep `mlx-community/ivrit-ai-whisper-large-v3-mlx` as the default model and leave all transcription arguments unchanged. After the single FFmpeg conversion, submit transcription and diarization to a two-worker `ThreadPoolExecutor` when both operations are enabled. Wait for both results before assigning speakers and writing output.

When `--no-diarize` or `--no-transcribe` is supplied, run the requested operation directly without creating an executor. This keeps single-stage behavior simple and avoids unnecessary concurrency overhead.

## Components and Data Flow

`main` remains responsible for orchestration:

1. Validate CLI arguments and convert the source to one 16 kHz mono WAV.
2. Run `transcribe` and `diarize` concurrently when both are requested.
3. Preserve the existing sequential single-operation branches.
4. Assign the maximum-overlap speaker to each transcription segment.
5. Write the requested output formats exactly as before.

The existing `transcribe`, `diarize`, speaker assignment, and writer functions retain their interfaces.

## Error Handling

Calling `Future.result()` propagates the original exception from either worker. The executor context waits for the other worker to finish before temporary audio is deleted, preventing a running worker from losing its input file. Existing conversion and CLI validation behavior remains unchanged.

## Testing

Add orchestration tests that replace the expensive model functions with controlled callables and verify:

- Both full-pipeline operations overlap in time.
- `--no-diarize` calls transcription only.
- `--no-transcribe` calls diarization only.
- The existing output behavior remains unchanged.

Run the complete test suite, then benchmark the supplied MP3 and compare segment/turn counts against the baseline. No transcript-quality trade-off is accepted: the default model and transcription options must remain unchanged.
