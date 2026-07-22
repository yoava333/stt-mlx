# stt-mlx

Local, MLX-accelerated speech-to-text with optional speaker diarization for Apple Silicon Macs. `stt-mlx` combines [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) with [Senko](https://github.com/narcotic-sh/senko) and writes speaker-attributed transcripts as text, subtitles, or JSON.

## Features

- MLX-accelerated Whisper transcription on Apple Silicon
- Optional speaker diarization
- TXT, SRT, JSON, and RTTM output
- Automatic conversion of FFmpeg-supported audio to 16 kHz mono WAV
- Configurable Whisper model and language

## Requirements

- Apple Silicon Mac
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)
- [FFmpeg](https://ffmpeg.org/)

Install the system tools with Homebrew:

```bash
brew install uv ffmpeg
```

## Installation

Install the CLI directly from GitHub:

```bash
uv tool install git+https://github.com/yoava333/stt-mlx.git
```

Upgrade an existing installation with:

```bash
uv tool upgrade stt-mlx
```

The first transcription and diarization runs download the selected model weights.

## Usage

Transcribe and diarize an audio file:

```bash
stt recording.mp3
```

Transcribe without diarization:

```bash
stt recording.mp3 --no-diarize
```

Run diarization without transcription:

```bash
stt recording.mp3 --no-transcribe
```

Select a language, model, output prefix, and transcript formats:

```bash
mkdir -p results
stt recording.mp3 \
  --language he \
  --model mlx-community/ivrit-ai-whisper-large-v3-mlx \
  --output-prefix results/interview \
  --formats txt,srt,json
```

Run `stt --help` for the complete CLI reference.

## Output files

With the default prefix for `recording.mp3`, transcript modes can create:

- `recording.diarized.txt` — readable speaker-grouped transcript with timestamps
- `recording.diarized.srt` — subtitles prefixed with speaker labels
- `recording.diarized.json` — transcript segments and diarization turns as structured data

Diarization-only mode creates `recording.rttm`.

`--formats` accepts a comma-separated subset of `txt`, `srt`, and `json`. `--output-prefix` is literal, so `--output-prefix report.v1` produces `report.v1.diarized.txt`.

## Development

```bash
git clone https://github.com/yoava333/stt-mlx.git
cd stt-mlx
uv sync --dev
uv run pytest
uvx ruff check src tests
uvx ruff format --check src tests
uv build
```

## Current limitations

- The project targets Apple Silicon because transcription uses MLX.
- Transcription and diarization currently run sequentially.
- Speaker attribution uses maximum timestamp overlap between transcript segments and diarization turns.
- Model downloads can be large and require network access on first use.
