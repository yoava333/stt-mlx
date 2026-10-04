# stt-mlx

Local, MLX-accelerated speech-to-text with optional speaker diarization for Apple Silicon Macs. `stt-mlx` combines [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) with [Senko](https://github.com/narcotic-sh/senko) and writes speaker-attributed transcripts as text, subtitles, or JSON.

## Features

- MLX-accelerated Whisper transcription on Apple Silicon
- Fused scaled-dot-product attention for lower GPU runtime
- Optional speaker diarization
- Persistent voice enrollment for named speaker tags
- TXT, SRT, JSON, and RTTM output
- Automatic conversion of FFmpeg-supported audio to 16 kHz mono WAV
- Configurable transcription engine (MLX Whisper or Hebrew Parakeet-TDT), model, and language

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

Clone the repository, then install the global CLI from the checkout:

```bash
./install.sh
```

Run the same script again after pulling changes to upgrade the installation:

```bash
./install.sh
```

The installer rebuilds the tool with Python 3.13. Do not install it with the
system-default Python: Senko's Core ML models do not load under Python 3.14.

The first transcription and diarization runs download the selected model weights.

## Usage

Transcribe and diarize an audio file:

```bash
stt recording.mp3
```

Specify the exact number of speakers when it is known:

```bash
stt recording.mp3 --num-speakers 3
```

Enroll a person once using a clean recording that contains only their voice:

```bash
stt yoav-voice-sample.m4a --enroll-speaker "Yoav"
```

Future diarized transcripts identify matching speaker tags automatically:

```text
=== Yoav ===
[00:00:03.200 --> 00:00:07.100] ...
```

Voice profiles are stored locally in
`~/.local/share/stt-mlx/speaker-profiles.json` (or under `$XDG_DATA_HOME` when
set). During every diarized run, an unmatched voice automatically creates a
persistent anonymous profile with a tag such as `speaker-a1b2c3d4`. Matching
voices reuse that tag in later recordings. Use `stt profiles rename` once you
know who it belongs to.

Each profile has a stable unique ID, a unique speaker tag, and one or more voice
samples. Explicitly enrolling the same tag again adds another sample to the
existing profile. Matching compares a detected speaker with every sample and
uses the strongest result, which supports enrollment across different recording
conditions without averaging distinct samples together. Weak or ambiguous
matches remain anonymous. Voice matching is intended for transcript labeling,
not identity authentication.

List the enrolled profiles and their sample counts:

```bash
stt profiles list
```

Rename a speaker tag using either its profile ID or exact current tag:

```bash
stt profiles rename "Yoav" "Yoav Aharoni"
```

Delete a misleading profile:

```bash
stt profiles delete "Wrong speaker"
```

Combine duplicate or similar profiles under one canonical tag:

```bash
stt profiles combine "Possible Yoav" "Yoav Aharoni"
```

The source profile's samples are added to the target profile, then the source
is removed. The target keeps its ID and speaker tag.

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

The default model is Hebrew-specific, so `stt` defaults to `--language he`. This
skips a redundant language-detection encoder pass. Pass another language when
using a model for other speech.

### Transcription engine

`stt` transcribes with MLX Whisper by default. Pass `--engine parakeet` to use
the Hebrew [Parakeet-TDT](https://huggingface.co/thewh1teagle/parakeet-tdt-0.6b-v3-he)
model instead, which runs on the GPU through PyTorch MPS:

```bash
stt recording.m4a --engine parakeet
```

Each engine has its own default model, so `--model` is only needed to override
it. Diarization, speaker profiles, and all output formats behave identically
across engines. Parakeet is Hebrew-only, so `--language` is ignored for it.

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
- Voice identification accuracy depends on the enrollment audio, microphone, noise, and recording conditions.
- Model downloads can be large and require network access on first use.
