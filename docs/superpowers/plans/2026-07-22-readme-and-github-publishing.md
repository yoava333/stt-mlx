# README and GitHub Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a usage-first README and publish the verified `main` branch as the public `yoava333/stt-mlx` GitHub repository.

**Architecture:** Documentation remains a single root `README.md` aligned with the actual argparse interface and `pyproject.toml`. Publishing uses the authenticated GitHub CLI only after local tests, linting, packaging, and credential-pattern checks succeed.

**Tech Stack:** Markdown, Python 3.12–3.13, uv, pytest, Ruff, Git, GitHub CLI

## Global Constraints

- Create `yoava333/stt-mlx` as a public repository with default branch `main`.
- Do not add a license, CI workflow, release, logo, screenshots, or implementation changes.
- Do not force-push, rewrite history, enable Pages, or create repository secrets.
- Do not print tokens or secret values during validation.
- Preserve all currently tracked design and implementation-plan documents.

---

### Task 1: Write and verify the usage-first README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the CLI contract in `src/stt_mlx/cli.py` and dependencies in `pyproject.toml`
- Produces: installable and runnable user instructions for `stt`

- [ ] **Step 1: Create the README with these exact sections**

````markdown
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
stt recording.mp3 \
  --language en \
  --model mlx-community/whisper-large-v3-turbo \
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
````

- [ ] **Step 2: Verify every documented command and option exists**

Run: `uv run stt --help`

Expected: help lists `--output-prefix`, `--model`, `--language`, `--no-diarize`, `--no-transcribe`, and `--formats`.

- [ ] **Step 3: Check Markdown hygiene**

Run: `git diff --check && test -s README.md`

Expected: exit 0 with a non-empty README and no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add usage-first README"
```

### Task 2: Verify and publish the repository

**Files:**
- Verify: all tracked files
- External target: `https://github.com/yoava333/stt-mlx`

**Interfaces:**
- Consumes: verified local `main` history
- Produces: public GitHub repository with `origin` tracking `main`

- [ ] **Step 1: Run local quality gates**

Run: `uv lock --check && uvx ruff check src tests && uvx ruff format --check src tests && uv run pytest -v && uv build`

Expected: lock resolution unchanged, Ruff clean, 19 tests pass, and source/wheel builds succeed.

- [ ] **Step 2: Scan tracked files for common credential material**

```bash
if git grep -IlE '(gh[opsu]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----)' -- . ':!uv.lock'; then
  echo "credential-like material found in tracked files"
  exit 1
fi
```

Expected: exit 0 with no matching filenames and no credential values printed.

- [ ] **Step 3: Confirm the target remains available**

Run: `gh repo view yoava333/stt-mlx --json name 2>/dev/null`

Expected: nonzero exit because the repository does not yet exist. If it exists, stop instead of overwriting or attaching to it.

- [ ] **Step 4: Create the public repository and push `main`**

```bash
gh repo create yoava333/stt-mlx \
  --public \
  --description "MLX-accelerated speech-to-text with speaker diarization for Apple Silicon" \
  --source . \
  --remote origin \
  --push
```

Expected: the repository is created, `origin` is added, and local `main` tracks `origin/main`.

- [ ] **Step 5: Verify the published state**

Run: `gh repo view yoava333/stt-mlx --json nameWithOwner,visibility,defaultBranchRef,url,description && gh api repos/yoava333/stt-mlx/readme --jq '.html_url' && git remote -v && git status --short --branch`

Expected: `nameWithOwner` is `yoava333/stt-mlx`, visibility is `PUBLIC`, default branch is `main`, the README URL is returned, `origin` uses the new repository, and the branch is clean and tracking `origin/main`.
