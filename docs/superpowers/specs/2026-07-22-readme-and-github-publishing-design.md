# README and GitHub Publishing Design

## Goal

Give `stt-mlx` a clear usage-first README and publish the existing `main` branch as the public GitHub repository `yoava333/stt-mlx`.

## README Audience and Structure

The README targets an Apple Silicon Mac user who wants local speech transcription with optional speaker diarization. It will explain the project in this order:

1. A concise description and feature summary.
2. Platform and software prerequisites: Apple Silicon macOS, Python 3.12 or 3.13, FFmpeg, and `uv`.
3. Installation from the GitHub repository.
4. A default transcription-and-diarization example.
5. Transcription-only and diarization-only examples.
6. Model, language, output-prefix, and format options.
7. The TXT, SRT, JSON, and RTTM output contracts.
8. First-run model download and local-processing expectations.
9. Development commands for syncing, testing, linting, formatting, and building.
10. Current limitations, including Apple Silicon/MLX requirements and sequential transcription/diarization execution.

The README will use commands that match the actual CLI and project metadata. It will not claim support, performance, licensing, or privacy guarantees that the repository does not establish.

## Repository Publishing

Create `https://github.com/yoava333/stt-mlx` as a public repository with the description “MLX-accelerated speech-to-text with speaker diarization for Apple Silicon.” Use the authenticated `yoava333` GitHub CLI account.

Add the new repository as the `origin` remote and push the local `main` branch with upstream tracking. Do not force-push, rewrite history, create releases, enable Pages, or add repository secrets.

## Validation

Before publishing:

- Confirm the worktree contains only intended README/specification/plan changes.
- Run the complete pytest suite.
- Run Ruff lint and formatting checks.
- Build the source distribution and wheel.
- Scan tracked files for common credential/private-key patterns without printing credential values.

After publishing:

- Verify `origin` points to `https://github.com/yoava333/stt-mlx.git`.
- Verify GitHub reports the repository as public and its default branch as `main`.
- Verify the remote README is visible.

## Scope

No license, CI workflow, release, package publication, logo, screenshots, or implementation changes are included. Existing design and plan documents remain tracked and will be published with the rest of the repository.
