#!/bin/sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required and was not found on PATH" >&2
    exit 1
fi

repository_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

uv tool install \
    --force \
    --reinstall \
    --python 3.13 \
    "$repository_dir"

echo "Installed stt with Python 3.13."
