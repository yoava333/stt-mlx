"""Per-speaker voiceprints built from diarization window embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_KEEP = 0.7


def window_labels(
    windows: Sequence[tuple[float, float]],
    segments: Sequence[tuple[float, float, str]],
) -> list[str | None]:
    """Label each window with the speaker whose segment contains its midpoint."""
    labels: list[str | None] = []
    for start, end in windows:
        middle = (start + end) / 2
        labels.append(
            next(
                (
                    speaker
                    for seg_start, seg_end, speaker in segments
                    if seg_start <= middle <= seg_end
                ),
                None,
            )
        )
    return labels


def voiceprints(
    embeddings: np.ndarray,
    labels: Sequence[str | None],
    keep: float = DEFAULT_KEEP,
) -> dict[str, list[float]]:
    """Average each speaker's cleanest windows into a unit-length voiceprint.

    Windows are length-normalized so loud or long windows do not dominate.
    Windows next to a speaker change often mix two voices, so they are
    skipped unless a speaker has no other windows. Of the remaining windows,
    only the ``keep`` fraction most similar to the speaker's mean is used.
    """
    units = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    label_array = np.array(labels, dtype=object)
    stable = np.zeros(len(labels), dtype=bool)
    stable[1:-1] = (label_array[1:-1] == label_array[:-2]) & (
        label_array[1:-1] == label_array[2:]
    )

    prints: dict[str, list[float]] = {}
    for speaker in dict.fromkeys(label for label in labels if label is not None):
        mask = label_array == speaker
        if (mask & stable).any():
            mask &= stable
        windows = units[mask]
        similarity = windows @ _unit(windows.mean(axis=0))
        windows = windows[similarity >= np.quantile(similarity, 1 - keep)]
        prints[speaker] = _unit(windows.mean(axis=0)).tolist()
    return prints


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
