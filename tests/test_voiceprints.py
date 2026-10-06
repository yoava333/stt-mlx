import math

import numpy as np
import pytest

from stt_mlx.voiceprints import voiceprints, window_labels


def test_window_labels_uses_segment_containing_window_midpoint() -> None:
    labels = window_labels(
        [(0.0, 1.5), (1.5, 3.0), (5.0, 6.5)],
        [(0.0, 1.9, "SPEAKER_01"), (1.9, 4.0, "SPEAKER_02")],
    )

    assert labels == ["SPEAKER_01", "SPEAKER_02", None]


def test_voiceprints_are_unit_length_and_ignore_window_magnitude() -> None:
    embeddings = np.array([[10.0, 0.0], [10.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = ["A", "A", "A", "A"]

    prints = voiceprints(embeddings, labels, keep=1.0)

    assert prints["A"] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])


def test_voiceprints_skip_windows_next_to_speaker_changes() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )
    embeddings[2] = [0.0, 1.0]  # last A window bleeds into B
    labels = ["A", "A", "A", "B", "B", "B"]

    prints = voiceprints(embeddings, labels, keep=1.0)

    assert prints["A"] == pytest.approx([1.0, 0.0])


def test_voiceprints_drop_windows_least_similar_to_speaker() -> None:
    embeddings = np.array([[1.0, 0.0]] * 9 + [[0.0, 1.0]] + [[1.0, 0.0]])
    labels = ["A"] * 11

    prints = voiceprints(embeddings, labels, keep=0.7)

    assert prints["A"] == pytest.approx([1.0, 0.0])


def test_voiceprints_fall_back_to_all_windows_of_short_speakers() -> None:
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = ["A", "A", "B", "A"]

    prints = voiceprints(embeddings, labels, keep=1.0)

    assert prints["B"] == pytest.approx([0.0, 1.0])


def test_voiceprints_ignore_unlabeled_windows() -> None:
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    prints = voiceprints(embeddings, ["A", "A", "A", None], keep=1.0)

    assert list(prints) == ["A"]
    assert prints["A"] == pytest.approx([1.0, 0.0])
