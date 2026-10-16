"""Speaker clustering of diarization window embeddings."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from stt_mlx.speaker_profiles import SpeakerProfile, fragment_speakers
from stt_mlx.voiceprints import unit

MAX_CLUSTER_WINDOWS = 6000


def cluster_windows(
    embeddings: np.ndarray,
    cluster: Callable[[np.ndarray], np.ndarray],
    profiles: Sequence[SpeakerProfile],
    max_windows: int = MAX_CLUSTER_WINDOWS,
) -> np.ndarray:
    """Label every window with a speaker cluster.

    Spectral clustering separates similar voices that density clustering
    merges, but its cost grows with the cube of the window count, so it runs
    on an evenly spaced sample of at most ``max_windows`` windows. Clusters
    that are fragments of an enrolled voice claimed by a closer cluster are
    dropped. Every window then joins the nearest remaining cluster centroid.
    """
    step = math.ceil(len(embeddings) / max_windows)
    units = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sample = units[::step]
    sample_labels = np.asarray(cluster(embeddings[::step]))

    centroids = {
        label: unit(sample[sample_labels == label].mean(axis=0))
        for label in np.unique(sample_labels).tolist()
    }
    fragments = fragment_speakers(
        {str(label): centroid.tolist() for label, centroid in centroids.items()},
        profiles,
    )
    kept = [label for label in centroids if str(label) not in fragments]
    nearest = (units @ np.array([centroids[label] for label in kept]).T).argmax(axis=1)
    return np.array(kept)[nearest]
