import numpy as np

from stt_mlx.clustering import cluster_windows
from stt_mlx.speaker_profiles import SpeakerProfile


def test_cluster_windows_clusters_a_subsample_and_assigns_every_window() -> None:
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]] * 3)
    seen: list[int] = []

    def cluster(sample: np.ndarray) -> np.ndarray:
        seen.append(len(sample))
        return (sample[:, 1] > sample[:, 0]).astype(int)

    labels = cluster_windows(embeddings, cluster, [], max_windows=6)

    assert seen == [6]
    assert labels.tolist() == [0, 0, 1, 1] * 3


def test_cluster_windows_clusters_everything_when_under_the_limit() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

    labels = cluster_windows(
        embeddings, lambda sample: np.array([7, 3, 3]), [], max_windows=3
    )

    assert labels.tolist() == [7, 3, 3]


def test_cluster_windows_reassigns_fragment_windows_to_remaining_speakers() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.996, 0.087], [0.643, 0.766], [0.0, 1.0], [0.0, 1.0]]
    )
    profiles = [SpeakerProfile("a", "Person A", [[1.0, 0.0]])]

    labels = cluster_windows(
        embeddings, lambda sample: np.array([0, 0, 1, 1, 2, 2]), profiles
    )

    assert labels.tolist() == [0, 0, 0, 2, 2, 2]
