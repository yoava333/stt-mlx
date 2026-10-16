import json
from pathlib import Path

import pytest

from stt_mlx.speaker_profiles import (
    SpeakerProfile,
    combine_profiles,
    cosine_similarity,
    create_anonymous_profile,
    delete_profile,
    enroll_profile,
    fragment_speakers,
    load_profiles,
    match_speakers,
    rename_profile,
)


def test_enroll_creates_stable_profile_and_adds_samples(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"

    first = enroll_profile(path, "Yoav", [1.0, 0.0])
    second = enroll_profile(path, "yoav", [0.8, 0.2])

    assert second == SpeakerProfile(first.id, "Yoav", [[1.0, 0.0], [0.8, 0.2]])
    assert load_profiles(path) == [second]


def test_create_anonymous_profile_uses_id_based_unique_tag(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"

    first = create_anonymous_profile(path, [1.0, 0.0])
    second = create_anonymous_profile(path, [0.0, 1.0])

    assert first.tag == f"speaker-{first.id[:8]}"
    assert second.tag == f"speaker-{second.id[:8]}"
    assert first.id != second.id
    assert load_profiles(path) == [first, second]


def test_rename_profile_preserves_id_and_embeddings(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    original = enroll_profile(path, "Yoav", [1.0, 0.0])

    renamed = rename_profile(path, original.id, "Yoav A")

    assert renamed == SpeakerProfile(original.id, "Yoav A", [[1.0, 0.0]])
    assert load_profiles(path) == [renamed]


def test_rename_rejects_duplicate_tag_case_insensitively(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    enroll_profile(path, "Yoav", [1.0, 0.0])
    enroll_profile(path, "Dana", [0.0, 1.0])

    with pytest.raises(ValueError, match="tag already exists"):
        rename_profile(path, "Dana", "YOAV")


def test_delete_profile_accepts_tag(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    profile = enroll_profile(path, "Yoav", [1.0, 0.0])

    assert delete_profile(path, "Yoav") == profile
    assert load_profiles(path) == []


def test_combine_profiles_keeps_target_id_tag_and_all_samples(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    source = enroll_profile(path, "Unknown Yoav", [0.8, 0.2])
    target = enroll_profile(path, "Yoav", [1.0, 0.0])

    combined = combine_profiles(path, source.id, target.id)

    assert combined == SpeakerProfile(target.id, "Yoav", [[1.0, 0.0], [0.8, 0.2]])
    assert load_profiles(path) == [combined]


def test_cosine_similarity_rejects_incompatible_vectors() -> None:
    assert cosine_similarity([1.0], [1.0, 2.0]) == -1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == -1.0


def test_match_speakers_uses_best_embedding_and_unique_assignment() -> None:
    matches = match_speakers(
        {
            "SPEAKER_01": [1.0, 0.0],
            "SPEAKER_02": [0.9, 0.1],
            "SPEAKER_03": [0.0, -1.0],
        },
        [SpeakerProfile("id", "Yoav", [[-1.0, 0.0], [1.0, 0.0]])],
        threshold=0.8,
    )

    assert matches == {"SPEAKER_01": "Yoav"}


def test_match_speakers_picks_best_profile_even_when_runner_up_is_close() -> None:
    matches = match_speakers(
        {"SPEAKER_01": [1.0, 0.0]},
        [
            SpeakerProfile("a", "Person A", [[0.999, 0.045]]),
            SpeakerProfile("b", "Person B", [[1.0, 0.0]]),
        ],
        threshold=0.7,
    )

    assert matches == {"SPEAKER_01": "Person B"}


def test_match_speakers_gives_contested_profile_to_closest_speaker() -> None:
    matches = match_speakers(
        {"SPEAKER_01": [0.9, 0.1], "SPEAKER_02": [1.0, 0.0]},
        [
            SpeakerProfile("a", "Person A", [[1.0, 0.0]]),
            SpeakerProfile("b", "Person B", [[0.0, 1.0]]),
        ],
        threshold=0.7,
    )

    assert matches == {"SPEAKER_02": "Person A"}


def test_match_speakers_default_threshold_rejects_unfamiliar_voice() -> None:
    profile = SpeakerProfile("id", "Yoav", [[1.0, 0.0]])

    assert match_speakers({"SPEAKER_01": [0.72, 0.694]}, [profile]) == {}
    assert match_speakers({"SPEAKER_01": [0.78, 0.626]}, [profile]) == {
        "SPEAKER_01": "Yoav"
    }


def test_match_speakers_rejects_weak_match() -> None:
    matches = match_speakers(
        {"SPEAKER_01": [1.0, 0.0]},
        [SpeakerProfile("id", "Yoav", [[0.6, 0.8]])],
        threshold=0.7,
    )

    assert matches == {}


def test_fragment_speakers_drops_weaker_claims_on_the_same_profile() -> None:
    fragments = fragment_speakers(
        {"A": [1.0, 0.0], "A_part": [0.95, 0.312], "B": [0.0, 1.0]},
        [
            SpeakerProfile("a", "Person A", [[0.0, -1.0], [1.0, 0.0]]),
            SpeakerProfile("b", "Person B", [[0.0, 1.0]]),
        ],
        threshold=0.9,
    )

    assert fragments == {"A_part"}


def test_fragment_speakers_keeps_voice_that_only_resembles_a_profile() -> None:
    profile = SpeakerProfile("a", "Person A", [[1.0, 0.0]])

    fragments = fragment_speakers(
        {"A": [1.0, 0.0], "Newcomer": [0.8, 0.6]}, [profile], threshold=0.85
    )

    assert fragments == set()


def test_fragment_speakers_default_threshold_is_stricter_than_matching() -> None:
    profile = SpeakerProfile("a", "Person A", [[1.0, 0.0]])
    centroids = {"A": [1.0, 0.0], "Other": [0.8, 0.6]}

    assert match_speakers({"Other": centroids["Other"]}, [profile]) == {
        "Other": "Person A"
    }
    assert fragment_speakers(centroids, [profile]) == set()
    assert fragment_speakers({**centroids, "Other": [0.9, 0.436]}, [profile]) == {
        "Other"
    }


def test_load_profiles_rejects_obsolete_store_format(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"Yoav": [1.0, 0.0]}))

    with pytest.raises(TypeError, match="invalid speaker profile store"):
        load_profiles(path)


def test_load_profiles_rejects_non_finite_embedding(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        '{"profiles": [{"id": "id", "tag": "Yoav", "embeddings": [[NaN]]}]}'
    )

    with pytest.raises(ValueError, match="invalid embedding"):
        load_profiles(path)
