"""Persistent voice profiles and speaker-centroid matching."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MATCH_THRESHOLD = 0.75
CONFIDENT_MATCH_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    id: str
    tag: str
    embeddings: list[list[float]]


def default_profiles_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local/share"
    return base / "stt-mlx" / "speaker-profiles.json"


def _embedding(values: object, profile: str) -> list[float]:
    if not isinstance(values, list):
        raise TypeError(f"invalid embeddings for speaker profile {profile!r}")
    vector = [float(value) for value in values]
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"invalid embedding for speaker profile {profile!r}")
    return vector


def load_profiles(path: Path) -> list[SpeakerProfile]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("profiles"), list
    ):
        raise TypeError(f"invalid speaker profile store: {path}")

    profiles: list[SpeakerProfile] = []
    ids: set[str] = set()
    tags: set[str] = set()
    for item in payload["profiles"]:
        if not isinstance(item, Mapping):
            raise TypeError(f"invalid speaker profile store: {path}")
        profile_id = item.get("id")
        tag = item.get("tag")
        raw_embeddings = item.get("embeddings")
        if (
            not isinstance(profile_id, str)
            or not profile_id
            or not isinstance(tag, str)
            or not tag
            or not isinstance(raw_embeddings, list)
            or not raw_embeddings
        ):
            raise ValueError(f"invalid speaker profile store: {path}")
        folded_tag = tag.casefold()
        if profile_id in ids or folded_tag in tags:
            raise ValueError(f"duplicate speaker profile ID or tag: {path}")
        embeddings = [_embedding(values, tag) for values in raw_embeddings]
        dimensions = {len(values) for values in embeddings}
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent embeddings for speaker profile {tag!r}")
        profiles.append(SpeakerProfile(profile_id, tag, embeddings))
        ids.add(profile_id)
        tags.add(folded_tag)
    return profiles


def save_profiles(path: Path, profiles: Sequence[SpeakerProfile]) -> None:
    payload = {
        "profiles": [
            {"id": profile.id, "tag": profile.tag, "embeddings": profile.embeddings}
            for profile in profiles
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_profile(
    profiles: Sequence[SpeakerProfile], selector: str
) -> SpeakerProfile:
    matches = [
        profile
        for profile in profiles
        if profile.id == selector or profile.tag == selector
    ]
    if not matches:
        raise ValueError(f"speaker profile not found: {selector}")
    return matches[0]


def enroll_profile(path: Path, tag: str, embedding: Sequence[float]) -> SpeakerProfile:
    profiles = load_profiles(path)
    vector = _embedding(list(embedding), tag)
    existing = next(
        (profile for profile in profiles if profile.tag.casefold() == tag.casefold()),
        None,
    )
    if existing is None:
        enrolled = SpeakerProfile(uuid.uuid4().hex, tag, [vector])
        profiles.append(enrolled)
    else:
        if len(existing.embeddings[0]) != len(vector):
            raise ValueError(
                f"embedding dimensions do not match profile {existing.tag!r}"
            )
        enrolled = SpeakerProfile(
            existing.id, existing.tag, [*existing.embeddings, vector]
        )
        profiles = [
            enrolled if profile.id == existing.id else profile for profile in profiles
        ]
    save_profiles(path, profiles)
    return enrolled


def create_anonymous_profile(path: Path, embedding: Sequence[float]) -> SpeakerProfile:
    profiles = load_profiles(path)
    vector = _embedding(list(embedding), "anonymous")
    profile_id = uuid.uuid4().hex
    profile = SpeakerProfile(profile_id, f"speaker-{profile_id[:8]}", [vector])
    profiles.append(profile)
    save_profiles(path, profiles)
    return profile


def rename_profile(path: Path, selector: str, new_tag: str) -> SpeakerProfile:
    profiles = load_profiles(path)
    profile = resolve_profile(profiles, selector)
    if any(
        other.id != profile.id and other.tag.casefold() == new_tag.casefold()
        for other in profiles
    ):
        raise ValueError(f"speaker tag already exists: {new_tag}")
    renamed = SpeakerProfile(profile.id, new_tag, profile.embeddings)
    save_profiles(
        path, [renamed if item.id == profile.id else item for item in profiles]
    )
    return renamed


def delete_profile(path: Path, selector: str) -> SpeakerProfile:
    profiles = load_profiles(path)
    profile = resolve_profile(profiles, selector)
    save_profiles(path, [item for item in profiles if item.id != profile.id])
    return profile


def combine_profiles(path: Path, source: str, target: str) -> SpeakerProfile:
    profiles = load_profiles(path)
    source_profile = resolve_profile(profiles, source)
    target_profile = resolve_profile(profiles, target)
    if source_profile.id == target_profile.id:
        raise ValueError("source and target speaker profiles must be different")
    source_dimensions = {len(values) for values in source_profile.embeddings}
    target_dimensions = {len(values) for values in target_profile.embeddings}
    if source_dimensions != target_dimensions:
        raise ValueError("source and target embedding dimensions do not match")
    combined = SpeakerProfile(
        target_profile.id,
        target_profile.tag,
        [*target_profile.embeddings, *source_profile.embeddings],
    )
    save_profiles(
        path,
        [
            combined if profile.id == target_profile.id else profile
            for profile in profiles
            if profile.id != source_profile.id
        ],
    )
    return combined


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot / (left_norm * right_norm)


def match_speakers(
    centroids: Mapping[str, Sequence[float]],
    profiles: Sequence[SpeakerProfile],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, str]:
    """Return one-to-one matches, assigning the most similar pairs first."""
    candidates: list[tuple[float, str, str, str]] = []
    for speaker, centroid in centroids.items():
        for profile in profiles:
            score = profile_score(centroid, profile)
            if score >= threshold:
                candidates.append((score, speaker, profile.id, profile.tag))

    matches: dict[str, str] = {}
    used_profiles: set[str] = set()
    for _score, speaker, profile_id, tag in sorted(candidates, reverse=True):
        if speaker not in matches and profile_id not in used_profiles:
            matches[speaker] = tag
            used_profiles.add(profile_id)
    return matches


def fragment_speakers(
    centroids: Mapping[str, Sequence[float]],
    profiles: Sequence[SpeakerProfile],
    threshold: float = CONFIDENT_MATCH_THRESHOLD,
) -> set[str]:
    """Return speakers that are split-off parts of another speaker's voice.

    A speaker is a fragment when its closest profile matches at least
    ``threshold`` and another speaker matches that same profile more closely.
    The threshold is stricter than matching so that a new voice which merely
    resembles an enrolled one is kept as its own speaker.
    """
    if not profiles:
        return set()
    claims: dict[str, list[tuple[float, str]]] = {}
    for speaker, centroid in centroids.items():
        score, profile_id = max(
            (profile_score(centroid, profile), profile.id) for profile in profiles
        )
        if score >= threshold:
            claims.setdefault(profile_id, []).append((score, speaker))
    return {
        speaker
        for claimants in claims.values()
        for _score, speaker in sorted(claimants, reverse=True)[1:]
    }


def profile_score(centroid: Sequence[float], profile: SpeakerProfile) -> float:
    """Return the similarity of a voice to the closest of a profile's samples."""
    return max(
        cosine_similarity(centroid, embedding) for embedding in profile.embeddings
    )
