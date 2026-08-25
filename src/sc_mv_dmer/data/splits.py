"""Fail-closed verifier for the externally approved primary split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sc_mv_dmer.data.manifests import DatasetManifest


class FrozenSplitManifestUnavailable(RuntimeError):
    """Raised when no approved frozen split artifact is available."""


class SplitMismatchError(ValueError):
    """Raised when a supplied split violates song-level coverage invariants."""


def load_frozen_split(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FrozenSplitManifestUnavailable(
            "FROZEN_SPLIT_MANIFEST_UNAVAILABLE: approved split artifact is missing"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SplitMismatchError("frozen split manifest must be an object")
    return payload


def verify_frozen_split(dataset: DatasetManifest, split: dict[str, Any]) -> dict[str, Any]:
    """Verify an existing split; never generate membership."""

    memberships = {name: tuple(split.get(name, ())) for name in ("train_song_ids", "validation_song_ids", "test_song_ids")}
    expected = set(dataset.primary_song_keys)
    assigned = set().union(*memberships.values())
    if any(len(values) != len(set(values)) for values in memberships.values()):
        raise SplitMismatchError("split contains duplicate song membership")
    if set(memberships["train_song_ids"]) & set(memberships["validation_song_ids"]):
        raise SplitMismatchError("train/validation song overlap")
    if set(memberships["train_song_ids"]) & set(memberships["test_song_ids"]):
        raise SplitMismatchError("train/test song overlap")
    if set(memberships["validation_song_ids"]) & set(memberships["test_song_ids"]):
        raise SplitMismatchError("validation/test song overlap")
    if assigned != expected:
        raise SplitMismatchError("split does not exactly cover primary song membership")
    return {
        "verdict": "PASS",
        "train_song_count": len(memberships["train_song_ids"]),
        "validation_song_count": len(memberships["validation_song_ids"]),
        "test_song_count": len(memberships["test_song_ids"]),
        "primary_song_count": len(expected),
    }
