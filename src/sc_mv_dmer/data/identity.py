"""Stable DEAM dataset, song and sample identities."""

from __future__ import annotations

from sc_mv_dmer.data.discovery import DEAM_DATASET_VERSION
from sc_mv_dmer.foundation.identity import stable_id


DATASET_NAME = "DEAM"
PRIMARY_EXCERPT_RULE_VERSION = "deam-primary-45s-v1"


def make_dataset_id() -> str:
    return stable_id("dataset", (DATASET_NAME, DEAM_DATASET_VERSION))


def make_song_id(dataset_id: str, logical_song_key: str) -> str:
    return stable_id("song", (dataset_id, logical_song_key))


def make_sample_id(song_id: str, excerpt_key: str = PRIMARY_EXCERPT_RULE_VERSION) -> str:
    return stable_id("sample", (song_id, excerpt_key, "0", "45"))
