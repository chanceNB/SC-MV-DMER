"""Deterministic first-freeze generator and fail-closed split verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sc_mv_dmer.data.manifests import DatasetManifest
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


SPLIT_CONTRACT_ID = "DEAM-PRIMARY-SONG-SPLIT/v1"
SPLIT_MANIFEST_ID = "primary-song-80-10-10-v1"
TRAIN_COUNT = 1395
VALIDATION_COUNT = 174
TEST_COUNT = 175


class FrozenSplitManifestUnavailable(RuntimeError):
    """Raised when no approved frozen split artifact is available."""


class SplitMismatchError(ValueError):
    """Raised when a supplied split violates frozen invariants."""


class SplitRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    song_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    logical_song_key: str = Field(pattern=r"^[0-9]+$")
    split: Literal["train", "validation", "test"]
    rank_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank_index: int = Field(ge=0)


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


def _rank_payload(dataset_id: str, song_id: str) -> dict[str, str]:
    return {"dataset_id": dataset_id, "song_id": song_id, "split_contract_id": SPLIT_CONTRACT_ID}


def _rank_digest(dataset_id: str, song_id: str) -> str:
    return sha256_canonical(_rank_payload(dataset_id, song_id))


def freeze_primary_split(dataset: DatasetManifest, *, dataset_manifest_logical_path: str, dataset_manifest_file_sha256: str) -> dict[str, Any]:
    """Create the one authorized first-freeze split from primary song identities."""

    if len(dataset.primary_records) != 1744 or len({record.song_id for record in dataset.primary_records}) != 1744:
        raise SplitMismatchError("primary dataset must contain exactly 1744 unique records")
    ranked = sorted((_rank_digest(dataset.dataset_id, record.song_id), record.song_id, record) for record in dataset.primary_records if record.population_role == "PRIMARY")
    if len(ranked) != 1744:
        raise SplitMismatchError("only PRIMARY records may enter the frozen split")
    records: list[dict[str, Any]] = []
    for index, (digest, song_id, source) in enumerate(ranked):
        split = "train" if index < TRAIN_COUNT else "validation" if index < TRAIN_COUNT + VALIDATION_COUNT else "test"
        records.append(SplitRecord(song_id=song_id, sample_id=source.sample_id, logical_song_key=source.logical_song_key, split=split, rank_digest=digest, rank_index=index).model_dump(mode="json"))
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "split_manifest_id": SPLIT_MANIFEST_ID,
        "split_contract_id": SPLIT_CONTRACT_ID,
        "dataset_manifest_id": dataset.manifest_id,
        "dataset_manifest_logical_path": dataset_manifest_logical_path,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_manifest_file_sha256": dataset_manifest_file_sha256,
        "primary_population_count": 1744,
        "train_count": TRAIN_COUNT,
        "validation_count": VALIDATION_COUNT,
        "test_count": TEST_COUNT,
        "train_song_ids": [item["song_id"] for item in records[:TRAIN_COUNT]],
        "validation_song_ids": [item["song_id"] for item in records[TRAIN_COUNT:TRAIN_COUNT + VALIDATION_COUNT]],
        "test_song_ids": [item["song_id"] for item in records[TRAIN_COUNT + VALIDATION_COUNT:]],
        "records": records,
        "canonicalization": {
            "encoding": "UTF-8",
            "json_object_key_order": "lexicographic ascending",
            "separators": "compact comma/colon",
            "numeric_representation": "JSON integers only",
            "rank_input_keys": ["dataset_id", "song_id", "split_contract_id"],
            "digest": "SHA-256 of canonical rank input",
            "ordering": "rank_digest ascending, then song_id ascending",
            "rank_index_base": 0,
            "membership_inputs_excluded": ["annotation", "filename", "path", "seed", "target", "model_result"],
        },
    }
    payload["split_sha256"] = sha256_canonical(payload)
    return payload


def verify_frozen_split(dataset: DatasetManifest, split: dict[str, Any], *, dataset_manifest_file_sha256: str | None = None) -> dict[str, Any]:
    """Verify registered split identity, ranking, quotas, coverage and checksum."""

    if split.get("split_manifest_id") != SPLIT_MANIFEST_ID or split.get("split_contract_id") != SPLIT_CONTRACT_ID:
        raise SplitMismatchError("split contract identity mismatch")
    if split.get("dataset_manifest_id") != dataset.manifest_id or split.get("dataset_manifest_sha256") != dataset.manifest_sha256:
        raise SplitMismatchError("split does not bind the dataset manifest identity/checksum")
    if dataset_manifest_file_sha256 is not None and split.get("dataset_manifest_file_sha256") != dataset_manifest_file_sha256:
        raise SplitMismatchError("split does not bind the dataset manifest file checksum")
    expected_counts = {"primary_population_count": 1744, "train_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT, "test_count": TEST_COUNT}
    if any(split.get(key) != value for key, value in expected_counts.items()):
        raise SplitMismatchError("split quota is not the frozen 1395/174/175 quota")
    unsigned = dict(split)
    observed_checksum = unsigned.pop("split_sha256", None)
    if not isinstance(observed_checksum, str) or sha256_canonical(unsigned) != observed_checksum:
        raise SplitMismatchError("split manifest checksum mismatch")
    raw_records = split.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != 1744:
        raise SplitMismatchError("split must contain exactly 1744 records")
    records = [SplitRecord.model_validate(item).model_dump(mode="json") for item in raw_records]
    expected_by_song = {record.song_id: record for record in dataset.primary_records if record.population_role == "PRIMARY"}
    if len(expected_by_song) != 1744 or {item["song_id"] for item in records} != set(expected_by_song):
        raise SplitMismatchError("split does not exactly cover PRIMARY songs")
    ranked = sorted((item["rank_digest"], item["song_id"]) for item in records)
    for index, item in enumerate(records):
        expected = expected_by_song[item["song_id"]]
        if item["sample_id"] != expected.sample_id or item["logical_song_key"] != expected.logical_song_key:
            raise SplitMismatchError("split record identity does not match dataset manifest")
        if item["rank_digest"] != _rank_digest(dataset.dataset_id, item["song_id"]):
            raise SplitMismatchError("split rank digest mismatch")
        if item["rank_index"] != index or (item["rank_digest"], item["song_id"]) != ranked[index]:
            raise SplitMismatchError("split ranking or rank_index mismatch")
    memberships = {name: tuple(split.get(name, ())) for name in ("train_song_ids", "validation_song_ids", "test_song_ids")}
    if any(len(values) != len(set(values)) for values in memberships.values()):
        raise SplitMismatchError("split contains duplicate song membership")
    if any(set(memberships[left]) & set(memberships[right]) for left, right in (("train_song_ids", "validation_song_ids"), ("train_song_ids", "test_song_ids"), ("validation_song_ids", "test_song_ids"))):
        raise SplitMismatchError("split song sets overlap")
    expected_sets = {
        "train_song_ids": tuple(item["song_id"] for item in records[:TRAIN_COUNT]),
        "validation_song_ids": tuple(item["song_id"] for item in records[TRAIN_COUNT:TRAIN_COUNT + VALIDATION_COUNT]),
        "test_song_ids": tuple(item["song_id"] for item in records[TRAIN_COUNT + VALIDATION_COUNT:]),
    }
    if memberships != expected_sets:
        raise SplitMismatchError("split membership arrays do not match ranked records")
    return {"verdict": "PASS", "primary_song_count": 1744, "train_song_count": TRAIN_COUNT, "validation_song_count": VALIDATION_COUNT, "test_song_count": TEST_COUNT, "song_overlap_count": 0, "rank_digest_rule": SPLIT_CONTRACT_ID}


def write_split_once(split: dict[str, Any], destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(canonical_json(split) + "\n")
