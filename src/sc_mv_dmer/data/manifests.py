"""Immutable machine-readable DEAM source and primary manifests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator

from sc_mv_dmer.foundation.canonical import sha256_canonical
from sc_mv_dmer.foundation.manifests import ImmutableRecord, freeze_value


class SourceArtifact(ImmutableRecord):
    relative_path: str = Field(min_length=1)
    role: Literal["AUDIO", "ANNOTATION"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    logical_song_key: str | None = None
    duplicate_content_group: str | None = None

    @field_validator("relative_path")
    @classmethod
    def _relative_only(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or path.drive or (len(value) >= 2 and value[1] == ":") or ".." in path.parts or value.startswith(("/", "\\\\")):
            raise ValueError("source artifact path must be repository-relative")
        return value


class SourceInventory(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    inventory_id: Literal["DEAM-SOURCE-INVENTORY-v1"] = "DEAM-SOURCE-INVENTORY-v1"
    dataset_name: Literal["DEAM"] = "DEAM"
    dataset_version: str = Field(min_length=1)
    audio_root_relative: Literal["DEAM_audio/MEMD_audio"] = "DEAM_audio/MEMD_audio"
    annotation_root_relative: Literal["DEAM_Annotations/annotations"] = (
        "DEAM_Annotations/annotations"
    )
    source_artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)
    audio_song_keys: tuple[str, ...] = Field(min_length=1)
    annotation_song_keys: tuple[str, ...] = Field(min_length=1)
    primary_song_keys: tuple[str, ...] = Field(min_length=1)
    long_song_keys: tuple[str, ...] = Field(min_length=1)
    primary_population_count: Literal[1744] = 1744
    long_song_count: Literal[58] = 58
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("source_artifacts", "audio_song_keys", "annotation_song_keys", mode="after")
    @classmethod
    def _freeze_sequences(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(value)

    def unsigned_payload(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        payload.pop("inventory_sha256")
        return payload


class DatasetRecord(ImmutableRecord):
    dataset_id: str = Field(min_length=1)
    song_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    logical_song_key: str = Field(pattern=r"^[0-9]+$")
    source_relative_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(gt=0)
    population_role: Literal["PRIMARY", "LONG_AUXILIARY"]
    excerpt_start_seconds: Literal[0] = 0
    excerpt_end_seconds: Literal[45] = 45

    @field_validator("source_relative_path")
    @classmethod
    def _relative_source_only(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or path.drive or (len(value) >= 2 and value[1] == ":") or ".." in path.parts or value.startswith(("/", "\\\\")):
            raise ValueError("dataset source path must be relative")
        return value


class DatasetManifest(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    manifest_id: Literal["DEAM-PRIMARY-v1"] = "DEAM-PRIMARY-v1"
    dataset_name: Literal["DEAM"] = "DEAM"
    dataset_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    inventory_logical_path: str = Field(min_length=1)
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_records: tuple[DatasetRecord, ...] = Field(min_length=1744, max_length=1744)
    primary_song_keys: tuple[str, ...] = Field(min_length=1744, max_length=1744)
    long_song_keys: tuple[str, ...] = Field(min_length=58, max_length=58)
    primary_population_count: Literal[1744] = 1744
    long_song_count: Literal[58] = 58
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("inventory_logical_path")
    @classmethod
    def _relative_inventory_only(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or path.drive or (len(value) >= 2 and value[1] == ":") or ".." in path.parts or value.startswith(("/", "\\\\")):
            raise ValueError("inventory logical path must be relative")
        return value

    @field_validator("primary_records", "primary_song_keys", "long_song_keys", mode="after")
    @classmethod
    def _freeze_sequences(cls, value: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(value)

    def unsigned_payload(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        payload.pop("manifest_sha256")
        return payload


def verify_inventory_checksum(inventory: SourceInventory) -> None:
    if sha256_canonical(inventory.unsigned_payload()) != inventory.inventory_sha256:
        raise ValueError("source inventory checksum mismatch")


def verify_dataset_manifest_checksum(manifest: DatasetManifest) -> None:
    if sha256_canonical(manifest.unsigned_payload()) != manifest.manifest_sha256:
        raise ValueError("dataset manifest checksum mismatch")
