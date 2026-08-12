"""Immutable, machine-readable records for run provenance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FrozenDict(dict[str, Any]):
    """A JSON-serializable dictionary which rejects all mutating operations."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        dict.__init__(self, values)

    @staticmethod
    def _immutable(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("frozen mapping cannot be mutated")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def freeze_value(value: Any) -> Any:
    """Copy a JSON-like value into an immutable representation."""

    if isinstance(value, Mapping):
        return FrozenDict({str(key): freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze_value(item) for item in value)
    return value


class ImmutableRecord(BaseModel):
    """Base model which rejects undeclared data and attribute reassignment."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class ArtifactRef(ImmutableRecord):
    artifact_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    logical_path: str | None = None
    version: str | None = None


class RunSpec(ImmutableRecord):
    """All immutable inputs and provenance required to identify a run."""

    run_id: str = Field(min_length=1)
    run_mode: str = Field(min_length=1)
    semantic_config_hash: str = Field(pattern=SHA256_PATTERN)
    resolved_config_hash: str = Field(pattern=SHA256_PATTERN)
    config_snapshot_ref: ArtifactRef
    config_provenance_ref: ArtifactRef
    git_commit: str = Field(min_length=7)
    git_dirty: bool
    data_ref: ArtifactRef
    split_ref: ArtifactRef
    feature_cache_ref: ArtifactRef | None = None
    seed: int
    environment: Mapping[str, Any]
    launch_command: tuple[str, ...] = Field(min_length=1)
    parent_run_id: str | None = None
    continuation_of_run_id: str | None = None
    resume_checkpoint_ref: ArtifactRef | None = None
    schema_versions: Mapping[str, str] = Field(min_length=1)

    @field_validator("environment", "schema_versions", mode="after")
    @classmethod
    def _freeze_mappings(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_value(value)

    @field_validator("launch_command", mode="after")
    @classmethod
    def _freeze_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not part for part in value):
            raise ValueError("launch command must not contain empty parts")
        return tuple(value)

    def model_post_init(self, __context: Any) -> None:
        if self.run_id in {self.parent_run_id, self.continuation_of_run_id}:
            raise ValueError("a run cannot be its own parent or continuation")
