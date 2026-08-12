"""One-way run finalization and immutable evidence lifecycle records."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, field_validator

from sc_mv_dmer.foundation.manifests import (
    SHA256_PATTERN,
    ArtifactRef,
    ImmutableRecord,
    RunSpec,
    freeze_value,
)


class TerminalRecordMutationError(RuntimeError):
    """Raised when code tries to change an already terminal run."""


class DuplicateRecordError(ValueError):
    """Raised when an append-only record ID is submitted twice."""


class UnknownDependencyError(ValueError):
    """Raised when a record refers to a node that is not registered."""


class TerminalState(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class HistoricalVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class RunManifest(ImmutableRecord):
    """A terminal run record. Its spec preserves complete input provenance."""

    spec: RunSpec
    terminal_state: TerminalState
    terminal_reason: str = Field(min_length=1)
    metrics: Mapping[str, Any] = Field(default_factory=dict)
    checkpoints: tuple[ArtifactRef, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    completed_at: datetime

    @property
    def run_id(self) -> str:
        return self.spec.run_id

    @field_validator("metrics", mode="after")
    @classmethod
    def _freeze_metrics(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_value(value)

    @field_validator("checkpoints", "artifacts", mode="after")
    @classmethod
    def _freeze_artifacts(cls, value: tuple[ArtifactRef, ...]) -> tuple[ArtifactRef, ...]:
        return tuple(value)


class ActiveRun:
    """An in-memory handle which may be finalized exactly once."""

    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized


def start_run(spec: RunSpec) -> ActiveRun:
    """Create an in-memory active handle without executing a run."""

    return ActiveRun(spec)


def finalize_run(
    active: ActiveRun,
    terminal: TerminalState,
    *,
    terminal_reason: str,
    metrics: Mapping[str, Any] | None = None,
    checkpoints: Sequence[ArtifactRef] = (),
    artifacts: Sequence[ArtifactRef] = (),
    completed_at: datetime | None = None,
) -> RunManifest:
    """Seal an active run. A continuation must use a new ``RunSpec`` ID."""

    if active.finalized:
        raise TerminalRecordMutationError("run has already been finalized")
    manifest = RunManifest(
        spec=active.spec,
        terminal_state=terminal,
        terminal_reason=terminal_reason,
        metrics=dict(metrics or {}),
        checkpoints=tuple(checkpoints),
        artifacts=tuple(artifacts),
        completed_at=completed_at or datetime.now(timezone.utc),
    )
    active._finalized = True
    return manifest


class GateEvaluationAttempt(ImmutableRecord):
    """A historical Gate evaluation, never a claim that an evaluation ran."""

    evaluation_id: str = Field(min_length=1)
    gate_version: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    definition_id: str = Field(min_length=1)
    verdict: HistoricalVerdict
    authority_kind: str = Field(pattern=r"^(automatic|manual)$")
    authority_metadata: Mapping[str, Any] = Field(default_factory=dict)
    evidence_commit: str = Field(min_length=7)
    evidence_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    source_run_ids: tuple[str, ...]
    predecessor_dependencies: tuple[str, ...]

    @field_validator("authority_metadata", mode="after")
    @classmethod
    def _freeze_authority(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_value(value)


class InvalidationRecord(ImmutableRecord):
    record_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    invalidated_by: str = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SupersessionRecord(ImmutableRecord):
    record_id: str = Field(min_length=1)
    superseded_node_id: str = Field(min_length=1)
    replacement_node_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
