"""One-way run finalization and immutable evidence lifecycle records."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

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


class AuthorityKind(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class HumanApprovalState(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"


class AutomaticAuthorityMetadata(ImmutableRecord):
    """Typed provenance for an automatic evaluation authority."""

    authority_kind: Literal["automatic"] = "automatic"
    automation_identity: str = Field(min_length=1)


class ManualAuthorityMetadata(ImmutableRecord):
    """Typed human-approval provenance; approval is never synthesized."""

    authority_kind: Literal["manual"] = "manual"
    approval_state: HumanApprovalState
    reviewer_identity: str | None = None
    approval_artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_approval_provenance(self) -> "ManualAuthorityMetadata":
        if self.approval_state is HumanApprovalState.APPROVED:
            if not self.reviewer_identity or not self.approval_artifact_sha256:
                raise ValueError(
                    "approved manual authority requires human reviewer identity "
                    "and immutable approval artifact checksum"
                )
            normalized_identity = self.reviewer_identity.strip().casefold()
            if "codex" in normalized_identity or "automatic" in normalized_identity:
                raise ValueError("human reviewer identity cannot be Codex or automatic")
        return self


AuthorityMetadata = Annotated[
    AutomaticAuthorityMetadata | ManualAuthorityMetadata,
    Field(discriminator="authority_kind"),
]


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

    def __init__(self, spec: RunSpec, lifecycle_store: "LifecycleStore") -> None:
        self.spec = spec
        self._lifecycle_store = lifecycle_store
        self._finalized = False

    @property
    def finalized(self) -> bool:
        return self._finalized


class LifecycleStore:
    """Explicit owner of run identities and their one-way active lifecycle."""

    def __init__(self) -> None:
        self._active_runs: dict[str, ActiveRun] = {}
        self._terminal_run_ids: set[str] = set()

    def start(self, spec: RunSpec) -> ActiveRun:
        if spec.run_id in self._active_runs or spec.run_id in self._terminal_run_ids:
            raise DuplicateRecordError(f"duplicate run ID: {spec.run_id}")
        active = ActiveRun(spec, self)
        self._active_runs[spec.run_id] = active
        return active

    def seal(self, active: ActiveRun) -> None:
        run_id = active.spec.run_id
        if (
            active._lifecycle_store is not self
            or self._active_runs.get(run_id) is not active
            or active.finalized
            or run_id in self._terminal_run_ids
        ):
            raise TerminalRecordMutationError("run has already been finalized or is not active")
        active._finalized = True
        del self._active_runs[run_id]
        self._terminal_run_ids.add(run_id)


def start_run(spec: RunSpec, lifecycle_store: LifecycleStore) -> ActiveRun:
    """Register a new active handle in an explicit lifecycle store."""

    return lifecycle_store.start(spec)


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

    manifest = RunManifest(
        spec=active.spec,
        terminal_state=terminal,
        terminal_reason=terminal_reason,
        metrics=dict(metrics or {}),
        checkpoints=tuple(checkpoints),
        artifacts=tuple(artifacts),
        completed_at=completed_at or datetime.now(timezone.utc),
    )
    active._lifecycle_store.seal(active)
    return manifest


class GateEvaluationAttempt(ImmutableRecord):
    """A historical Gate evaluation, never a claim that an evaluation ran."""

    evaluation_id: str = Field(min_length=1)
    gate_version: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    definition_id: str = Field(min_length=1)
    verdict: HistoricalVerdict
    authority_kind: AuthorityKind
    authority_metadata: AuthorityMetadata
    evidence_commit: str = Field(min_length=7)
    evidence_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    source_run_ids: tuple[str, ...]
    predecessor_dependencies: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_authority(self) -> "GateEvaluationAttempt":
        if self.authority_metadata.authority_kind != self.authority_kind.value:
            raise ValueError("authority kind must match typed authority metadata")
        if (
            self.authority_kind is AuthorityKind.MANUAL
            and self.verdict is HistoricalVerdict.PASS
            and (
                not isinstance(self.authority_metadata, ManualAuthorityMetadata)
                or self.authority_metadata.approval_state is not HumanApprovalState.APPROVED
            )
        ):
            raise ValueError("manual PASS requires verified human approval provenance")
        return self


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
