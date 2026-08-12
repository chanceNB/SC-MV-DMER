"""Append-only evidence registry and effective-validity resolution."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from sc_mv_dmer.foundation.lifecycle import (
    DuplicateRecordError,
    GateEvaluationAttempt,
    InvalidationRecord,
    RunManifest,
    SupersessionRecord,
    UnknownDependencyError,
)
from sc_mv_dmer.foundation.manifests import ImmutableRecord


class EffectiveValidityStatus(str, Enum):
    VALID = "VALID"
    INVALIDATED = "INVALIDATED"
    STALE_DEPENDENCY = "STALE_DEPENDENCY"


class EffectiveValidity(ImmutableRecord):
    status: EffectiveValidityStatus
    reason: str = Field(min_length=1)
    invalidated_sources: tuple[str, ...] = ()
    dependency_path: tuple[str, ...] = ()
    replacement_ref: str | None = None


class Registry:
    """In-memory append-only registry for declared evidence relationships."""

    def __init__(self) -> None:
        self._nodes: dict[str, Any] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._invalidations: dict[str, InvalidationRecord] = {}
        self._supersessions: dict[str, SupersessionRecord] = {}

    def register_node(self, node_id: str, record: Any | None = None) -> None:
        if not node_id:
            raise ValueError("node ID must not be empty")
        if node_id in self._nodes:
            raise DuplicateRecordError(f"duplicate node ID: {node_id}")
        self._nodes[node_id] = record if record is not None else {"node_id": node_id}
        self._dependencies[node_id] = set()

    def node_record(self, node_id: str) -> Any:
        return self._nodes.get(node_id)

    def add_run(self, manifest: RunManifest) -> None:
        self.register_node(manifest.run_id, manifest)

    def add_gate_attempt(self, attempt: GateEvaluationAttempt) -> None:
        self.register_node(attempt.evaluation_id, attempt)
        for predecessor in attempt.predecessor_dependencies:
            self.add_dependency(attempt.evaluation_id, predecessor)

    def add_dependency(self, node_id: str, predecessor_id: str) -> None:
        if node_id not in self._nodes:
            raise UnknownDependencyError(f"unknown dependency node: {node_id}")
        if predecessor_id not in self._nodes:
            raise UnknownDependencyError(f"unknown dependency node: {predecessor_id}")
        self._dependencies[node_id].add(predecessor_id)

    def append_invalidation(self, record: InvalidationRecord) -> None:
        if record.record_id in self._invalidations or record.record_id in self._supersessions:
            raise DuplicateRecordError(f"duplicate record ID: {record.record_id}")
        self._require_node(record.node_id)
        self._invalidations[record.record_id] = record

    def append_supersession(self, record: SupersessionRecord) -> None:
        if record.record_id in self._supersessions or record.record_id in self._invalidations:
            raise DuplicateRecordError(f"duplicate record ID: {record.record_id}")
        self._require_node(record.superseded_node_id)
        self._require_node(record.replacement_node_id)
        self._supersessions[record.record_id] = record

    def active_gate_attempt(self, definition_id: str) -> GateEvaluationAttempt | None:
        superseded = {record.superseded_node_id for record in self._supersessions.values()}
        candidates = [
            node
            for node in self._nodes.values()
            if isinstance(node, GateEvaluationAttempt)
            and node.definition_id == definition_id
            and node.evaluation_id not in superseded
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda attempt: attempt.attempt_number)

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise UnknownDependencyError(f"unknown dependency node: {node_id}")

    def _direct_record(self, node_id: str) -> InvalidationRecord | SupersessionRecord | None:
        for record in self._invalidations.values():
            if record.node_id == node_id:
                return record
        for record in self._supersessions.values():
            if record.superseded_node_id == node_id:
                return record
        return None

    def _predecessors(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._dependencies.get(node_id, ())))


def resolve_effective_validity(node_id: str, registry: Registry) -> EffectiveValidity:
    """Resolve lifecycle validity without rewriting any stored historical record."""

    def resolve(current: str, visiting: tuple[str, ...]) -> EffectiveValidity:
        if current not in registry._nodes:
            return EffectiveValidity(
                status=EffectiveValidityStatus.STALE_DEPENDENCY,
                reason=f"unknown dependency node: {current}",
                invalidated_sources=(current,),
                dependency_path=visiting + (current,),
            )
        if current in visiting:
            return EffectiveValidity(
                status=EffectiveValidityStatus.STALE_DEPENDENCY,
                reason="dependency cycle detected; validity fails closed",
                invalidated_sources=(),
                dependency_path=visiting + (current,),
            )

        direct = registry._direct_record(current)
        if isinstance(direct, InvalidationRecord):
            return EffectiveValidity(
                status=EffectiveValidityStatus.INVALIDATED,
                reason=direct.reason,
                invalidated_sources=(current,),
                dependency_path=(current,),
            )
        if isinstance(direct, SupersessionRecord):
            return EffectiveValidity(
                status=EffectiveValidityStatus.INVALIDATED,
                reason=direct.reason,
                invalidated_sources=(current,),
                dependency_path=(current,),
                replacement_ref=direct.replacement_node_id,
            )

        for predecessor in registry._predecessors(current):
            predecessor_validity = resolve(predecessor, visiting + (current,))
            if predecessor_validity.status is not EffectiveValidityStatus.VALID:
                path = predecessor_validity.dependency_path
                if not path or path[0] != current:
                    path = (current,) + path
                return EffectiveValidity(
                    status=EffectiveValidityStatus.STALE_DEPENDENCY,
                    reason=f"stale dependency: {predecessor_validity.reason}",
                    invalidated_sources=predecessor_validity.invalidated_sources,
                    dependency_path=path,
                    replacement_ref=predecessor_validity.replacement_ref,
                )
        return EffectiveValidity(
            status=EffectiveValidityStatus.VALID,
            reason="no direct invalidation or stale predecessor",
        )

    return resolve(node_id, ())
