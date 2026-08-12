from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sc_mv_dmer.foundation.lifecycle import (
    DuplicateRecordError,
    GateEvaluationAttempt,
    HistoricalVerdict,
    InvalidationRecord,
    SupersessionRecord,
)
from sc_mv_dmer.foundation.validity import (
    EffectiveValidityStatus,
    Registry,
    UnknownDependencyError,
    resolve_effective_validity,
)


def add_nodes(registry: Registry, *node_ids: str) -> None:
    for node_id in node_ids:
        registry.register_node(node_id)


def test_invalidated_predecessor_makes_downstream_stale_without_rewriting_history() -> None:
    registry = Registry()
    add_nodes(registry, "source", "downstream")
    registry.add_dependency("downstream", "source")
    registry.append_invalidation(
        InvalidationRecord(
            record_id="invalidation-001",
            node_id="source",
            reason="source checksum is no longer trusted",
            invalidated_by="reviewer-001",
            recorded_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        )
    )

    result = resolve_effective_validity("downstream", registry)

    assert result.status is EffectiveValidityStatus.STALE_DEPENDENCY
    assert result.invalidated_sources == ("source",)
    assert result.dependency_path == ("downstream", "source")
    assert registry.node_record("source") is not None


def test_invalidation_only_affects_dependency_descendants() -> None:
    registry = Registry()
    add_nodes(registry, "source", "affected", "unaffected")
    registry.add_dependency("affected", "source")
    registry.append_invalidation(
        InvalidationRecord(
            record_id="invalidation-001",
            node_id="source",
            reason="source no longer valid",
            invalidated_by="reviewer-001",
        )
    )

    assert resolve_effective_validity("affected", registry).status is EffectiveValidityStatus.STALE_DEPENDENCY
    assert resolve_effective_validity("unaffected", registry).status is EffectiveValidityStatus.VALID


def test_replacement_provenance_is_returned_for_superseded_node() -> None:
    registry = Registry()
    add_nodes(registry, "old", "replacement", "dependent")
    registry.add_dependency("dependent", "old")
    registry.append_supersession(
        SupersessionRecord(
            record_id="supersession-001",
            superseded_node_id="old",
            replacement_node_id="replacement",
            reason="corrected evidence bundle",
        )
    )

    result = resolve_effective_validity("dependent", registry)

    assert result.status is EffectiveValidityStatus.STALE_DEPENDENCY
    assert result.replacement_ref == "replacement"
    assert result.dependency_path == ("dependent", "old")


def test_old_superseded_gate_pass_is_not_an_active_predecessor() -> None:
    registry = Registry()
    old = GateEvaluationAttempt(
        evaluation_id="gate-eval-old",
        gate_version="1.0",
        attempt_number=1,
        definition_id="gate-definition",
        verdict=HistoricalVerdict.PASS,
        authority_kind="automatic",
        authority_metadata={"rule": "v1"},
        evidence_commit="1234567890abcdef",
        evidence_bundle_sha256="d" * 64,
        source_run_ids=("run-001",),
        predecessor_dependencies=(),
    )
    new = GateEvaluationAttempt(
        evaluation_id="gate-eval-new",
        gate_version="2.0",
        attempt_number=1,
        definition_id="gate-definition",
        verdict=HistoricalVerdict.BLOCKED,
        authority_kind="automatic",
        authority_metadata={"rule": "v2"},
        evidence_commit="fedcba0987654321",
        evidence_bundle_sha256="e" * 64,
        source_run_ids=("run-001",),
        predecessor_dependencies=(),
    )
    registry.add_gate_attempt(old)
    registry.add_gate_attempt(new)
    registry.append_supersession(
        SupersessionRecord(
            record_id="supersession-001",
            superseded_node_id=old.evaluation_id,
            replacement_node_id=new.evaluation_id,
            reason="Gate definition v2 supersedes v1 evidence",
        )
    )

    assert registry.active_gate_attempt("gate-definition") == new
    assert resolve_effective_validity(old.evaluation_id, registry).status is EffectiveValidityStatus.INVALIDATED


def test_unknown_dependency_and_cycles_fail_closed() -> None:
    registry = Registry()
    add_nodes(registry, "known", "cycle-a", "cycle-b")
    with pytest.raises(UnknownDependencyError, match="unknown"):
        registry.add_dependency("known", "missing")

    registry.add_dependency("cycle-a", "cycle-b")
    registry.add_dependency("cycle-b", "cycle-a")
    result = resolve_effective_validity("cycle-a", registry)

    assert result.status is EffectiveValidityStatus.STALE_DEPENDENCY
    assert "cycle" in result.reason.lower()


def test_records_are_append_only_and_human_approval_is_never_synthesized() -> None:
    registry = Registry()
    registry.register_node("node")
    record = InvalidationRecord(
        record_id="invalidation-001",
        node_id="node",
        reason="manual review pending",
        invalidated_by="reviewer-001",
    )
    registry.append_invalidation(record)
    with pytest.raises(DuplicateRecordError, match="duplicate"):
        registry.append_invalidation(record)

    pending = GateEvaluationAttempt(
        evaluation_id="gate-eval-pending",
        gate_version="1.0",
        attempt_number=1,
        definition_id="human-gate",
        verdict=HistoricalVerdict.INSUFFICIENT_EVIDENCE,
        authority_kind="manual",
        authority_metadata={"human_criterion_status": "PENDING"},
        evidence_commit="1234567890abcdef",
        evidence_bundle_sha256="f" * 64,
        source_run_ids=(),
        predecessor_dependencies=(),
    )

    assert pending.authority_metadata["human_criterion_status"] == "PENDING"
    assert "approved_by" not in pending.authority_metadata
