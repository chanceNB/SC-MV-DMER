from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from sc_mv_dmer.foundation.lifecycle import (
    DuplicateRecordError,
    AutomaticAuthorityMetadata,
    GateEvaluationAttempt,
    HistoricalVerdict,
    HumanApprovalState,
    InvalidationRecord,
    ManualAuthorityMetadata,
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


def test_multiple_stale_predecessors_report_all_invalidated_sources_and_paths() -> None:
    registry = Registry()
    add_nodes(registry, "source-a", "source-b", "downstream")
    registry.add_dependency("downstream", "source-a")
    registry.add_dependency("downstream", "source-b")
    registry.append_invalidation(
        InvalidationRecord(
            record_id="invalidation-a",
            node_id="source-a",
            reason="source A no longer valid",
            invalidated_by="reviewer-001",
        )
    )
    registry.append_invalidation(
        InvalidationRecord(
            record_id="invalidation-b",
            node_id="source-b",
            reason="source B no longer valid",
            invalidated_by="reviewer-001",
        )
    )

    result = resolve_effective_validity("downstream", registry)

    assert result.status is EffectiveValidityStatus.STALE_DEPENDENCY
    assert result.invalidated_sources == ("source-a", "source-b")
    assert result.dependency_paths == (
        ("downstream", "source-a"),
        ("downstream", "source-b"),
    )


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
        authority_metadata=AutomaticAuthorityMetadata(automation_identity="gate-engine-v1"),
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
        authority_metadata=AutomaticAuthorityMetadata(automation_identity="gate-engine-v2"),
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
        authority_metadata=ManualAuthorityMetadata(
            approval_state=HumanApprovalState.PENDING,
        ),
        evidence_commit="1234567890abcdef",
        evidence_bundle_sha256="f" * 64,
        source_run_ids=(),
        predecessor_dependencies=(),
    )

    assert pending.authority_metadata.approval_state is HumanApprovalState.PENDING
    assert pending.authority_metadata.reviewer_identity is None


def test_manual_pass_requires_verified_human_identity_and_approval_artifact() -> None:
    values = {
        "evaluation_id": "gate-eval-approved",
        "gate_version": "1.0",
        "attempt_number": 1,
        "definition_id": "human-gate",
        "verdict": HistoricalVerdict.PASS,
        "authority_kind": "manual",
        "evidence_commit": "1234567890abcdef",
        "evidence_bundle_sha256": "a" * 64,
        "source_run_ids": (),
        "predecessor_dependencies": (),
    }
    with pytest.raises(ValidationError, match="manual PASS"):
        GateEvaluationAttempt(
            **values,
            authority_metadata=ManualAuthorityMetadata(
                approval_state=HumanApprovalState.PENDING,
            ),
        )

    approved = GateEvaluationAttempt(
        **values,
        authority_metadata=ManualAuthorityMetadata(
            approval_state=HumanApprovalState.APPROVED,
            reviewer_identity="reviewer-001",
            approval_artifact_sha256="b" * 64,
        ),
    )

    assert approved.authority_metadata.reviewer_identity == "reviewer-001"


@pytest.mark.parametrize(
    "identity",
    ["codex", "automatic", "Codex Automation", "automatic-gate-runner"],
)
def test_manual_approval_rejects_automatic_or_codex_identities(identity: str) -> None:
    with pytest.raises(ValidationError, match="human reviewer"):
        ManualAuthorityMetadata(
            approval_state=HumanApprovalState.APPROVED,
            reviewer_identity=identity,
            approval_artifact_sha256="b" * 64,
        )
