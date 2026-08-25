"""Automatic formal closure evaluator for RG-01.

The evaluator is deliberately read-only with respect to upstream artifacts.  It
checks the current-effective E1.1, E1.2-v2 and E1.3-v2 lineage before creating
an append-only GateEvaluationAttempt.  No temporal operation is performed here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from sc_mv_dmer.features.mert_timegrid import (
    load_timegrid_binding,
    verify_timegrid_binding,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.dimensions import (
    load_upstream_manifest,
    verify_downstream_binding,
)
from sc_mv_dmer.foundation.lifecycle import (
    AutomaticAuthorityMetadata,
    GateEvaluationAttempt,
    HistoricalVerdict,
)
from sc_mv_dmer.foundation.validity import (
    EffectiveValidityStatus,
    Registry,
    resolve_effective_validity,
)


class RG01EvaluationError(ValueError):
    """Raised when an RG-01 evidence bundle cannot be loaded."""


@dataclass(frozen=True)
class RG01Evidence:
    """Logical paths consumed by the formal RG-01 evaluator."""

    repository_root: Path
    e11_report: str = "reports/experiments/e1-1-mert-real-frame-rate.json"
    e11_qualification: str = "evidence/qualifications/e1-1-mert-frame-rate-verification.json"
    e12_binding: str = "manifests/dimensions/mert-downstream-dimensions-v2.json"
    e12_qualification: str = "evidence/qualifications/e1-2-mert-downstream-dimension-derivation-v2.json"
    e13_timegrid: str = "manifests/timegrid/mert-2hz-timegrid-v2.json"
    e13_qualification: str = "evidence/qualifications/e1-3-mert-timegrid-binding-v2.json"
    upstream_manifest: str = "manifests/upstream/mert-v1-95m-primary-v1.json"

    def path(self, logical_path: str) -> Path:
        candidate = (self.repository_root / logical_path).resolve()
        root = self.repository_root.resolve()
        if root not in candidate.parents:
            raise RG01EvaluationError(f"evidence path escapes repository: {logical_path}")
        return candidate


def _read_json(evidence: RG01Evidence, logical_path: str) -> dict[str, Any]:
    path = evidence.path(logical_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RG01EvaluationError(f"cannot load JSON evidence: {logical_path}") from exc
    if not isinstance(payload, dict):
        raise RG01EvaluationError(f"JSON evidence is not an object: {logical_path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qualification_checksum(payload: dict[str, Any]) -> bool:
    observed = payload.get("qualification_sha256")
    if not isinstance(observed, str):
        return False
    unsigned = dict(payload)
    unsigned.pop("qualification_sha256", None)
    return sha256_canonical(unsigned) == observed


def _git_state(root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RG01EvaluationError("unable to establish Git evidence state") from exc
    if len(commit) < 7:
        raise RG01EvaluationError("Git HEAD is not a valid commit")
    return commit, dirty


def _criterion(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"criterion": name, "passed": bool(passed), "detail": detail}


def _audit(evidence: RG01Evidence) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...]]:
    root = evidence.repository_root.resolve()
    e11 = _read_json(evidence, evidence.e11_report)
    e11q = _read_json(evidence, evidence.e11_qualification)
    e12q = _read_json(evidence, evidence.e12_qualification)
    e13q = _read_json(evidence, evidence.e13_qualification)
    e12 = _read_json(evidence, evidence.e12_binding)
    e13 = _read_json(evidence, evidence.e13_timegrid)
    upstream = load_upstream_manifest(evidence.path(evidence.upstream_manifest))
    criteria: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        criteria.append(_criterion(name, passed, detail))

    # File hashes are part of the lineage, not merely display metadata.
    source_paths = {
        "e11_report": evidence.e11_report,
        "e11_qualification": evidence.e11_qualification,
        "e12_binding": evidence.e12_binding,
        "e12_qualification": evidence.e12_qualification,
        "e13_timegrid": evidence.e13_timegrid,
        "e13_qualification": evidence.e13_qualification,
        "upstream_manifest": evidence.upstream_manifest,
    }
    source_hashes = {
        key: _file_sha256(evidence.path(path)) for key, path in source_paths.items()
    }
    add("git_clean_preflight", not _git_state(root)[1], "formal evaluation must start from clean Git")
    add("e11_pass", e11q.get("e1_1_verdict") == "PASS" and e11.get("structural_consistency") == "PASS", "E1.1 verdict and structural consistency")
    add("e11_qualification_checksum", _qualification_checksum(e11q), "E1.1 qualification canonical checksum")
    add("e11_observations", _valid_e11_observations(e11), "P1/P2 finite, exact-length, T_raw=3374, D=768")
    add("e11_report_binding", e11q.get("report_logical_path") == evidence.e11_report and e11q.get("report_file_sha256") == source_hashes["e11_report"], "E1.1 report path and checksum")
    add("pinned_upstream", _valid_upstream(upstream, e11), "pinned repository, revision and snapshot identity")

    e12_valid = False
    try:
        from sc_mv_dmer.foundation.dimensions import DownstreamDimensionBinding
        e12_model = DownstreamDimensionBinding.model_validate(e12)
        verify_downstream_binding(e12_model)
        endpoints = {node.node_id for node in e12_model.nodes}
        edges_complete = len(e12_model.nodes) == 40 and len(e12_model.edges) == 60 and all(
            edge.source_node_id in endpoints and edge.target_node_id in endpoints
            for edge in e12_model.edges
        )
        e12_valid = (
            e12_model.observed_t_raw == 3374
            and e12_model.hidden_dimension == 768
            and e12_model.timesnet_raw_temporal_input_dimension == 3374
            and e12_model.canonical_downstream_t == 90
            and e12_model.required_temporal_reduction.ratio_numerator == 3374
            and e12_model.required_temporal_reduction.ratio_denominator == 90
            and e12_model.required_temporal_reduction.operation == "SAMPLE_DOMAIN_FRAME_CENTER_BIN_MEAN"
            and edges_complete
            and set(e12_model.forbidden_downstream_actions) >= {"TRIM", "PADDING", "INTERPOLATION", "RESHAPE", "ADAPTIVE_POOLING"}
        )
    except Exception:
        e12_model = None
    add("e12_binding", e12_valid, "v2 dimension binding checksum, DAG completeness and T=90")
    add("e12_qualification", e12q.get("e1_2_verdict") == "PASS" and e12q.get("e1_3_readiness") == "READY" and _qualification_checksum(e12q) and e12q.get("binding_logical_path") == evidence.e12_binding and e12q.get("binding_file_sha256") == source_hashes["e12_binding"], "current E1.2 v2 qualification and binding checksum")

    e13_valid = False
    try:
        e13_model = load_timegrid_binding(evidence.path(evidence.e13_timegrid))
        verify_timegrid_binding(e13_model)
        frames = e13_model.frames
        e13_valid = (
            e13_model.raw_frame_count == 3374
            and e13_model.expected_frame_count == 3374
            and e13_model.valid_frame_count == 3374
            and e13_model.assigned_frame_count == 3374
            and e13_model.unassigned_frame_count == 0
            and e13_model.duplicate_assignment_count == 0
            and e13_model.valid_bin_count == 90
            and len(e13_model.bins) == 90
            and all(item.valid_bin and item.frame_count > 0 for item in e13_model.bins)
            and len(frames) == 3374
            and [item.frame_index for item in frames] == list(range(3374))
            and all(item.valid_frame and item.assignment_count == 1 for item in frames)
            and sum(e13_model.bin_frame_counts) == 3374
            and set(e13_model.forbidden_operations) >= {"TRIM", "PADDING", "INTERPOLATION", "RESHAPE", "ADAPTIVE_POOLING"}
        )
    except Exception:
        e13_model = None
    add("e13_timegrid", e13_valid, "v2 exact-once frame accounting and 90 non-empty bins")
    add("e13_qualification", e13q.get("e1_3_verdict") == "PASS" and e13q.get("e1_4_readiness") == "READY" and _qualification_checksum(e13q) and e13q.get("timegrid_logical_path") == evidence.e13_timegrid and e13q.get("timegrid_file_sha256") == source_hashes["e13_timegrid"], "current E1.3 v2 qualification and TimeGrid checksum")

    add("lineage", e12q.get("predecessor_qualification") == evidence.e11_qualification and e13q.get("predecessor_qualification") == evidence.e12_qualification, "E1.1 -> E1.2 v2 -> E1.3 v2 predecessor chain")
    add("forbidden_operation_firewall", _forbidden_firewall(e12, e13), "no trim/padding/interpolation/reshape/adaptive pooling semantics")

    source_artifacts = {path: _file_sha256(evidence.path(path)) for path in source_paths.values()}
    predecessor_dependencies = (
        evidence.e11_qualification,
        evidence.e12_qualification,
        evidence.e13_qualification,
        evidence.upstream_manifest,
    )
    return (
        {"criteria": criteria, "source_artifacts": source_artifacts, "predecessor_dependencies": predecessor_dependencies},
        source_hashes,
        predecessor_dependencies,
    )


def _valid_e11_observations(e11: dict[str, Any]) -> bool:
    observations = [e11.get("p1"), e11.get("p2")]
    if any(not isinstance(item, dict) for item in observations):
        return False
    for item in observations:
        shape = item.get("layer5_shape")
        if not (
            item.get("observed_frame_count") == 3374
            and item.get("hidden_dimension") == 768
            and item.get("observed_fps") == 3374 / 45.0
            and item.get("input_sample_rate") == 24000
            and item.get("input_sample_count") == 1080000
            and item.get("input_duration_seconds") == 45.0
            and shape == [1, 3374, 768]
            and item.get("layer6_shape") == [1, 3374, 768]
            and item.get("layer5_finite") is True
            and item.get("layer6_finite") is True
        ):
            return False
    structural_fields = (
        "observed_frame_count",
        "hidden_dimension",
        "observed_fps",
        "input_sample_rate",
        "input_sample_count",
        "input_duration_seconds",
        "input_tensor_shape",
        "layer5_shape",
        "layer6_shape",
        "layer5_finite",
        "layer6_finite",
    )
    return all(observations[0].get(field) == observations[1].get(field) for field in structural_fields)


def _valid_upstream(upstream: dict[str, Any], e11: dict[str, Any]) -> bool:
    return (
        upstream.get("repository_id") == "m-a-p/MERT-v1-95M"
        and upstream.get("revision") == "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
        and upstream.get("snapshot_aggregate_sha256") == e11.get("snapshot_aggregate_sha256")
        and upstream.get("semantic_identity_sha256") == "341484e496d82b55422a638b10ad24ce2e5d1fa507627dad1c849049c9f7c39d"
    )


def _forbidden_firewall(e12: dict[str, Any], e13: dict[str, Any]) -> bool:
    e12_ops = set(e12.get("forbidden_downstream_actions", ()))
    e13_ops = set(e13.get("forbidden_operations", ()))
    return {"TRIM", "PADDING", "INTERPOLATION", "RESHAPE", "ADAPTIVE_POOLING"}.issubset(e12_ops | e13_ops)


def evaluate_rg01_report(evidence: RG01Evidence) -> dict[str, Any]:
    """Return the complete deterministic audit report without writing files."""

    audit, source_hashes, predecessors = _audit(evidence)
    commit, dirty = _git_state(evidence.repository_root.resolve())
    passed = all(item["passed"] for item in audit["criteria"])
    bundle_payload = {
        "definition_id": "RG-01",
        "gate_version": "RG-01-v1",
        "criteria": audit["criteria"],
        "source_artifacts": audit["source_artifacts"],
        "predecessor_dependencies": list(predecessors),
    }
    bundle_sha256 = sha256_canonical(bundle_payload)
    registry = Registry()
    for node in predecessors:
        registry.register_node(node)
    registry.add_dependency(evidence.e12_qualification, evidence.e11_qualification)
    registry.add_dependency(evidence.e13_qualification, evidence.e12_qualification)
    statuses = {
        node: resolve_effective_validity(node, registry).status.value for node in predecessors
    }
    evaluation_id = "RG-01-E1.4-attempt-1"
    attempt = GateEvaluationAttempt(
        evaluation_id=evaluation_id,
        gate_version="RG-01-v1",
        attempt_number=1,
        definition_id="RG-01",
        verdict=HistoricalVerdict.PASS if passed else HistoricalVerdict.FAIL,
        authority_kind="automatic",
        authority_metadata=AutomaticAuthorityMetadata(
            automation_identity="sc_mv_dmer.rg01.formal_evaluator.v1"
        ),
        evidence_commit=commit,
        evidence_bundle_sha256=bundle_sha256,
        source_run_ids=("E1.1:P1_CANONICAL_EXACT_LENGTH", "E1.1:P2_REAL_DEAM"),
        predecessor_dependencies=predecessors,
    )
    return {
        "schema_version": "1.0",
        "evaluation_id": evaluation_id,
        "gate_version": "RG-01-v1",
        "definition_id": "RG-01",
        "attempt_number": 1,
        "verdict": attempt.verdict.value,
        "current_effective_verdict": attempt.verdict.value if passed and not dirty else "STALE_DEPENDENCY",
        "evidence_commit": commit,
        "observed_git_dirty": dirty,
        "authority": attempt.authority_metadata.model_dump(mode="json"),
        "evidence_bundle_sha256": bundle_sha256,
        "source_run_ids": list(attempt.source_run_ids),
        "predecessor_dependencies": list(predecessors),
        "predecessor_effective_validity": statuses,
        "source_artifacts": audit["source_artifacts"],
        "criteria": audit["criteria"],
        "e1_4_verdict": attempt.verdict.value,
        "e2_unlocked": bool(passed and not dirty),
        "attempt": attempt.model_dump(mode="json"),
        "_bundle_payload": bundle_payload,
    }


def evaluate_rg01(evidence: RG01Evidence) -> GateEvaluationAttempt:
    """Evaluate RG-01 and return its immutable historical attempt."""

    return GateEvaluationAttempt.model_validate(evaluate_rg01_report(evidence)["attempt"])


def write_rg01_evidence(evidence: RG01Evidence, *, attempt_path: Path, qualification_path: Path, report_path: Path, text_report_path: Path) -> GateEvaluationAttempt:
    """Run RG-01 and write each terminal artifact exactly once."""

    report = evaluate_rg01_report(evidence)
    attempt = GateEvaluationAttempt.model_validate(report["attempt"])
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    qualification_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text_report_path.parent.mkdir(parents=True, exist_ok=True)
    gate_payload = attempt.model_dump(mode="json")
    qualification = dict(report)
    qualification.pop("attempt", None)
    qualification.pop("_bundle_payload", None)
    qualification["qualification_sha256"] = sha256_canonical(qualification)
    for path, payload in ((attempt_path, gate_payload), (qualification_path, qualification), (report_path, report)):
        with Path(path).open("x", encoding="utf-8") as output:
            output.write(canonical_json(payload) + "\n")
    text = [
        "=== RG-01 FORMAL CLOSURE (E1.4) ===",
        f"evaluation_id={attempt.evaluation_id}",
        f"gate_version={attempt.gate_version}",
        f"verdict={attempt.verdict.value}",
        f"evidence_commit={attempt.evidence_commit}",
        f"evidence_bundle_sha256={attempt.evidence_bundle_sha256}",
        f"current_effective_verdict={report['current_effective_verdict']}",
    ] + [f"{item['criterion']}={item['passed']} :: {item['detail']}" for item in report["criteria"]]
    with Path(text_report_path).open("x", encoding="utf-8") as output:
        output.write("\n".join(text) + "\n")
    return attempt
