"""Immutable E0-minimum bootstrap qualification bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from sc_mv_dmer.data.probes import DeamProbeManifest
from sc_mv_dmer.features.mert_upstream import (
    PINNED_MERT_REPOSITORY,
    PINNED_MERT_REVISION,
    MertUpstreamStatus,
)
from sc_mv_dmer.foundation.capabilities import CapabilityRole, DependencyValidity
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.manifests import ImmutableRecord
from sc_mv_dmer.foundation.preflight import PreflightObservation
from sc_mv_dmer.foundation.stages import FROZEN_CONTRACT_BLOCKER


class E1Readiness(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class QualificationWriteError(FileExistsError):
    """Raised when attempting to overwrite terminal qualification evidence."""


class E0MinimumQualificationBundle(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    implementation_git_commit: str
    observed_git_dirty: bool
    config_schema_version: str
    research_spec_version: str
    semantic_config_hash: str
    resolved_config_hash: str
    variants_catalog_sha256: str
    stages_catalog_sha256: str
    full_catalog_blocker: str
    full_catalog_blocker_deferred: bool
    probe_manifest_id: str
    probe_manifest_sha256: str
    dataset_id: str
    song_id: str
    sample_id: str
    source_relative_path: str
    source_sha256: str
    probe_seconds: Literal[45]
    target_sample_rate_hz: Literal[24000]
    target_sample_count: Literal[1080000]
    probe_preparation_status: Literal["SPECIFIED_NOT_EXECUTED"]
    mert_repository: str
    mert_revision: str
    mert_upstream_identity: str
    mert_local_verification_status: str
    e0_implementation_status: Literal["COMPLETE"]
    e1_1_readiness: E1Readiness
    e1_1_blocked_roles: tuple[str, ...]
    e1_1_blocked_reasons: tuple[str, ...]
    formal_execution_ready: bool
    paper_eligible: bool
    qualification_sha256: str


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_probe_contract(probe: DeamProbeManifest) -> None:
    if (
        probe.probe_seconds != 45
        or probe.target_sample_rate_hz != 24_000
        or probe.target_sample_count != 1_080_000
        or probe.preparation_status != "SPECIFIED_NOT_EXECUTED"
        or probe.source_duration_seconds < 45
    ):
        raise ValueError("probe manifest does not satisfy the exact E0 45s/24kHz contract")


def _assert_pinned_mert_identity(status: MertUpstreamStatus) -> None:
    if status.repository != PINNED_MERT_REPOSITORY or status.revision != PINNED_MERT_REVISION:
        raise ValueError("MERT status does not match the pinned repository/revision identity")


def load_probe_manifest(path: Path) -> DeamProbeManifest:
    """Load one previously registered probe manifest without touching its source audio."""

    return DeamProbeManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_mert_status(path: Path) -> MertUpstreamStatus:
    """Load a status artifact while retaining only the status contract fields."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MERT status artifact must be an object")
    fields = MertUpstreamStatus.model_fields
    return MertUpstreamStatus.model_validate(
        {key: value for key, value in payload.items() if key in fields}
    )


def _external_prerequisite_reasons(
    prerequisites: Mapping[CapabilityRole | str, object],
) -> tuple[str, ...]:
    required: tuple[CapabilityRole | str, ...] = (
        "E0_MINIMUM_BOOTSTRAP",
        CapabilityRole.MERT_UPSTREAM_IDENTITY,
        CapabilityRole.MERT_PROBE_IDENTITY,
        CapabilityRole.MERT_PROBE_REPRESENTATION,
    )
    return tuple(
        f"E1_1_PREREQUISITE_NOT_VALID:{dependency if isinstance(dependency, str) else dependency.value}"
        for dependency in required
        if prerequisites.get(dependency) is not DependencyValidity.VALID
    )


def qualify_e0_minimum_bootstrap(
    observation: PreflightObservation,
    probe: DeamProbeManifest,
    mert_status: MertUpstreamStatus,
    e1_1_prerequisite_validity: Mapping[CapabilityRole | str, object],
) -> E0MinimumQualificationBundle:
    """Build E0 evidence only; this reads declared artifacts and never runs MERT."""

    _assert_probe_contract(probe)
    _assert_pinned_mert_identity(mert_status)
    root = _repository_root()
    external_reasons = _external_prerequisite_reasons(e1_1_prerequisite_validity)
    missing_local_roles = tuple(mert_status.missing_roles)
    blocked_reasons = external_reasons + tuple(
        f"MERT_LOCAL_ROLE_NOT_VERIFIED:{role}" for role in missing_local_roles
    )
    if mert_status.local_verification_status != "VERIFIED":
        blocked_reasons += (
            f"MERT_LOCAL_VERIFICATION_NOT_VERIFIED:{mert_status.local_verification_status}",
        )
    readiness = E1Readiness.BLOCKED if blocked_reasons else E1Readiness.READY
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "implementation_git_commit": observation.implementation_git_commit,
        "observed_git_dirty": observation.observed_git_dirty,
        "config_schema_version": observation.config_schema_version,
        "research_spec_version": observation.research_spec_version,
        "semantic_config_hash": observation.semantic_config_hash,
        "resolved_config_hash": observation.resolved_config_hash,
        "variants_catalog_sha256": _file_sha256(root / "configs" / "catalog" / "variants.yaml"),
        "stages_catalog_sha256": _file_sha256(root / "configs" / "catalog" / "stages.yaml"),
        "full_catalog_blocker": FROZEN_CONTRACT_BLOCKER,
        "full_catalog_blocker_deferred": True,
        "probe_manifest_id": probe.sample_id,
        "probe_manifest_sha256": sha256_canonical(probe.model_dump(mode="json")),
        "dataset_id": probe.dataset_id,
        "song_id": probe.song_id,
        "sample_id": probe.sample_id,
        "source_relative_path": probe.source_relative_path,
        "source_sha256": probe.source_sha256,
        "probe_seconds": 45,
        "target_sample_rate_hz": 24_000,
        "target_sample_count": 1_080_000,
        "probe_preparation_status": probe.preparation_status,
        "mert_repository": mert_status.repository,
        "mert_revision": mert_status.revision,
        "mert_upstream_identity": mert_status.upstream_identity,
        "mert_local_verification_status": mert_status.local_verification_status,
        "e0_implementation_status": "COMPLETE",
        "e1_1_readiness": readiness,
        "e1_1_blocked_roles": missing_local_roles,
        "e1_1_blocked_reasons": blocked_reasons,
        "formal_execution_ready": False,
        "paper_eligible": False,
    }
    return E0MinimumQualificationBundle(
        **payload,
        qualification_sha256=sha256_canonical(payload),
    )


def write_terminal_qualification(
    bundle: E0MinimumQualificationBundle, destination: Path
) -> None:
    """Write a new terminal artifact once; corrections require a different path/version."""

    destination = Path(destination)
    if destination.exists():
        raise QualificationWriteError(f"terminal qualification already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(bundle.model_dump(mode="json")) + "\n", encoding="utf-8")
