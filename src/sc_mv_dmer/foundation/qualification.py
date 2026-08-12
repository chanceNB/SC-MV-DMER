"""Immutable E0-minimum bootstrap qualification bundles."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator

from sc_mv_dmer.data.discovery import DEAM_DATASET_VERSION
from sc_mv_dmer.data.probes import (
    PROBE_RULE_VERSION,
    DeamProbeManifest,
)
from sc_mv_dmer.features.mert_upstream import (
    PINNED_MERT_REPOSITORY,
    PINNED_MERT_REVISION,
    MertUpstreamStatus,
    UpstreamModelBlocker,
    bind_mert_formal,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from pydantic import field_validator

from sc_mv_dmer.foundation.manifests import FrozenDict, ImmutableRecord
from sc_mv_dmer.foundation.preflight import PreflightObservation
from sc_mv_dmer.foundation.identity import stable_id
from sc_mv_dmer.foundation.stages import (
    FROZEN_CONTRACT_BLOCKER,
    audit_stage_authority,
    load_stage_authority_catalog,
)


class E1Readiness(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


class QualificationWriteError(FileExistsError):
    """Raised when attempting to overwrite terminal qualification evidence."""


class E0MinimumQualificationBundle(ImmutableRecord):
    schema_version: Literal["1.1"] = "1.1"
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
    stage_authority_coverage: tuple[str, ...]
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
    e1_1_prerequisite_validity: Mapping[str, str]
    formal_execution_ready: bool
    paper_eligible: bool
    supersedes: str | None = None
    correction_reason: str | None = None
    qualification_sha256: str

    @field_validator("e1_1_prerequisite_validity", mode="after")
    @classmethod
    def _freeze_prerequisite_validity(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return FrozenDict(value)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


_QUALIFICATION_SCHEMA_PATHS = {
    "1.0": "schemas/e0_minimum_qualification.v1.schema.json",
    "1.1": "schemas/e0_minimum_qualification.schema.json",
}


def load_qualification_schema(artifact: Mapping[str, object]) -> dict[str, object]:
    """Select the immutable schema contract addressed by an artifact's version."""

    version = artifact.get("schema_version")
    path = _QUALIFICATION_SCHEMA_PATHS.get(version)
    if path is None:
        raise ValueError(f"unsupported qualification schema version: {version!r}")
    loaded = json.loads((_repository_root() / path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("qualification schema must be an object")
    return loaded


def validate_qualification_artifact(artifact: Mapping[str, object]) -> None:
    """Validate an artifact with its selected full Draft 2020-12 contract."""

    schema = load_qualification_schema(artifact)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(dict(artifact))


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


def _probe_identity_is_valid(probe: DeamProbeManifest, data_root: Path) -> bool:
    relative_path = Path(probe.source_relative_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return False
    match = re.fullmatch(r"DEAM_audio/MEMD_audio/(\d+)\.mp3", probe.source_relative_path)
    if match is None:
        return False
    dataset_id = stable_id("dataset", ("DEAM", DEAM_DATASET_VERSION))
    song_id = stable_id("song", (dataset_id, match.group(1)))
    sample_id = stable_id("sample", (song_id, PROBE_RULE_VERSION, "0", "45"))
    source = Path(data_root).resolve() / relative_path
    if not source.is_file() or _file_sha256(source) != probe.source_sha256:
        return False
    return (
        probe.dataset_id == dataset_id
        and probe.song_id == song_id
        and probe.sample_id == sample_id
    )


def _probe_representation_is_valid(probe: DeamProbeManifest) -> bool:
    try:
        semantic = json.loads(probe.semantic_identity_json)
    except json.JSONDecodeError:
        return False
    if not isinstance(semantic, dict) or sha256_canonical(semantic) != probe.semantic_identity_sha256:
        return False
    return (
        semantic
        == {
            "dataset_id": probe.dataset_id,
            "song_id": probe.song_id,
            "source_relative_path": probe.source_relative_path,
            "source_sha256": probe.source_sha256,
            "probe_seconds": 45,
            "target_sample_rate_hz": 24_000,
            "target_sample_count": 1_080_000,
            "rule_version": PROBE_RULE_VERSION,
        }
        and probe.probe_seconds == 45
        and probe.target_sample_rate_hz == 24_000
        and probe.target_sample_count == 1_080_000
        and probe.preparation_status == "SPECIFIED_NOT_EXECUTED"
    )


def _derive_e1_1_prerequisite_validity(
    observation: PreflightObservation,
    probe: DeamProbeManifest,
    mert_status: MertUpstreamStatus,
    data_root: Path,
) -> dict[str, str]:
    """Derive the four E1.1 prerequisites from observed artifacts, never caller input."""

    mert_valid = True
    try:
        bind_mert_formal(mert_status)
    except UpstreamModelBlocker:
        mert_valid = False
    return {
        "E0_MINIMUM_BOOTSTRAP": "VALID"
        if observation.formal_allowed
        and observation.config_schema_version
        and observation.research_spec_version
        and observation.semantic_config_hash
        and observation.resolved_config_hash
        else "NOT_VALID",
        "MERT_UPSTREAM_IDENTITY": "VALID" if mert_valid else "NOT_VALID",
        "MERT_PROBE_IDENTITY": "VALID"
        if _probe_identity_is_valid(probe, data_root)
        else "NOT_VALID",
        "MERT_PROBE_REPRESENTATION": "VALID"
        if _probe_representation_is_valid(probe)
        else "NOT_VALID",
    }


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


def _external_prerequisite_reasons(prerequisites: Mapping[str, str]) -> tuple[str, ...]:
    required = (
        "E0_MINIMUM_BOOTSTRAP",
        "MERT_UPSTREAM_IDENTITY",
        "MERT_PROBE_IDENTITY",
        "MERT_PROBE_REPRESENTATION",
    )
    return tuple(
        f"E1_1_PREREQUISITE_NOT_VALID:{dependency}"
        for dependency in required
        if prerequisites.get(dependency) != "VALID"
    )


def qualify_e0_minimum_bootstrap(
    observation: PreflightObservation,
    probe: DeamProbeManifest,
    mert_status: MertUpstreamStatus,
    data_root: Path,
    *,
    supersedes: str | None = None,
    correction_reason: str | None = None,
) -> E0MinimumQualificationBundle:
    """Build E0 evidence only; this reads declared artifacts and never runs MERT."""

    _assert_probe_contract(probe)
    _assert_pinned_mert_identity(mert_status)
    if (supersedes is None) != (correction_reason is None):
        raise ValueError("qualification supersession requires both path and correction reason")
    root = _repository_root()
    required_stage_ids = (
        "E0_MINIMUM_BOOTSTRAP",
        "E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
    )
    coverage = audit_stage_authority(
        load_stage_authority_catalog(root / "configs" / "catalog" / "stages.yaml"),
        required_stage_ids,
    )
    prerequisite_validity = _derive_e1_1_prerequisite_validity(
        observation, probe, mert_status, data_root
    )
    external_reasons = _external_prerequisite_reasons(prerequisite_validity)
    missing_local_roles = tuple(mert_status.missing_roles)
    blocked_reasons = external_reasons + tuple(
        f"MERT_LOCAL_ROLE_NOT_VERIFIED:{role}" for role in missing_local_roles
    )
    try:
        bind_mert_formal(mert_status)
    except UpstreamModelBlocker:
        blocked_reasons += (
            "MERT_FORMAL_BINDING_FAILED",
        )
    readiness = E1Readiness.BLOCKED if blocked_reasons else E1Readiness.READY
    payload: dict[str, Any] = {
        "schema_version": "1.1",
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
        "stage_authority_coverage": coverage.covered_stage_ids,
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
        "e1_1_prerequisite_validity": prerequisite_validity,
        "formal_execution_ready": observation.formal_allowed and readiness is E1Readiness.READY,
        "paper_eligible": False,
        "supersedes": supersedes,
        "correction_reason": correction_reason,
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as artifact:
            artifact.write(canonical_json(bundle.model_dump(mode="json")) + "\n")
    except FileExistsError as exc:
        raise QualificationWriteError(
            f"terminal qualification already exists: {destination}"
        ) from exc
