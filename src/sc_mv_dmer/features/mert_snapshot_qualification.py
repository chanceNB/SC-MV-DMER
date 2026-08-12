"""Machine-readable E1.0 inventory and qualification for pinned MERT bytes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from jsonschema import Draft202012Validator
from pydantic import Field

from sc_mv_dmer.features.mert_snapshot import (
    PINNED_MERT_REPOSITORY,
    PINNED_MERT_REVISION,
    PinnedMertUpstreamManifest,
    PinnedSnapshotFile,
    SnapshotBindingError,
    verify_pinned_mert_manifest,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.manifests import ImmutableRecord


_ACQUISITION_ALLOWLIST = (
    "config.json",
    "preprocessor_config.json",
    "configuration_MERT.py",
    "modeling_MERT.py",
    "pytorch_model.bin",
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class E10QualificationError(ValueError):
    """Raised when formal E1.0 qualification cannot be issued."""


class MertPinnedSnapshotInventory(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    repository_id: Literal["m-a-p/MERT-v1-95M"] = PINNED_MERT_REPOSITORY
    revision: Literal[PINNED_MERT_REVISION] = PINNED_MERT_REVISION
    model_root_runtime: str = Field(min_length=1)
    acquisition_mechanism: Literal["huggingface_hub.snapshot_download"] = (
        "huggingface_hub.snapshot_download"
    )
    acquisition_version: str = Field(min_length=1)
    acquisition_allowlist: tuple[str, ...] = Field(min_length=5, max_length=5)
    files: tuple[PinnedSnapshotFile, ...] = Field(min_length=5, max_length=5)
    required_files_complete: Literal[True] = True
    fairseq_checkpoint_status: Literal["NOT_ACQUIRED_NOT_CONSUMED"] = (
        "NOT_ACQUIRED_NOT_CONSUMED"
    )
    snapshot_aggregate_sha256: str = Field(pattern=_SHA256_PATTERN)
    supersedes: str = Field(min_length=1)
    inventory_sha256: str = Field(pattern=_SHA256_PATTERN)


class E10MertUpstreamBindingQualification(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    qualification_id: Literal["E1-0-MERT-UPSTREAM-BINDING-v1"] = (
        "E1-0-MERT-UPSTREAM-BINDING-v1"
    )
    git_commit: str = Field(min_length=7)
    git_dirty: Literal[False] = False
    repository_id: Literal["m-a-p/MERT-v1-95M"] = PINNED_MERT_REPOSITORY
    revision: Literal[PINNED_MERT_REVISION] = PINNED_MERT_REVISION
    manifest_logical_path: str = Field(min_length=1)
    manifest_file_sha256: str = Field(pattern=_SHA256_PATTERN)
    inventory_logical_path: str = Field(min_length=1)
    inventory_file_sha256: str = Field(pattern=_SHA256_PATTERN)
    predecessor_qualification: str = Field(min_length=1)
    predecessor_qualification_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_upstream_identity: str = Field(pattern=_SHA256_PATTERN)
    snapshot_aggregate_sha256: str = Field(pattern=_SHA256_PATTERN)
    custom_code_aggregate_sha256: str = Field(pattern=_SHA256_PATTERN)
    canonical_weight_sha256: str = Field(pattern=_SHA256_PATTERN)
    criteria: dict[str, bool]
    e1_0_verdict: Literal["PASS"] = "PASS"
    e1_1_readiness: Literal["READY"] = "READY"
    model_instantiated: Literal[False] = False
    mert_forward_executed: Literal[False] = False
    frame_rate_derived: Literal[False] = False
    t_raw_derived: Literal[False] = False
    timesnet_derived: Literal[False] = False
    binning_2hz_executed: Literal[False] = False
    rg_01_attempted: Literal[False] = False
    qualification_sha256: str = Field(pattern=_SHA256_PATTERN)


def build_snapshot_inventory(
    manifest: PinnedMertUpstreamManifest,
    snapshot_root: Path,
    *,
    acquisition_mechanism: str,
    acquisition_version: str,
    supersedes: str,
) -> MertPinnedSnapshotInventory:
    """Record acquisition/runtime provenance after re-verifying actual bytes."""

    if acquisition_mechanism != "huggingface_hub.snapshot_download":
        raise E10QualificationError("unapproved acquisition mechanism")
    try:
        verify_pinned_mert_manifest(manifest, snapshot_root)
    except SnapshotBindingError as exc:
        raise E10QualificationError("snapshot verification failed") from exc
    payload = {
        "schema_version": "1.0",
        "repository_id": PINNED_MERT_REPOSITORY,
        "revision": PINNED_MERT_REVISION,
        "model_root_runtime": str(Path(snapshot_root).resolve()),
        "acquisition_mechanism": acquisition_mechanism,
        "acquisition_version": acquisition_version,
        "acquisition_allowlist": _ACQUISITION_ALLOWLIST,
        "files": [item.model_dump(mode="json") for item in manifest.files],
        "required_files_complete": True,
        "fairseq_checkpoint_status": "NOT_ACQUIRED_NOT_CONSUMED",
        "snapshot_aggregate_sha256": manifest.snapshot_aggregate_sha256,
        "supersedes": supersedes,
    }
    return MertPinnedSnapshotInventory(
        **payload,
        inventory_sha256=sha256_canonical(payload),
    )


def build_e10_binding_qualification(
    manifest: PinnedMertUpstreamManifest,
    inventory: MertPinnedSnapshotInventory,
    snapshot_root: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    manifest_logical_path: str,
    manifest_file_sha256: str,
    inventory_logical_path: str,
    inventory_file_sha256: str,
    predecessor_qualification: str,
    predecessor_qualification_sha256: str,
) -> E10MertUpstreamBindingQualification:
    """Issue PASS only for a clean, byte-verified, static E1.0 binding."""

    if git_dirty:
        raise E10QualificationError("formal E1.0 qualification requires clean Git")
    try:
        verify_pinned_mert_manifest(manifest, snapshot_root)
    except SnapshotBindingError as exc:
        raise E10QualificationError("snapshot verification failed") from exc
    if (
        inventory.repository_id != manifest.repository_id
        or inventory.revision != manifest.revision
        or inventory.snapshot_aggregate_sha256 != manifest.snapshot_aggregate_sha256
        or tuple(inventory.files) != tuple(manifest.files)
        or not inventory.required_files_complete
    ):
        raise E10QualificationError("inventory does not match pinned manifest")
    criteria = {
        "exact_revision_verified": True,
        "all_required_bytes_present": True,
        "all_local_sha256_registered": True,
        "custom_model_code_pinned": True,
        "canonical_transformers_weight_identified": True,
        "processor_identity_pinned": True,
        "snapshot_aggregate_valid": True,
        "local_files_only_required": True,
    }
    payload = {
        "schema_version": "1.0",
        "qualification_id": "E1-0-MERT-UPSTREAM-BINDING-v1",
        "git_commit": git_commit,
        "git_dirty": False,
        "repository_id": manifest.repository_id,
        "revision": manifest.revision,
        "manifest_logical_path": manifest_logical_path,
        "manifest_file_sha256": manifest_file_sha256,
        "inventory_logical_path": inventory_logical_path,
        "inventory_file_sha256": inventory_file_sha256,
        "predecessor_qualification": predecessor_qualification,
        "predecessor_qualification_sha256": predecessor_qualification_sha256,
        "semantic_upstream_identity": manifest.semantic_identity_sha256,
        "snapshot_aggregate_sha256": manifest.snapshot_aggregate_sha256,
        "custom_code_aggregate_sha256": manifest.custom_code_aggregate_sha256,
        "canonical_weight_sha256": manifest.weight_sha256,
        "criteria": criteria,
        "e1_0_verdict": "PASS",
        "e1_1_readiness": "READY",
        "model_instantiated": False,
        "mert_forward_executed": False,
        "frame_rate_derived": False,
        "t_raw_derived": False,
        "timesnet_derived": False,
        "binning_2hz_executed": False,
        "rg_01_attempted": False,
    }
    return E10MertUpstreamBindingQualification(
        **payload,
        qualification_sha256=sha256_canonical(payload),
    )


def validate_e10_artifact(artifact: dict[str, object], schema_path: Path) -> None:
    """Validate an E1.0 artifact with the full Draft 2020-12 contract."""

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(artifact)


def write_immutable_e10_artifact(record: ImmutableRecord, destination: Path) -> None:
    """Write inventory/qualification once using exclusive creation."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(canonical_json(record.model_dump(mode="json")) + "\n")
