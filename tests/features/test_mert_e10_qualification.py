from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from sc_mv_dmer.features.mert_snapshot import (
    PINNED_MERT_REVISION,
    build_pinned_mert_manifest,
    write_immutable_manifest,
)
from sc_mv_dmer.features.mert_snapshot_qualification import (
    E10QualificationError,
    build_e10_binding_qualification,
    build_snapshot_inventory,
    validate_e10_artifact,
    write_immutable_e10_artifact,
)
from tests.features.test_mert_snapshot_binding import snapshot_fixture


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def bound_fixture(tmp_path: Path):
    root = snapshot_fixture(tmp_path / "snapshot")
    manifest = build_pinned_mert_manifest(root, revision=PINNED_MERT_REVISION)
    inventory = build_snapshot_inventory(
        manifest,
        root,
        acquisition_mechanism="huggingface_hub.snapshot_download",
        acquisition_version="0.36.0",
        supersedes="reports/preflight/mert-primary-local-status-v3.json",
    )
    return root, manifest, inventory


def test_inventory_records_runtime_root_but_not_in_snapshot_identity(tmp_path: Path) -> None:
    """Runtime placement belongs to provenance and cannot contaminate snapshot identity."""

    root, manifest, inventory = bound_fixture(tmp_path)

    assert inventory.model_root_runtime == str(root.resolve())
    assert inventory.snapshot_aggregate_sha256 == manifest.snapshot_aggregate_sha256
    assert inventory.required_files_complete is True
    assert inventory.acquisition_allowlist == (
        "config.json",
        "preprocessor_config.json",
        "configuration_MERT.py",
        "modeling_MERT.py",
        "pytorch_model.bin",
    )
    assert str(root.resolve()) not in manifest.model_dump_json()


def test_clean_verified_binding_passes_and_only_makes_e1_1_ready(tmp_path: Path) -> None:
    """E1.0 success may open E1.1 but must not claim any forward-derived evidence."""

    root, manifest, inventory = bound_fixture(tmp_path)
    qualification = build_e10_binding_qualification(
        manifest,
        inventory,
        root,
        git_commit="a" * 40,
        git_dirty=False,
        manifest_logical_path="manifests/upstream/mert-v1-95m-primary-v1.json",
        manifest_file_sha256="b" * 64,
        inventory_logical_path="reports/preflight/mert-pinned-snapshot-inventory.json",
        inventory_file_sha256="c" * 64,
        predecessor_qualification="evidence/qualifications/step-0-minimum-bootstrap-v4.json",
        predecessor_qualification_sha256="d" * 64,
    )

    assert qualification.e1_0_verdict == "PASS"
    assert qualification.e1_1_readiness == "READY"
    assert qualification.model_instantiated is False
    assert qualification.mert_forward_executed is False
    assert qualification.frame_rate_derived is False
    assert qualification.t_raw_derived is False
    assert qualification.rg_01_attempted is False
    assert qualification.snapshot_aggregate_sha256 == manifest.snapshot_aggregate_sha256


def test_dirty_git_or_changed_snapshot_blocks_qualification(tmp_path: Path) -> None:
    """A dirty execution or post-bind byte change cannot produce formal PASS evidence."""

    root, manifest, inventory = bound_fixture(tmp_path)
    kwargs = dict(
        manifest_logical_path="manifests/upstream/mert-v1-95m-primary-v1.json",
        manifest_file_sha256="b" * 64,
        inventory_logical_path="reports/preflight/mert-pinned-snapshot-inventory.json",
        inventory_file_sha256="c" * 64,
        predecessor_qualification="evidence/qualifications/step-0-minimum-bootstrap-v4.json",
        predecessor_qualification_sha256="d" * 64,
    )
    with pytest.raises(E10QualificationError, match="clean Git"):
        build_e10_binding_qualification(
            manifest, inventory, root, git_commit="a" * 40, git_dirty=True, **kwargs
        )

    with (root / "modeling_MERT.py").open("ab") as changed:
        changed.write(b"tampered")
    with pytest.raises(E10QualificationError, match="snapshot verification"):
        build_e10_binding_qualification(
            manifest, inventory, root, git_commit="a" * 40, git_dirty=False, **kwargs
        )


def test_versioned_schemas_validate_manifest_inventory_and_qualification(
    tmp_path: Path,
) -> None:
    """Machine-readable E1.0 artifacts must satisfy their committed schemas."""

    root, manifest, inventory = bound_fixture(tmp_path)
    qualification = build_e10_binding_qualification(
        manifest,
        inventory,
        root,
        git_commit="a" * 40,
        git_dirty=False,
        manifest_logical_path="manifests/upstream/mert-v1-95m-primary-v1.json",
        manifest_file_sha256="b" * 64,
        inventory_logical_path="reports/preflight/mert-pinned-snapshot-inventory.json",
        inventory_file_sha256="c" * 64,
        predecessor_qualification="evidence/qualifications/step-0-minimum-bootstrap-v4.json",
        predecessor_qualification_sha256="d" * 64,
    )

    validate_e10_artifact(
        manifest.model_dump(mode="json"),
        REPOSITORY_ROOT / "schemas" / "pinned_mert_upstream_manifest.schema.json",
    )
    validate_e10_artifact(
        inventory.model_dump(mode="json"),
        REPOSITORY_ROOT / "schemas" / "mert_pinned_snapshot_inventory.schema.json",
    )
    validate_e10_artifact(
        qualification.model_dump(mode="json"),
        REPOSITORY_ROOT / "schemas" / "e1_0_mert_upstream_binding.schema.json",
    )

    malformed = qualification.model_dump(mode="json")
    malformed["git_dirty"] = "false"
    with pytest.raises(ValidationError):
        validate_e10_artifact(
            malformed,
            REPOSITORY_ROOT / "schemas" / "e1_0_mert_upstream_binding.schema.json",
        )


def test_e10_artifact_writes_are_exclusive(tmp_path: Path) -> None:
    """Inventory and qualification corrections must use new append-only paths."""

    _, _, inventory = bound_fixture(tmp_path)
    destination = tmp_path / "inventory.json"
    write_immutable_e10_artifact(inventory, destination)
    before = destination.read_bytes()

    with pytest.raises(FileExistsError):
        write_immutable_e10_artifact(inventory, destination)

    assert destination.read_bytes() == before


def test_written_manifest_contains_no_absolute_snapshot_root(tmp_path: Path) -> None:
    """The tracked formal manifest cannot serialize a machine-local model root."""

    root, manifest, _ = bound_fixture(tmp_path)
    destination = tmp_path / "manifest.json"
    write_immutable_manifest(manifest, destination)

    assert str(root.resolve()) not in destination.read_text(encoding="utf-8")
