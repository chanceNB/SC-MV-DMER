from __future__ import annotations

import json
import hashlib
import sys
import subprocess
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from sc_mv_dmer.data.discovery import DEAM_DATASET_VERSION, DeamSource
from sc_mv_dmer.data.probes import DeamProbeManifest, build_probe
from sc_mv_dmer.features.mert_upstream import MertUpstreamStatus
from sc_mv_dmer.foundation.config import RunMode, resolve_config
from sc_mv_dmer.foundation.preflight import observe_preflight
from sc_mv_dmer.foundation.qualification import (
    E1Readiness,
    QualificationWriteError,
    load_qualification_schema,
    qualify_e0_minimum_bootstrap,
    validate_qualification_artifact,
    write_terminal_qualification,
)
from sc_mv_dmer.cli import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = REPOSITORY_ROOT / "configs" / "research" / "defaults.yaml"


def probe_root(tmp_path: Path) -> tuple[Path, DeamProbeManifest]:
    root = tmp_path / "deam"
    source_path = root / "DEAM_audio" / "MEMD_audio" / "2.mp3"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"real-probe-source")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    dataset_id = "dataset_" + "a" * 64
    song_id = "song_" + "b" * 64
    return root, build_probe(
        DeamSource(
            dataset_id=dataset_id,
            song_id=song_id,
            source_relative_path="DEAM_audio/MEMD_audio/2.mp3",
            source_sha256=digest,
            source_size_bytes=source_path.stat().st_size,
            duration_seconds=45.0,
        )
    )


def mert_status(*, verified: bool) -> MertUpstreamStatus:
    required = ("config", "processor", "weights")
    return MertUpstreamStatus(
        repository="m-a-p/MERT-v1-95M",
        revision="12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
        license="cc-by-nc-4.0",
        required_roles=required,
        local_files={role: {"source_relative_path": role, "sha256": "f" * 64} for role in required}
        if verified
        else {},
        expected_checksums={role: "f" * 64 for role in required} if verified else {},
        missing_roles=() if verified else required,
        checksum_mismatches=(),
        local_verification_status="VERIFIED" if verified else "BLOCKED_MISSING_LOCAL_BYTES",
        upstream_identity="a" * 64,
    )


def qualification(tmp_path: Path, *, status: MertUpstreamStatus, supersedes: str | None = None):
    data_root, probe = probe_root(tmp_path)
    return qualify_e0_minimum_bootstrap(
        observe_preflight(
            REPOSITORY_ROOT,
            resolve_config([DEFAULTS], {}, RunMode.FORMAL),
            git_commit="a" * 40,
            git_dirty=False,
        ),
        probe,
        status,
        data_root,
        supersedes=supersedes,
        correction_reason="MERT checksum registry correction" if supersedes else None,
    )


def test_missing_local_mert_bytes_complete_e0_but_block_e1_1_with_exact_roles(tmp_path: Path) -> None:
    """Treating missing model bytes as E0 failure or E1 ready would misstate the boundary."""

    bundle = qualification(tmp_path, status=mert_status(verified=False))

    assert bundle.e0_implementation_status == "COMPLETE"
    assert bundle.e1_1_readiness is E1Readiness.BLOCKED
    assert bundle.e1_1_blocked_roles == ("config", "processor", "weights")
    assert bundle.stage_authority_coverage == (
        "E0_MINIMUM_BOOTSTRAP",
        "E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
    )
    assert bundle.e1_1_prerequisite_validity == {
        "E0_MINIMUM_BOOTSTRAP": "VALID",
        "MERT_UPSTREAM_IDENTITY": "NOT_VALID",
        "MERT_PROBE_IDENTITY": "NOT_VALID",
        "MERT_PROBE_REPRESENTATION": "VALID",
    }
    with pytest.raises(TypeError):
        bundle.e1_1_prerequisite_validity["E0_MINIMUM_BOOTSTRAP"] = "NOT_VALID"
    assert bundle.formal_execution_ready is False
    assert bundle.paper_eligible is False
    assert bundle.full_catalog_blocker_deferred is True


def test_caller_forged_verified_status_remains_blocked_by_code_controlled_registry(tmp_path: Path) -> None:
    """Trusting a caller-supplied VERIFIED string would bypass formal MERT binding."""

    bundle = qualification(tmp_path, status=mert_status(verified=True))

    assert bundle.e1_1_readiness is E1Readiness.BLOCKED
    assert bundle.formal_execution_ready is False
    assert "MERT_FORMAL_BINDING_FAILED" in bundle.e1_1_blocked_reasons


def test_tampered_real_probe_source_blocks_probe_identity(tmp_path: Path) -> None:
    """A changed source under the supplied runtime root must invalidate probe identity."""

    data_root, probe = probe_root(tmp_path)
    (data_root / probe.source_relative_path).write_bytes(b"tampered")
    bundle = qualify_e0_minimum_bootstrap(
        observe_preflight(REPOSITORY_ROOT, resolve_config([DEFAULTS], {}, RunMode.FORMAL), git_commit="a" * 40, git_dirty=False),
        probe, mert_status(verified=False), data_root,
    )

    assert bundle.e1_1_prerequisite_validity["MERT_PROBE_IDENTITY"] == "NOT_VALID"


def test_forged_probe_semantic_identity_blocks_probe_representation(tmp_path: Path) -> None:
    """A manifest with a mismatched semantic identity hash cannot authorize representation."""

    data_root, registered_probe = probe_root(tmp_path)
    forged = registered_probe.model_copy(update={"semantic_identity_sha256": "0" * 64})
    bundle = qualify_e0_minimum_bootstrap(
        observe_preflight(REPOSITORY_ROOT, resolve_config([DEFAULTS], {}, RunMode.FORMAL), git_commit="a" * 40, git_dirty=False),
        forged, mert_status(verified=False), data_root,
    )

    assert bundle.e1_1_prerequisite_validity["MERT_PROBE_REPRESENTATION"] == "NOT_VALID"


def test_late_blockers_are_not_e1_1_predecessors(tmp_path: Path) -> None:
    """Adding full split, annotation, PMEmo, or late modules would improperly widen E1.1."""

    bundle = qualification(tmp_path, status=mert_status(verified=True))

    assert "E1_1_PREREQUISITE_NOT_VALID:full_split" not in bundle.e1_1_blocked_reasons


def test_terminal_qualification_write_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    """Overwriting terminal evidence would erase its provenance."""

    destination = tmp_path / "qualification.json"
    bundle = qualification(tmp_path, status=mert_status(verified=False))
    write_terminal_qualification(bundle, destination)

    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored["qualification_sha256"] == bundle.qualification_sha256
    with pytest.raises(QualificationWriteError, match="terminal qualification"):
        write_terminal_qualification(bundle, destination)


def test_corrected_qualification_records_supersession_without_overwrite(tmp_path: Path) -> None:
    """Replacing a terminal evidence file instead of appending a successor loses history."""

    bundle = qualification(
        tmp_path, status=mert_status(verified=False),
        supersedes="evidence/qualifications/step-0-minimum-bootstrap.json",
    )

    assert bundle.supersedes == "evidence/qualifications/step-0-minimum-bootstrap.json"
    assert bundle.correction_reason == "MERT checksum registry correction"


def test_qualification_does_not_import_or_execute_model_code(tmp_path: Path) -> None:
    """Preflight must remain an artifact reader, not a MERT execution entrypoint."""
    # Model tests legitimately import torch during collection. A fresh process
    # verifies this boundary without depending on unrelated collection order.
    code = """
import runpy, sys
from pathlib import Path
namespace = runpy.run_path(sys.argv[1])
assert 'torch' not in sys.modules
namespace['qualification'](Path(sys.argv[2]), status=namespace['mert_status'](verified=False))
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
                            cwd=REPOSITORY_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_qualify_step_cli_reads_artifacts_only_and_writes_terminal_bundle(tmp_path: Path) -> None:
    """Removing the CLI reader would leave E0 qualification impossible to record."""

    output = tmp_path / "qualification.json"

    assert main(
        [
            "qualify-step",
            "--step",
            "0",
            "--mode",
            "debug",
            "--probe-manifest",
            str(REPOSITORY_ROOT / "manifests" / "probes" / "deam-e1-real-45s-v1.json"),
            "--mert-status",
            str(REPOSITORY_ROOT / "reports" / "preflight" / "mert-primary-local-status-v2.json"),
            "--data-root",
            "E:\\DEAM",
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["e1_1_readiness"] == "BLOCKED"


def test_qualify_step_cli_records_explicit_correction_supersession(tmp_path: Path) -> None:
    """Omitting CLI correction metadata would overwrite history without an audit link."""

    output = tmp_path / "qualification-v2.json"

    assert main(
        [
            "qualify-step",
            "--step", "0", "--mode", "debug",
            "--probe-manifest", str(REPOSITORY_ROOT / "manifests" / "probes" / "deam-e1-real-45s-v1.json"),
            "--mert-status", str(REPOSITORY_ROOT / "reports" / "preflight" / "mert-primary-local-status-v3.json"),
            "--data-root", "E:\\DEAM",
            "--output", str(output),
            "--supersedes", "evidence/qualifications/step-0-minimum-bootstrap.json",
            "--correction-reason", "pinned checksum trust anchored to code-controlled registry; MERT status v3",
        ]
    ) == 0

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["supersedes"] == "evidence/qualifications/step-0-minimum-bootstrap.json"
    assert stored["correction_reason"] == "pinned checksum trust anchored to code-controlled registry; MERT status v3"


def test_current_qualification_schema_covers_every_bundle_field_and_value(tmp_path: Path) -> None:
    """A strict current schema missing a bundle field would reject its own output."""

    schema = json.loads(
        (REPOSITORY_ROOT / "schemas" / "e0_minimum_qualification.schema.json").read_text(
            encoding="utf-8"
        )
    )
    evidence = qualification(tmp_path, status=mert_status(verified=False)).model_dump(mode="json")
    fields = set(__import__("sc_mv_dmer.foundation.qualification", fromlist=["E0MinimumQualificationBundle"]).E0MinimumQualificationBundle.model_fields)

    assert schema["additionalProperties"] is False
    assert fields == set(schema["properties"])
    assert fields == set(schema["required"])
    assert set(evidence) <= set(schema["properties"])
    assert set(schema["required"]) <= set(evidence)
    assert schema["properties"]["schema_version"]["const"] == evidence["schema_version"]
    assert evidence["e1_1_readiness"] in schema["properties"]["e1_1_readiness"]["enum"]
    for key in (
        "semantic_config_hash", "resolved_config_hash", "variants_catalog_sha256",
        "stages_catalog_sha256", "probe_manifest_sha256", "source_sha256",
        "mert_upstream_identity", "qualification_sha256",
    ):
        assert len(evidence[key]) == 64


@pytest.mark.parametrize(
    "artifact_name",
    (
        "step-0-minimum-bootstrap.json",
        "step-0-minimum-bootstrap-v2.json",
        "step-0-minimum-bootstrap-v3.json",
    ),
)
def test_legacy_qualification_artifacts_select_and_validate_against_v1_schema(
    artifact_name: str,
) -> None:
    """A v1 artifact must keep selecting its legacy contract after current schema evolves."""

    artifact = json.loads(
        (REPOSITORY_ROOT / "evidence" / "qualifications" / artifact_name).read_text(
            encoding="utf-8"
        )
    )
    schema = load_qualification_schema(artifact)

    assert schema["$id"].endswith("e0_minimum_qualification.v1.schema.json")
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    validate_qualification_artifact(artifact)


def test_current_v1_1_bundle_selects_strict_current_schema(tmp_path: Path) -> None:
    """New qualification output must select a strict 1.1 schema with all fields required."""

    bundle = qualification(tmp_path, status=mert_status(verified=False))
    artifact = bundle.model_dump(mode="json")
    schema = load_qualification_schema(artifact)

    assert artifact["schema_version"] == "1.1"
    assert schema["$id"].endswith("e0_minimum_qualification.schema.json")
    assert schema["title"] == "E0MinimumQualificationBundle v1.1"
    assert set(schema["required"]) == set(artifact)
    validate_qualification_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observed_git_dirty", "false"),
        ("implementation_git_commit", ""),
        ("e1_1_blocked_roles", [1]),
        ("stage_authority_coverage", "E0_MINIMUM_BOOTSTRAP"),
    ),
)
def test_draft202012_schema_rejects_malformed_current_qualification_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    """Relaxing JSON Schema type/item/minLength checks would admit malformed evidence."""

    artifact = qualification(tmp_path, status=mert_status(verified=False)).model_dump(mode="json")
    artifact[field] = value

    with pytest.raises(JsonSchemaValidationError):
        validate_qualification_artifact(artifact)
