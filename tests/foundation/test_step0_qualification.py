from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sc_mv_dmer.data.probes import DeamProbeManifest
from sc_mv_dmer.features.mert_upstream import MertUpstreamStatus
from sc_mv_dmer.foundation.capabilities import CapabilityRole, DependencyValidity
from sc_mv_dmer.foundation.config import RunMode, resolve_config
from sc_mv_dmer.foundation.preflight import observe_preflight
from sc_mv_dmer.foundation.qualification import (
    E1Readiness,
    QualificationWriteError,
    qualify_e0_minimum_bootstrap,
    write_terminal_qualification,
)
from sc_mv_dmer.cli import main


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = REPOSITORY_ROOT / "configs" / "research" / "defaults.yaml"


def probe() -> DeamProbeManifest:
    return DeamProbeManifest(
        dataset_id="dataset_" + "a" * 64,
        song_id="song_" + "b" * 64,
        sample_id="sample_" + "c" * 64,
        source_relative_path="DEAM_audio/MEMD_audio/2.mp3",
        source_sha256="d" * 64,
        source_size_bytes=3,
        source_duration_seconds=45.1,
        crop_resample_contract="crop [0,45)s; resample only if needed to 24000 Hz; not executed",
        preparation_status="SPECIFIED_NOT_EXECUTED",
        semantic_identity_json='{"probe":true}',
        semantic_identity_sha256="e" * 64,
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


def valid_prerequisites() -> dict[CapabilityRole | str, DependencyValidity]:
    return {
        "E0_MINIMUM_BOOTSTRAP": DependencyValidity.VALID,
        CapabilityRole.MERT_UPSTREAM_IDENTITY: DependencyValidity.VALID,
        CapabilityRole.MERT_PROBE_IDENTITY: DependencyValidity.VALID,
        CapabilityRole.MERT_PROBE_REPRESENTATION: DependencyValidity.VALID,
    }


def qualification(*, status: MertUpstreamStatus, prerequisites=None, supersedes: str | None = None):
    return qualify_e0_minimum_bootstrap(
        observe_preflight(
            REPOSITORY_ROOT,
            resolve_config([DEFAULTS], {}, RunMode.FORMAL),
            git_commit="a" * 40,
            git_dirty=False,
        ),
        probe(),
        status,
        valid_prerequisites() if prerequisites is None else prerequisites,
        supersedes=supersedes,
        correction_reason="MERT checksum registry correction" if supersedes else None,
    )


def test_missing_local_mert_bytes_complete_e0_but_block_e1_1_with_exact_roles() -> None:
    """Treating missing model bytes as E0 failure or E1 ready would misstate the boundary."""

    bundle = qualification(status=mert_status(verified=False))

    assert bundle.e0_implementation_status == "COMPLETE"
    assert bundle.e1_1_readiness is E1Readiness.BLOCKED
    assert bundle.e1_1_blocked_roles == ("config", "processor", "weights")
    assert bundle.stage_authority_coverage == (
        "E0_MINIMUM_BOOTSTRAP",
        "E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
    )
    assert bundle.e1_1_prerequisite_validity == {
        "E0_MINIMUM_BOOTSTRAP": "VALID",
        "MERT_UPSTREAM_IDENTITY": "VALID",
        "MERT_PROBE_IDENTITY": "VALID",
        "MERT_PROBE_REPRESENTATION": "VALID",
    }
    with pytest.raises(TypeError):
        bundle.e1_1_prerequisite_validity["E0_MINIMUM_BOOTSTRAP"] = "NOT_VALID"
    assert bundle.formal_execution_ready is False
    assert bundle.paper_eligible is False
    assert bundle.full_catalog_blocker_deferred is True


def test_caller_forged_verified_status_remains_blocked_by_code_controlled_registry() -> None:
    """Trusting a caller-supplied VERIFIED string would bypass formal MERT binding."""

    bundle = qualification(status=mert_status(verified=True))

    assert bundle.e1_1_readiness is E1Readiness.BLOCKED
    assert bundle.formal_execution_ready is False
    assert "MERT_FORMAL_BINDING_FAILED" in bundle.e1_1_blocked_reasons


@pytest.mark.parametrize("dependency", tuple(valid_prerequisites()))
def test_missing_or_stale_e1_1_prerequisite_blocks(dependency) -> None:
    """Unknown, missing, or stale E1.1 prerequisites must fail closed."""

    prerequisites = valid_prerequisites()
    del prerequisites[dependency]
    assert qualification(status=mert_status(verified=True), prerequisites=prerequisites).e1_1_readiness is E1Readiness.BLOCKED
    prerequisites = valid_prerequisites()
    prerequisites[dependency] = DependencyValidity.STALE_DEPENDENCY
    assert qualification(status=mert_status(verified=True), prerequisites=prerequisites).e1_1_readiness is E1Readiness.BLOCKED


def test_late_blockers_are_not_e1_1_predecessors() -> None:
    """Adding full split, annotation, PMEmo, or late modules would improperly widen E1.1."""

    bundle = qualification(
        status=mert_status(verified=True),
        prerequisites={**valid_prerequisites(), "full_split": DependencyValidity.INVALID},
    )

    assert "E1_1_PREREQUISITE_NOT_VALID:full_split" not in bundle.e1_1_blocked_reasons


def test_terminal_qualification_write_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    """Overwriting terminal evidence would erase its provenance."""

    destination = tmp_path / "qualification.json"
    bundle = qualification(status=mert_status(verified=False))
    write_terminal_qualification(bundle, destination)

    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored["qualification_sha256"] == bundle.qualification_sha256
    with pytest.raises(QualificationWriteError, match="terminal qualification"):
        write_terminal_qualification(bundle, destination)


def test_corrected_qualification_records_supersession_without_overwrite() -> None:
    """Replacing a terminal evidence file instead of appending a successor loses history."""

    bundle = qualification(
        status=mert_status(verified=False),
        supersedes="evidence/qualifications/step-0-minimum-bootstrap.json",
    )

    assert bundle.supersedes == "evidence/qualifications/step-0-minimum-bootstrap.json"
    assert bundle.correction_reason == "MERT checksum registry correction"


def test_qualification_does_not_import_or_execute_model_code() -> None:
    """Preflight must remain an artifact reader, not a MERT execution entrypoint."""

    qualification(status=mert_status(verified=False))

    assert "torch" not in sys.modules


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
            "--output", str(output),
            "--supersedes", "evidence/qualifications/step-0-minimum-bootstrap.json",
            "--correction-reason", "pinned checksum trust anchored to code-controlled registry; MERT status v3",
        ]
    ) == 0

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["supersedes"] == "evidence/qualifications/step-0-minimum-bootstrap.json"
    assert stored["correction_reason"] == "pinned checksum trust anchored to code-controlled registry; MERT status v3"
