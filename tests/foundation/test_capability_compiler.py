from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from sc_mv_dmer.foundation.capabilities import (
    ActivationStatus,
    CapabilityRole,
    DependencyValidity,
    ExecutionContext,
    ExecutionRole,
    NotApplicableRoleActivationError,
    Phase,
    Readiness,
    ScientificExistence,
    UnknownCapabilityContextError,
    compile_capabilities,
    load_variant_catalog,
    select_variant_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VARIANTS_CATALOG = REPOSITORY_ROOT / "configs" / "catalog" / "variants.yaml"
STAGES_CATALOG = REPOSITORY_ROOT / "configs" / "catalog" / "stages.yaml"


@pytest.fixture
def profile():
    return select_variant_profile(load_variant_catalog(VARIANTS_CATALOG), "E1_MERT_PROBE")


def e1_1_validity() -> dict[CapabilityRole | str, DependencyValidity]:
    return {
        "E0_MINIMUM_BOOTSTRAP": DependencyValidity.VALID,
        CapabilityRole.MERT_UPSTREAM_IDENTITY: DependencyValidity.VALID,
        CapabilityRole.MERT_PROBE_IDENTITY: DependencyValidity.VALID,
        CapabilityRole.MERT_PROBE_REPRESENTATION: DependencyValidity.VALID,
    }


def test_e0_compilation_has_no_active_forward_training_or_gate_execution(profile) -> None:
    """Removing an E0 prohibition would incorrectly authorize execution."""

    manifest = compile_capabilities(
        profile,
        ExecutionContext(
            stage_id="E0_MINIMUM_BOOTSTRAP",
            phase=Phase.BOOTSTRAP,
            execution_role=ExecutionRole.INFERENCE,
        ),
        dependency_validity={},
    )

    assert manifest.role_capabilities[CapabilityRole.MERT_UPSTREAM_IDENTITY].activation is ActivationStatus.ACTIVE
    assert manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD].activation is ActivationStatus.INACTIVE
    assert manifest.role_capabilities[CapabilityRole.MERT_OUTPUT_DIMENSION_DERIVATION].activation is ActivationStatus.INACTIVE
    assert manifest.role_capabilities[CapabilityRole.TIMESNET_PERIOD_DERIVATION].activation is ActivationStatus.INACTIVE
    assert manifest.role_capabilities[CapabilityRole.RG_01_EVALUATION].activation is ActivationStatus.INACTIVE
    assert manifest.role_capabilities[CapabilityRole.TRAINING].activation is ActivationStatus.NOT_APPLICABLE
    assert CapabilityRole.MERT_REAL_FORWARD in manifest.forbidden_roles
    assert CapabilityRole.RG_01_EVALUATION in manifest.forbidden_roles


def test_e1_1_activates_only_real_forward_measurement_roles(profile) -> None:
    """Adding a later E1.2/E1.4 role to E1.1 would widen this authority."""

    manifest = compile_capabilities(
        profile,
        ExecutionContext(
            stage_id="E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
            phase=Phase.MEASUREMENT,
            execution_role=ExecutionRole.INFERENCE,
        ),
        dependency_validity=e1_1_validity(),
    )

    assert manifest.role_capabilities[CapabilityRole.MERT_UPSTREAM_IDENTITY].activation is ActivationStatus.ACTIVE
    assert manifest.role_capabilities[CapabilityRole.MERT_PROBE_IDENTITY].activation is ActivationStatus.ACTIVE
    assert manifest.role_capabilities[CapabilityRole.MERT_PROBE_REPRESENTATION].activation is ActivationStatus.ACTIVE
    assert manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD].activation is ActivationStatus.ACTIVE
    assert manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD].readiness is Readiness.READY
    assert manifest.role_capabilities[CapabilityRole.MERT_OUTPUT_DIMENSION_DERIVATION].activation is ActivationStatus.INACTIVE
    assert manifest.role_capabilities[CapabilityRole.TIMESNET_PERIOD_DERIVATION].activation is ActivationStatus.INACTIVE
    assert manifest.role_capabilities[CapabilityRole.RG_01_EVALUATION].activation is ActivationStatus.INACTIVE


@pytest.mark.parametrize(
    "external_dependency",
    (
        "E0_MINIMUM_BOOTSTRAP",
        CapabilityRole.MERT_UPSTREAM_IDENTITY,
        CapabilityRole.MERT_PROBE_IDENTITY,
        CapabilityRole.MERT_PROBE_REPRESENTATION,
    ),
)
@pytest.mark.parametrize("mode", ("missing", "nonvalid"))
def test_e1_1_real_forward_requires_every_external_authority_dependency(
    profile, external_dependency, mode
) -> None:
    """Any absent or non-VALID declared external prerequisite blocks first forward."""

    dependency_validity = e1_1_validity()
    if mode == "missing":
        del dependency_validity[external_dependency]
    else:
        dependency_validity[external_dependency] = DependencyValidity.STALE_DEPENDENCY

    manifest = compile_capabilities(
        profile,
        ExecutionContext(
            stage_id="E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
            phase=Phase.MEASUREMENT,
            execution_role=ExecutionRole.INFERENCE,
        ),
        dependency_validity=dependency_validity,
    )

    assert manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD].readiness is Readiness.DEPENDENCY_BLOCKED


def test_late_modules_remain_not_applicable_in_every_registered_context(profile) -> None:
    """Changing a probe-only late module to PRESENT must fail this closed-world contract."""

    for context in (
        ExecutionContext(
            stage_id="E0_MINIMUM_BOOTSTRAP",
            phase=Phase.BOOTSTRAP,
            execution_role=ExecutionRole.INFERENCE,
        ),
        ExecutionContext(
            stage_id="E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
            phase=Phase.MEASUREMENT,
            execution_role=ExecutionRole.INFERENCE,
        ),
    ):
        manifest = compile_capabilities(profile, context, dependency_validity={})
        for role in (
            CapabilityRole.SENSOR,
            CapabilityRole.MARKOV,
            CapabilityRole.FUSION,
            CapabilityRole.QWEN,
            CapabilityRole.LM_HEAD,
            CapabilityRole.PARSER,
            CapabilityRole.GENERATION,
            CapabilityRole.COT,
            CapabilityRole.EVENT_INJECTION,
            CapabilityRole.TRAINING,
            CapabilityRole.OPTIMIZER,
        ):
            capability = manifest.role_capabilities[role]
            assert capability.scientific_existence is ScientificExistence.NOT_APPLICABLE_BY_DESIGN
            assert capability.activation is ActivationStatus.NOT_APPLICABLE
            assert capability.readiness is None


def test_requested_not_applicable_role_is_rejected(profile) -> None:
    """Treating an N/A role as an activation request must never become executable."""

    with pytest.raises(NotApplicableRoleActivationError, match="SENSOR"):
        compile_capabilities(
            profile,
            ExecutionContext(
                stage_id="E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
                phase=Phase.MEASUREMENT,
                execution_role=ExecutionRole.INFERENCE,
                requested_roles=(CapabilityRole.SENSOR,),
            ),
            dependency_validity={},
        )


def test_unknown_variant_role_stage_and_phase_fail_closed(profile) -> None:
    """Untranscribed catalog values must not acquire implicit authority."""

    with pytest.raises(UnknownCapabilityContextError, match="UNKNOWN_STAGE"):
        compile_capabilities(
            profile,
            ExecutionContext(
                stage_id="UNKNOWN_STAGE",
                phase=Phase.BOOTSTRAP,
                execution_role=ExecutionRole.INFERENCE,
            ),
            dependency_validity={},
        )
    with pytest.raises(UnknownCapabilityContextError, match="MEASUREMENT"):
        compile_capabilities(
            profile,
            ExecutionContext(
                stage_id="E0_MINIMUM_BOOTSTRAP",
                phase=Phase.MEASUREMENT,
                execution_role=ExecutionRole.INFERENCE,
            ),
            dependency_validity={},
        )
    with pytest.raises(ValueError, match="UNKNOWN_VARIANT"):
        select_variant_profile(load_variant_catalog(VARIANTS_CATALOG), "UNKNOWN_VARIANT")
    with pytest.raises(ValueError, match="UNKNOWN_ROLE"):
        load_variant_catalog(
            {
                "catalog_version": "1.0",
                "scope": "E1_MINIMUM_ONLY",
                "full_m0_m20_coverage": False,
                "future_catalog_blocker_ref": "SGA-M0M20-EV-3R+",
                "profiles": [
                    {
                        "variant_id": "UNKNOWN_VARIANT",
                        "variant_version": "v1",
                        "role_map": {"UNKNOWN_ROLE": "PRESENT"},
                        "forbidden_roles": [],
                    }
                ],
            }
        )


def test_e1_1_ignores_undeclared_self_validity_for_first_forward(profile) -> None:
    """The first real forward cannot depend on its own output validity."""

    manifest = compile_capabilities(
        profile,
        ExecutionContext(
            stage_id="E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
            phase=Phase.MEASUREMENT,
            execution_role=ExecutionRole.INFERENCE,
        ),
        dependency_validity={
            **e1_1_validity(),
            CapabilityRole.MERT_REAL_FORWARD: DependencyValidity.STALE_DEPENDENCY,
        },
    )

    assert manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD].activation is ActivationStatus.ACTIVE
    assert manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD].readiness is Readiness.READY


def test_compiled_manifest_is_immutable(profile) -> None:
    """Allowing runtime mutation would let a CLI turn a forbidden role active."""

    manifest = compile_capabilities(
        profile,
        ExecutionContext(
            stage_id="E0_MINIMUM_BOOTSTRAP",
            phase=Phase.BOOTSTRAP,
            execution_role=ExecutionRole.INFERENCE,
        ),
        dependency_validity={},
    )

    with pytest.raises(TypeError):
        manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD] = None
    with pytest.raises(ValidationError):
        manifest.role_capabilities[CapabilityRole.MERT_REAL_FORWARD].activation = ActivationStatus.ACTIVE


def test_catalog_metadata_and_schemas_are_strict_and_machine_readable() -> None:
    """Relaxing catalog/schema strictness would permit unsupported capability authority."""

    catalog = yaml.safe_load(VARIANTS_CATALOG.read_text(encoding="utf-8"))
    assert catalog["scope"] == "E1_MINIMUM_ONLY"
    assert catalog["full_m0_m20_coverage"] is False
    assert catalog["future_catalog_blocker_ref"] == "SGA-M0M20-EV-3R+"
    assert yaml.safe_load(STAGES_CATALOG.read_text(encoding="utf-8"))["scope"] == "E1_MINIMUM_ONLY"

    for name in (
        "variant_capability_profile.schema.json",
        "compiled_execution_capability.schema.json",
        "stage_authority.schema.json",
    ):
        schema = json.loads((REPOSITORY_ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        assert schema["properties"]["schema_version"]["const"] == "1.0"
