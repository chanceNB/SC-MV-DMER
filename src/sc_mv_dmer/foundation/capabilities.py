"""Closed-world E1-minimum capability compilation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from sc_mv_dmer.foundation.manifests import FrozenDict, ImmutableRecord


class ScientificExistence(str, Enum):
    PRESENT = "PRESENT"
    NOT_APPLICABLE_BY_DESIGN = "NOT_APPLICABLE_BY_DESIGN"


class ActivationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Readiness(str, Enum):
    READY = "READY"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    GATE_BLOCKED = "GATE_BLOCKED"
    STALE_DEPENDENCY = "STALE_DEPENDENCY"
    INVALID = "INVALID"


class DependencyValidity(str, Enum):
    VALID = "VALID"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"
    GATE_BLOCKED = "GATE_BLOCKED"
    STALE_DEPENDENCY = "STALE_DEPENDENCY"
    INVALID = "INVALID"


class CapabilityRole(str, Enum):
    MERT_UPSTREAM_IDENTITY = "MERT_UPSTREAM_IDENTITY"
    MERT_PROBE_IDENTITY = "MERT_PROBE_IDENTITY"
    MERT_PROBE_REPRESENTATION = "MERT_PROBE_REPRESENTATION"
    MERT_REAL_FORWARD = "MERT_REAL_FORWARD"
    MERT_OUTPUT_DIMENSION_DERIVATION = "MERT_OUTPUT_DIMENSION_DERIVATION"
    TIMESNET_PERIOD_DERIVATION = "TIMESNET_PERIOD_DERIVATION"
    RG_01_EVALUATION = "RG_01_EVALUATION"
    SENSOR = "SENSOR"
    MARKOV = "MARKOV"
    FUSION = "FUSION"
    QWEN = "QWEN"
    LM_HEAD = "LM_HEAD"
    PARSER = "PARSER"
    GENERATION = "GENERATION"
    COT = "COT"
    EVENT_INJECTION = "EVENT_INJECTION"
    TRAINING = "TRAINING"
    OPTIMIZER = "OPTIMIZER"


class Phase(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    MEASUREMENT = "MEASUREMENT"


class ExecutionRole(str, Enum):
    INFERENCE = "INFERENCE"
    TRAINING = "TRAINING"


class UnknownCapabilityContextError(ValueError):
    """Raised when a context lacks a transcribed authority row."""


class NotApplicableRoleActivationError(ValueError):
    """Raised when execution requests a role excluded by design."""


class ExecutionContext(ImmutableRecord):
    stage_id: str = Field(min_length=1)
    phase: Phase
    execution_role: ExecutionRole
    requested_roles: tuple[CapabilityRole, ...] = ()


class VariantCapabilityProfile(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    variant_id: str = Field(min_length=1)
    variant_version: str = Field(min_length=1)
    role_map: Mapping[CapabilityRole, ScientificExistence]
    forbidden_roles: tuple[CapabilityRole, ...]

    @field_validator("role_map", mode="after")
    @classmethod
    def _freeze_exact_role_map(
        cls, value: Mapping[CapabilityRole, ScientificExistence]
    ) -> Mapping[CapabilityRole, ScientificExistence]:
        if set(value) != set(CapabilityRole):
            missing = sorted(role.value for role in set(CapabilityRole) - set(value))
            unknown = sorted(str(role) for role in set(value) - set(CapabilityRole))
            raise ValueError(f"role map must be exact; missing={missing}, unknown={unknown}")
        return FrozenDict(value)

    def model_post_init(self, __context: Any) -> None:
        if len(set(self.forbidden_roles)) != len(self.forbidden_roles):
            raise ValueError("forbidden roles must not contain duplicates")
        not_applicable = {
            role
            for role, existence in self.role_map.items()
            if existence is ScientificExistence.NOT_APPLICABLE_BY_DESIGN
        }
        if set(self.forbidden_roles) != not_applicable:
            raise ValueError("forbidden roles must exactly match NOT_APPLICABLE_BY_DESIGN roles")


class RoleExecutionCapability(ImmutableRecord):
    scientific_existence: ScientificExistence
    activation: ActivationStatus
    readiness: Readiness | None = None


class CompiledExecutionCapabilityManifest(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    variant_id: str
    variant_version: str
    context: ExecutionContext
    role_capabilities: Mapping[CapabilityRole, RoleExecutionCapability]
    forbidden_roles: tuple[CapabilityRole, ...]

    @field_validator("role_capabilities", mode="after")
    @classmethod
    def _freeze_capabilities(
        cls, value: Mapping[CapabilityRole, RoleExecutionCapability]
    ) -> Mapping[CapabilityRole, RoleExecutionCapability]:
        return FrozenDict(value)


class _VariantCatalog(ImmutableRecord):
    catalog_version: Literal["1.0"]
    scope: Literal["E1_MINIMUM_ONLY"]
    full_m0_m20_coverage: Literal[False]
    future_catalog_blocker_ref: str = Field(min_length=1)
    profiles: tuple[VariantCapabilityProfile, ...] = Field(min_length=1)

    def model_post_init(self, __context: Any) -> None:
        identifiers = [profile.variant_id for profile in self.profiles]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("variant catalog must not contain duplicate variant IDs")


def _load_yaml_or_mapping(source: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Path):
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        data = source
    if not isinstance(data, Mapping):
        raise ValueError("catalog must be a mapping")
    return data


def load_variant_catalog(source: Path | Mapping[str, Any]) -> tuple[VariantCapabilityProfile, ...]:
    """Load the explicitly scoped E1-only variant catalog."""

    return _VariantCatalog.model_validate(_load_yaml_or_mapping(source)).profiles


def select_variant_profile(
    profiles: tuple[VariantCapabilityProfile, ...], variant_id: str
) -> VariantCapabilityProfile:
    """Select one transcribed profile or fail closed."""

    for profile in profiles:
        if profile.variant_id == variant_id:
            return profile
    raise ValueError(f"unknown variant: {variant_id}")


def _stage_catalog_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "catalog" / "stages.yaml"


def _readiness_for(validity: DependencyValidity) -> Readiness:
    if validity is DependencyValidity.VALID:
        return Readiness.READY
    return Readiness(validity.value)


def _is_explicitly_valid(validity: object) -> bool:
    return validity is DependencyValidity.VALID


def compile_capabilities(
    profile: VariantCapabilityProfile,
    context: ExecutionContext,
    dependency_validity: Mapping[CapabilityRole | str, object],
) -> CompiledExecutionCapabilityManifest:
    """Compile an immutable manifest from an exact profile and stage authority row."""

    from sc_mv_dmer.foundation.stages import load_stage_authority_catalog

    rows = load_stage_authority_catalog(_stage_catalog_path())
    row = next((candidate for candidate in rows if candidate.stage_id == context.stage_id), None)
    if row is None:
        raise UnknownCapabilityContextError(f"unknown stage: {context.stage_id}")
    if row.phase is not context.phase or row.execution_role is not context.execution_role:
        raise UnknownCapabilityContextError(
            f"unregistered context for {context.stage_id}: {context.phase.value}/{context.execution_role.value}"
        )
    if row.variant_id != profile.variant_id:
        raise UnknownCapabilityContextError(
            f"stage {context.stage_id} does not authorize variant {profile.variant_id}"
        )
    allowed_dependency_keys = set(CapabilityRole) | set(row.required_predecessor_stage_ids)
    unknown_dependencies = set(dependency_validity) - allowed_dependency_keys
    if unknown_dependencies:
        raise ValueError(f"unknown capability dependency: {sorted(map(str, unknown_dependencies))}")
    for role in context.requested_roles:
        if profile.role_map[role] is ScientificExistence.NOT_APPLICABLE_BY_DESIGN:
            raise NotApplicableRoleActivationError(
                f"NOT_APPLICABLE_BY_DESIGN role cannot be activated: {role.value}"
            )
        if role not in row.active_roles:
            raise UnknownCapabilityContextError(
                f"role {role.value} is not active in stage {context.stage_id}"
            )

    authority_dependencies = (
        *row.required_predecessor_stage_ids,
        *row.required_identity_roles,
    )
    authority_is_valid = all(
        _is_explicitly_valid(dependency_validity.get(dependency))
        for dependency in authority_dependencies
    )
    capabilities: dict[CapabilityRole, RoleExecutionCapability] = {}
    for role, existence in profile.role_map.items():
        if existence is ScientificExistence.NOT_APPLICABLE_BY_DESIGN:
            capabilities[role] = RoleExecutionCapability(
                scientific_existence=existence,
                activation=ActivationStatus.NOT_APPLICABLE,
            )
        elif role in row.active_roles:
            role_dependencies_are_valid = all(
                _is_explicitly_valid(dependency_validity.get(dependency))
                for dependency in row.role_dependencies.get(role, ())
            )
            readiness = (
                Readiness.READY
                if authority_is_valid and role_dependencies_are_valid
                else Readiness.DEPENDENCY_BLOCKED
            )
            capabilities[role] = RoleExecutionCapability(
                scientific_existence=existence,
                activation=ActivationStatus.ACTIVE,
                readiness=readiness,
            )
        else:
            capabilities[role] = RoleExecutionCapability(
                scientific_existence=existence,
                activation=ActivationStatus.INACTIVE,
            )
    return CompiledExecutionCapabilityManifest(
        variant_id=profile.variant_id,
        variant_version=profile.variant_version,
        context=context,
        role_capabilities=capabilities,
        forbidden_roles=row.forbidden_roles,
    )
