"""Stage authority rows limited to the E1-minimum catalog."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from sc_mv_dmer.foundation.capabilities import CapabilityRole, ExecutionRole, Phase
from sc_mv_dmer.foundation.manifests import FrozenDict, ImmutableRecord


FROZEN_CONTRACT_BLOCKER = "IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: SGA-M0M20-EV-3R+"


class FrozenContractCoverageError(ValueError):
    """Raised instead of synthesizing frozen-contract stage authority."""


class StageAuthorityRow(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    stage_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    phase: Phase
    execution_role: ExecutionRole
    active_roles: tuple[CapabilityRole, ...]
    forbidden_roles: tuple[CapabilityRole, ...]
    required_predecessor_stage_ids: tuple[str, ...] = ()
    required_identity_roles: tuple[CapabilityRole, ...] = ()
    role_dependencies: Mapping[CapabilityRole, tuple[CapabilityRole, ...]] = Field(
        default_factory=dict
    )

    @field_validator("role_dependencies", mode="after")
    @classmethod
    def _freeze_role_dependencies(
        cls, value: Mapping[CapabilityRole, tuple[CapabilityRole, ...]]
    ) -> Mapping[CapabilityRole, tuple[CapabilityRole, ...]]:
        return FrozenDict({role: tuple(dependencies) for role, dependencies in value.items()})

    def model_post_init(self, __context: Any) -> None:
        if len(set(self.active_roles)) != len(self.active_roles):
            raise ValueError("active roles must not contain duplicates")
        if len(set(self.forbidden_roles)) != len(self.forbidden_roles):
            raise ValueError("forbidden roles must not contain duplicates")
        if set(self.active_roles) & set(self.forbidden_roles):
            raise ValueError("a role cannot be both active and forbidden")
        for role, dependencies in self.role_dependencies.items():
            if role not in self.active_roles:
                raise ValueError("role dependencies require an active role")
            if role in dependencies:
                raise ValueError("role dependencies must not contain self dependencies")


class StageAuthorityCoverage(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    covered_stage_ids: tuple[str, ...]
    missing_stage_ids: tuple[str, ...]


class _StageCatalog(ImmutableRecord):
    catalog_version: Literal["1.0"]
    scope: Literal["E1_MINIMUM_ONLY"]
    full_m0_m20_coverage: Literal[False]
    future_catalog_blocker_ref: Literal["SGA-M0M20-EV-3R+"]
    stages: tuple[StageAuthorityRow, ...] = Field(min_length=1)

    def model_post_init(self, __context: Any) -> None:
        stage_ids = [row.stage_id for row in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("stage catalog must not contain duplicate stage IDs")


def load_stage_authority_catalog(source: Path | Mapping[str, Any]) -> tuple[StageAuthorityRow, ...]:
    """Load only the two explicitly transcribed E1-minimum rows."""

    if isinstance(source, Path):
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    else:
        data = source
    if not isinstance(data, Mapping):
        raise ValueError("stage catalog must be a mapping")
    return _StageCatalog.model_validate(data).stages


def audit_stage_authority(
    rows: tuple[StageAuthorityRow, ...], required_stage_ids: tuple[str, ...]
) -> StageAuthorityCoverage:
    """Report only transcribed coverage; full catalog requests fail closed."""

    if any(stage_id.startswith("M") for stage_id in required_stage_ids):
        raise FrozenContractCoverageError(FROZEN_CONTRACT_BLOCKER)
    available = {row.stage_id for row in rows}
    missing = tuple(stage_id for stage_id in required_stage_ids if stage_id not in available)
    if missing:
        raise FrozenContractCoverageError(
            f"{FROZEN_CONTRACT_BLOCKER}; untranscribed stage IDs: {', '.join(missing)}"
        )
    return StageAuthorityCoverage(
        covered_stage_ids=tuple(required_stage_ids),
        missing_stage_ids=(),
    )
