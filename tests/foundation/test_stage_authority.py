from __future__ import annotations

from pathlib import Path

import pytest

from sc_mv_dmer.foundation.capabilities import CapabilityRole, ExecutionRole, Phase
from sc_mv_dmer.foundation.stages import (
    FrozenContractCoverageError,
    StageAuthorityRow,
    audit_stage_authority,
    load_stage_authority_catalog,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGES_CATALOG = REPOSITORY_ROOT / "configs" / "catalog" / "stages.yaml"


def test_e1_minimum_stage_authority_audit_covers_only_explicit_subset() -> None:
    """Dropping either transcribed row would make the E1 minimum catalog incomplete."""

    rows = load_stage_authority_catalog(STAGES_CATALOG)

    coverage = audit_stage_authority(
        rows,
        required_stage_ids=(
            "E0_MINIMUM_BOOTSTRAP",
            "E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
        ),
    )

    assert coverage.covered_stage_ids == (
        "E0_MINIMUM_BOOTSTRAP",
        "E1_1_MERT_REAL_OUTPUT_FRAME_RATE_PRINT",
    )
    assert coverage.missing_stage_ids == ()


def test_full_catalog_request_raises_exact_frozen_contract_coverage_error() -> None:
    """Inventing an M0-M20 authority row would conceal the frozen-contract gap."""

    rows = load_stage_authority_catalog(STAGES_CATALOG)

    with pytest.raises(
        FrozenContractCoverageError,
        match="IMPLEMENTATION_BLOCKED_BY_FROZEN_CONTRACT: SGA-M0M20-EV-3R\\+",
    ):
        audit_stage_authority(rows, required_stage_ids=("M0", "M20"))


def test_stage_authority_rejects_unpermitted_or_duplicate_roles() -> None:
    """An authority row must not authorize a duplicate or forbidden model role."""

    with pytest.raises(ValueError, match="duplicate"):
        StageAuthorityRow(
            stage_id="E0_MINIMUM_BOOTSTRAP",
            variant_id="E1_MERT_PROBE",
            phase=Phase.BOOTSTRAP,
            execution_role=ExecutionRole.INFERENCE,
            active_roles=(CapabilityRole.MERT_PROBE_IDENTITY, CapabilityRole.MERT_PROBE_IDENTITY),
            forbidden_roles=(CapabilityRole.MERT_REAL_FORWARD,),
        )
    with pytest.raises(ValueError, match="cannot be both active and forbidden"):
        StageAuthorityRow(
            stage_id="E0_MINIMUM_BOOTSTRAP",
            variant_id="E1_MERT_PROBE",
            phase=Phase.BOOTSTRAP,
            execution_role=ExecutionRole.INFERENCE,
            active_roles=(CapabilityRole.MERT_REAL_FORWARD,),
            forbidden_roles=(CapabilityRole.MERT_REAL_FORWARD,),
        )
