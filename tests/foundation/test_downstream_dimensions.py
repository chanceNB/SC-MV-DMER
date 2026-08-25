from __future__ import annotations

import pytest

from sc_mv_dmer.features.mert_temporal_geometry import derive_temporal_geometry
from sc_mv_dmer.foundation.dimensions import (
    DimensionBindingError,
    compile_downstream_dimensions,
    verify_downstream_binding,
)


def _geometry():
    return derive_temporal_geometry(
        input_sample_rate_hz=24_000,
        input_sample_count=1_080_000,
        kernels=(10, 3, 3, 3, 3, 2, 2),
        strides=(5, 2, 2, 2, 2, 2, 2),
        source_config_sha256="a" * 64,
        source_custom_code_aggregate_sha256="b" * 64,
        source_geometry_identity="fixture",
    )


def _e11() -> dict[str, object]:
    observation = {
        "observed_frame_count": 3374,
        "observed_fps": 3374 / 45.0,
        "hidden_dimension": 768,
    }
    return {"e1_1_verdict": "PASS", "p1": observation, "p2": observation.copy()}


def _manifest() -> dict[str, object]:
    return {
        "repository_id": "m-a-p/MERT-v1-95M",
        "semantic_identity_sha256": "c" * 64,
        "snapshot_aggregate_sha256": "d" * 64,
    }


def test_dimension_binding_resolves_measured_to_all_downstream_shapes() -> None:
    binding = compile_downstream_dimensions(
        e11_evidence=_e11(),
        e11_evidence_sha256="e" * 64,
        predecessor_qualification="evidence/qualifications/e1-1-mert-frame-rate-verification.json",
        predecessor_qualification_sha256="f" * 64,
        upstream_manifest=_manifest(),
        upstream_manifest_sha256="1" * 64,
        temporal_geometry=_geometry(),
    )
    verify_downstream_binding(binding)
    shapes = {node.semantic_name: node.shape for node in binding.nodes}
    assert shapes["TimesNet raw temporal input"] == (3374, 768)
    assert shapes["X_deep"] == (90, 768)
    assert shapes["F_cat"] == (90, 1024)
    assert shapes["F_fused"] == (90, 256)
    assert shapes["Y_va"] == (90, 2)
    assert binding.timesnet_period_interpretation.period_set_numeric is None
    assert binding.required_temporal_reduction.ratio_numerator == 3374
    assert binding.required_temporal_reduction.ratio_denominator == 90


def test_dimension_binding_rejects_e11_frame_mismatch() -> None:
    e11 = _e11()
    e11["p1"] = {**e11["p1"], "observed_frame_count": 2250, "observed_fps": 50.0}
    with pytest.raises(DimensionBindingError, match="structural observations differ"):
        compile_downstream_dimensions(
            e11_evidence=e11,
            e11_evidence_sha256="e" * 64,
            predecessor_qualification="e1-1",
            predecessor_qualification_sha256="f" * 64,
            upstream_manifest=_manifest(),
            upstream_manifest_sha256="1" * 64,
            temporal_geometry=_geometry(),
        )

