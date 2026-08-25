from __future__ import annotations

from sc_mv_dmer.features.mert_temporal_geometry import derive_temporal_geometry
from sc_mv_dmer.features.mert_timegrid import (
    compile_mert_timegrid,
    verify_timegrid_binding,
)


def test_standard_excerpt_maps_every_3374_frame_once_to_90_nonempty_bins() -> None:
    geometry = derive_temporal_geometry(
        input_sample_rate_hz=24_000,
        input_sample_count=1_080_000,
        kernels=(10, 3, 3, 3, 3, 2, 2),
        strides=(5, 2, 2, 2, 2, 2, 2),
        source_config_sha256="a" * 64,
        source_custom_code_aggregate_sha256="b" * 64,
        source_geometry_identity="fixture",
    )
    binding = compile_mert_timegrid(
        geometry=geometry,
        dimension_binding_logical_path="manifests/dimensions/mert-downstream-dimensions-v1.json",
        dimension_binding_sha256="c" * 64,
        predecessor_qualification="e1-2",
        predecessor_qualification_sha256="d" * 64,
        temporal_geometry_sha256="e" * 64,
    )

    verify_timegrid_binding(binding)
    assert binding.raw_frame_count == 3374
    assert binding.unassigned_frame_count == 0
    assert binding.duplicate_assignment_count == 0
    assert sum(binding.bin_frame_counts) == 3374
    assert min(binding.bin_frame_counts) == 37
    assert max(binding.bin_frame_counts) == 38
    assert len(binding.frames) == 3374
    assert len(binding.bins) == 90
    assert all(frame.assignment_count == 1 for frame in binding.frames)
    assert all(item.valid_bin for item in binding.bins)

