from __future__ import annotations

from sc_mv_dmer.features.mert_temporal_geometry import (
    derive_temporal_geometry,
    frame_supports,
)


def test_pinned_geometry_derives_3374_frames_320_stride_400_receptive_field() -> None:
    geometry = derive_temporal_geometry(
        input_sample_rate_hz=24_000,
        input_sample_count=1_080_000,
        kernels=(10, 3, 3, 3, 3, 2, 2),
        strides=(5, 2, 2, 2, 2, 2, 2),
        source_config_sha256="a" * 64,
        source_custom_code_aggregate_sha256="b" * 64,
        source_geometry_identity="fixture",
    )

    assert geometry.expected_frame_count == 3374
    assert geometry.effective_stride_samples == 320
    assert geometry.effective_receptive_field_samples == 400
    assert geometry.first_frame_center_sample_numerator == 399
    assert geometry.first_frame_center_sample_denominator == 2
    assert geometry.expected_last_support_end_exclusive_samples == 1_079_760
    assert geometry.tail_uncovered_samples == 240

    supports = frame_supports(geometry)
    assert len(supports) == 3374
    assert supports[0] == {
        "frame_index": 0,
        "support_start_sample": 0,
        "support_end_exclusive_sample": 400,
        "center_sample_numerator": 399,
        "center_sample_denominator": 2,
    }
    assert supports[-1]["support_start_sample"] == 1_079_360
    assert supports[-1]["support_end_exclusive_sample"] == 1_079_760


def test_nonzero_padding_or_dilation_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="zero padding"):
        derive_temporal_geometry(
            input_sample_rate_hz=24_000,
            input_sample_count=1_080_000,
            kernels=(3,),
            strides=(2,),
            paddings=(1,),
            source_config_sha256="a" * 64,
            source_custom_code_aggregate_sha256="b" * 64,
            source_geometry_identity="fixture",
        )

