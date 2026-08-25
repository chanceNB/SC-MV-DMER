"""E1.3 exact sample-domain MERT-to-2Hz TimeGrid binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from sc_mv_dmer.features.mert_temporal_geometry import (
    MertTemporalGeometry,
    frame_supports,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.manifests import ImmutableRecord, freeze_value


class TimeGridBindingError(ValueError):
    """Raised when raw frame membership cannot satisfy RG-01 exactly."""


class TimeGridFrame(ImmutableRecord):
    frame_index: int = Field(ge=0)
    support_start_sample: int
    support_end_exclusive_sample: int
    center_sample_numerator: int
    center_sample_denominator: Literal[2] = 2
    center_time_numerator: int
    center_time_denominator: int
    assigned_bin_index: int = Field(ge=0)
    assignment_count: Literal[1] = 1
    valid_frame: Literal[True] = True


class TimeGridBin(ImmutableRecord):
    bin_index: int = Field(ge=0)
    start_sample_inclusive: int = Field(ge=0)
    end_sample_exclusive: int = Field(gt=0)
    start_time_numerator: int = Field(ge=0)
    end_time_numerator: int = Field(gt=0)
    time_denominator: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    valid_bin: Literal[True] = True


class MertTimeGridBinding(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    binding_id: Literal["MERT-2HZ-TIMEGRID-v1"] = "MERT-2HZ-TIMEGRID-v1"
    qualification_id: Literal["E1-3-MERT-TIMEGRID-2HZ-BINDING-v1"] = (
        "E1-3-MERT-TIMEGRID-2HZ-BINDING-v1"
    )
    predecessor_qualification: str = Field(min_length=1)
    predecessor_qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension_binding_logical_path: str = Field(min_length=1)
    dimension_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temporal_geometry: MertTemporalGeometry
    temporal_geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: int = Field(gt=0)
    input_sample_count: int = Field(gt=0)
    duration_seconds_numerator: Literal[45] = 45
    duration_seconds_denominator: Literal[1] = 1
    raw_frame_count: int = Field(gt=0)
    expected_frame_count: int = Field(gt=0)
    bin_count: Literal[90] = 90
    bin_duration_seconds_numerator: Literal[1] = 1
    bin_duration_seconds_denominator: Literal[2] = 2
    bins: tuple[TimeGridBin, ...] = Field(min_length=90, max_length=90)
    frames: tuple[TimeGridFrame, ...] = Field(min_length=1)
    valid_frame_count: int = Field(gt=0)
    invalid_frame_count: int = Field(ge=0)
    assigned_frame_count: int = Field(gt=0)
    unassigned_frame_count: Literal[0] = 0
    duplicate_assignment_count: Literal[0] = 0
    bin_frame_counts: tuple[int, ...] = Field(min_length=90, max_length=90)
    min_bin_frame_count: int = Field(gt=0)
    max_bin_frame_count: int = Field(gt=0)
    valid_bin_count: Literal[90] = 90
    aggregation_rule: Literal["MEAN_OVER_ASSIGNED_VALID_FRAMES"] = (
        "MEAN_OVER_ASSIGNED_VALID_FRAMES"
    )
    bin_boundary_rule: Literal["HALF_OPEN_SAMPLE_DOMAIN"] = "HALF_OPEN_SAMPLE_DOMAIN"
    assignment_formula: str = Field(min_length=1)
    mask_semantics: Literal["EXPLICIT_VALID_FRAME_AND_BIN_MASKS"] = (
        "EXPLICIT_VALID_FRAME_AND_BIN_MASKS"
    )
    forbidden_operations: tuple[
        Literal[
            "TRIM",
            "PADDING",
            "DROP",
            "CLAMP",
            "FRAME_COPY",
            "FILL",
            "INTERPOLATION",
            "ADAPTIVE_POOLING",
            "RESHAPE",
        ], ...
    ]
    mapping_verdict: Literal["PASS"] = "PASS"
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("bin_frame_counts", mode="after")
    @classmethod
    def _counts_match_bins(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(count <= 0 for count in value):
            raise ValueError("all canonical 2Hz bins must be non-empty")
        return value


class E13TimeGridQualification(ImmutableRecord):
    """Terminal E1.3 qualification; it does not authorize E2 by itself."""

    schema_version: Literal["1.0"] = "1.0"
    qualification_id: Literal["E1-3-MERT-TIMEGRID-2HZ-BINDING-v1"] = (
        "E1-3-MERT-TIMEGRID-2HZ-BINDING-v1"
    )
    implementation_git_commit: str = Field(min_length=7)
    observed_git_dirty: bool
    predecessor_qualification: str = Field(min_length=1)
    predecessor_qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension_binding_logical_path: str = Field(min_length=1)
    dimension_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timegrid_logical_path: str = Field(min_length=1)
    timegrid_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timegrid_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temporal_geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_frame_count: int = Field(gt=0)
    expected_frame_count: int = Field(gt=0)
    bin_count: Literal[90] = 90
    assigned_frame_count: int = Field(gt=0)
    unassigned_frame_count: Literal[0] = 0
    duplicate_assignment_count: Literal[0] = 0
    min_bin_frame_count: int = Field(gt=0)
    max_bin_frame_count: int = Field(gt=0)
    aggregation_rule: Literal["MEAN_OVER_ASSIGNED_VALID_FRAMES"] = (
        "MEAN_OVER_ASSIGNED_VALID_FRAMES"
    )
    forbidden_actions: tuple[
        Literal[
            "TRIM",
            "PADDING",
            "DROP",
            "CLAMP",
            "FRAME_COPY",
            "FILL",
            "INTERPOLATION",
            "ADAPTIVE_POOLING",
            "RESHAPE",
        ], ...
    ]
    e1_3_verdict: Literal["PASS"] = "PASS"
    e1_4_readiness: Literal["READY"] = "READY"
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _bin_index(
    *,
    center_sample_numerator: int,
    center_sample_denominator: int,
    sample_rate_hz: int,
    bin_duration_seconds_numerator: int,
    bin_duration_seconds_denominator: int,
) -> int:
    """Compute floor(center / bin width) using integer arithmetic only."""

    numerator = center_sample_numerator * bin_duration_seconds_denominator
    denominator = (
        center_sample_denominator
        * sample_rate_hz
        * bin_duration_seconds_numerator
    )
    return numerator // denominator


def compile_mert_timegrid(
    *,
    geometry: MertTemporalGeometry,
    dimension_binding_logical_path: str,
    dimension_binding_sha256: str,
    predecessor_qualification: str,
    predecessor_qualification_sha256: str,
    temporal_geometry_sha256: str,
) -> MertTimeGridBinding:
    """Bind every valid MERT frame to exactly one of 90 half-open bins."""

    if geometry.input_sample_rate_hz != 24_000:
        raise TimeGridBindingError("RG-01 TimeGrid requires 24 kHz MERT input")
    if geometry.input_sample_count != 1_080_000:
        raise TimeGridBindingError("RG-01 TimeGrid requires 1,080,000 input samples")
    supports = frame_supports(geometry)
    if not supports:
        raise TimeGridBindingError("geometry has no raw frames")
    frame_records: list[TimeGridFrame] = []
    counts = [0] * 90
    invalid_count = 0
    for support in supports:
        start = support["support_start_sample"]
        end = support["support_end_exclusive_sample"]
        valid = 0 <= start and end <= geometry.input_sample_count
        if not valid:
            invalid_count += 1
            continue
        frame_bin = _bin_index(
            center_sample_numerator=support["center_sample_numerator"],
            center_sample_denominator=support["center_sample_denominator"],
            sample_rate_hz=geometry.input_sample_rate_hz,
            bin_duration_seconds_numerator=1,
            bin_duration_seconds_denominator=2,
        )
        if frame_bin < 0 or frame_bin >= 90:
            raise TimeGridBindingError(
                f"frame {support['frame_index']} center falls outside the 45s grid"
            )
        counts[frame_bin] += 1
        frame_records.append(
            TimeGridFrame(
                frame_index=support["frame_index"],
                support_start_sample=start,
                support_end_exclusive_sample=end,
                center_sample_numerator=support["center_sample_numerator"],
                center_time_numerator=support["center_sample_numerator"],
                center_time_denominator=geometry.input_sample_rate_hz * 2,
                assigned_bin_index=frame_bin,
            )
        )
    if invalid_count:
        raise TimeGridBindingError(
            "pinned 45s contract produced a frame outside source support; "
            "padding cannot be used to repair it"
        )
    if len(frame_records) != geometry.expected_frame_count:
        raise TimeGridBindingError("valid frame count differs from expected geometry count")
    if sum(counts) != geometry.expected_frame_count:
        raise TimeGridBindingError("frame assignment does not account for every raw frame")
    if any(count <= 0 for count in counts):
        raise TimeGridBindingError("one or more canonical 2Hz bins is empty")

    bins = tuple(
        TimeGridBin(
            bin_index=index,
            start_sample_inclusive=index * 12_000,
            end_sample_exclusive=(index + 1) * 12_000,
            start_time_numerator=index,
            end_time_numerator=index + 1,
            time_denominator=2,
            frame_count=count,
        )
        for index, count in enumerate(counts)
    )
    payload = {
        "schema_version": "1.0",
        "binding_id": "MERT-2HZ-TIMEGRID-v1",
        "qualification_id": "E1-3-MERT-TIMEGRID-2HZ-BINDING-v1",
        "predecessor_qualification": predecessor_qualification,
        "predecessor_qualification_sha256": predecessor_qualification_sha256,
        "dimension_binding_logical_path": dimension_binding_logical_path,
        "dimension_binding_sha256": dimension_binding_sha256,
        "temporal_geometry": geometry.model_dump(mode="json"),
        "temporal_geometry_sha256": temporal_geometry_sha256,
        "sample_rate_hz": geometry.input_sample_rate_hz,
        "input_sample_count": geometry.input_sample_count,
        "duration_seconds_numerator": 45,
        "duration_seconds_denominator": 1,
        "raw_frame_count": geometry.expected_frame_count,
        "expected_frame_count": geometry.expected_frame_count,
        "bin_count": 90,
        "bin_duration_seconds_numerator": 1,
        "bin_duration_seconds_denominator": 2,
        "bins": [item.model_dump(mode="json") for item in bins],
        "frames": [item.model_dump(mode="json") for item in frame_records],
        "valid_frame_count": len(frame_records),
        "invalid_frame_count": invalid_count,
        "assigned_frame_count": len(frame_records),
        "unassigned_frame_count": 0,
        "duplicate_assignment_count": 0,
        "bin_frame_counts": counts,
        "min_bin_frame_count": min(counts),
        "max_bin_frame_count": max(counts),
        "valid_bin_count": 90,
        "aggregation_rule": "MEAN_OVER_ASSIGNED_VALID_FRAMES",
        "bin_boundary_rule": "HALF_OPEN_SAMPLE_DOMAIN",
        "assignment_formula": (
            "floor(center_sample_numerator * bin_duration_denominator / "
            "(center_sample_denominator * sample_rate_hz * bin_duration_numerator))"
        ),
        "mask_semantics": "EXPLICIT_VALID_FRAME_AND_BIN_MASKS",
        "forbidden_operations": [
            "TRIM",
            "PADDING",
            "DROP",
            "CLAMP",
            "FRAME_COPY",
            "FILL",
            "INTERPOLATION",
            "ADAPTIVE_POOLING",
            "RESHAPE",
        ],
        "mapping_verdict": "PASS",
    }
    return MertTimeGridBinding(
        **payload,
        binding_sha256=sha256_canonical(payload),
    )


def verify_timegrid_binding(binding: MertTimeGridBinding) -> None:
    payload = binding.model_dump(mode="json")
    observed = payload.pop("binding_sha256")
    if sha256_canonical(payload) != observed:
        raise TimeGridBindingError("TimeGrid binding checksum mismatch")
    if binding.raw_frame_count != binding.expected_frame_count:
        raise TimeGridBindingError("raw/expected frame counts differ")
    if binding.valid_frame_count != binding.raw_frame_count:
        raise TimeGridBindingError("not all raw frames are valid source-supported frames")
    if binding.unassigned_frame_count != 0 or binding.duplicate_assignment_count != 0:
        raise TimeGridBindingError("TimeGrid has unassigned or duplicate frames")
    if sum(binding.bin_frame_counts) != binding.raw_frame_count:
        raise TimeGridBindingError("bin counts do not sum to raw frame count")
    if len(binding.frames) != binding.raw_frame_count or len(binding.bins) != 90:
        raise TimeGridBindingError("TimeGrid record cardinality mismatch")


def write_immutable_timegrid_binding(
    binding: MertTimeGridBinding, destination: Path
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(canonical_json(binding.model_dump(mode="json")) + "\n")


def load_timegrid_binding(path: Path) -> MertTimeGridBinding:
    return MertTimeGridBinding.model_validate_json(Path(path).read_text(encoding="utf-8"))
