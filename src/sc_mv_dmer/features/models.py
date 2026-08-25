"""Typed raw/binned feature and pseudo-label records."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sc_mv_dmer.foundation.canonical import sha256_canonical


ViewName = Literal["deep", "mel", "mfcc", "chroma"]


def ndarray_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return sha256_canonical({"dtype": str(value.dtype), "shape": list(value.shape), "bytes_sha256": __import__("hashlib").sha256(value.tobytes(order="C")).hexdigest()})


class FeatureFrameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    dataset_id: str = Field(min_length=1)
    song_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    view: ViewName
    rule_version: str = Field(min_length=1)
    values: np.ndarray
    frame_center_sample_numerators: tuple[int, ...] = Field(min_length=1)
    frame_center_sample_denominator: int = Field(gt=0)
    sample_rate_hz: int = Field(gt=0)
    support_length_samples: int = Field(gt=0)
    hop_length_samples: int = Field(gt=0)
    feature_dim: int = Field(gt=0)
    dtype: str = Field(min_length=1)
    finite: bool
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    waveform_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_array(self) -> "FeatureFrameRecord":
        if self.values.ndim != 2 or self.values.shape[1] != self.feature_dim:
            raise ValueError("feature values must be [raw_frames, feature_dim]")
        if self.values.shape[0] != len(self.frame_center_sample_numerators):
            raise ValueError("timestamp count must equal raw frame count")
        if str(self.values.dtype) != self.dtype:
            raise ValueError("dtype provenance does not match values")
        if bool(np.isfinite(self.values).all()) != self.finite:
            raise ValueError("finite provenance does not match values")
        if ndarray_sha256(self.values) != self.content_sha256:
            raise ValueError("feature content checksum mismatch")
        return self


class BinnedFeatureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    dataset_id: str = Field(min_length=1)
    song_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    view: ViewName
    values: np.ndarray
    bin_count: Literal[90] = 90
    feature_dim: int = Field(gt=0)
    bin_counts: tuple[int, ...] = Field(min_length=90, max_length=90)
    raw_eligible_count: int = Field(gt=0)
    unassigned_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    timegrid_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dimension_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregation_rule: Literal["MEAN_OVER_ASSIGNED_VALID_FRAMES"] = "MEAN_OVER_ASSIGNED_VALID_FRAMES"
    finite: bool
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_binned(self) -> "BinnedFeatureRecord":
        if self.values.shape != (90, self.feature_dim):
            raise ValueError("binned feature values must have shape [90, feature_dim]")
        if len(self.bin_counts) != 90 or sum(self.bin_counts) != self.raw_eligible_count:
            raise ValueError("bin counts do not account for every raw frame")
        if self.unassigned_count != 0 or self.duplicate_count != 0 or min(self.bin_counts) <= 0:
            raise ValueError("RG-01 binner has uncovered, duplicate or empty bins")
        if bool(np.isfinite(self.values).all()) != self.finite:
            raise ValueError("finite provenance does not match values")
        if ndarray_sha256(self.values) != self.content_sha256:
            raise ValueError("binned feature content checksum mismatch")
        return self


class PseudoLabelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    dataset_id: str = Field(min_length=1)
    song_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule_version: str = Field(min_length=1)
    targets: dict[str, np.ndarray]
    target_dimensions: dict[str, int]
    valid_target_mask: np.ndarray
    segment_boundaries_seconds: tuple[tuple[float, float], ...] = Field(min_length=1)
    timegrid_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_labels(self) -> "PseudoLabelRecord":
        if set(self.targets) != {"rms", "brightness", "mode", "key"}:
            raise ValueError("pseudo-label bundle must contain RMS/brightness/mode/key")
        if self.valid_target_mask.shape != (90,):
            raise ValueError("pseudo-label target mask must be [90]")
        for name, value in self.targets.items():
            if value.shape != (90, self.target_dimensions[name]):
                raise ValueError(f"pseudo-label shape mismatch: {name}")
            if not np.isfinite(value).all():
                raise ValueError(f"pseudo-label contains non-finite values: {name}")
        payload = {name: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": ndarray_sha256(value)} for name, value in sorted(self.targets.items())}
        payload["mask"] = {"dtype": str(self.valid_target_mask.dtype), "shape": list(self.valid_target_mask.shape), "sha256": ndarray_sha256(self.valid_target_mask)}
        if sha256_canonical(payload) != self.content_sha256:
            raise ValueError("pseudo-label content checksum mismatch")
        return self
