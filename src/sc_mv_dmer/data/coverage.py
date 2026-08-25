"""Independent input/target/Sensor/Event validity masks."""

from __future__ import annotations

import hashlib

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def mask_sha256(mask: np.ndarray) -> str:
    value = np.asarray(mask, dtype=np.bool_)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


class EvaluationInputCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    input_coverage_mask: np.ndarray
    target_validity_mask: np.ndarray
    sensor_validity_mask: np.ndarray
    event_validity_mask: np.ndarray
    input_coverage_mask_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_validity_mask_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sensor_validity_mask_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_validity_mask_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_masks(self) -> "EvaluationInputCoverage":
        masks = (self.input_coverage_mask, self.target_validity_mask, self.sensor_validity_mask, self.event_validity_mask)
        if any(mask.shape != (90,) or mask.dtype != np.bool_ for mask in masks):
            raise ValueError("all coverage masks must be independent boolean [90] arrays")
        hashes = (self.input_coverage_mask_hash, self.target_validity_mask_hash, self.sensor_validity_mask_hash, self.event_validity_mask_hash)
        if any(mask_sha256(mask) != digest for mask, digest in zip(masks, hashes)):
            raise ValueError("coverage mask checksum mismatch")
        return self


def compile_input_coverage(
    input_coverage_mask: np.ndarray,
    target_validity_mask: np.ndarray | None = None,
    sensor_validity_mask: np.ndarray | None = None,
    event_validity_mask: np.ndarray | None = None,
) -> EvaluationInputCoverage:
    input_mask = np.asarray(input_coverage_mask, dtype=np.bool_)
    target = np.ones(90, dtype=np.bool_) if target_validity_mask is None else np.asarray(target_validity_mask, dtype=np.bool_)
    sensor = np.ones(90, dtype=np.bool_) if sensor_validity_mask is None else np.asarray(sensor_validity_mask, dtype=np.bool_)
    event = np.ones(90, dtype=np.bool_) if event_validity_mask is None else np.asarray(event_validity_mask, dtype=np.bool_)
    return EvaluationInputCoverage(
        input_coverage_mask=input_mask,
        target_validity_mask=target,
        sensor_validity_mask=sensor,
        event_validity_mask=event,
        input_coverage_mask_hash=mask_sha256(input_mask),
        target_validity_mask_hash=mask_sha256(target),
        sensor_validity_mask_hash=mask_sha256(sensor),
        event_validity_mask_hash=mask_sha256(event),
    )
