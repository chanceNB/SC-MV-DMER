"""Pinned MERT temporal geometry and exact sample-domain frame supports.

The geometry here describes only the seven waveform feature-extractor
convolutions used by the pinned MERT snapshot.  Transformer positional
convolutions are not part of the waveform-to-frame time axis.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import Field, field_validator

from sc_mv_dmer.foundation.canonical import canonical_json
from sc_mv_dmer.foundation.manifests import ImmutableRecord, freeze_value


class TemporalGeometryError(ValueError):
    """Raised when a temporal contract cannot be derived exactly."""


class ConvLayerGeometry(ImmutableRecord):
    layer_index: int = Field(ge=0)
    kernel_size: int = Field(gt=0)
    stride: int = Field(gt=0)
    padding: int = Field(ge=0)
    dilation: int = Field(gt=0)
    input_length: int = Field(gt=0)
    output_length: int = Field(gt=0)
    cumulative_stride_samples: int = Field(gt=0)
    receptive_field_samples: int = Field(gt=0)
    support_start_offset_samples: int
    support_end_offset_exclusive_samples: int
    length_formula: str = Field(min_length=1)


class MertTemporalGeometry(ImmutableRecord):
    """A fully resolved valid-convolution temporal contract."""

    contract_id: Literal["MERTTimeAxisContract"] = "MERTTimeAxisContract"
    contract_version: Literal["MERT-TIME-AXIS-v1"] = "MERT-TIME-AXIS-v1"
    input_sample_rate_hz: int = Field(gt=0)
    input_sample_count: int = Field(gt=0)
    convolution_layer_count: int = Field(gt=0)
    layers: tuple[ConvLayerGeometry, ...] = Field(min_length=1)
    effective_stride_samples: int = Field(gt=0)
    effective_receptive_field_samples: int = Field(gt=0)
    first_frame_support_start_samples: int
    first_frame_support_end_exclusive_samples: int
    first_frame_center_sample_numerator: int
    first_frame_center_sample_denominator: Literal[2] = 2
    expected_frame_count: int = Field(gt=0)
    expected_last_support_end_exclusive_samples: int
    tail_uncovered_samples: int
    geometry_rule: Literal["VALID_CONVOLUTION_NO_PADDING"] = (
        "VALID_CONVOLUTION_NO_PADDING"
    )
    source_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_custom_code_aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_geometry_identity: str = Field(min_length=1)

    @field_validator("layers")
    @classmethod
    def _layers_match_count(
        cls, value: tuple[ConvLayerGeometry, ...]
    ) -> tuple[ConvLayerGeometry, ...]:
        if not value:
            raise ValueError("at least one convolution layer is required")
        if tuple(layer.layer_index for layer in value) != tuple(range(len(value))):
            raise ValueError("convolution layer indices must be contiguous from zero")
        return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _conv_output_length(
    input_length: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> int:
    numerator = input_length + 2 * padding - dilation * (kernel_size - 1) - 1
    if numerator < 0:
        raise TemporalGeometryError("convolution kernel has no valid input support")
    return numerator // stride + 1


def derive_temporal_geometry(
    *,
    input_sample_rate_hz: int,
    input_sample_count: int,
    kernels: Sequence[int],
    strides: Sequence[int],
    paddings: Sequence[int] | None = None,
    dilations: Sequence[int] | None = None,
    source_config_sha256: str,
    source_custom_code_aggregate_sha256: str,
    source_geometry_identity: str,
) -> MertTemporalGeometry:
    """Derive lengths, support offsets and rational frame centers.

    Padding and dilation are accepted as explicit inputs so a future pinned
    revision cannot silently inherit a different formula.  The current
    contract rejects anything other than valid (zero-padding) convolutions.
    """

    kernels = tuple(int(value) for value in kernels)
    strides = tuple(int(value) for value in strides)
    if not kernels or len(kernels) != len(strides):
        raise TemporalGeometryError("kernel and stride lists must be non-empty and equal")
    paddings = tuple(0 for _ in kernels) if paddings is None else tuple(int(v) for v in paddings)
    dilations = tuple(1 for _ in kernels) if dilations is None else tuple(int(v) for v in dilations)
    if len(paddings) != len(kernels) or len(dilations) != len(kernels):
        raise TemporalGeometryError("padding/dilation lists must match kernel list")
    if any(value != 0 for value in paddings):
        raise TemporalGeometryError("MERTTimeAxisContract requires zero padding")
    if any(value != 1 for value in dilations):
        raise TemporalGeometryError("MERTTimeAxisContract requires unit dilation")
    if input_sample_rate_hz <= 0 or input_sample_count <= 0:
        raise TemporalGeometryError("input rate and count must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", source_config_sha256):
        raise TemporalGeometryError("source config checksum is not a SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", source_custom_code_aggregate_sha256):
        raise TemporalGeometryError("source custom-code checksum is not a SHA-256")

    current_length = input_sample_count
    cumulative_stride = 1
    receptive_field = 1
    support_start = 0
    layers: list[ConvLayerGeometry] = []
    for index, (kernel, stride, padding, dilation) in enumerate(
        zip(kernels, strides, paddings, dilations)
    ):
        if kernel <= 0 or stride <= 0 or dilation <= 0:
            raise TemporalGeometryError("kernel, stride and dilation must be positive")
        output_length = _conv_output_length(
            current_length, kernel, stride, padding, dilation
        )
        receptive_field += dilation * (kernel - 1) * cumulative_stride
        support_start -= padding * cumulative_stride
        cumulative_stride *= stride
        support_end = support_start + receptive_field
        layers.append(
            ConvLayerGeometry(
                layer_index=index,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                input_length=current_length,
                output_length=output_length,
                cumulative_stride_samples=cumulative_stride,
                receptive_field_samples=receptive_field,
                support_start_offset_samples=support_start,
                support_end_offset_exclusive_samples=support_end,
                length_formula=(
                    "floor((L + 2*padding - dilation*(kernel-1) - 1) / stride) + 1"
                ),
            )
        )
        current_length = output_length

    first_center_numerator = 2 * support_start + receptive_field - 1
    last_support_end = support_start + (current_length - 1) * cumulative_stride + receptive_field
    return MertTemporalGeometry(
        input_sample_rate_hz=input_sample_rate_hz,
        input_sample_count=input_sample_count,
        convolution_layer_count=len(layers),
        layers=tuple(layers),
        effective_stride_samples=cumulative_stride,
        effective_receptive_field_samples=receptive_field,
        first_frame_support_start_samples=support_start,
        first_frame_support_end_exclusive_samples=support_start + receptive_field,
        first_frame_center_sample_numerator=first_center_numerator,
        expected_frame_count=current_length,
        expected_last_support_end_exclusive_samples=last_support_end,
        tail_uncovered_samples=input_sample_count - last_support_end,
        source_config_sha256=source_config_sha256,
        source_custom_code_aggregate_sha256=source_custom_code_aggregate_sha256,
        source_geometry_identity=source_geometry_identity,
    )


def load_pinned_mert_geometry(
    snapshot_root: Path,
    *,
    input_sample_rate_hz: int,
    input_sample_count: int,
    expected_config_sha256: str | None = None,
    expected_custom_code_aggregate_sha256: str | None = None,
) -> MertTemporalGeometry:
    """Read the pinned config and verify its feature-extractor implementation.

    The pinned implementation uses ``nn.Conv1d`` without padding/dilation
    arguments.  Those defaults are checked in the source before deriving the
    contract, rather than being assumed from a README or a historical FPS.
    """

    root = Path(snapshot_root)
    config_path = root / "config.json"
    modeling_path = root / "modeling_MERT.py"
    if not config_path.is_file() or not modeling_path.is_file():
        raise TemporalGeometryError("pinned MERT config/modeling bytes are missing")
    config_sha256 = _sha256_file(config_path)
    if expected_config_sha256 is not None and config_sha256 != expected_config_sha256:
        raise TemporalGeometryError("pinned MERT config checksum mismatch")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemporalGeometryError("pinned MERT config is not valid JSON") from exc
    try:
        kernels = tuple(int(value) for value in config["conv_kernel"])
        strides = tuple(int(value) for value in config["conv_stride"])
        sample_rate = int(config["sample_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TemporalGeometryError("pinned MERT config lacks temporal geometry") from exc
    if sample_rate != input_sample_rate_hz:
        raise TemporalGeometryError("config sample rate differs from the probe contract")

    modeling_text = modeling_path.read_text(encoding="utf-8")
    # Both feature-extractor Conv1d constructors intentionally omit these
    # keyword arguments; rejecting an explicit nonzero/nonunit value catches
    # source drift without importing or instantiating the model.
    if "class MERTModel" not in modeling_text or "HubertFeatureEncoder(config)" not in modeling_text:
        raise TemporalGeometryError("pinned MERT model does not bind HubertFeatureEncoder")
    # The feature-extractor classes are inherited from the pinned Transformers
    # runtime.  Their Conv1d constructors omit padding/dilation, so these
    # arguments must not be supplied by the custom MERT wrapper itself.
    if "padding=config.conv_kernel" in modeling_text or "dilation=config.conv" in modeling_text:
        raise TemporalGeometryError("feature-extractor temporal padding/dilation drifted")
    custom_code_sha256 = expected_custom_code_aggregate_sha256 or (
        "0" * 64
    )
    return derive_temporal_geometry(
        input_sample_rate_hz=input_sample_rate_hz,
        input_sample_count=input_sample_count,
        kernels=kernels,
        strides=strides,
        source_config_sha256=config_sha256,
        source_custom_code_aggregate_sha256=custom_code_sha256,
        source_geometry_identity=(
            "pinned MERT HubertFeatureEncoder Conv1d layers; "
            "nn.Conv1d defaults padding=0,dilation=1"
        ),
    )


def frame_supports(geometry: MertTemporalGeometry) -> tuple[dict[str, int], ...]:
    """Return exact integer/rational support metadata for every raw frame."""

    result: list[dict[str, int]] = []
    for frame_index in range(geometry.expected_frame_count):
        start = (
            geometry.first_frame_support_start_samples
            + frame_index * geometry.effective_stride_samples
        )
        end = start + geometry.effective_receptive_field_samples
        center_numerator = (
            geometry.first_frame_center_sample_numerator
            + 2 * frame_index * geometry.effective_stride_samples
        )
        result.append(
            {
                "frame_index": frame_index,
                "support_start_sample": start,
                "support_end_exclusive_sample": end,
                "center_sample_numerator": center_numerator,
                "center_sample_denominator": geometry.first_frame_center_sample_denominator,
            }
        )
    return tuple(result)


def geometry_payload(geometry: MertTemporalGeometry) -> dict[str, object]:
    """Return a JSON-ready geometry payload for downstream evidence."""

    return geometry.model_dump(mode="json")


def geometry_sha256(geometry: MertTemporalGeometry) -> str:
    """Hash a geometry contract independently of any downstream artifact."""

    from sc_mv_dmer.foundation.canonical import sha256_canonical

    return sha256_canonical(geometry_payload(geometry))
