"""Strict contracts for the E1.1 MERT real frame-rate observation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import Field

from sc_mv_dmer.foundation.manifests import ImmutableRecord


E11_INPUT_SAMPLE_RATE_HZ = 24_000
E11_INPUT_SAMPLE_COUNT = 1_080_000
E11_INPUT_DURATION_SECONDS = 45.0


class E11VerificationError(ValueError):
    """Raised before evidence is issued when an E1.1 contract is violated."""


class ResearchLayerMapping(ImmutableRecord):
    hidden_states_count: int = Field(gt=0)
    index_convention: Literal[
        "hidden_states[0]=encoder_input; hidden_states[n]=transformer_block_n_output"
    ]
    research_layer_5_actual_index: int = Field(ge=0)
    research_layer_5_block_identity: Literal["encoder.layers[4] / transformer_block_5"]
    research_layer_6_actual_index: int = Field(ge=0)
    research_layer_6_block_identity: Literal["encoder.layers[5] / transformer_block_6"]


class MertProbeObservation(ImmutableRecord):
    probe_id: str = Field(min_length=1)
    probe_type: Literal["P1_CANONICAL", "P2_REAL_DEAM"]
    input_sample_rate: Literal[24000]
    input_sample_count: Literal[1080000]
    input_duration_seconds: Literal[45.0]
    input_tensor_shape: tuple[int, int]
    hidden_states_count: int = Field(gt=0)
    research_layer_5_actual_index: int = Field(ge=0)
    research_layer_5_block_identity: str = Field(min_length=1)
    research_layer_6_actual_index: int = Field(ge=0)
    research_layer_6_block_identity: str = Field(min_length=1)
    index_convention: str = Field(min_length=1)
    layer5_shape: tuple[int, int, int]
    layer6_shape: tuple[int, int, int]
    observed_frame_count: int = Field(gt=0)
    hidden_dimension: int = Field(gt=0)
    observed_fps: float = Field(gt=0)
    layer5_finite: Literal[True]
    layer6_finite: Literal[True]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pinned_manifest_file(
    manifest_path: Path,
    *,
    expected_sha256: str,
) -> None:
    """Reject a manifest whose bytes are not the pre-authorized E1.0 bytes."""

    actual = _sha256_file(Path(manifest_path))
    if actual != expected_sha256:
        raise E11VerificationError(
            f"manifest SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )


def validate_registered_source(
    *,
    data_root: Path,
    source_relative_path: str,
    expected_sha256: str,
) -> Path:
    """Resolve and hash the registered DEAM probe without changing its bytes."""

    root = Path(data_root).resolve()
    source = (root / Path(source_relative_path)).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise E11VerificationError("source audio escapes configured data root") from error
    if not source.is_file():
        raise E11VerificationError(f"source audio missing: {source}")
    actual = _sha256_file(source)
    if actual != expected_sha256:
        raise E11VerificationError(
            f"source audio SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )
    return source


def validate_preprocessed_contract(
    *,
    sample_rate_hz: int,
    sample_count: int,
    duration_seconds: float,
) -> None:
    """Require the exact frozen E1.1 waveform representation."""

    if sample_rate_hz != E11_INPUT_SAMPLE_RATE_HZ:
        raise E11VerificationError(
            f"preprocessed sample rate must be {E11_INPUT_SAMPLE_RATE_HZ:,} Hz"
        )
    if sample_count != E11_INPUT_SAMPLE_COUNT:
        raise E11VerificationError(
            f"preprocessed sample count must be {E11_INPUT_SAMPLE_COUNT:,}"
        )
    if duration_seconds != E11_INPUT_DURATION_SECONDS:
        raise E11VerificationError(
            f"preprocessed duration must be {E11_INPUT_DURATION_SECONDS} seconds"
        )


def resolve_research_layer_mapping(
    *,
    hidden_states_count: int,
    block_to_hidden_state_candidates: Mapping[int, Sequence[int]],
) -> ResearchLayerMapping:
    """Resolve block-to-returned-index mapping from runtime tensor identity evidence."""

    resolved: dict[int, int] = {}
    for block_number in (5, 6):
        candidates = tuple(block_to_hidden_state_candidates.get(block_number, ()))
        if len(candidates) != 1:
            raise E11VerificationError(
                f"research layer {block_number} mapping is ambiguous: {candidates}"
            )
        index = candidates[0]
        if index < 0 or index >= hidden_states_count:
            raise E11VerificationError(
                f"research layer {block_number} index is outside hidden-state collection"
            )
        resolved[block_number] = index
    if resolved[5] == resolved[6]:
        raise E11VerificationError("research layer mapping is ambiguous: shared index")
    return ResearchLayerMapping(
        hidden_states_count=hidden_states_count,
        index_convention=(
            "hidden_states[0]=encoder_input; "
            "hidden_states[n]=transformer_block_n_output"
        ),
        research_layer_5_actual_index=resolved[5],
        research_layer_5_block_identity="encoder.layers[4] / transformer_block_5",
        research_layer_6_actual_index=resolved[6],
        research_layer_6_block_identity="encoder.layers[5] / transformer_block_6",
    )


def build_probe_observation(
    *,
    probe_id: str,
    probe_type: Literal["P1_CANONICAL", "P2_REAL_DEAM"],
    input_sample_rate: int,
    input_sample_count: int,
    input_tensor_shape: tuple[int, int],
    hidden_states_count: int,
    layer_mapping: ResearchLayerMapping,
    layer5_shape: tuple[int, int, int],
    layer6_shape: tuple[int, int, int],
    layer5_finite: bool,
    layer6_finite: bool,
) -> MertProbeObservation:
    """Validate actual tensor metadata and derive only the observed frame rate."""

    validate_preprocessed_contract(
        sample_rate_hz=input_sample_rate,
        sample_count=input_sample_count,
        duration_seconds=input_sample_count / input_sample_rate,
    )
    if input_tensor_shape != (1, E11_INPUT_SAMPLE_COUNT):
        raise E11VerificationError(
            f"input tensor shape must be (1, {E11_INPUT_SAMPLE_COUNT})"
        )
    if hidden_states_count != layer_mapping.hidden_states_count:
        raise E11VerificationError("hidden-state collection changed after layer mapping")
    if layer5_shape[1] != layer6_shape[1]:
        raise E11VerificationError("layer5/layer6 temporal dimensions differ")
    if layer5_shape[2] != layer6_shape[2]:
        raise E11VerificationError("layer5/layer6 hidden dimensions differ")
    if layer5_shape[0] != 1 or layer6_shape[0] != 1:
        raise E11VerificationError("layer5/layer6 batch dimension must be one")
    if not layer5_finite or not layer6_finite:
        raise E11VerificationError("layer5/layer6 contains non-finite values")

    observed_frame_count = layer5_shape[1]
    return MertProbeObservation(
        probe_id=probe_id,
        probe_type=probe_type,
        input_sample_rate=input_sample_rate,
        input_sample_count=input_sample_count,
        input_duration_seconds=E11_INPUT_DURATION_SECONDS,
        input_tensor_shape=input_tensor_shape,
        hidden_states_count=hidden_states_count,
        research_layer_5_actual_index=layer_mapping.research_layer_5_actual_index,
        research_layer_5_block_identity=layer_mapping.research_layer_5_block_identity,
        research_layer_6_actual_index=layer_mapping.research_layer_6_actual_index,
        research_layer_6_block_identity=layer_mapping.research_layer_6_block_identity,
        index_convention=layer_mapping.index_convention,
        layer5_shape=layer5_shape,
        layer6_shape=layer6_shape,
        observed_frame_count=observed_frame_count,
        hidden_dimension=layer5_shape[2],
        observed_fps=observed_frame_count / E11_INPUT_DURATION_SECONDS,
        layer5_finite=True,
        layer6_finite=True,
    )


def require_structural_consistency(
    p1: MertProbeObservation,
    p2: MertProbeObservation,
) -> None:
    """Reject divergent P1/P2 structural observations without adaptation."""

    if p1.observed_frame_count != p2.observed_frame_count:
        raise E11VerificationError("P1/P2 frame counts differ")
    if p1.hidden_dimension != p2.hidden_dimension:
        raise E11VerificationError("P1/P2 hidden dimensions differ")
    if p1.hidden_states_count != p2.hidden_states_count:
        raise E11VerificationError("P1/P2 hidden-state collection counts differ")
    if (
        p1.research_layer_5_actual_index != p2.research_layer_5_actual_index
        or p1.research_layer_6_actual_index != p2.research_layer_6_actual_index
    ):
        raise E11VerificationError("P1/P2 research layer mappings differ")
