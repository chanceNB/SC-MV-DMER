"""E1.2 downstream dimension binding compiled from E1.1 evidence.

This module contains shape metadata only.  It never creates a model, changes
the measured MERT frame count, or performs temporal resampling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping

from pydantic import Field, field_validator

from sc_mv_dmer.features.mert_temporal_geometry import (
    MertTemporalGeometry,
    TemporalGeometryError,
    geometry_sha256,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.manifests import ImmutableRecord, freeze_value


class DimensionBindingError(ValueError):
    """Raised when E1.1 evidence cannot authorize E1.2 derivation."""


class DimensionNode(ImmutableRecord):
    node_id: str = Field(min_length=1)
    semantic_name: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    source_class: Literal["MEASURED", "DERIVED", "FROZEN_CONSTANT"]
    formula: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    dependencies: tuple[str, ...]

    @field_validator("shape")
    @classmethod
    def _shape_is_nonnegative(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(dimension < 0 for dimension in value):
            raise ValueError("dimension shapes cannot contain negative values")
        return value


class DimensionEdge(ImmutableRecord):
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    source_class: Literal["MEASURED", "DERIVED", "FROZEN_CONSTANT"]


class TimesNetPeriodInterpretation(ImmutableRecord):
    mode: Literal["DYNAMIC_FFT_TOP_K"] = "DYNAMIC_FFT_TOP_K"
    raw_temporal_length: int = Field(gt=0)
    period_set_numeric: tuple[int, ...] | None = None
    period_set_formula: str = Field(min_length=1)
    period_units: Literal["RAW_MERT_FRAMES"] = "RAW_MERT_FRAMES"
    static_period_constants_authorized: Literal[False] = False


class TemporalReductionBinding(ImmutableRecord):
    input_temporal_length: int = Field(gt=0)
    output_temporal_length: int = Field(gt=0)
    ratio_numerator: int = Field(gt=0)
    ratio_denominator: int = Field(gt=0)
    ratio_decimal: float = Field(gt=0)
    operation: Literal["SAMPLE_DOMAIN_FRAME_CENTER_BIN_MEAN"] = (
        "SAMPLE_DOMAIN_FRAME_CENTER_BIN_MEAN"
    )
    target_bin_duration_seconds: Literal[0.5] = 0.5
    target_bin_count: int = Field(gt=0)
    forbidden_operations: tuple[
        Literal[
            "TRIM",
            "PADDING",
            "INTERPOLATION",
            "RESHAPE",
            "ADAPTIVE_POOLING",
        ], ...
    ]


class DownstreamDimensionBinding(ImmutableRecord):
    schema_version: Literal["1.0"] = "1.0"
    binding_id: Literal["MERT-DOWNSTREAM-DIMENSIONS-v1"] = (
        "MERT-DOWNSTREAM-DIMENSIONS-v1"
    )
    qualification_id: Literal["E1-2-MERT-DOWNSTREAM-DIMENSION-DERIVATION-v1"] = (
        "E1-2-MERT-DOWNSTREAM-DIMENSION-DERIVATION-v1"
    )
    predecessor_qualification: str = Field(min_length=1)
    predecessor_qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    e1_1_evidence_logical_path: str = Field(min_length=1)
    e1_1_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_manifest_logical_path: str = Field(min_length=1)
    upstream_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_semantic_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_snapshot_aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temporal_geometry: MertTemporalGeometry
    temporal_geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_t_raw: int = Field(gt=0)
    observed_fps: float = Field(gt=0)
    hidden_dimension: int = Field(gt=0)
    timesnet_raw_temporal_input_dimension: int = Field(gt=0)
    timesnet_period_interpretation: TimesNetPeriodInterpretation
    required_temporal_reduction: TemporalReductionBinding
    canonical_downstream_t: int = Field(gt=0)
    nodes: tuple[DimensionNode, ...] = Field(min_length=1)
    edges: tuple[DimensionEdge, ...] = Field(min_length=1)
    all_downstream_temporal_dimensions: Mapping[str, int]
    forbidden_downstream_actions: tuple[
        Literal[
            "MERT_T_RAW_REWRITE",
            "TRIM",
            "PADDING",
            "INTERPOLATION",
            "RESHAPE",
            "ADAPTIVE_POOLING",
        ], ...
    ]
    derivation_verdict: Literal["PASS"] = "PASS"
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("all_downstream_temporal_dimensions", mode="after")
    @classmethod
    def _freeze_temporal_dimensions(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        return freeze_value(value)


class E12DimensionQualification(ImmutableRecord):
    """Terminal E1.2 qualification that authorizes E1.3 only."""

    schema_version: Literal["1.0"] = "1.0"
    qualification_id: Literal["E1-2-MERT-DOWNSTREAM-DIMENSION-DERIVATION-v1"] = (
        "E1-2-MERT-DOWNSTREAM-DIMENSION-DERIVATION-v1"
    )
    implementation_git_commit: str = Field(min_length=7)
    observed_git_dirty: bool
    predecessor_qualification: str = Field(min_length=1)
    predecessor_qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    e1_1_evidence_logical_path: str = Field(min_length=1)
    e1_1_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_logical_path: str = Field(min_length=1)
    binding_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temporal_geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_t_raw: int = Field(gt=0)
    observed_fps: float = Field(gt=0)
    canonical_downstream_t: Literal[90] = 90
    forbidden_actions: tuple[
        Literal[
            "MERT_T_RAW_REWRITE",
            "TRIM",
            "PADDING",
            "INTERPOLATION",
            "RESHAPE",
            "ADAPTIVE_POOLING",
        ], ...
    ]
    e1_2_verdict: Literal["PASS"] = "PASS"
    e1_3_readiness: Literal["READY"] = "READY"
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_e11_pass(e11_evidence: Mapping[str, object]) -> tuple[int, float, int]:
    if e11_evidence.get("e1_1_verdict") != "PASS":
        raise DimensionBindingError("E1.1 is not PASS")
    p1 = e11_evidence.get("p1")
    p2 = e11_evidence.get("p2")
    if not isinstance(p1, Mapping) or not isinstance(p2, Mapping):
        raise DimensionBindingError("E1.1 P1/P2 observations are missing")
    values = []
    for label, observation in (("p1", p1), ("p2", p2)):
        try:
            count = int(observation["observed_frame_count"])
            fps = float(observation["observed_fps"])
            dimension = int(observation["hidden_dimension"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DimensionBindingError(f"E1.1 {label} observation is incomplete") from exc
        if count <= 0 or dimension <= 0 or fps <= 0:
            raise DimensionBindingError(f"E1.1 {label} observation is non-positive")
        values.append((count, fps, dimension))
    if values[0] != values[1]:
        raise DimensionBindingError("E1.1 P1/P2 structural observations differ")
    count, fps, dimension = values[0]
    if abs(fps - count / 45.0) > 1e-12:
        raise DimensionBindingError("E1.1 observed_fps is not T_raw/45")
    if dimension != 768:
        raise DimensionBindingError("E1.1 hidden dimension is not the frozen 768")
    return count, fps, dimension


def _node(
    node_id: str,
    semantic_name: str,
    shape: tuple[int, ...],
    *,
    source_class: Literal["MEASURED", "DERIVED", "FROZEN_CONSTANT"],
    formula: str,
    dependencies: tuple[str, ...],
    dtype: str = "float32",
    unit: str = "tensor_dimension",
) -> DimensionNode:
    return DimensionNode(
        node_id=node_id,
        semantic_name=semantic_name,
        shape=shape,
        dtype=dtype,
        unit=unit,
        source_class=source_class,
        formula=formula,
        contract_version="MERT-DOWNSTREAM-DIMENSIONS-v1",
        dependencies=dependencies,
    )


def compile_downstream_dimensions(
    *,
    e11_evidence: Mapping[str, object],
    e11_evidence_sha256: str,
    predecessor_qualification: str,
    predecessor_qualification_sha256: str,
    upstream_manifest: Mapping[str, object],
    upstream_manifest_sha256: str,
    temporal_geometry: MertTemporalGeometry,
) -> DownstreamDimensionBinding:
    """Compile the complete shape DAG without changing raw temporal length."""

    raw_t, observed_fps, hidden_dimension = _assert_e11_pass(e11_evidence)
    if raw_t != temporal_geometry.expected_frame_count:
        raise DimensionBindingError(
            "pinned geometry expected frame count differs from E1.1 T_raw"
        )
    if temporal_geometry.input_sample_count != 1_080_000:
        raise DimensionBindingError("E1.2 requires the exact 45s/24kHz sample count")
    if upstream_manifest.get("repository_id") != "m-a-p/MERT-v1-95M":
        raise DimensionBindingError("upstream manifest repository identity mismatch")
    semantic_identity = str(upstream_manifest.get("semantic_identity_sha256", ""))
    snapshot_identity = str(upstream_manifest.get("snapshot_aggregate_sha256", ""))
    if len(semantic_identity) != 64 or len(snapshot_identity) != 64:
        raise DimensionBindingError("upstream manifest identities are incomplete")

    t = 90
    reduction = TemporalReductionBinding(
        input_temporal_length=raw_t,
        output_temporal_length=t,
        ratio_numerator=raw_t,
        ratio_denominator=t,
        ratio_decimal=raw_t / t,
        target_bin_count=t,
        forbidden_operations=(
            "TRIM",
            "PADDING",
            "INTERPOLATION",
            "RESHAPE",
            "ADAPTIVE_POOLING",
        ),
    )
    period_interpretation = TimesNetPeriodInterpretation(
        raw_temporal_length=raw_t,
        period_set_numeric=None,
        period_set_formula=(
            "runtime FFT top-k periods over the measured raw MERT sequence; "
            "period values are measured/derived by the TimesNet forward and "
            "are not static constants in E1.2"
        ),
    )

    nodes: list[DimensionNode] = [
        _node(
            "mert_raw_output",
            "X_mert",
            (raw_t, hidden_dimension),
            source_class="MEASURED",
            formula="E1.1 p1/p2 layer5_shape[1:3]",
            dependencies=("E1.1",),
        ),
        _node(
            "timesnet_raw_input",
            "TimesNet raw temporal input",
            (raw_t, hidden_dimension),
            source_class="DERIVED",
            formula="TimesNet input temporal length = measured T_raw; no rewrite",
            dependencies=("mert_raw_output",),
        ),
        _node(
            "x_deep",
            "X_deep",
            (t, hidden_dimension),
            source_class="DERIVED",
            formula="sample-domain frame-center assignment to 90 half-open bins; per-bin mean",
            dependencies=("timesnet_raw_input",),
        ),
        _node(
            "w_deep_projection",
            "W_deep",
            (hidden_dimension, 256),
            source_class="FROZEN_CONSTANT",
            formula="frozen deep projection 768 -> 256",
            dependencies=("mert_raw_output",),
            dtype="float32",
            unit="weight_shape",
        ),
        _node(
            "e_deep",
            "E_deep",
            (t, 256),
            source_class="DERIVED",
            formula="X_deep @ W_deep followed by the frozen one-layer Transformer encoder",
            dependencies=("x_deep", "w_deep_projection"),
        ),
        _node(
            "x_mel",
            "X_mel",
            (t, 128),
            source_class="FROZEN_CONSTANT",
            formula="frozen Mel contract after 2Hz bin mean",
            dependencies=("canonical_t",),
        ),
        _node(
            "x_mfcc",
            "X_mfcc",
            (t, 40),
            source_class="FROZEN_CONSTANT",
            formula="frozen MFCC contract after 2Hz bin mean",
            dependencies=("canonical_t",),
        ),
        _node(
            "x_chroma",
            "X_chroma",
            (t, 12),
            source_class="FROZEN_CONSTANT",
            formula="frozen Chroma contract after 2Hz bin mean",
            dependencies=("canonical_t",),
        ),
        _node(
            "e_mel",
            "E_mel",
            (t, 256),
            source_class="FROZEN_CONSTANT",
            formula="Mel projection 128 -> 256 plus one-layer Transformer encoder",
            dependencies=("x_mel",),
        ),
        _node(
            "e_mfcc",
            "E_mfcc",
            (t, 256),
            source_class="FROZEN_CONSTANT",
            formula="MFCC projection 40 -> 256 plus one-layer Transformer encoder",
            dependencies=("x_mfcc",),
        ),
        _node(
            "e_chroma",
            "E_chroma",
            (t, 256),
            source_class="FROZEN_CONSTANT",
            formula="Chroma projection 12 -> 256 plus one-layer Transformer encoder",
            dependencies=("x_chroma",),
        ),
    ]
    for view in ("deep", "mel", "mfcc", "chroma"):
        nodes.append(
            _node(
                f"gamma_{view}",
                f"gamma_{view}",
                (t, 8),
                source_class="FROZEN_CONSTANT",
                formula="eight-state soft clustering over the encoded view",
                dependencies=(f"e_{view}",),
            )
        )
        nodes.append(
            _node(
                f"gamma_tilde_{view}",
                f"gamma_tilde_{view}",
                (t, 8),
                source_class="FROZEN_CONSTANT",
                formula="shared-A Markov posterior recursion",
                dependencies=(f"gamma_{view}", "markov_A", "markov_pi"),
            )
        )
    nodes.extend(
        (
            _node(
                "markov_A",
                "A",
                (8, 8),
                source_class="FROZEN_CONSTANT",
                formula="shared 8-state transition matrix",
                dependencies=("canonical_t",),
                unit="weight_shape",
            ),
            _node(
                "markov_pi",
                "pi",
                (8,),
                source_class="FROZEN_CONSTANT",
                formula="shared 8-state initial distribution",
                dependencies=("canonical_t",),
                unit="weight_shape",
            ),
        )
    )
    for view in ("mel", "mfcc", "chroma"):
        nodes.append(
            _node(
                f"state_score_{view}",
                f"s_{view}",
                (t, t),
                source_class="FROZEN_CONSTANT",
                formula="gamma_tilde_deep @ gamma_tilde_view.T",
                dependencies=("gamma_tilde_deep", f"gamma_tilde_{view}"),
            )
        )
        nodes.append(
            _node(
                f"cross_attention_{view}",
                f"F_deep_from_{view}",
                (t, 256),
                source_class="FROZEN_CONSTANT",
                formula="state-conditioned cross attention with deep Query",
                dependencies=("e_deep", f"e_{view}", f"state_score_{view}"),
            )
        )
    nodes.extend(
        (
            _node(
                "fusion_concat",
                "F_cat",
                (t, 1024),
                source_class="FROZEN_CONSTANT",
                formula="concat [E_deep, F_deep_from_mel, F_deep_from_mfcc, F_deep_from_chroma]",
                dependencies=(
                    "e_deep",
                    "cross_attention_mel",
                    "cross_attention_mfcc",
                    "cross_attention_chroma",
                ),
            ),
            _node(
                "f_fused",
                "F_fused",
                (t, 256),
                source_class="FROZEN_CONSTANT",
                formula="frozen fusion projector 1024 -> 256 + GELU + LayerNorm",
                dependencies=("fusion_concat",),
            ),
            _node(
                "sensor_mode",
                "e_mode",
                (t, 1),
                source_class="FROZEN_CONSTANT",
                formula="mode sensor output",
                dependencies=("e_chroma",),
            ),
            _node(
                "sensor_key",
                "e_key",
                (t, 12),
                source_class="FROZEN_CONSTANT",
                formula="key sensor output",
                dependencies=("e_chroma",),
            ),
            _node(
                "sensor_brightness",
                "e_bright",
                (t, 1),
                source_class="FROZEN_CONSTANT",
                formula="brightness sensor output",
                dependencies=("e_mfcc",),
            ),
            _node(
                "sensor_rms",
                "e_rms",
                (t, 1),
                source_class="FROZEN_CONSTANT",
                formula="RMS sensor output",
                dependencies=("e_mel",),
            ),
            _node(
                "chatts_input",
                "ChatTS input",
                (t, 256),
                source_class="DERIVED",
                formula="F_fused temporal sequence",
                dependencies=("f_fused",),
            ),
            _node(
                "chatts_tokens_7b",
                "T_ts_7B",
                (t, 3584),
                source_class="FROZEN_CONSTANT",
                formula="7B ChatTS Patch Encoder output dimension",
                dependencies=("chatts_input",),
            ),
            _node(
                "chatts_tokens_14b",
                "T_ts_14B",
                (t, 5120),
                source_class="FROZEN_CONSTANT",
                formula="14B ChatTS Patch Encoder output dimension",
                dependencies=("chatts_input",),
            ),
            _node(
                "y_va",
                "Y_va",
                (t, 2),
                source_class="FROZEN_CONSTANT",
                formula="VA head emits one Valence/Arousal pair per canonical time position",
                dependencies=("chatts_tokens_7b", "chatts_tokens_14b"),
            ),
            _node(
                "y_answer",
                "Y_answer",
                (t, 2),
                source_class="FROZEN_CONSTANT",
                formula="ANSWER parser contract emits one VA pair per canonical time position",
                dependencies=("canonical_t",),
            ),
        )
    )

    edges = tuple(
        DimensionEdge(
            source_node_id=node_id,
            target_node_id=dependency,
            relation="DEPENDS_ON",
            formula="target derives from source under the node formula",
            source_class="DERIVED" if node_id in {"timesnet_raw_input", "x_deep", "e_deep", "chatts_input"} else "FROZEN_CONSTANT",
        )
        for node in nodes
        for dependency in node.dependencies
        for node_id in (node.node_id,)
    )
    temporal_dimensions = {
        "mert_raw_output": raw_t,
        "timesnet_raw_input": raw_t,
        "x_deep": t,
        "e_deep": t,
        "x_mel": t,
        "e_mel": t,
        "x_mfcc": t,
        "e_mfcc": t,
        "x_chroma": t,
        "e_chroma": t,
        "fusion_concat": t,
        "f_fused": t,
        "chatts_tokens_7b": t,
        "chatts_tokens_14b": t,
        "y_va": t,
        "y_answer": t,
    }
    payload = {
        "schema_version": "1.0",
        "binding_id": "MERT-DOWNSTREAM-DIMENSIONS-v1",
        "qualification_id": "E1-2-MERT-DOWNSTREAM-DIMENSION-DERIVATION-v1",
        "predecessor_qualification": predecessor_qualification,
        "predecessor_qualification_sha256": predecessor_qualification_sha256,
        "e1_1_evidence_logical_path": "reports/experiments/e1-1-mert-real-frame-rate.json",
        "e1_1_evidence_sha256": e11_evidence_sha256,
        "upstream_manifest_logical_path": "manifests/upstream/mert-v1-95m-primary-v1.json",
        "upstream_manifest_sha256": upstream_manifest_sha256,
        "upstream_semantic_identity_sha256": semantic_identity,
        "upstream_snapshot_aggregate_sha256": snapshot_identity,
        "temporal_geometry": temporal_geometry.model_dump(mode="json"),
        "temporal_geometry_sha256": geometry_sha256(temporal_geometry),
        "observed_t_raw": raw_t,
        "observed_fps": observed_fps,
        "hidden_dimension": hidden_dimension,
        "timesnet_raw_temporal_input_dimension": raw_t,
        "timesnet_period_interpretation": period_interpretation.model_dump(mode="json"),
        "required_temporal_reduction": reduction.model_dump(mode="json"),
        "canonical_downstream_t": t,
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "edges": [edge.model_dump(mode="json") for edge in edges],
        "all_downstream_temporal_dimensions": temporal_dimensions,
        "forbidden_downstream_actions": [
            "MERT_T_RAW_REWRITE",
            "TRIM",
            "PADDING",
            "INTERPOLATION",
            "RESHAPE",
            "ADAPTIVE_POOLING",
        ],
        "derivation_verdict": "PASS",
    }
    return DownstreamDimensionBinding(
        **payload,
        binding_sha256=sha256_canonical(payload),
    )


def verify_downstream_binding(binding: DownstreamDimensionBinding) -> None:
    """Verify the terminal checksum and all E1.2 invariants."""

    payload = binding.model_dump(mode="json")
    observed = payload.pop("binding_sha256")
    if sha256_canonical(payload) != observed:
        raise DimensionBindingError("downstream dimension binding checksum mismatch")
    if binding.observed_t_raw != binding.temporal_geometry.expected_frame_count:
        raise DimensionBindingError("dimension binding geometry/T_raw mismatch")
    if binding.timesnet_raw_temporal_input_dimension != binding.observed_t_raw:
        raise DimensionBindingError("TimesNet raw input rewrote T_raw")
    if binding.canonical_downstream_t != 90:
        raise DimensionBindingError("canonical downstream T is not the frozen 90")
    if binding.required_temporal_reduction.ratio_numerator != binding.observed_t_raw:
        raise DimensionBindingError("MERT-to-2Hz ratio does not use observed T_raw")
    if binding.required_temporal_reduction.ratio_denominator != 90:
        raise DimensionBindingError("MERT-to-2Hz target is not the frozen 90 bins")


def write_immutable_dimension_binding(
    binding: DownstreamDimensionBinding, destination: Path
) -> None:
    """Write a binding once; corrections require a new version/path."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        output.write(canonical_json(binding.model_dump(mode="json")) + "\n")


def load_upstream_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DimensionBindingError("upstream manifest must be a JSON object")
    return payload


def file_sha256(path: Path) -> str:
    return _sha256_file(path)
