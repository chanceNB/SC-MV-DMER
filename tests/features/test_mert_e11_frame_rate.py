from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sc_mv_dmer.features.mert_frame_rate import (
    E11VerificationError,
    build_probe_observation,
    require_structural_consistency,
    resolve_research_layer_mapping,
    validate_pinned_manifest_file,
    validate_preprocessed_contract,
    validate_registered_source,
)
from sc_mv_dmer.features.mert_e11_execution import (
    map_block_outputs_to_returned_hidden_states,
)


def test_wrong_upstream_manifest_fails_before_execution(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(E11VerificationError, match="manifest SHA-256 mismatch"):
        validate_pinned_manifest_file(
            manifest,
            expected_sha256="0" * 64,
        )


def test_wrong_registered_source_hash_fails(tmp_path: Path) -> None:
    source = tmp_path / "DEAM_audio" / "MEMD_audio" / "2.mp3"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"tampered")

    with pytest.raises(E11VerificationError, match="source audio SHA-256 mismatch"):
        validate_registered_source(
            data_root=tmp_path,
            source_relative_path="DEAM_audio/MEMD_audio/2.mp3",
            expected_sha256=hashlib.sha256(b"registered").hexdigest(),
        )


def test_wrong_preprocessed_sample_count_fails() -> None:
    with pytest.raises(E11VerificationError, match="1,080,000"):
        validate_preprocessed_contract(
            sample_rate_hz=24_000,
            sample_count=1_079_999,
            duration_seconds=45.0,
        )


def test_ambiguous_research_layer_mapping_fails() -> None:
    with pytest.raises(E11VerificationError, match="ambiguous"):
        resolve_research_layer_mapping(
            hidden_states_count=13,
            block_to_hidden_state_candidates={5: (5, 6), 6: (6,)},
        )


def test_layer_temporal_shape_mismatch_fails() -> None:
    mapping = resolve_research_layer_mapping(
        hidden_states_count=13,
        block_to_hidden_state_candidates={5: (5,), 6: (6,)},
    )

    with pytest.raises(E11VerificationError, match="temporal dimensions differ"):
        build_probe_observation(
            probe_id="P1",
            probe_type="P1_CANONICAL",
            input_sample_rate=24_000,
            input_sample_count=1_080_000,
            input_tensor_shape=(1, 1_080_000),
            hidden_states_count=13,
            layer_mapping=mapping,
            layer5_shape=(1, 2249, 768),
            layer6_shape=(1, 2248, 768),
            layer5_finite=True,
            layer6_finite=True,
        )


def test_nonfinite_hidden_state_fails() -> None:
    mapping = resolve_research_layer_mapping(
        hidden_states_count=13,
        block_to_hidden_state_candidates={5: (5,), 6: (6,)},
    )

    with pytest.raises(E11VerificationError, match="non-finite"):
        build_probe_observation(
            probe_id="P1",
            probe_type="P1_CANONICAL",
            input_sample_rate=24_000,
            input_sample_count=1_080_000,
            input_tensor_shape=(1, 1_080_000),
            hidden_states_count=13,
            layer_mapping=mapping,
            layer5_shape=(1, 2249, 768),
            layer6_shape=(1, 2249, 768),
            layer5_finite=False,
            layer6_finite=True,
        )


def test_observed_fps_is_derived_from_observed_frames_not_assumed() -> None:
    mapping = resolve_research_layer_mapping(
        hidden_states_count=13,
        block_to_hidden_state_candidates={5: (5,), 6: (6,)},
    )
    observation = build_probe_observation(
        probe_id="P1",
        probe_type="P1_CANONICAL",
        input_sample_rate=24_000,
        input_sample_count=1_080_000,
        input_tensor_shape=(1, 1_080_000),
        hidden_states_count=13,
        layer_mapping=mapping,
        layer5_shape=(1, 2249, 768),
        layer6_shape=(1, 2249, 768),
        layer5_finite=True,
        layer6_finite=True,
    )

    assert observation.observed_frame_count == 2249
    assert observation.observed_fps == 2249 / 45.0


def test_p1_p2_frame_count_mismatch_fails() -> None:
    mapping = resolve_research_layer_mapping(
        hidden_states_count=13,
        block_to_hidden_state_candidates={5: (5,), 6: (6,)},
    )
    common = {
        "input_sample_rate": 24_000,
        "input_sample_count": 1_080_000,
        "input_tensor_shape": (1, 1_080_000),
        "hidden_states_count": 13,
        "layer_mapping": mapping,
        "layer5_finite": True,
        "layer6_finite": True,
    }
    p1 = build_probe_observation(
        probe_id="P1",
        probe_type="P1_CANONICAL",
        layer5_shape=(1, 2249, 768),
        layer6_shape=(1, 2249, 768),
        **common,
    )
    p2 = build_probe_observation(
        probe_id="P2",
        probe_type="P2_REAL_DEAM",
        layer5_shape=(1, 2248, 768),
        layer6_shape=(1, 2248, 768),
        **common,
    )

    with pytest.raises(E11VerificationError, match="P1/P2 frame counts differ"):
        require_structural_consistency(p1, p2)


def test_runtime_tensor_identity_maps_blocks_without_assumed_index() -> None:
    encoder_input = object()
    block_1 = object()
    block_5 = object()
    block_6 = object()
    block_12 = object()

    candidates = map_block_outputs_to_returned_hidden_states(
        returned_hidden_states=(encoder_input, block_1, block_5, block_6, block_12),
        captured_block_outputs={5: block_5, 6: block_6},
    )

    assert candidates == {5: (2,), 6: (3,)}


def test_runtime_tensor_identity_missing_block_fails() -> None:
    with pytest.raises(E11VerificationError, match="not present"):
        map_block_outputs_to_returned_hidden_states(
            returned_hidden_states=(object(), object()),
            captured_block_outputs={5: object(), 6: object()},
        )
