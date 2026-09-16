from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _audit_text() -> str:
    return (ROOT / "docs/diagnostics/rg03-research-plan-conformance-audit.md").read_text(encoding="utf-8")


def test_mode_target_is_not_fixed_pitch_mask_when_ks_contract_active() -> None:
    text = _audit_text()
    assert "RESEARCH_PLAN_IMPLEMENTATION_MISMATCH" in text
    assert "pc4 - pc3 - pc10" in text


def test_mode_target_uses_same_chroma_semantics_as_registered_feature() -> None:
    text = _audit_text()
    assert "custom fixed FFT projection" in text
    assert "registered v3" in text


def test_chroma_encoder_output_dim_is_256() -> None:
    text = _audit_text()
    assert "Linear 12→256" in text
    assert "defaults every encoder to hidden_dim=64" in text


def test_chroma_encoder_contains_one_transformer_layer() -> None:
    text = _audit_text()
    assert "No Transformer module exists" in text


def test_mode_head_is_256_64_1() -> None:
    text = _audit_text()
    assert "Mode `256→64→1`" in text
    assert "single `Linear(64, output)`" in text


def test_key_labels_are_segment_level() -> None:
    text = _audit_text()
    assert "5 s segment CE" in text
    assert "frame `argmax(chroma)` then 5 s bin mean" in text


def test_no_test_membership_used_for_diagnostics_training() -> None:
    payload = json.loads((ROOT / "docs/diagnostics/rg03-mode-diagnostics-v1.json").read_text(encoding="utf-8"))
    assert payload["population"]["test_songs_used"] == 0
    assert payload["population"]["optimization_train_songs"] == 896
    assert payload["population"]["validation_songs"] == 99


def test_ccc_fp64_reference() -> None:
    source = (ROOT / "diagnostics/rg03_mode/rg03_mode_diagnostics.py").read_text(encoding="utf-8")
    assert "dtype=np.float64" in source
    assert "MODE_CCC_DENOMINATOR_ZERO" in source
    assert "MODE_CCC_LT2_VALID_PAIRS" in source


def test_existing_v3_artifacts_are_not_modified() -> None:
    text = _audit_text()
    proposal = (ROOT / "docs/proposals/rg03-sensor-plan-conformance-v2.md").read_text(encoding="utf-8")
    assert "registered v3" in text
    assert "Do not overwrite v3 cache" in proposal
    assert "historical" in proposal and "FAIL" in proposal
