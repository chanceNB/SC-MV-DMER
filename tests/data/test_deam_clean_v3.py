import hashlib
import json
import os
from pathlib import Path

import pytest

from sc_mv_dmer.foundation.canonical import sha256_canonical


def test_v3_snapshot_has_995_train_and_58_test_without_mutating_v2():
    repo = Path(__file__).parents[2]
    historical_root_value = os.environ.get("SC_MV_DMER_HISTORICAL_ARTIFACT_ROOT")
    if not historical_root_value:
        pytest.fail("MISSING_IMMUTABLE_HISTORICAL_ARTIFACT: set SC_MV_DMER_HISTORICAL_ARTIFACT_ROOT")
    historical_root = Path(historical_root_value)
    v2_path = historical_root / "manifests/datasets/deam-pdmer-clean-v2.json"
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3_path = repo / "manifests/datasets/deam-pdmer-clean-v3.json"
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    assert v2["manifest_id"] == "DEAM-PDMER-CLEAN-v2"
    assert v3["manifest_id"] == "DEAM-PDMER-CLEAN-v3"
    assert v3["supersedes_manifest_id"] == v2["manifest_id"]
    assert v3["supersedes_manifest_sha256"] == v2["manifest_sha256"]
    assert v2["manifest_id"] == "DEAM-PDMER-CLEAN-v2"
    assert v3["population"]["retained_train_primary_count"] == 995
    assert v3["population"]["retained_test_long_song_count"] == 58
    assert v3["split_policy"]["validation"] == "UNDEFINED_PENDING_SEPARATE_FREEZE"
    assert v3["audio_policy"]["excluded_short_primary_logical_song_keys"] == ["1174", "1200", "1273", "1493", "1789"]
    assert len(v3["records"]) == 1802
    assert sha256_canonical({k: v for k, v in v3.items() if k != "manifest_sha256"}) == v3["manifest_sha256"]


def test_v3_preflight_and_training_contract_are_non_training_artifacts():
    repo = Path(__file__).parents[2]
    evidence = json.loads((repo / "evidence/gates/rg-02/deam-pdmer-clean-v3-preflight-v1.json").read_text(encoding="utf-8"))
    budget = json.loads((repo / "reports/data/deam-pdmer-clean-v3-training-contract-preflight.json").read_text(encoding="utf-8"))
    assert evidence["verdict"] == "PASS"
    assert evidence["scope"] == "DATA_PREFLIGHT_ONLY"
    assert evidence["checks"]["feature_cache_binding"]["published"] is False
    assert evidence["checks"]["pseudo_label_cache_binding"]["published"] is False
    assert budget["training_authorized"] is False
    assert budget["checkpoint_published"] is False
    assert budget["resolved"]["budget"]["samples_per_epoch"] == 995
