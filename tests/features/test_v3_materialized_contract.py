import hashlib
import json
from pathlib import Path

import numpy as np

from sc_mv_dmer.foundation.canonical import sha256_canonical


def test_v3_feature_and_pseudo_caches_bind_full_population_without_validation(tmp_path):
    repo = Path(__file__).parents[2]
    feature_path = repo / "manifests/features/deam-four-view-cache-v3.json"
    pseudo_path = repo / "manifests/features/deam-pseudo-label-bundle-v3.json"
    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
    assert feature["dataset_manifest_logical_path"] == "manifests/datasets/deam-pdmer-clean-v3.json"
    assert feature["dataset_manifest_sha256"] == pseudo["dataset_manifest_sha256"]
    assert feature["record_count"] == pseudo["record_count"] == 1053
    assert feature["population_counts"] == pseudo["population_counts"] == {"TRAIN_PRIMARY": 995, "TEST_LONG_SONG": 58}
    assert feature["validation_policy"] == pseudo["validation_policy"] == "UNDEFINED_PENDING_SEPARATE_FREEZE"
    assert sha256_canonical({k: v for k, v in feature.items() if k != "manifest_sha256"}) == feature["manifest_sha256"]
    assert sha256_canonical({k: v for k, v in pseudo.items() if k != "manifest_sha256"}) == pseudo["manifest_sha256"]

    roles = {row["population_role"] for row in feature["records"]}
    assert roles == {"TRAIN_PRIMARY", "TEST_LONG_SONG"}
    for row in feature["records"][:3]:
        assert row["views"]["deep"]["shape"] == [90, 768]
        assert row["views"]["mel"]["shape"] == [90, 128]
        assert row["views"]["mfcc"]["shape"] == [90, 40]
        assert row["views"]["chroma"]["shape"] == [90, 12]
        assert all(sum(row["views"][name]["bin_counts"]) == row["views"][name]["raw_eligible_count"] for name in ("deep", "mel", "mfcc", "chroma"))

    pseudo_roles = {row["population_role"] for row in pseudo["records"]}
    assert pseudo_roles == roles
    for row in pseudo["records"][:3]:
        payload = repo / row["payload_logical_path"]
        assert payload.is_file()
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == row["payload_sha256"]
        with np.load(payload) as arrays:
            assert arrays["rms"].shape == (90, 1)
            assert arrays["brightness"].shape == (90, 1)
            assert arrays["mode"].shape == (90, 1)
            assert arrays["key"].shape == (90, 12)
            assert arrays["valid_target_mask"].shape == (90,)
