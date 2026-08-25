import pytest
from pathlib import Path

from sc_mv_dmer.data.manifests import DatasetManifest
from sc_mv_dmer.data.splits import (
    FrozenSplitManifestUnavailable,
    freeze_primary_split,
    load_frozen_split,
    verify_frozen_split,
)


def test_missing_approved_split_fails_closed(tmp_path) -> None:
    with pytest.raises(FrozenSplitManifestUnavailable, match="FROZEN_SPLIT_MANIFEST_UNAVAILABLE"):
        load_frozen_split(tmp_path / "missing-split.json")


def test_first_freeze_is_hash_ranked_and_has_fixed_quota() -> None:
    root = Path(__file__).parents[2]
    manifest_path = root / "manifests/datasets/deam-primary-v1.json"
    dataset = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    split = freeze_primary_split(
        dataset,
        dataset_manifest_logical_path="manifests/datasets/deam-primary-v1.json",
        dataset_manifest_file_sha256="a" * 64,
    )
    result = verify_frozen_split(dataset, split, dataset_manifest_file_sha256="a" * 64)
    assert result["verdict"] == "PASS"
    assert (split["train_count"], split["validation_count"], split["test_count"]) == (1395, 174, 175)
    assert split["records"][0]["rank_index"] == 0
