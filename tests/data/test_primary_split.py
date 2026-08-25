import pytest

from sc_mv_dmer.data.splits import FrozenSplitManifestUnavailable, load_frozen_split


def test_missing_approved_split_fails_closed(tmp_path) -> None:
    with pytest.raises(FrozenSplitManifestUnavailable, match="FROZEN_SPLIT_MANIFEST_UNAVAILABLE"):
        load_frozen_split(tmp_path / "missing-split.json")
