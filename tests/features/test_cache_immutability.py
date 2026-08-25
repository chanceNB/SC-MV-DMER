import numpy as np
import pytest

from sc_mv_dmer.features.cache import CacheSpec, ImmutableCacheError, write_cache
from sc_mv_dmer.features.models import BinnedFeatureRecord, ndarray_sha256


def _spec():
    return CacheSpec(
        cache_id="fixture", version="v1", logical_root="artifacts/features/fixture",
        dataset_manifest_logical_path="manifests/datasets/x.json", dataset_manifest_sha256="a" * 64,
        split_manifest_logical_path="manifests/splits/x.json", split_manifest_sha256="b" * 64,
        dimensions_logical_path="manifests/dimensions/x.json", dimensions_sha256="c" * 64,
        timegrid_logical_path="manifests/timegrid/x.json", timegrid_sha256="d" * 64,
        upstream_manifest_logical_path="manifests/upstream/x.json", upstream_manifest_sha256="e" * 64,
        rg01_evidence_logical_path="evidence/gates/rg-01/x.json", rg01_evidence_sha256="f" * 64,
    )


def _record(value=1.0):
    values = np.full((90, 2), value, dtype=np.float32)
    feature = BinnedFeatureRecord(
        dataset_id="d", song_id="s", sample_id="x", source_audio_sha256="a" * 64,
        view="mel", values=values, feature_dim=2, bin_counts=(1,) * 90,
        raw_eligible_count=90, unassigned_count=0, duplicate_count=0,
        timegrid_binding_sha256="b" * 64, dimension_binding_sha256="c" * 64,
        finite=True, content_sha256=ndarray_sha256(values),
    )
    return {"dataset_id": "d", "song_id": "s", "sample_id": "x", "source_audio_sha256": "a" * 64,
            "views": {"deep": feature.model_copy(update={"view": "deep", "feature_dim": 2}),
                      "mel": feature, "mfcc": feature.model_copy(update={"view": "mfcc"}),
                      "chroma": feature.model_copy(update={"view": "chroma"})}}


def test_finalized_cache_version_cannot_change_content(tmp_path):
    path = tmp_path / "manifests" / "features" / "cache.json"
    write_cache([_record()], _spec(), path)
    with pytest.raises(ImmutableCacheError):
        write_cache([_record(2.0)], _spec(), path)
