import numpy as np

from sc_mv_dmer.features.cache import CacheSpec, write_cache
from sc_mv_dmer.features.models import BinnedFeatureRecord, ndarray_sha256
from sc_mv_dmer.gates.rg02 import evaluate_rg02_automatic


def test_one_bad_record_fails_automatic_rg02(tmp_path):
    values = np.zeros((90, 2), dtype=np.float32)
    rec = BinnedFeatureRecord(dataset_id="d", song_id="song-17", sample_id="sample-17", source_audio_sha256="a" * 64, view="mel", values=values, feature_dim=2, bin_counts=(1,) * 90, raw_eligible_count=90, unassigned_count=0, duplicate_count=0, timegrid_binding_sha256="b" * 64, dimension_binding_sha256="c" * 64, finite=True, content_sha256=ndarray_sha256(values))
    views = {name: rec.model_copy(update={"view": name, "feature_dim": 2 if name == "mel" else 2}) for name in ("deep", "mel", "mfcc", "chroma")}
    spec = CacheSpec(cache_id="fixture", version="v1", logical_root="artifacts/features/fixture", dataset_manifest_logical_path="d", dataset_manifest_sha256="a"*64, split_manifest_logical_path="s", split_manifest_sha256="b"*64, dimensions_logical_path="x", dimensions_sha256="c"*64, timegrid_logical_path="t", timegrid_sha256="d"*64, upstream_manifest_logical_path="u", upstream_manifest_sha256="e"*64, rg01_evidence_logical_path="r", rg01_evidence_sha256="f"*64)
    path = tmp_path / "manifests" / "features" / "cache.json"
    manifest = write_cache([{"dataset_id":"d", "song_id":"song-17", "sample_id":"sample-17", "source_audio_sha256":"a"*64, "views":views}], spec, path)
    result = evaluate_rg02_automatic(manifest, expected_count=1)
    assert result["verdict"] == "PASS"
    result = evaluate_rg02_automatic(manifest, expected_count=1744)
    assert result["verdict"] == "FAIL"
