import numpy as np

from sc_mv_dmer.features.binning import bin_feature
from sc_mv_dmer.features.models import FeatureFrameRecord, ndarray_sha256


def test_every_eligible_frame_assigned_once():
    values = np.arange(1935 * 2, dtype=np.float32).reshape(1935, 2)
    centers = tuple(2 * (1024 + 512 * i) for i in range(1935))
    record = FeatureFrameRecord(
        dataset_id="d", song_id="s", sample_id="x", source_audio_sha256="a" * 64,
        view="mel", rule_version="v", values=values, frame_center_sample_numerators=centers,
        frame_center_sample_denominator=2, sample_rate_hz=22050, support_length_samples=2048,
        hop_length_samples=512, feature_dim=2, dtype="float32", finite=True,
        content_sha256=ndarray_sha256(values), waveform_sha256="b" * 64,
    )
    result = bin_feature(record, timegrid_binding_sha256="c" * 64)
    assert result.values.shape == (90, 2)
    assert result.unassigned_count == result.duplicate_count == 0
    assert sum(result.bin_counts) == 1935
    assert min(result.bin_counts) > 0
