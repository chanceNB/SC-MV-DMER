import numpy as np
import pytest
from pydantic import ValidationError

from sc_mv_dmer.features.models import FeatureFrameRecord, ndarray_sha256


def test_feature_record_requires_dimension_dependency_hash() -> None:
    with pytest.raises(ValidationError):
        FeatureFrameRecord(
            dataset_id="d", song_id="s", sample_id="x", source_audio_sha256="a" * 64,
            view="mel", rule_version="v", values=np.zeros((10, 128), dtype=np.float32),
            frame_center_sample_numerators=tuple(range(10)), frame_center_sample_denominator=1,
            sample_rate_hz=22050, support_length_samples=2048, hop_length_samples=512,
            feature_dim=128, dtype="float32", finite=True,
            content_sha256="0" * 64, waveform_sha256="b" * 64,
        )


def test_feature_record_validates_content_hash() -> None:
    values = np.zeros((2, 3), dtype=np.float32)
    record = FeatureFrameRecord(
        dataset_id="d", song_id="s", sample_id="x", source_audio_sha256="a" * 64,
        view="mel", rule_version="v", values=values,
        frame_center_sample_numerators=(0, 1), frame_center_sample_denominator=1,
        sample_rate_hz=22050, support_length_samples=2, hop_length_samples=1,
        feature_dim=3, dtype="float32", finite=True, content_sha256=ndarray_sha256(values), waveform_sha256="b" * 64,
    )
    assert record.values.shape == (2, 3)
