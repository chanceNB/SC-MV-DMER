import hashlib

import numpy as np

from sc_mv_dmer.features.pseudo_labels import derive_pseudo_labels
from sc_mv_dmer.features.resample import WaveformRecord


def test_pseudo_labels_have_rule_and_segment_identity():
    wave = np.sin(np.linspace(0, 500, 22050 * 45, endpoint=False)).astype(np.float32)
    record = WaveformRecord(
        dataset_id="d", song_id="s", sample_id="x", source_audio_sha256="a" * 64,
        source_relative_path="audio/x.mp3", waveform=wave, sample_rate_hz=22050,
        original_sample_rate_hz=22050, waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(), duration_seconds=45,
    )
    labels = derive_pseudo_labels(record, timegrid_binding_sha256="b" * 64)
    assert set(labels.targets) == {"rms", "brightness", "mode", "key"}
    assert labels.targets["rms"].shape == (90, 1)
    assert labels.targets["key"].shape == (90, 12)
    assert len(labels.segment_boundaries_seconds) == 9
