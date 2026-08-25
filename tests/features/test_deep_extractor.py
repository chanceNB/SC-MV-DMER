import hashlib

import numpy as np

from sc_mv_dmer.features.deep import extract_deep
from sc_mv_dmer.features.resample import WaveformRecord


def test_deep_extractor_uses_registered_layers_and_90_bins():
    wave = np.zeros(1_080_000, dtype=np.float32)
    record = WaveformRecord(
        dataset_id="d", song_id="s", sample_id="x", source_audio_sha256="a" * 64,
        source_relative_path="audio/x.mp3", waveform=wave, sample_rate_hz=24000,
        original_sample_rate_hz=24000, waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(), duration_seconds=45,
    )
    layers = {"hidden_states": [np.zeros((1, 3374, 768), dtype=np.float32) for _ in range(7)]}
    out = extract_deep(record, layers, "b" * 64)
    assert out.values.shape == (90, 768)
    assert out.raw_eligible_count == 3374
    assert out.timegrid_binding_sha256 == "b" * 64
