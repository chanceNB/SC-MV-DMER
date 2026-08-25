import hashlib

import numpy as np

from sc_mv_dmer.features.handcrafted import extract_handcrafted
from sc_mv_dmer.features.resample import WaveformRecord


def _waveform():
    wave = np.sin(np.linspace(0, 1000, 22050 * 45, endpoint=False)).astype(np.float32)
    return WaveformRecord(
        dataset_id="dataset", song_id="song", sample_id="sample", source_audio_sha256="a" * 64,
        source_relative_path="audio/song.mp3", waveform=wave, sample_rate_hz=22050,
        original_sample_rate_hz=22050,
        waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(), duration_seconds=45,
    )


def test_handcrafted_shapes_and_shared_timing():
    out = extract_handcrafted(_waveform())
    assert out["mel"].values.shape[1] == 128
    assert out["mfcc"].values.shape[1] == 40
    assert out["chroma"].values.shape[1] == 12
    assert out["mel"].waveform_sha256 == out["mfcc"].waveform_sha256 == out["chroma"].waveform_sha256
    assert out["mel"].hop_length_samples == out["mfcc"].hop_length_samples == 512
    assert out["chroma"].values.shape[0] == out["mel"].values.shape[0]
    nonzero = np.sum(np.abs(out["chroma"].values), axis=1)
    assert np.all((nonzero <= 1.0001) | (nonzero < 1e-8))
