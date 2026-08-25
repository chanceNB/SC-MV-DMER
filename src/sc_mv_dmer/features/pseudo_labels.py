"""Deterministic RMS/brightness/mode/key pseudo-label derivation."""

from __future__ import annotations

import numpy as np

from sc_mv_dmer.features.models import PseudoLabelRecord, ndarray_sha256
from sc_mv_dmer.features.resample import WaveformRecord


def derive_pseudo_labels(waveform: WaveformRecord, *, timegrid_binding_sha256: str = "0" * 64, rule_version: str = "deam-pseudo-label-v1", segment_duration_seconds: int = 5) -> PseudoLabelRecord:
    sr = waveform.sample_rate_hz
    hop = 512
    n_fft = 2048
    frame_count = 1 + (len(waveform.waveform) - n_fft) // hop
    frames = np.stack([waveform.waveform[i : i + n_fft] for i in range(0, frame_count * hop, hop)], axis=0)
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32)
    power = spec**2
    rms_raw = np.sqrt(np.mean(frames**2, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bright_raw = (power * freqs[None, :]).sum(axis=1) / np.maximum(power.sum(axis=1), 1e-12)
    chroma = np.zeros((frame_count, 12), dtype=np.float32)
    for idx in range(12):
        lo, hi = 2.0 ** ((idx - 9) / 12) * 440.0, 2.0 ** ((idx - 8) / 12) * 440.0
        chroma[:, idx] = power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
    chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-12)
    centers = np.asarray([i + n_fft // 2 for i in range(0, frame_count * hop, hop)], dtype=float) / sr
    bin_indices = np.floor(centers / 0.5).astype(int)
    targets: dict[str, np.ndarray] = {}
    binned = lambda values: np.stack([values[bin_indices == i].mean(axis=0) for i in range(90)], axis=0).reshape(90, -1).astype(np.float32)
    targets["rms"] = binned(rms_raw)
    targets["brightness"] = binned(bright_raw)
    major = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    minor = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    major_strength = chroma @ major
    minor_strength = chroma @ minor
    targets["mode"] = binned((major_strength - minor_strength)[:, None])
    key_raw = np.eye(12, dtype=np.float32)[np.argmax(chroma, axis=1)]
    targets["key"] = binned(key_raw)
    mask = np.ones(90, dtype=np.bool_)
    payload = {name: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": ndarray_sha256(value)} for name, value in sorted(targets.items())}
    payload["mask"] = {"dtype": str(mask.dtype), "shape": list(mask.shape), "sha256": ndarray_sha256(mask)}
    boundaries = tuple((float(i * segment_duration_seconds), float((i + 1) * segment_duration_seconds)) for i in range(9))
    return PseudoLabelRecord(dataset_id=waveform.dataset_id, song_id=waveform.song_id, sample_id=waveform.sample_id, source_audio_sha256=waveform.source_audio_sha256, rule_version=rule_version, targets=targets, target_dimensions={"rms": 1, "brightness": 1, "mode": 1, "key": 12}, valid_target_mask=mask, segment_boundaries_seconds=boundaries, timegrid_binding_sha256=timegrid_binding_sha256, content_sha256=__import__("sc_mv_dmer.foundation.canonical", fromlist=["sha256_canonical"]).sha256_canonical(payload))
