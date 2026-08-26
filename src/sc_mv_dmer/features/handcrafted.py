"""Shared 22.05 kHz Mel, MFCC and deterministic CQT-like Chroma extraction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sc_mv_dmer.features.models import FeatureFrameRecord, ndarray_sha256
from sc_mv_dmer.features.resample import WaveformRecord


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def _mel_to_hz(value: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_points = _mel_to_hz(np.linspace(float(_hz_to_mel(0.0)), float(_hz_to_mel(sr / 2)), n_mels + 2))
    bins = np.floor((n_fft + 1) * mel_points / sr).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for idx in range(n_mels):
        left, center, right = int(bins[idx]), int(bins[idx + 1]), int(bins[idx + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for point in range(max(0, left), min(center, bank.shape[1])):
            bank[idx, point] = (point - left) / max(center - left, 1)
        for point in range(max(0, center), min(right, bank.shape[1])):
            bank[idx, point] = (right - point) / max(right - center, 1)
    return bank


def _dct_type_ii(values: np.ndarray, coefficients: int) -> np.ndarray:
    n = values.shape[-1]
    k = np.arange(coefficients, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    basis = np.cos(np.pi / n * (i + 0.5) * k)
    basis[0] *= 1.0 / math.sqrt(n)
    if coefficients > 1:
        basis[1:] *= math.sqrt(2.0 / n)
    return values @ basis.T


def _cqt_like(power: np.ndarray, freqs: np.ndarray, sr: int, bins_per_octave: int = 36) -> np.ndarray:
    # Fixed frequency-domain projection: no learned/adaptive frontend and no librosa dependency.
    low = 32.70319566257483
    centers = low * (2.0 ** (np.arange(8 * bins_per_octave) / bins_per_octave))
    out = np.zeros((power.shape[0], centers.shape[0]), dtype=np.float32)
    for i, center in enumerate(centers):
        width = center * (2.0 ** (0.5 / bins_per_octave) - 2.0 ** (-0.5 / bins_per_octave))
        weights = np.maximum(0.0, 1.0 - np.abs(freqs - center) / max(width, 1e-6))
        total = weights.sum()
        if total > 0:
            out[:, i] = (power * weights[None, :]).sum(axis=1) / total
    bins_per_pitch = bins_per_octave // 12
    chroma = np.zeros((power.shape[0], 12), dtype=np.float32)
    for octave in range(8):
        octave_start = octave * bins_per_octave
        for pitch_class in range(12):
            start = octave_start + pitch_class * bins_per_pitch
            chroma[:, pitch_class] += out[:, start : start + bins_per_pitch].mean(axis=1)
    norms = np.sum(np.abs(chroma), axis=1, keepdims=True)
    return np.divide(chroma, norms, out=np.zeros_like(chroma), where=norms > 1e-12)


def extract_handcrafted(waveform: WaveformRecord, *, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128, n_mfcc: int = 40, include_c0: bool = True) -> dict[str, FeatureFrameRecord]:
    if waveform.sample_rate_hz != 22050:
        raise ValueError("handcrafted views require the shared 22.05 kHz waveform")
    values = waveform.waveform
    frame_count = 1 + (len(values) - n_fft) // hop_length
    if frame_count <= 0:
        raise ValueError("waveform is shorter than one STFT frame")
    frames = np.stack([values[start : start + n_fft] for start in range(0, frame_count * hop_length, hop_length)], axis=0)
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / waveform.sample_rate_hz)
    mel_power = power @ _mel_filterbank(waveform.sample_rate_hz, n_fft, n_mels).T
    mel_db = (10.0 * np.log10(np.maximum(mel_power, 1e-10))).astype(np.float32)
    mfcc = _dct_type_ii(np.log(np.maximum(mel_power, 1e-10)), n_mfcc).astype(np.float32)
    chroma = _cqt_like(power, freqs, waveform.sample_rate_hz)
    centers = tuple(2 * (start + n_fft // 2) for start in range(0, frame_count * hop_length, hop_length))
    common: dict[str, Any] = {
        "dataset_id": waveform.dataset_id,
        "song_id": waveform.song_id,
        "sample_id": waveform.sample_id,
        "source_audio_sha256": waveform.source_audio_sha256,
        "frame_center_sample_numerators": centers,
        "frame_center_sample_denominator": 2,
        "sample_rate_hz": waveform.sample_rate_hz,
        "support_length_samples": n_fft,
        "hop_length_samples": hop_length,
        "waveform_sha256": waveform.waveform_sha256,
    }
    output: dict[str, FeatureFrameRecord] = {}
    for view, array, rule in (("mel", mel_db, "mel-power-to-db-v1"), ("mfcc", mfcc, f"mfcc-dct2-ortho-c0-{str(include_c0).lower()}-v1"), ("chroma", chroma, "cqt36-chroma-l1-v1")):
        array = np.ascontiguousarray(array, dtype=np.float32)
        output[view] = FeatureFrameRecord(**common, view=view, rule_version=rule, values=array, feature_dim=array.shape[1], dtype=str(array.dtype), finite=bool(np.isfinite(array).all()), content_sha256=ndarray_sha256(array))
    return output
