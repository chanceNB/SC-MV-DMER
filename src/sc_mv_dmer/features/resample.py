"""Deterministic source decoding and resampling for Step 2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field
from scipy.signal import resample_poly


class WaveformRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    dataset_id: str = "unbound"
    song_id: str
    sample_id: str
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_relative_path: str
    waveform: np.ndarray
    sample_rate_hz: int
    original_sample_rate_hz: int
    waveform_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    channel_policy: Literal["MEAN_CHANNELS"] = "MEAN_CHANNELS"
    duration_seconds: float


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wave_hash(waveform: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(waveform, dtype=np.float32).tobytes()).hexdigest()


def load_waveform(
    path: Path,
    *,
    song_id: str,
    sample_id: str,
    source_relative_path: str,
    target_sample_rate_hz: int,
    source_audio_sha256: str | None = None,
    duration_seconds: int = 45,
) -> WaveformRecord:
    path = Path(path)
    source_hash = source_audio_sha256 or _file_sha256(path)
    samples, source_rate = sf.read(path, always_2d=True, dtype="float32")
    required = int(source_rate * duration_seconds)
    if samples.shape[0] < required:
        raise ValueError(f"source is shorter than required {duration_seconds}s: {path}")
    mono = samples[:required].mean(axis=1, dtype=np.float32)
    if source_rate != target_sample_rate_hz:
        import math

        divisor = math.gcd(source_rate, target_sample_rate_hz)
        up = target_sample_rate_hz // divisor
        down = source_rate // divisor
        mono = resample_poly(mono, up, down).astype(np.float32, copy=False)
    target_count = target_sample_rate_hz * duration_seconds
    if mono.shape[0] < target_count:
        raise ValueError("resampler produced fewer than exact duration samples")
    mono = np.ascontiguousarray(mono[:target_count], dtype=np.float32)
    return WaveformRecord(
        dataset_id="unbound",
        song_id=song_id,
        sample_id=sample_id,
        source_audio_sha256=source_hash,
        source_relative_path=source_relative_path,
        waveform=mono,
        sample_rate_hz=target_sample_rate_hz,
        original_sample_rate_hz=source_rate,
        waveform_sha256=_wave_hash(mono),
        duration_seconds=float(duration_seconds),
    )
