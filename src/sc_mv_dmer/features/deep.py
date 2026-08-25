"""RG-01-bound pinned MERT Deep feature extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from sc_mv_dmer.features.models import BinnedFeatureRecord, ndarray_sha256
from sc_mv_dmer.features.resample import WaveformRecord


def _binding_hash(binding: Any) -> str:
    if isinstance(binding, str):
        return binding
    for name in ("binding_sha256", "dimension_binding_sha256"):
        value = getattr(binding, name, None)
        if isinstance(value, str):
            return value
    return "0" * 64


def _hidden_layers(bundle: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(bundle, Mapping):
        hidden = bundle.get("hidden_states")
        if hidden is None:
            hidden = bundle.get("layers")
        if isinstance(hidden, Mapping):
            return np.asarray(hidden[5]), np.asarray(hidden[6])
        if hidden is not None and len(hidden) > 6:
            return np.asarray(hidden[5]), np.asarray(hidden[6])
    hidden = getattr(bundle, "hidden_states", None)
    if hidden is not None and len(hidden) > 6:
        return np.asarray(hidden[5]), np.asarray(hidden[6])
    model = getattr(bundle, "model", bundle)
    processor = getattr(bundle, "processor", None)
    waveform = getattr(bundle, "waveform", None)
    if processor is None or waveform is None:
        raise ValueError("MERT bundle must provide hidden_states or processor/model/waveform")
    import torch

    inputs = processor(waveform, sampling_rate=24000, return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True)
    states = output.hidden_states
    return states[5].detach().cpu().numpy(), states[6].detach().cpu().numpy()


def extract_deep(waveform: WaveformRecord, bundle: Any, binding: Any) -> BinnedFeatureRecord:
    if waveform.sample_rate_hz != 24000 or waveform.waveform.shape[0] != 1_080_000:
        raise ValueError("Deep view requires exact 24 kHz, 45 second waveform")
    layer5, layer6 = _hidden_layers(bundle)
    layer5 = np.asarray(layer5)
    layer6 = np.asarray(layer6)
    if layer5.ndim == 3:
        layer5 = layer5[0]
    if layer6.ndim == 3:
        layer6 = layer6[0]
    if layer5.shape != layer6.shape or layer5.ndim != 2 or layer5.shape[1] != 768:
        raise ValueError("registered MERT layers 5/6 must both be [3374, 768]")
    if layer5.shape[0] != 3374:
        raise ValueError("MERT Deep extraction must preserve measured T_raw=3374")
    values_raw = ((layer5.astype(np.float32) + layer6.astype(np.float32)) / 2.0).astype(np.float32)
    frame_index = np.arange(3374, dtype=np.int64)
    centers = (399 + 640 * frame_index) / 2.0
    bin_index = np.floor(centers / (24000 * 0.5)).astype(int)
    counts = tuple(int(np.sum(bin_index == i)) for i in range(90))
    if min(counts) <= 0 or bin_index.min() < 0 or bin_index.max() >= 90:
        raise ValueError("Deep frames do not cover all RG-01 bins exactly")
    values = np.stack([values_raw[bin_index == i].mean(axis=0) for i in range(90)], axis=0)
    return BinnedFeatureRecord(
        dataset_id="unbound",
        song_id=waveform.song_id,
        sample_id=waveform.sample_id,
        source_audio_sha256=waveform.source_audio_sha256,
        view="deep",
        values=values,
        feature_dim=768,
        bin_counts=counts,
        raw_eligible_count=3374,
        unassigned_count=0,
        duplicate_count=0,
        timegrid_binding_sha256=_binding_hash(binding),
        dimension_binding_sha256=_binding_hash(binding),
        finite=True,
        content_sha256=ndarray_sha256(values),
    )
