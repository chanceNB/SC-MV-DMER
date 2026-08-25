"""Exact RG-01 half-open sample-domain binning."""

from __future__ import annotations

import numpy as np

from sc_mv_dmer.features.mert_timegrid import TimeGridBindingError
from sc_mv_dmer.features.models import BinnedFeatureRecord, FeatureFrameRecord, ndarray_sha256


def bin_feature(record: FeatureFrameRecord, grid=None, *, timegrid_binding_sha256: str | None = None, bin_count: int = 90, duration_seconds: int = 45) -> BinnedFeatureRecord:
    if bin_count != 90 or duration_seconds != 45:
        raise ValueError("RG-01 feature binning is fixed at 90 half-second bins over 45 seconds")
    if timegrid_binding_sha256 is None:
        timegrid_binding_sha256 = getattr(grid, "binding_sha256", None) or getattr(grid, "timegrid_binding_sha256", None) or "0" * 64
    centers = np.asarray(record.frame_center_sample_numerators, dtype=np.int64)
    # numerator/denominator timestamp belongs to [i*sr/2, (i+1)*sr/2).
    bins = np.floor((centers / record.frame_center_sample_denominator) / (record.sample_rate_hz * 0.5)).astype(int)
    valid = (bins >= 0) & (bins < bin_count)
    if not bool(valid.all()):
        raise TimeGridBindingError("raw feature frame center falls outside RG-01 45s grid")
    counts = tuple(int(np.sum(bins == index)) for index in range(bin_count))
    if min(counts) <= 0:
        raise TimeGridBindingError("RG-01 bin has no assigned raw feature frame")
    values = np.stack([record.values[bins == index].mean(axis=0) for index in range(bin_count)], axis=0).astype(np.float32)
    return BinnedFeatureRecord(
        dataset_id=record.dataset_id,
        song_id=record.song_id,
        sample_id=record.sample_id,
        source_audio_sha256=record.source_audio_sha256,
        view=record.view,
        values=values,
        feature_dim=values.shape[1],
        bin_counts=counts,
        raw_eligible_count=int(valid.sum()),
        unassigned_count=0,
        duplicate_count=0,
        timegrid_binding_sha256=timegrid_binding_sha256,
        dimension_binding_sha256="0" * 64,
        finite=True,
        content_sha256=ndarray_sha256(values),
    )
