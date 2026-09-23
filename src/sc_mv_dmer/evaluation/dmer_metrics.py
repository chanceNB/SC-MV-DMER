"""Dimension-masked, equal-song DMER metrics with explicit degeneracy handling."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


DIMENSIONS = ("valence", "arousal")
METRICS = ("ccc", "pcc", "rmse")


def _metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[dict, list[str]]:
    if len(target) == 0:
        return dict.fromkeys(METRICS), ["zero_valid_points"]
    constant_p = bool(np.all(prediction == prediction[0]))
    constant_y = bool(np.all(target == target[0]))
    # Using the repeated value avoids fictitious variance from mean roundoff.
    p_mean = prediction[0] if constant_p else prediction.mean()
    y_mean = target[0] if constant_y else target.mean()
    centered_p, centered_y = prediction - p_mean, target - y_mean
    p_var, y_var = np.mean(centered_p ** 2), np.mean(centered_y ** 2)
    covariance = np.mean(centered_p * centered_y)
    denominator = p_var + y_var + (p_mean - y_mean) ** 2
    diagnostics = []
    if p_var == 0: diagnostics.append("constant_prediction")
    if y_var == 0: diagnostics.append("constant_target")
    if len(target) == 1: diagnostics.append("single_valid_point")
    # Identical constant sequences are a perfect match by explicit convention.
    ccc = 1.0 if denominator == 0 else float(2 * covariance / denominator)
    pcc = 0.0 if p_var == 0 or y_var == 0 else float(covariance / np.sqrt(p_var * y_var))
    result = {"ccc": ccc, "pcc": pcc, "rmse": float(np.sqrt(np.mean((prediction - target) ** 2)))}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite metrics from valid values")
    return result, diagnostics


def evaluate_songs(records: Iterable[dict]) -> dict:
    """Evaluate unique songs; any zero-point song dimension invalidates its macro.

    CCC uses population variance. PCC is zero for constant series, with an
    explicit diagnostic. Nonfinite *masked-out* values never enter metrics;
    nonfinite valid predictions/targets are errors rather than omitted points.
    """
    per_song, invalid, diagnostics = [], [], []
    seen = set()
    pooled_p = {name: [] for name in DIMENSIONS}
    pooled_y = {name: [] for name in DIMENSIONS}
    counts = dict.fromkeys(DIMENSIONS, 0)
    for record in records:
        song = record["song_id"]
        if song in seen: raise ValueError(f"duplicate song_id: {song}")
        seen.add(song)
        prediction = np.asarray(record["prediction"], dtype=np.float64)
        target = np.asarray(record["target"], dtype=np.float64)
        raw_mask = np.asarray(record["mask"])
        if prediction.ndim != 2 or prediction.shape[1] != 2 or prediction.shape != target.shape or target.shape != raw_mask.shape:
            raise ValueError("prediction, target and mask must share shape [N,2]")
        if raw_mask.dtype != np.bool_:
            raise ValueError("mask must be boolean")
        if not np.isfinite(prediction[raw_mask]).all() or not np.isfinite(target[raw_mask]).all():
            raise ValueError(f"nonfinite valid prediction or target for {song}")
        entry = {"song_id": song, "valid_point_counts": {}}
        for index, name in enumerate(DIMENSIONS):
            p, y = prediction[raw_mask[:, index], index], target[raw_mask[:, index], index]
            scores, degenerate = _metrics(p, y)
            entry[name] = scores
            entry["valid_point_counts"][name] = len(y)
            counts[name] += len(y)
            pooled_p[name].append(p)
            pooled_y[name].append(y)
            if len(y) == 0:
                invalid.append({"song_id": song, "dimension": name, "reason": "zero_valid_points"})
            diagnostics.extend({"song_id": song, "dimension": name, "kind": kind} for kind in degenerate)
        per_song.append(entry)
    if not per_song:
        raise ValueError("cannot evaluate an empty song collection")
    macro, pooled = {}, {}
    for name in DIMENSIONS:
        if any(item["dimension"] == name for item in invalid):
            macro[name] = dict.fromkeys(METRICS)
        else:
            macro[name] = {metric: float(np.mean([r[name][metric] for r in per_song])) for metric in METRICS}
        pooled[name], pooled_degenerate = _metrics(np.concatenate(pooled_p[name]), np.concatenate(pooled_y[name]))
        diagnostics.extend({"song_id": "__pooled__", "dimension": name, "kind": kind} for kind in pooled_degenerate)
    return {
        "status": "INVALID" if invalid else "VALID", "song_count": len(per_song),
        "valid_point_counts": counts, "macro": macro, "pooled": pooled,
        "selection_score": None if invalid else float(np.mean([macro[name]["ccc"] for name in DIMENSIONS])),
        "per_song": per_song, "invalid_song_dimensions": invalid, "degenerate_diagnostics": diagnostics,
        "conventions": {"dimension_order": list(DIMENSIONS), "macro_weighting": "equal_song", "ccc_variance": "population", "constant_pcc": 0, "identical_constant_ccc": 1, "zero_valid_dimension": "invalid_macro_and_selection"},
    }
