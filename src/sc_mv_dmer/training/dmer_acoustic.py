"""Full-population DMER training primitives, separate from legacy RG-03.

Invalid targets are masked before arithmetic. Invalid predictions fail closed;
they never improve metrics by disappearing from the evaluated population.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

SEEDS = (52826381, 128866372, 1616929435, 1871035633, 1460830465)
VIEW_DIMS = {"deep": 768, "mel": 128, "mfcc": 40, "chroma": 12}


@dataclass
class TrainNormalizer:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]

    @classmethod
    def fit(cls, rows: Iterable[dict]) -> "TrainNormalizer":
        sums, squares, counts = {}, {}, {}
        for row in rows:
            if row["split"] != "train":
                raise ValueError("normalization may fit train rows only")
            mask = np.asarray(row["audio_mask"], dtype=bool)
            for name, values in row["views"].items():
                x = np.asarray(values, dtype=np.float64)[mask]
                if not len(x) or not np.isfinite(x).all():
                    raise ValueError("normalization needs finite valid train features")
                sums[name] = sums.get(name, 0) + x.sum(axis=0)
                squares[name] = squares.get(name, 0) + np.square(x).sum(axis=0)
                counts[name] = counts.get(name, 0) + len(x)
        if not counts:
            raise ValueError("empty train normalization population")
        mean = {k: sums[k] / counts[k] for k in counts}
        std = {k: np.sqrt(np.maximum(squares[k] / counts[k] - mean[k] ** 2, 0))
               for k in counts}
        return cls(mean, {k: np.maximum(v, 1e-6) for k, v in std.items()})

    def transform(self, views: dict, audio_mask: np.ndarray) -> dict:
        result = {}
        for name, values in views.items():
            x = np.asarray(values, dtype=np.float32)
            result[name] = np.where(np.asarray(audio_mask)[:, None],
                                    (x - self.mean[name]) / self.std[name], 0).astype(np.float32)
            if not np.isfinite(result[name]).all():
                raise ValueError("nonfinite normalized feature")
        return result

    def to_dict(self) -> dict:
        return {"fit_role": "train", "mean": {k: v.tolist() for k, v in self.mean.items()},
                "std": {k: v.tolist() for k, v in self.std.items()}}


def masked_va_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("VA loss requires equally shaped [batch,time,2] arrays")
    mask = mask.bool()
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("valid prediction and target must be finite")
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("each song and dimension needs supervision")
    delta = torch.where(mask, prediction, 0) - torch.where(mask, target, 0)
    return (delta.square().sum(dim=1) / counts).mean()


def smoothness_loss(prediction: Tensor, mask: Tensor) -> Tensor:
    adjacent = mask[:, 1:].bool() & mask[:, :-1].bool()
    if not torch.isfinite(prediction[mask.bool()]).all():
        raise ValueError("valid prediction must be finite")
    clean = torch.where(mask.bool(), prediction, 0)
    delta = torch.where(adjacent, clean[:, 1:] - clean[:, :-1], 0)
    counts = adjacent.sum(dim=1)
    present = counts > 0
    if not present.any():
        return clean.sum() * 0
    return (delta.square().sum(dim=1) / counts.clamp_min(1))[present].mean()


def fit_constant(rows: Iterable[dict]) -> np.ndarray:
    totals, counts = np.zeros(2), np.zeros(2, dtype=np.int64)
    for row in rows:
        if row["split"] != "train":
            raise ValueError("constant baseline may fit train rows only")
        y, mask = np.asarray(row["target"], dtype=np.float64), np.asarray(row["mask"], dtype=bool)
        if y.shape != mask.shape or y.ndim != 2 or y.shape[1] != 2 or not np.isfinite(y[mask]).all():
            raise ValueError("invalid train targets")
        totals += np.where(mask, y, 0).sum(axis=0)
        counts += mask.sum(axis=0)
    if (counts == 0).any():
        raise ValueError("empty train target dimension")
    return totals / counts


def gate_against_crnn(candidate: dict, reference: dict, tolerance: float = .02) -> dict:
    for report in (candidate, reference):
        for name in ("valence", "arousal"):
            value = report.get("macro", {}).get(name, {}).get("ccc")
            if report.get("status") == "INVALID" or not isinstance(value, (int, float)) or not np.isfinite(value):
                return {"verdict": "INVALID", "reason": "missing or invalid song-equal CCC", "deltas": None}
    deltas = {name: float(candidate["macro"][name]["ccc"] - reference["macro"][name]["ccc"])
              for name in ("valence", "arousal")}
    if not all(np.isfinite(x) for x in deltas.values()):
        return {"verdict": "INVALID", "deltas": deltas}
    verdict = "FAIL" if min(deltas.values()) < -tolerance - 1e-12 else (
        "REACHED" if min(deltas.values()) >= 0 else "NEAR")
    return {"verdict": verdict, "deltas": deltas, "tolerance": tolerance,
            "selection_role": "validation", "aggregation": "song_equal"}


def validate_feature_binding(dataset: dict, cache: dict) -> None:
    if cache.get("feature_version") != "deam-dmer-features-v2":
        raise ValueError("legacy feature caches are not eligible for full DMER")
    if cache.get("dataset_internal_sha256") != dataset["manifest_sha256"]:
        raise ValueError("feature cache does not bind full dataset hash")
    if cache.get("full_population_ready") is not True or cache.get("four_view_ready") is not True:
        raise ValueError("full-population feature materialization is incomplete")
    rows = cache.get("windows", [])
    if not rows or len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("feature windows missing or duplicated")
    if {r["song_id"] for r in rows} != {r["song_id"] for r in dataset["records"]}:
        raise ValueError("feature cache does not cover all 1802 songs")
    from sc_mv_dmer.data.dmer_protocol import make_windows
    expected = {w["window_id"]: w for r in dataset["records"] for w in make_windows(r)}
    if {r["window_id"] for r in rows} != set(expected):
        raise ValueError("feature cache does not cover all required audio windows")
    for row in rows:
        if any(row.get(k) != expected[row["window_id"]][k] for k in ("song_id", "logical_song_key", "start_ms")):
            raise ValueError("feature window identity does not match its source song")
