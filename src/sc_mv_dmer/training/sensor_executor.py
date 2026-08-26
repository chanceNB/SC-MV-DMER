"""Sensor-only model and loss contracts for RG-03.

The module intentionally contains no data loading, optimizer, checkpoint, or
training loop. Those concerns belong to the RG-03 runner after pre-flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VIEW_DIMS = {"mel": 128, "mfcc": 40, "chroma": 12}
TARGETS = ("rms", "brightness", "mode", "key")
KEY_CLASS_COUNT = 12
KEY_SEGMENT_COUNT = 9
FRAMES_PER_KEY_SEGMENT = 10


class HandcraftedViewEncoder(nn.Module):
    """Small trainable encoder kept on the Sensor supervision path."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, values: Tensor) -> Tensor:
        return self.normalization(self.activation(self.projection(values)))


class SensorOnlyModel(nn.Module):
    """Sensor branch consuming only Mel/MFCC/Chroma encoder outputs.

    ``sensor`` uses the clean encoded views. ``downstream_views`` is an
    optional diagnostic branch where View Dropout is applied; it is never fed
    back into Sensor heads.
    """

    def __init__(self, hidden_dim: int = 64, view_dropout_p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= view_dropout_p < 1.0:
            raise ValueError("view_dropout_p must be in [0, 1)")
        self.encoders = nn.ModuleDict({name: HandcraftedViewEncoder(dim, hidden_dim) for name, dim in VIEW_DIMS.items()})
        self.view_dropout = nn.Dropout(view_dropout_p)
        self.sensor_heads = nn.ModuleDict(
            {
                "rms": nn.Linear(hidden_dim, 1),
                "brightness": nn.Linear(hidden_dim, 1),
                "mode": nn.Linear(hidden_dim, 1),
                "key": nn.Linear(hidden_dim, KEY_CLASS_COUNT),
            }
        )

    def encode_views(self, views: Mapping[str, Tensor]) -> dict[str, Tensor]:
        if set(views) != set(VIEW_DIMS):
            raise ValueError(f"Sensor branch requires exactly {tuple(VIEW_DIMS)} views")
        encoded: dict[str, Tensor] = {}
        for name, dimension in VIEW_DIMS.items():
            value = views[name]
            if value.ndim != 3 or value.shape[-1] != dimension:
                raise ValueError(f"{name} input must have shape [batch,time,{dimension}]")
            encoded[name] = self.encoders[name](value)
        return encoded

    def forward(self, views: Mapping[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        encoded = self.encode_views(views)
        sensor = {
            "rms": self.sensor_heads["rms"](encoded["mel"]),
            "brightness": self.sensor_heads["brightness"](encoded["mfcc"]),
            "mode": self.sensor_heads["mode"](encoded["chroma"]),
            "key": self.sensor_heads["key"](encoded["chroma"]),
        }
        downstream = {name: self.view_dropout(value) for name, value in encoded.items()}
        return {"sensor": sensor, "encoded_views": encoded, "downstream_views": downstream}


class TrainingExecutionNotAuthorized(RuntimeError):
    """Raised when a caller attempts to start training from the contract facade."""


@dataclass(frozen=True)
class SensorFormalExecutor:
    """Execution facade whose default and only tested operation is planning.

    Actual optimization is intentionally a separate authorized operation. This
    prevents a pre-flight/dry-run command from silently creating checkpoints.
    """

    optimization_train_songs: int = 896
    validation_songs: int = 99
    test_songs: int = 58
    validation_metric: str = "validation CE"
    cadence: str = "every epoch"
    patience: int = 10
    min_delta: float = 1e-4
    seed_set: str = "SC-MV-DMER-FORMAL-S5/v1"

    def dry_run(self) -> dict[str, object]:
        return {
            "optimization_train": self.optimization_train_songs,
            "validation": self.validation_songs,
            "test": self.test_songs,
            "selection_metric": self.validation_metric,
            "cadence": self.cadence,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "seed_set": self.seed_set,
            "training_started": False,
            "checkpoint_published": False,
        }

    def fit(self, *args: object, **kwargs: object) -> None:
        raise TrainingExecutionNotAuthorized(
            "RG-03 training is not authorized by the contract-validation command"
        )


@dataclass(frozen=True)
class LossTerm:
    numerator: Tensor
    denominator: int
    eligibility: bool

    @property
    def value(self) -> Tensor | None:
        return self.numerator / self.denominator if self.eligibility else None


@dataclass(frozen=True)
class SensorLossResult:
    terms: dict[str, LossTerm]
    total: Tensor | None


class SensorLossContract:
    """DLC-P5-compatible Sensor losses with denominator-safe reductions."""

    @staticmethod
    def _continuous(prediction: Tensor, target: Tensor, mask: Tensor) -> LossTerm:
        aligned_mask = mask.unsqueeze(-1) if mask.ndim + 1 == prediction.ndim else mask
        valid = aligned_mask.to(torch.bool) & torch.isfinite(prediction) & torch.isfinite(target)
        values = (prediction - target).pow(2).reshape(-1)[valid.reshape(-1)]
        numerator = values.sum() if values.numel() else prediction.sum() * 0.0
        return LossTerm(numerator=numerator, denominator=int(values.numel()), eligibility=values.numel() > 0)

    @staticmethod
    def pool_key_frames(key_frames: Tensor) -> Tensor:
        if key_frames.ndim != 3 or key_frames.shape[1] != 90 or key_frames.shape[2] != KEY_CLASS_COUNT:
            raise ValueError("Key frames must have shape [batch,90,12]")
        return key_frames.reshape(key_frames.shape[0], KEY_SEGMENT_COUNT, FRAMES_PER_KEY_SEGMENT, KEY_CLASS_COUNT).mean(dim=2)

    @staticmethod
    def pool_key_targets(key_frames: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor]:
        if frame_mask.shape != key_frames.shape[:2]:
            raise ValueError("Key frame mask must have shape [batch,90]")
        reshaped_mask = frame_mask.to(torch.bool).reshape(frame_mask.shape[0], KEY_SEGMENT_COUNT, FRAMES_PER_KEY_SEGMENT)
        finite = torch.isfinite(key_frames).all(dim=-1).reshape(frame_mask.shape[0], KEY_SEGMENT_COUNT, FRAMES_PER_KEY_SEGMENT)
        valid_frames = reshaped_mask & finite
        count = valid_frames.sum(dim=-1)
        valid_segment = count >= 5
        weighted = key_frames.reshape(key_frames.shape[0], KEY_SEGMENT_COUNT, FRAMES_PER_KEY_SEGMENT, KEY_CLASS_COUNT) * valid_frames.unsqueeze(-1)
        pooled = weighted.sum(dim=2) / count.clamp_min(1).unsqueeze(-1)
        return pooled, valid_segment

    @classmethod
    def key_ce(cls, logits: Tensor, key_targets: Tensor, frame_mask: Tensor) -> LossTerm:
        pooled_logits = cls.pool_key_frames(logits)
        pooled_targets, valid_segment = cls.pool_key_targets(key_targets, frame_mask)
        labels = pooled_targets.argmax(dim=-1)
        finite = torch.isfinite(pooled_logits).all(dim=-1) & torch.isfinite(pooled_targets).all(dim=-1)
        valid = valid_segment & finite
        losses = F.cross_entropy(pooled_logits.reshape(-1, KEY_CLASS_COUNT), labels.reshape(-1), reduction="none").reshape_as(valid)
        selected = losses[valid]
        numerator = selected.sum() if selected.numel() else logits.sum() * 0.0
        return LossTerm(numerator=numerator, denominator=int(selected.numel()), eligibility=selected.numel() > 0)

    @classmethod
    def compute(cls, predictions: Mapping[str, Tensor], targets: Mapping[str, Tensor], masks: Mapping[str, Tensor]) -> SensorLossResult:
        terms = {
            name: cls._continuous(predictions[name], targets[name], masks[name])
            for name in ("rms", "brightness", "mode")
        }
        terms["key"] = cls.key_ce(predictions["key"], targets["key"], masks["key"])
        active = [term.value for term in terms.values() if term.eligibility]
        total = torch.stack(active).sum() if active else None
        return SensorLossResult(terms=terms, total=total)
