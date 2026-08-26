"""Deterministic non-architectural training-budget calculations for v3."""

from __future__ import annotations

import math
from typing import Any

from sc_mv_dmer.foundation.canonical import sha256_canonical


def build_v3_training_contract(*, manifest_sha256: str, batch_size: int = 4, gradient_accumulation: int = 4, logical_epochs: int = 5, warmup_fraction: float = 0.10) -> dict[str, Any]:
    if batch_size <= 0 or gradient_accumulation <= 0 or logical_epochs <= 0:
        raise ValueError("training budget values must be positive")
    samples = 995
    batches_per_epoch = math.ceil(samples / batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation)
    total_micro_batches = batches_per_epoch * logical_epochs
    total_updates = updates_per_epoch * logical_epochs
    warmup_updates = math.ceil(total_updates * warmup_fraction)
    checkpoint_interval = updates_per_epoch
    semantic = {"dataset_manifest_logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "dataset_manifest_sha256": manifest_sha256, "train_population_count": samples, "test_population_count": 58, "validation_policy": "UNDEFINED_PENDING_SEPARATE_FREEZE", "sampler_policy": "GLOBAL_PERMUTATION_EXACT_ONCE", "sample_identity": "song_id/sample_id", "architecture_identity": "A1-v1", "loss_definition_identity": "LOR-FBC-3R+++", "input_contract": {"duration_seconds": 45, "time_grid_hz": 2, "canonical_t": 90, "mert_hidden_dimension": 768, "views": ["deep", "mel", "mfcc", "chroma"]}}
    resolved = semantic | {"runtime": {"batch_size": batch_size, "gradient_accumulation": gradient_accumulation, "logical_epochs": logical_epochs, "warmup_fraction": warmup_fraction, "checkpointing": True}, "budget": {"samples_per_epoch": samples, "batches_per_epoch": batches_per_epoch, "updates_per_epoch": updates_per_epoch, "total_micro_batches": total_micro_batches, "total_updates": total_updates, "warmup_updates": warmup_updates, "checkpoint_interval_updates": checkpoint_interval, "checkpoint_boundaries_updates": [checkpoint_interval * i for i in range(1, logical_epochs + 1)]}}
    return {"schema_version": "1.0", "contract_id": "DEAM-PDMER-CLEAN-v3-TRAINING-BUDGET-PREFLIGHT-v1", "semantic_config_hash": sha256_canonical(semantic), "resolved_config_hash": sha256_canonical(resolved), "semantic": semantic, "resolved": resolved, "training_authorized": False, "checkpoint_published": False}
