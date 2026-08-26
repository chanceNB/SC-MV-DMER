"""Authorized RG-03 Sensor-only formal training.

This module is intentionally independent of the Markov/Fusion/Deep paths. It
loads the already finalized v3 cache, trains only the handcrafted Sensor
encoders and heads, and writes append-only run artifacts after strict
pre-flight has passed.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sc_mv_dmer.foundation.canonical import sha256_canonical
from sc_mv_dmer.gates.rg03 import run_preflight
from sc_mv_dmer.training.sensor_executor import SensorLossContract, SensorOnlyModel


SEEDS = (52826381, 128866372, 1616929435, 1871035633, 1460830465)
TARGETS = ("rms", "brightness", "mode", "key")
CONTINUOUS = ("rms", "brightness", "mode")
VOCABULARY = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SensorSample:
    song_id: str
    sample_id: str
    role: str
    views: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    valid_mask: np.ndarray


@dataclass(frozen=True)
class TargetStats:
    mean: dict[str, float]
    std: dict[str, float]
    baseline: dict[str, float]
    prior: np.ndarray


@dataclass(frozen=True)
class SensorRunResult:
    seed: int
    run_dir: Path
    curve_path: Path
    best_checkpoint: Path
    best_epoch: int
    best_step: int
    best_validation_ce: float
    metrics: dict[str, Any]


def _sample_index(repo_root: Path, artifact_root: Path) -> dict[str, SensorSample]:
    feature = _json(repo_root / "manifests/features/deam-four-view-cache-v3.json")
    pseudo = _json(repo_root / "manifests/features/deam-pseudo-label-bundle-v3.json")
    pseudo_by_sample = {row["sample_id"]: row for row in pseudo["records"]}
    result: dict[str, SensorSample] = {}
    for row in feature["records"]:
        pseudo_row = pseudo_by_sample[row["sample_id"]]
        feature_path = artifact_root / row["payload_logical_path"]
        pseudo_path = artifact_root / pseudo_row["payload_logical_path"]
        with np.load(feature_path) as features, np.load(pseudo_path) as labels:
            views = {name: np.asarray(features[name], dtype=np.float32) for name in ("mel", "mfcc", "chroma")}
            targets = {name: np.asarray(labels[name], dtype=np.float32) for name in TARGETS}
            valid = np.asarray(labels["valid_target_mask"], dtype=bool)
        result[row["song_id"]] = SensorSample(row["song_id"], row["sample_id"], row["population_role"], views, targets, valid)
    return result


def _memberships(repo_root: Path) -> tuple[list[str], list[str], list[str]]:
    split = _json(repo_root / "manifests/splits/deam-pdmer-clean-v3-validation-v1.json")
    return list(split["optimization_train_song_ids"]), list(split["validation_song_ids"]), list(split["test_song_ids"])


def _fit_stats(samples: list[SensorSample], norm: dict[str, Any], key_evidence: dict[str, Any]) -> TargetStats:
    means = {name: float(norm["normalization"]["targets"][name]["mean"]) for name in CONTINUOUS}
    std = {name: float(norm["normalization"]["targets"][name]["std_population_ddof0"]) for name in CONTINUOUS}
    baseline = {name: float(norm["constant_baseline"][name if name != "rms" else "energy"]["value"]) for name in ("rms", "brightness")}
    baseline["mode"] = means["mode"]
    counts = np.asarray(key_evidence["prior"]["class_counts"], dtype=np.float64)
    prior = (counts + 1.0) / float(counts.sum() + len(counts))
    return TargetStats(means, std, baseline, prior)


def _tensor_batch(samples: list[SensorSample], stats: TargetStats, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
    views = {name: torch.from_numpy(np.stack([sample.views[name] for sample in samples])).to(device) for name in ("mel", "mfcc", "chroma")}
    targets: dict[str, Tensor] = {}
    masks: dict[str, Tensor] = {}
    for name in CONTINUOUS:
        values = np.stack([sample.targets[name] for sample in samples]).astype(np.float32)
        values = (values - stats.mean[name]) / max(stats.std[name], 1e-6)
        targets[name] = torch.from_numpy(values).to(device)
        masks[name] = torch.from_numpy(np.stack([sample.valid_mask for sample in samples])).to(device)
    targets["key"] = torch.from_numpy(np.stack([sample.targets["key"] for sample in samples])).to(device)
    masks["key"] = torch.from_numpy(np.stack([sample.valid_mask for sample in samples])).to(device)
    return views, targets, masks


def _iter_batches(ids: list[str], batch_size: int, generator: torch.Generator) -> list[list[str]]:
    order = torch.randperm(len(ids), generator=generator).tolist()
    return [[ids[index] for index in order[start : start + batch_size]] for start in range(0, len(ids), batch_size)]


def _validation_metrics(model: SensorOnlyModel, samples: list[SensorSample], stats: TargetStats, device: torch.device) -> tuple[dict[str, float], dict[str, int]]:
    model.eval()
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in TARGETS}
    targets: dict[str, list[np.ndarray]] = {name: [] for name in TARGETS}
    masks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), 16):
            batch = samples[start : start + 16]
            views, batch_targets, batch_masks = _tensor_batch(batch, stats, device)
            output = model(views)["sensor"]
            for name in CONTINUOUS:
                pred = output[name].detach().cpu().numpy() * stats.std[name] + stats.mean[name]
                predictions[name].append(pred)
                targets[name].append(np.stack([sample.targets[name] for sample in batch]))
            predictions["key"].append(output["key"].detach().cpu().numpy())
            targets["key"].append(np.stack([sample.targets["key"] for sample in batch]))
            masks.append(np.stack([sample.valid_mask for sample in batch]))
    flat_mask = np.concatenate(masks, axis=0).astype(bool)
    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in CONTINUOUS:
        pred = np.concatenate(predictions[name], axis=0).reshape(-1)
        target = np.concatenate(targets[name], axis=0).reshape(-1)
        valid = flat_mask.reshape(-1) & np.isfinite(pred) & np.isfinite(target)
        pred, target = pred[valid].astype(np.float64), target[valid].astype(np.float64)
        counts[name] = int(pred.size)
        metrics[f"{name}_mse"] = float(np.mean((pred - target) ** 2)) if pred.size else float("nan")
        if name == "mode":
            if pred.size < 2:
                metrics["mode_ccc"] = float("nan")
            else:
                mean_p, mean_t = pred.mean(), target.mean()
                var_p, var_t = np.mean((pred - mean_p) ** 2), np.mean((target - mean_t) ** 2)
                cov = np.mean((pred - mean_p) * (target - mean_t))
                denom = var_p + var_t + (mean_p - mean_t) ** 2
                metrics["mode_ccc"] = float(2 * cov / denom) if denom > 0 else float("nan")
    logits = np.concatenate(predictions["key"], axis=0).reshape(-1, 12)
    key_targets = np.concatenate(targets["key"], axis=0).reshape(-1, 12)
    pooled_logits = logits.reshape(-1, 9, 10, 12).mean(axis=2).reshape(-1, 12)
    pooled_targets = key_targets.reshape(-1, 9, 10, 12)
    frame_valid = flat_mask.reshape(-1, 9, 10)
    segment_valid = frame_valid.sum(axis=2) >= 5
    pooled_targets = np.divide((pooled_targets * frame_valid[..., None]).sum(axis=2), np.maximum(frame_valid.sum(axis=2), 1)[..., None]).reshape(-1, 12)
    valid_segments = segment_valid.reshape(-1) & np.isfinite(pooled_logits).all(axis=1) & np.isfinite(pooled_targets).all(axis=1)
    labels = pooled_targets[valid_segments].argmax(axis=1)
    log_probs = pooled_logits[valid_segments] - np.logaddexp.reduce(pooled_logits[valid_segments], axis=1, keepdims=True)
    metrics["key_ce"] = float(-log_probs[np.arange(len(labels)), labels].mean()) if labels.size else float("nan")
    metrics["key_prior_ce"] = float(-np.log(stats.prior[labels]).mean()) if labels.size else float("nan")
    counts["key_valid_segments"] = int(labels.size)
    return metrics, counts


def _loss(model: SensorOnlyModel, samples: list[SensorSample], stats: TargetStats, device: torch.device) -> Tensor:
    views, targets, masks = _tensor_batch(samples, stats, device)
    predictions = model(views)["sensor"]
    result = SensorLossContract.compute(predictions, targets, masks)
    if result.total is None:
        raise RuntimeError("L_sensor has zero eligible terms")
    return result.total


def train_formal(*, repo_root: Path, artifact_root: Path, output_root: Path, device: str = "cuda", max_epochs: int = 100, batch_size: int = 16) -> dict[str, Any]:
    preflight = run_preflight(repo_root, artifact_root=artifact_root)
    if preflight.verdict != "PASS":
        raise RuntimeError(f"RG-03 strict preflight failed: {preflight.blockers}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite append-only output root: {output_root}")
    output_root.mkdir(parents=True)
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    index = _sample_index(repo_root, artifact_root)
    train_ids, valid_ids, test_ids = _memberships(repo_root)
    norm = _json(repo_root / "evidence/data/deam-pdmer-clean-v3-normalization-baseline-v1.json")
    key_evidence = _json(repo_root / "evidence/data/deam-pdmer-clean-v3-key-contract-v1.json")
    stats = _fit_stats([index[song] for song in train_ids], norm, key_evidence)
    train_samples = [index[song] for song in train_ids]
    valid_samples = [index[song] for song in valid_ids]
    results: list[SensorRunResult] = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = SensorOnlyModel(hidden_dim=64, view_dropout_p=0.0).to(device_obj)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        run_dir = output_root / f"seed-{seed}"
        run_dir.mkdir()
        generator = torch.Generator().manual_seed(seed)
        curve: list[dict[str, Any]] = []
        best_ce = float("inf")
        best_epoch = 0
        best_step = 0
        best_checkpoint = run_dir / "best.pt"
        stale = 0
        global_step = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch_ids in _iter_batches(train_ids, batch_size, generator):
                loss = _loss(model, [index[song] for song in batch_ids], stats, device_obj)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
            scheduler.step()
            metrics, counts = _validation_metrics(model, valid_samples, stats, device_obj)
            epoch_record = {"epoch": epoch, "global_step": global_step, "train_loss": float(loss.detach().cpu()), "learning_rate": float(optimizer.param_groups[0]["lr"]), "validation": metrics, "valid_counts": counts}
            curve.append(epoch_record)
            checkpoint = run_dir / f"epoch-{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "seed": seed, "epoch": epoch, "global_step": global_step}, checkpoint)
            if metrics["key_ce"] < best_ce - 1e-4:
                best_ce, best_epoch, best_step, stale = metrics["key_ce"], epoch, global_step, 0
                torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch, "global_step": global_step, "validation_ce": best_ce}, best_checkpoint)
            else:
                stale += 1
            if stale >= 10:
                break
        model.load_state_dict(torch.load(best_checkpoint, map_location=device_obj, weights_only=False)["model"])
        final_metrics, final_counts = _validation_metrics(model, valid_samples, stats, device_obj)
        test_metrics, test_counts = _validation_metrics(model, [index[song] for song in test_ids], stats, device_obj)
        curve_path = run_dir / "learning-curve.json"
        curve_payload = {"schema_version": "1.0", "seed": seed, "selection": {"metric": "validation CE", "patience": 10, "min_delta": 1e-4, "tie_break": ["earliest epoch", "smallest global step"]}, "records": curve, "best_epoch": best_epoch, "best_global_step": best_step, "best_validation_ce": best_ce, "curve_sha256": sha256_canonical({"schema_version": "1.0", "seed": seed, "selection": {"metric": "validation CE", "patience": 10, "min_delta": 1e-4, "tie_break": ["earliest epoch", "smallest global step"]}, "records": curve, "best_epoch": best_epoch, "best_global_step": best_step, "best_validation_ce": best_ce})}
        curve_path.write_text(json.dumps(curve_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(SensorRunResult(seed, run_dir, curve_path, best_checkpoint, best_epoch, best_step, best_ce, {"validation": final_metrics, "validation_counts": final_counts, "test": test_metrics, "test_counts": test_counts, "baseline": stats.baseline}))
    evidence = {"schema_version": "1.0", "evidence_id": "RG-03-SENSOR-FORMAL-v1", "verdict": "FORMAL_COMPLETED", "scope": "SENSOR_ONLY", "dataset_manifest": {"logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "manifest_sha256": "d4c539945c5477e5ca325e9535f6f100918341a9292460d533377c493ee8473d"}, "validation_contract": "DEAM-PDMER-CLEAN-v3-SONG-VALIDATION/v1", "population": {"optimization_train": 896, "validation": 99, "test": 58}, "feature_cache_manifest_sha256": "0f7b0e699184208bcb0e719d8288bfc3c1f5c9ade989a83f90d1e892985a7089", "pseudo_label_cache_manifest_sha256": "8979085e00668295c992c506d04ff4ab7231a0728f78f41d6dcb29af023345dd", "decision": "DEC-0031", "seed_set": list(SEEDS), "architecture": {"identity": "A1-v1", "sensor_inputs": ["E_mel", "E_mfcc", "E_chroma"], "forbidden_inputs": ["F_fused", "Markov output"], "view_dropout_sensor_supervision": False}, "selection": {"metric": "validation CE", "cadence": "every epoch", "patience": 10, "min_delta": 1e-4, "tie_break": ["earliest epoch", "smallest global step"]}, "normalization": {"fit_split": "optimization_train", "metric_space": "inverse-transformed raw target space", "test_used": False}, "baseline": {"fit_split": "optimization_train", "test_used": False}, "runs": []}
    for result in results:
        checkpoints = sorted(result.run_dir.glob("epoch-*.pt")) + [result.best_checkpoint]
        evidence["runs"].append({"seed": result.seed, "run_logical_path": f"runs/rg-03-sensor-formal-v1/seed-{result.seed}", "learning_curve_sha256": _sha256_file(result.curve_path), "best_checkpoint_sha256": _sha256_file(result.best_checkpoint), "best_epoch": result.best_epoch, "best_global_step": result.best_step, "checkpoint_count": len(checkpoints), "checkpoint_sha256": {path.name: _sha256_file(path) for path in checkpoints}, "metrics": result.metrics})
    evidence["evidence_sha256"] = sha256_canonical(evidence)
    evidence_path = output_root / "rg-03-formal-evidence-v1.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    return {"evidence_path": str(evidence_path), "evidence_sha256": evidence["evidence_sha256"], "runs": evidence["runs"], "preflight": "PASS", "training_started": True, "checkpoint_published": True}
