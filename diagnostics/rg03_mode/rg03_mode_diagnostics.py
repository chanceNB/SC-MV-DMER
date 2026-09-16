"""RG-03 Mode root-cause diagnostics.

This module is deliberately diagnostic-only.  It reads the registered v3
manifests and immutable payload root, and it admits only the 896 optimization
songs and 99 validation songs.  It never opens a test payload, writes a
checkpoint, or emits Gate evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr
from torch import nn


DATASET_ID = "dataset_480f00e52c71f8a4f7c2681f2a59d3b11c105d915e3faf508bd06970ee846dae"
DATASET_MANIFEST_SHA256 = "d4c539945c5477e5ca325e9535f6f100918341a9292460d533377c493ee8473d"
FEATURE_MANIFEST = "manifests/features/deam-four-view-cache-v3.json"
PSEUDO_MANIFEST = "manifests/features/deam-pseudo-label-bundle-v3.json"
VALIDATION_MANIFEST = "manifests/splits/deam-pdmer-clean-v3-validation-v1.json"
FORMAL_EVIDENCE = "evidence/gates/rg-03/rg-03-sensor-formal-v2.json"
CONTRACT_ID = "DEAM-PDMER-CLEAN-v3-SONG-VALIDATION/v1"


def _read(root: Path, logical: str) -> dict[str, Any]:
    return json.loads((root / logical).read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ccc(pred: np.ndarray, target: np.ndarray) -> tuple[float, str | None]:
    """MEP-CCC2-style pooled moments, diagnostic implementation only."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    finite = np.isfinite(p) & np.isfinite(t)
    p, t = p[finite], t[finite]
    if p.size < 2:
        return float("nan"), "MODE_CCC_LT2_VALID_PAIRS"
    mp, mt = np.mean(p, dtype=np.float64), np.mean(t, dtype=np.float64)
    vp, vt = np.mean((p - mp) ** 2, dtype=np.float64), np.mean((t - mt) ** 2, dtype=np.float64)
    if vt == 0.0:
        return float("nan"), "MODE_CCC_ZERO_VARIANCE_TARGET"
    if vp == 0.0:
        return float("nan"), "MODE_CCC_ZERO_VARIANCE_PREDICTION"
    denom = vp + vt + (mp - mt) ** 2
    if denom == 0.0:
        return float("nan"), "MODE_CCC_DENOMINATOR_ZERO"
    cov = np.mean((p - mp) * (t - mt), dtype=np.float64)
    value = float(2.0 * cov / denom)
    return (value, None) if np.isfinite(value) else (float("nan"), "MODE_CCC_NONFINITE_RESULT")


def _load_arrays(root: Path, artifact_root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    split = _read(root, VALIDATION_MANIFEST)
    train = set(split["optimization_train_song_ids"])
    valid = set(split["validation_song_ids"])
    allowed = train | valid
    assert len(train) == 896 and len(valid) == 99 and len(allowed) == 995
    features = _read(root, FEATURE_MANIFEST)
    labels = _read(root, PSEUDO_MANIFEST)
    feature_rows = {r["song_id"]: r for r in features["records"] if r["song_id"] in allowed}
    label_rows = {r["song_id"]: r for r in labels["records"] if r["song_id"] in allowed}
    assert set(feature_rows) == allowed and set(label_rows) == allowed
    # Explicitly reject accidental test access.  No test payload path is opened.
    assert all(r["population_role"] == "TRAIN_PRIMARY" for r in feature_rows.values())
    assert all(r["population_role"] == "TRAIN_PRIMARY" for r in label_rows.values())
    chroma, mode, roles = {}, {}, {}
    for song_id in sorted(allowed):
        fpath = artifact_root / feature_rows[song_id]["payload_logical_path"]
        lpath = artifact_root / label_rows[song_id]["payload_logical_path"]
        if _file_sha256(fpath) != feature_rows[song_id]["payload_sha256"]:
            raise RuntimeError(f"feature checksum mismatch: {song_id}")
        if _file_sha256(lpath) != label_rows[song_id]["payload_sha256"]:
            raise RuntimeError(f"pseudo checksum mismatch: {song_id}")
        with np.load(fpath) as zf, np.load(lpath) as zl:
            chroma[song_id] = np.asarray(zf["chroma"], dtype=np.float64)
            mode[song_id] = np.asarray(zl["mode"], dtype=np.float64).reshape(90)
        roles[song_id] = "optimization_train" if song_id in train else "validation"
    return chroma, mode, roles, split


def _flat(chroma: dict[str, np.ndarray], mode: dict[str, np.ndarray], roles: dict[str, str], role: str) -> tuple[np.ndarray, np.ndarray]:
    ids = [s for s in sorted(chroma) if roles[s] == role]
    return np.concatenate([chroma[s] for s in ids]), np.concatenate([mode[s] for s in ids])


def _summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    qs = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return {"n": int(x.size), "mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x)), "p01": float(qs[0]), "p05": float(qs[1]), "p25": float(qs[2]), "p50": float(qs[3]), "p75": float(qs[4]), "p95": float(qs[5]), "p99": float(qs[6])}


def _regression_diagnostics(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> dict[str, Any]:
    x1 = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    xv = np.concatenate([x_valid, np.ones((x_valid.shape[0], 1))], axis=1)
    outputs: dict[str, Any] = {}
    for name, alpha in (("linear", 0.0), ("ridge_1e-2", 1e-2), ("ridge_1", 1.0)):
        gram = x1.T @ x1
        if alpha:
            gram = gram + alpha * np.eye(gram.shape[0])
            gram[-1, -1] -= alpha
        beta = np.linalg.solve(gram + 1e-10 * np.eye(gram.shape[0]), x1.T @ y_train)
        pred = xv @ beta
        outputs[name] = _prediction_metrics(pred, y_valid)

    torch.manual_seed(20260916)
    model = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
    optim = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tx, ty = torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    vx = torch.from_numpy(x_valid.astype(np.float32))
    for _ in range(80):
        optim.zero_grad(set_to_none=True)
        loss = (model(tx) - ty).pow(2).mean()
        loss.backward()
        optim.step()
    pred = model(vx).detach().numpy().reshape(-1)
    outputs["small_mlp_2x32"] = _prediction_metrics(pred, y_valid)
    return outputs


def _prediction_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p, t = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    c, reason = ccc(p, t)
    return {"mse": float(np.mean((p - t) ** 2)), "pearson": float(pearsonr(p, t).statistic), "ccc": c, "ccc_reason": reason, "prediction_mean": float(np.mean(p)), "prediction_std": float(np.std(p)), "target_mean": float(np.mean(t)), "target_std": float(np.std(t))}


class ResearchPlanModePrototype(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(12, 256)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.transformer(self.projection(x))).squeeze(-1)


def _prototype(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> dict[str, Any]:
    torch.manual_seed(20260916)
    model = ResearchPlanModePrototype()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    tx, ty = torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))
    vx, vy = torch.from_numpy(x_valid.astype(np.float32)), torch.from_numpy(y_valid.astype(np.float32))
    curve = []
    # One diagnostic run only; no checkpoint is written.
    for epoch in range(1, 51):
        model.train()
        order = torch.randperm(tx.shape[0])
        for start in range(0, tx.shape[0], 32):
            idx = order[start : start + 32]
            opt.zero_grad(set_to_none=True)
            loss = (model(tx[idx]) - ty[idx]).pow(2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(vx).numpy().reshape(-1)
        curve.append({"epoch": epoch, "train_mse_last_batch": float(loss.detach()), **_prediction_metrics(pred, vy.numpy().reshape(-1))})
    return {"architecture": "Linear(12,256)-TransformerEncoder(1,d_model=256,nhead=4,ffn=512)-MLP(256,64,1)", "curve": curve, "final": curve[-1]}


def _gradient_conflict(x_train: np.ndarray, y_train: np.ndarray, key_train: np.ndarray, *, epochs: int = 5) -> dict[str, Any]:
    """D1/D2 current-topology comparison on a fixed train-only batch."""
    torch.manual_seed(20260916)
    x = torch.from_numpy(x_train[:64].astype(np.float32))
    y = torch.from_numpy(y_train[:64].astype(np.float32)).unsqueeze(-1)
    key = torch.from_numpy(key_train[:64].astype(np.float32))
    results: dict[str, Any] = {}
    for name, with_key in (("D1_mode_only", False), ("D2_mode_plus_key", True)):
        torch.manual_seed(20260916)
        enc = nn.Sequential(nn.Linear(12, 64), nn.GELU(), nn.LayerNorm(64))
        mode_head = nn.Linear(64, 1)
        key_head = nn.Linear(64, 12)
        opt = torch.optim.Adam(list(enc.parameters()) + list(mode_head.parameters()) + list(key_head.parameters()), lr=1e-3)
        rows = []
        cosines = []
        for epoch in range(1, epochs + 1):
            h = enc(x)
            mode_loss = (mode_head(h) - y).pow(2).mean()
            key_labels = key.argmax(dim=-1)
            key_loss = nn.functional.cross_entropy(key_head(h).reshape(-1, 12), key_labels.reshape(-1))
            # Per-task encoder gradients at the same parameter point.
            gm = torch.autograd.grad(mode_loss, tuple(enc.parameters()), retain_graph=True)
            gk = torch.autograd.grad(key_loss, tuple(enc.parameters()), retain_graph=True)
            vm = torch.cat([g.reshape(-1) for g in gm]); vk = torch.cat([g.reshape(-1) for g in gk])
            cosines.append(float(nn.functional.cosine_similarity(vm, vk, dim=0)))
            total = mode_loss + (key_loss if with_key else 0.0)
            opt.zero_grad(set_to_none=True); total.backward(); opt.step()
            rows.append({"epoch": epoch, "train_mode_loss": float(mode_loss.detach()), "train_key_loss": float(key_loss.detach()) if with_key else None})
        results[name] = {"curve": rows, "gradient_cosine_mean": float(np.mean(cosines)), "gradient_cosine_median": float(np.median(cosines)), "gradient_cosine_negative_ratio": float(np.mean(np.asarray(cosines) < 0)), "gradient_cosines": cosines}
    return results


def _formal_selection(root: Path, artifact_root: Path) -> dict[str, Any]:
    """Compare CE-selected checkpoint with max observed Mode CCC."""
    evidence = _read(root, FORMAL_EVIDENCE)
    out = []
    run_root = artifact_root.parent / "rg03-sensor-formal-v2"
    for run in evidence["runs"]:
        curve = run_root / f"seed-{run['seed']}" / "learning-curve.json"
        if not curve.is_file():
            out.append({"seed": run["seed"], "status": "MISSING_LEARNING_CURVE"})
            continue
        records = json.loads(curve.read_text(encoding="utf-8"))["records"]
        mode = [(float(r["validation"]["mode_ccc"]), int(r["epoch"]), int(r["global_step"]), float(r["validation"]["key_ce"])) for r in records]
        selected = int(run["best_epoch"])
        maximum = max(mode, key=lambda x: (x[0], -x[1], -x[2]))
        selected_row = next(x for x in mode if x[1] == selected)
        out.append({"seed": run["seed"], "selected_epoch": selected, "selected_mode_ccc": selected_row[0], "max_mode_ccc_epoch": maximum[1], "max_mode_ccc": maximum[0], "delta_max_minus_selected": maximum[0] - selected_row[0]})
    return {"selection_metric": "validation CE", "runs": out}


def run(root: Path, artifact_root: Path, output: Path) -> dict[str, Any]:
    chroma, mode, roles, split = _load_arrays(root, artifact_root)
    x_train, y_train = _flat(chroma, mode, roles, "optimization_train")
    x_valid, y_valid = _flat(chroma, mode, roles, "validation")
    # Load key only for the same 896 train songs; no test records are opened.
    pseudo = _read(root, PSEUDO_MANIFEST)
    train_ids = {s for s, r in roles.items() if r == "optimization_train"}
    key_rows = {r["song_id"]: r for r in pseudo["records"] if r["song_id"] in train_ids}
    key_train = np.concatenate([np.asarray(np.load(artifact_root / key_rows[s]["payload_logical_path"])["key"], dtype=np.float64) for s in sorted(train_ids)])
    train_song_ids = sorted(s for s, r in roles.items() if r == "optimization_train")
    valid_song_ids = sorted(s for s, r in roles.items() if r == "validation")
    x_train_seq = np.stack([chroma[s] for s in train_song_ids])
    y_train_seq = np.stack([mode[s] for s in train_song_ids])
    x_valid_seq = np.stack([chroma[s] for s in valid_song_ids])
    y_valid_seq = np.stack([mode[s] for s in valid_song_ids])
    formula_pred = x_valid[:, 4] - x_valid[:, 3] - x_valid[:, 10]
    formula = _prediction_metrics(formula_pred, y_valid)
    stats = {"optimization_train": _summary(y_train), "validation": _summary(y_valid), "mode_target_variance": {"optimization_train": float(np.var(y_train)), "validation": float(np.var(y_valid))}}
    result: dict[str, Any] = {
        "schema_version": "rg03-mode-diagnostics-v1",
        "status": "DEBUG_DIAGNOSTIC_ONLY",
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "validation_contract": CONTRACT_ID,
        "population": {"optimization_train_songs": 896, "validation_songs": 99, "test_songs_used": 0},
        "source": {"feature_manifest": FEATURE_MANIFEST, "pseudo_manifest": PSEUDO_MANIFEST, "artifact_root_config": "SC_MV_DMER_ARTIFACT_ROOT", "absolute_paths_serialized": False},
        "A_frozen_target_statistics": stats,
        "B_formula_audit": {"formula": "binned_chroma[pc4] - binned_chroma[pc3] - binned_chroma[pc10]", "formula_reduction": "major/minor masks cancel pc1,pc5,pc7; only pc4,+ and pc3/pc10,- remain", **formula},
        "C_information_ceiling_current_binned_chroma": _regression_diagnostics(x_train, y_train, x_valid, y_valid),
        "D_mse_ccc_mismatch": {"best_debug_model": "small_mlp_2x32", **_regression_diagnostics(x_train, y_train, x_valid, y_valid)["small_mlp_2x32"]},
        "E_gradient_conflict": _gradient_conflict(x_train, y_train, key_train),
        "F_checkpoint_selection_conflict": _formal_selection(root, artifact_root),
        "G_research_plan_mode_prototype": _prototype(x_train_seq, y_train_seq, x_valid_seq, y_valid_seq),
    }
    # The prototype is trained on one frame sequence per row; retain scalar metrics.
    result["G_research_plan_mode_prototype"]["final"] = {k: v for k, v in result["G_research_plan_mode_prototype"]["final"].items() if k not in {"ccc_reason"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = run(args.repo_root, args.artifact_root, args.output)
    print(json.dumps({"output": str(args.output), "status": result["status"], "test_songs_used": result["population"]["test_songs_used"]}, sort_keys=True))


if __name__ == "__main__":
    main()
