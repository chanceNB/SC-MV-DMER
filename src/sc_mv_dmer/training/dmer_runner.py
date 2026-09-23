"""Train/validation-only acoustic experiments on the revised full DEAM protocol."""
from __future__ import annotations

import json
import math
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sc_mv_dmer.data.deam_dmer_full import file_sha256
from sc_mv_dmer.data.dmer_protocol import align_targets, make_windows, verify_split
from sc_mv_dmer.evaluation.dmer_metrics import evaluate_songs
from sc_mv_dmer.foundation.canonical import sha256_canonical
from sc_mv_dmer.training.dmer_acoustic import (
    SEEDS, VIEW_DIMS, TrainNormalizer, fit_constant, masked_va_loss,
    smoothness_loss, validate_feature_binding,
)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def load_rows(manifest_path: Path, split_path: Path, targets_path: Path,
              cache_path: Path | None = None) -> tuple[dict, dict, list[dict]]:
    manifest, split = _json(manifest_path), _json(split_path)
    verify_split(manifest, split)
    if file_sha256(targets_path) != manifest["artifacts"]["dmer_targets"]["file_sha256"]:
        raise ValueError("target artifact checksum mismatch")
    roles = {r["song_id"]: r["split"] for r in split["records"]}
    if any(role not in ("train", "validation", "test", "long_test") for role in roles.values()):
        raise ValueError("unknown split role")
    # Test targets remain in the immutable full-population artifact so its
    # manifest checksum can be verified, but do not JSON-decode their labels.
    # The split manifest supplies the only field needed to skip those records.
    song_id_pattern = re.compile(r'"song_id"\s*:\s*"([^"]+)"')
    target_records = []
    with Path(targets_path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            match = song_id_pattern.search(line)
            if match is None:
                raise ValueError("target row is missing song_id")
            song_id = match.group(1)
            if song_id not in roles:
                raise ValueError("target row is not registered in split manifest")
            if roles[song_id] in ("train", "validation"):
                record = json.loads(line)
                if record.get("song_id") != song_id:
                    raise ValueError("target row song_id parsing mismatch")
                target_records.append(record)
    targets = {r["song_id"]: r for r in target_records}
    expected_target_ids = {song_id for song_id, role in roles.items() if role in ("train", "validation")}
    if len(target_records) != len(expected_target_ids) or len(targets) != len(expected_target_ids) or set(targets) != expected_target_ids:
        raise ValueError("train/validation target population mismatch")
    features = {}
    if cache_path is not None:
        cache = _json(cache_path)
        validate_feature_binding(manifest, cache)
        if cache.get("dataset_manifest_sha256") != file_sha256(manifest_path):
            raise ValueError("feature cache source manifest file checksum mismatch")
        unsigned = {k: v for k, v in cache.items() if k != "manifest_sha256"}
        if cache.get("manifest_sha256") != sha256_canonical(unsigned):
            raise ValueError("feature manifest checksum mismatch")
        requested_windows = {window["window_id"] for record in manifest["records"]
                             if roles[record["song_id"]] in ("train", "validation")
                             for window in make_windows(record)}
        for row in cache["windows"]:
            if row["window_id"] not in requested_windows:
                continue
            path = (cache_path.parent / row["path"]).resolve()
            if not path.is_relative_to(cache_path.parent.resolve()):
                raise ValueError("feature path escapes registered cache root")
            if file_sha256(path) != row["sha256"]:
                raise ValueError(f"feature payload checksum mismatch: {row['window_id']}")
            features[row["window_id"]] = str(path)
        if set(features) != requested_windows:
            raise ValueError("train/validation feature population mismatch")
    rows = []
    for record in manifest["records"]:
        if roles[record["song_id"]] not in ("train", "validation"):
            continue
        for window in make_windows(record):
            target, mask = align_targets(targets[record["song_id"]], window)
            row = dict(window, split=roles[record["song_id"]], target=target, mask=mask)
            if cache_path is not None:
                row["path"] = features[window["window_id"]]
            rows.append(row)
    return manifest, split, rows


class WindowDataset(Dataset):
    def __init__(self, rows: list[dict], normalizer: TrainNormalizer,
                 raw_mert: bool = False, preload: bool = True):
        self.rows, self.normalizer, self.raw_mert = rows, normalizer, raw_mert
        self.cached = [self._views(row) for row in rows] if preload else None

    def _views(self, row: dict) -> tuple[dict, np.ndarray]:
        with np.load(row["path"], allow_pickle=False) as a:
            mask = np.asarray(a["audio_mask"], dtype=bool)
            views = {k: np.asarray(a[k], dtype=np.float32) for k in VIEW_DIMS}
        for name, dimension in VIEW_DIMS.items():
            if views[name].shape != (90, dimension):
                raise ValueError(f"bad feature shape: {name}")
        if mask.shape != (90,) or not mask.any():
            raise ValueError("empty/malformed audio mask")
        # The protocol mask describes the registered canonical grid.  Decoded
        # PCM may be shorter, so the feature mask is allowed to be stricter;
        # supervision itself must never survive beyond actual audio support.
        target_mask = np.asarray(row.get("mask"), dtype=bool)
        output_mask = mask[30:90]
        if target_mask.shape != (60, 2):
            raise ValueError("malformed target mask")
        if np.any(target_mask & ~output_mask[:, None]):
            raise ValueError("target supervision falls outside decoded audio")
        return self.normalizer.transform(views, mask), mask

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        views, mask = self.cached[index] if self.cached is not None else self._views(row)
        item = {"views": views, "audio_mask": mask, "target": row["target"], "mask": row["mask"],
                "song_id": row["song_id"], "output_time_ms": np.asarray(row["output_time_ms"], dtype=np.int64)}
        if self.raw_mert:
            with np.load(row["path"], allow_pickle=False) as a:
                item.update(mert_layers=a["mert_layers"].astype(np.float32),
                            mert_time_seconds=a["mert_time_seconds"], mert_mask=a["mert_mask"])
        return item


def _inputs(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor, dict]:
    views = {k: v.to(device=device, dtype=torch.float32) for k, v in batch["views"].items()}
    kwargs = {k: batch[k].to(device) for k in ("mert_layers", "mert_time_seconds", "mert_mask") if k in batch}
    return views, batch["audio_mask"].to(device), kwargs


def train_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    total, count = 0., 0
    for batch in loader:
        views, audio_mask, kwargs = _inputs(batch, device)
        target, mask = batch["target"].to(device), batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(views, audio_mask, **kwargs)
        loss = masked_va_loss(out["prediction"], target, mask) + .05 * smoothness_loss(out["prediction"], mask)
        entropies = out.get("row_entropy", {})
        if entropies:
            entropy_loss = torch.stack([torch.relu(.5 - h).mean() for h in entropies.values()]).mean()
            loss = loss + .01 * entropy_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(target)
        count += len(target)
    if not count:
        raise ValueError("empty training loader")
    return total / count


def stitch_song_predictions(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        entry = grouped.setdefault(row["song_id"], {})
        for i, timestamp in enumerate(row["output_time_ms"]):
            timestamp = int(timestamp)
            if timestamp in entry:
                raise ValueError("duplicate prediction timestamp within song")
            entry[timestamp] = (row["prediction"][i], row["target"][i], row["mask"][i])
    result = []
    for song_id, entry in sorted(grouped.items()):
        times = sorted(entry)
        arrays = list(zip(*(entry[t] for t in times)))
        result.append({"song_id": song_id, "output_time_ms": times,
                       "prediction": np.asarray(arrays[0]), "target": np.asarray(arrays[1]),
                       "mask": np.asarray(arrays[2], dtype=bool)})
    return result


@torch.no_grad()
def predict(model, loader, device) -> tuple[dict, list[dict]]:
    model.eval()
    records = []
    for batch in loader:
        views, mask, kwargs = _inputs(batch, device)
        pred = model(views, mask, **kwargs)["prediction"].cpu().numpy()
        for i, song in enumerate(batch["song_id"]):
            records.append({"song_id": song, "prediction": pred[i], "target": batch["target"][i].numpy(),
                            "mask": batch["mask"][i].numpy(), "output_time_ms": batch["output_time_ms"][i].tolist()})
    songs = stitch_song_predictions(records)
    return evaluate_songs(songs), songs


def save_predictions(path: Path, songs: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for song in songs:
            json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in song.items()}, f, allow_nan=False)
            f.write("\n")


def run_experiment(*, manifest_path: Path, split_path: Path, targets_path: Path,
                   output: Path, variant: str, cache_path: Path | None = None,
                   seed: int = SEEDS[0], epochs: int = 50, patience: int = 10,
                   batch_size: int = 16, device: str = "cuda", lr: float = 3e-4,
                   hidden_dim: int = 256, use_timesnet: bool = False,
                   raw_mert: bool = False, mode: str = "development") -> dict:
    if mode not in ("development", "formal") or epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("invalid experiment settings")
    if output.exists():
        raise FileExistsError("run output is immutable; choose a new run directory")
    if seed not in SEEDS:
        raise ValueError("seed outside the registered S5 set")
    repo = Path(__file__).resolve().parents[3]
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if mode == "formal" and git_status.strip():
        raise ValueError("formal experiments require a clean versioned workspace")
    if variant != "constant" and cache_path is None:
        raise ValueError("acoustic training requires full new feature cache")
    manifest, split, rows = load_rows(manifest_path, split_path, targets_path, cache_path)
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if (len(train), len(validation)) != (1395, 174):
        raise ValueError("unexpected train/validation population")
    output.mkdir(parents=True)
    spec = {"dataset_manifest_sha256": manifest["manifest_sha256"], "split_sha256": split["split_sha256"],
            "feature_manifest_file_sha256": file_sha256(cache_path) if cache_path else None,
            "variant": variant, "seed": seed, "epochs": epochs, "patience": patience,
            "batch_size": batch_size, "device": device, "lr": lr, "weight_decay": 1e-4,
            "hidden_dim": hidden_dim, "use_timesnet": use_timesnet, "raw_mert": raw_mert or use_timesnet,
            "mode": mode, "paper_eligible": mode == "formal", "evaluation_role": "validation",
            "loss": "song-dimension-equal MSE + 0.05 adjacent smoothness + optional 0.01 normalized entropy floor(0.5)",
            "selection": "maximum validation macro (CCC_V+CCC_A)/2; earliest exact tie",
            "loaded_split_roles": ["train", "validation"], "test_accessed_for_evaluation": False,
            "test_features_opened": False, "test_targets_decoded": False,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "git_status": git_status, "code_sha256": file_sha256(Path(__file__))}
    _write(output / "run_spec.json", spec)
    started = time.monotonic()
    try:
        if variant == "constant":
            value = fit_constant(train)
            songs = stitch_song_predictions([dict(r, prediction=np.broadcast_to(value, (60, 2))) for r in validation])
            metrics = evaluate_songs(songs)
            _write(output / "constant.json", {"value": value.tolist(), "fit_role": "train", "song_count": 1395})
            best_epoch, training_started = None, False
        else:
            from sc_mv_dmer.models import AcousticModel
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            target_device = torch.device(device)
            def normalized_rows():
                for r in train:
                    with np.load(r["path"], allow_pickle=False) as a:
                        yield {"split": "train", "audio_mask": a["audio_mask"],
                               "views": {k: a[k] for k in VIEW_DIMS}}
            normalizer = TrainNormalizer.fit(normalized_rows())
            _write(output / "normalization.json", normalizer.to_dict() | {"split_sha256": split["split_sha256"]})
            train_data = WindowDataset(train, normalizer, raw_mert or use_timesnet)
            validation_data = WindowDataset(validation, normalizer, raw_mert or use_timesnet)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
            val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
            model = AcousticModel(variant=variant, hidden_dim=hidden_dim, use_timesnet=use_timesnet).to(target_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            best_score, best_epoch, stale = -float("inf"), 0, 0
            for epoch in range(1, epochs + 1):
                loss = train_epoch(model, train_loader, optimizer, target_device)
                metrics, songs = predict(model, val_loader, target_device)
                score = metrics["selection_score"]
                if score is None or not math.isfinite(score):
                    raise ValueError("invalid validation selection score")
                with (output / "curve.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch, "train_loss": loss, "validation_score": score,
                                        "macro": metrics["macro"], "elapsed_seconds": time.monotonic() - started}) + "\n")
                print(json.dumps({"epoch": epoch, "loss": loss, "validation_ccc": score}), flush=True)
                checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "spec": spec}
                torch.save(checkpoint, output / "last.pt")
                if score > best_score:
                    best_score, best_epoch, stale = score, epoch, 0
                    torch.save(checkpoint, output / "best.pt")
                else:
                    stale += 1
                if stale >= patience: break
            model.load_state_dict(torch.load(output / "best.pt", map_location=target_device, weights_only=False)["state_dict"])
            metrics, songs = predict(model, val_loader, target_device)
            training_started = True
        _write(output / "validation_metrics.json", metrics)
        save_predictions(output / "validation_predictions.jsonl", songs)
        final = {"status": "SUCCEEDED", "training_started": training_started, "best_epoch": best_epoch,
                 "elapsed_seconds": time.monotonic() - started, "paper_eligible": spec["paper_eligible"],
                 "test_evaluated": False, "metrics_file_sha256": file_sha256(output / "validation_metrics.json")}
        _write(output / "final.json", final)
        file_hashes = {path.name: file_sha256(path) for path in sorted(output.iterdir())
                       if path.is_file() and path.name != "run_manifest.json"}
        _write(output / "run_manifest.json", {
            "schema_version": "1.0", "status": final["status"], "run_spec": spec,
            "population": {"dataset_songs_registered": len(manifest["records"]),
                           "train_songs_loaded": len(train), "validation_songs_loaded": len(validation),
                           "test_songs_loaded": 0, "long_test_songs_loaded": 0},
            "access_policy": {"loaded_split_roles": ["train", "validation"],
                              "test_targets_decoded": False, "test_features_opened": False,
                              "test_evaluated": False},
            "outputs_sha256": file_hashes})
        return final
    except BaseException as exc:
        failed = {"status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                  "error": str(exc), "elapsed_seconds": time.monotonic() - started}
        _write(output / "final.json", failed)
        _write(output / "run_manifest.json", {"schema_version": "1.0", **failed, "run_spec": spec,
                                                "loaded_split_roles": ["train", "validation"],
                                                "test_accessed_for_evaluation": False})
        raise
