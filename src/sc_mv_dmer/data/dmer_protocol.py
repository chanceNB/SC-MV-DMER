"""Full-population DMER split and absolute-time window contract.

Windows contain 45 seconds of input and predict their final 30 seconds at 2 Hz.
Full-song windows advance 30 seconds, so output times never overlap. Labels do
not influence window membership, and valence/arousal retain separate masks.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from sc_mv_dmer.data.splits import SPLIT_CONTRACT_ID, TRAIN_COUNT, VALIDATION_COUNT, TEST_COUNT
from sc_mv_dmer.foundation.canonical import sha256_canonical


SPLIT_MANIFEST_ID = "deam-dmer-full-v1-primary-song-80-10-10-v1"
WINDOW_CONTRACT_ID = "DEAM-DMER-45S-CONTEXT-30S-OUTPUT/v1"
DIMENSIONS = ("valence", "arousal")


def _verify_manifest(manifest: dict) -> None:
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if sha256_canonical(unsigned) != manifest.get("manifest_sha256"):
        raise ValueError("dataset manifest checksum mismatch")
    if manifest.get("manifest_id") != "DEAM-DMER-FULL-v1":
        raise ValueError("expected DEAM-DMER-FULL-v1 manifest")
    records = manifest.get("records", [])
    if len(records) != 1802:
        raise ValueError("full DMER requires all 1802 songs")
    for field in ("song_id", "sample_id", "logical_song_key"):
        if len({r[field] for r in records}) != 1802:
            raise ValueError(f"duplicate {field} in dataset manifest")
    if sum(r["population_kind"] == "SHORT_CLIP" for r in records) != 1744 or sum(r["population_kind"] == "FULL_SONG" for r in records) != 58:
        raise ValueError("full DMER requires 1744 short clips and 58 full songs")
    targets = manifest["artifacts"]["dmer_targets"]
    digest = targets.get("file_sha256", "")
    if targets.get("record_count") != 1802 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid target artifact registration")


def build_split(manifest: dict) -> dict:
    """Freeze the approved song split; the historical rank contract is reused."""
    _verify_manifest(manifest)
    ranked = sorted(
        (sha256_canonical({"dataset_id": manifest["dataset_id"], "song_id": r["song_id"], "split_contract_id": SPLIT_CONTRACT_ID}), r["song_id"], r)
        for r in manifest["records"] if r["population_kind"] == "SHORT_CLIP"
    )
    records = []
    for index, (digest, _, record) in enumerate(ranked):
        name = "train" if index < TRAIN_COUNT else "validation" if index < TRAIN_COUNT + VALIDATION_COUNT else "test"
        records.append({k: record[k] for k in ("song_id", "sample_id", "logical_song_key")} | {"split": name, "rank_digest": digest, "rank_index": index})
    for record in sorted((r for r in manifest["records"] if r["population_kind"] == "FULL_SONG"), key=lambda r: int(r["logical_song_key"])):
        records.append({k: record[k] for k in ("song_id", "sample_id", "logical_song_key")} | {"split": "long_test"})
    result = {
        "schema_version": "1.0", "split_manifest_id": SPLIT_MANIFEST_ID,
        "split_contract_id": SPLIT_CONTRACT_ID, "dataset_manifest_id": manifest["manifest_id"],
        "dataset_id": manifest["dataset_id"], "dataset_manifest_sha256": manifest["manifest_sha256"],
        "target_file_sha256": manifest["artifacts"]["dmer_targets"]["file_sha256"],
        "primary_population_count": 1744, "train_count": TRAIN_COUNT,
        "validation_count": VALIDATION_COUNT, "test_count": TEST_COUNT, "long_test_count": 58,
        "records": records,
        "ranking": "SHA256(canonical({dataset_id,song_id,split_contract_id})); digest ascending then song_id ascending; no random seed",
    }
    for name in ("train", "validation", "test", "long_test"):
        result[f"{name}_song_ids"] = [r["song_id"] for r in records if r["split"] == name]
    result["split_sha256"] = sha256_canonical(result)
    return result


def verify_split(manifest: dict, split: dict) -> dict:
    """Check source integrity, immutable identities, quotas, ranking and coverage."""
    _verify_manifest(manifest)
    unsigned = {k: v for k, v in split.items() if k != "split_sha256"}
    if sha256_canonical(unsigned) != split.get("split_sha256"):
        raise ValueError("split checksum mismatch")
    if split != build_split(manifest):
        raise ValueError("split membership, source binding or contract mismatch")
    return {"verdict": "PASS", "song_count": 1802, "train_count": TRAIN_COUNT, "validation_count": VALIDATION_COUNT, "test_count": TEST_COUNT, "long_test_count": 58, "song_overlap_count": 0}


def verify_target_file(manifest: dict, path: Path) -> str:
    """Check actual target bytes before any protocol materialization or loading."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != manifest["artifacts"]["dmer_targets"]["file_sha256"]:
        raise ValueError("target file checksum mismatch")
    return digest


def make_windows(record: dict) -> list[dict]:
    """Generate windows from audio duration only, including partially covered bins."""
    duration_ms = float(record["audio_duration_seconds"]) * 1000.0
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("audio duration must be positive and finite")
    kind = record["population_kind"]
    if kind == "SHORT_CLIP":
        starts = [0]
    elif kind == "FULL_SONG":
        starts = []
        start = 0
        while start + 15000 < duration_ms:
            starts.append(start)
            start += 30000
    else:
        raise ValueError(f"unknown population kind: {kind}")
    windows = []
    for start in starts:
        times = list(range(start, start + 45000, 500))
        coverage = [max(0.0, min(500.0, duration_ms - t)) / 500.0 for t in times]
        mask = [value > 0 for value in coverage]
        windows.append({
            "window_id": f"{record['logical_song_key']}@{start:09d}",
            "song_id": record["song_id"], "logical_song_key": record["logical_song_key"],
            "start_ms": start, "input_time_ms": times, "output_time_ms": times[30:],
            "audio_coverage": coverage, "audio_mask": mask, "output_audio_mask": mask[30:],
        })
    return windows


def align_targets(target: dict, window: dict) -> tuple[np.ndarray, np.ndarray]:
    """Select native labels at output times; missing values stay unsupervised."""
    for key in ("song_id", "logical_song_key"):
        if target.get(key) != window.get(key):
            raise ValueError(f"target/window {key} mismatch")
    times = window["output_time_ms"]
    audio_mask = window["output_audio_mask"]
    if len(times) != 60 or len(audio_mask) != 60:
        raise ValueError("window requires 60 output timestamps and masks")
    values = np.zeros((60, 2), dtype=np.float32)
    mask = np.zeros((60, 2), dtype=np.bool_)
    for dim, name in enumerate(DIMENSIONS):
        view = target["views"][name]
        native_times, native_values, native_mask = view["time_ms"], view["values"], view["supervision_mask"]
        if not (len(native_times) == len(native_values) == len(native_mask)) or len(set(native_times)) != len(native_times):
            raise ValueError(f"invalid {name} target time/value/mask lengths or duplicate time")
        lookup = {t: (value, valid) for t, value, valid in zip(native_times, native_values, native_mask)}
        for index, time_ms in enumerate(times):
            value, valid = lookup.get(time_ms, (None, False))
            if valid:
                if value is None or not math.isfinite(value):
                    raise ValueError(f"nonfinite valid {name} target")
                if audio_mask[index]:
                    values[index, dim] = value
                    mask[index, dim] = True
    return values, mask
