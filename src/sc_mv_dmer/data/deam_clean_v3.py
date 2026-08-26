"""Build the immutable DEAM-PDMER-CLEAN-v3 successor dataset.

This transformation consumes the already-frozen v2 snapshot and removes only
the five retained primary songs that fail the exact-45-second audio contract.
The v2 manifest and processed artifacts are never overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.identity import stable_id


V3_VERSION = "deam-pdmer-clean-v3"
V3_MANIFEST_ID = "DEAM-PDMER-CLEAN-v3"
V2_MANIFEST_ID = "DEAM-PDMER-CLEAN-v2"
EXCLUDED_SHORT_PRIMARY = ("1174", "1200", "1273", "1493", "1789")
TRAIN_PRIMARY_COUNT = 995
TEST_LONG_COUNT = 58
DMER_COUNT = 1053
PDMER_TASK_COUNT = 99


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stable_dataset_id() -> str:
    return stable_id("dataset", ("DEAM", V3_VERSION, "deam-cleaning-workerid-15s-v2", "exclude-short-primary-exact-45s-v1"))


def _song_id(dataset_id: str, key: str) -> str:
    return stable_id("song", (dataset_id, key))


def _sample_id(song_id: str, role: str) -> str:
    boundary = "0-45s" if role == "TRAIN_PRIMARY" else "15s-full-song"
    return stable_id("sample", (song_id, role, boundary))


def _task_id(dataset_id: str, worker_id: str) -> str:
    return stable_id("pdmer_task", (dataset_id, worker_id))


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable v3 artifact: {path}")
    path.write_text(text, encoding="utf-8")


def materialize_v3_from_v2(*, v2_manifest_path: Path, v2_processed_root: Path, output_root: Path, repo_manifest_path: Path, cleaning_audit_path: Path) -> dict[str, Any]:
    v2 = _load(v2_manifest_path)
    if v2.get("manifest_id") != V2_MANIFEST_ID:
        raise ValueError("v3 builder requires the registered DEAM-PDMER-CLEAN-v2 manifest")
    v2_dmer_path = Path(v2_processed_root) / "dmer_targets.jsonl"
    v2_pdmer_path = Path(v2_processed_root) / "pdmer_tasks.jsonl"
    if not v2_dmer_path.is_file() or not v2_pdmer_path.is_file():
        raise FileNotFoundError("v2 processed DMER/PDMER artifacts are required")
    v2_dmer = [json.loads(line) for line in v2_dmer_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    v2_pdmer = [json.loads(line) for line in v2_pdmer_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset_id = _stable_dataset_id()
    records: list[dict[str, Any]] = []
    old_to_new_song: dict[str, str] = {}
    for record in v2["records"]:
        key = record["logical_song_key"]
        role = record["population_role"]
        song_id = _song_id(dataset_id, key)
        sample_id = _sample_id(song_id, role)
        old_to_new_song[key] = song_id
        item = json.loads(json.dumps(record))
        item.update({"dataset_id": dataset_id, "song_id": song_id, "sample_id": sample_id})
        if role == "TRAIN_PRIMARY" and key in EXCLUDED_SHORT_PRIMARY:
            item["status"] = "EXCLUDED"
            item["exclusion_reason"] = "SHORT_AUDIO_EXACT_45S"
        records.append(item)

    retained_keys = {record["logical_song_key"] for record in records if record["status"] == "RETAINED"}
    dmer: list[dict[str, Any]] = []
    for row in v2_dmer:
        key = row["logical_song_key"]
        if key not in retained_keys:
            continue
        item = json.loads(json.dumps(row))
        item.update({"dataset_id": dataset_id, "song_id": old_to_new_song[key], "sample_id": _sample_id(old_to_new_song[key], item["population_role"])})
        dmer.append(item)

    pdmer: list[dict[str, Any]] = []
    assignment_count = 0
    for task in v2_pdmer:
        songs: list[dict[str, Any]] = []
        for song in task["songs"]:
            key = song["logical_song_key"]
            if key not in retained_keys:
                continue
            item = json.loads(json.dumps(song))
            item.update({"song_id": old_to_new_song[key], "sample_id": _sample_id(old_to_new_song[key], item["population_role"])})
            songs.append(item)
        if not songs:
            continue
        assignment_count += len(songs)
        pdmer.append({"task_id": _task_id(dataset_id, task["worker_id"]), "dataset_id": dataset_id, "worker_id": task["worker_id"], "target_rule_version": task["target_rule_version"], "song_count": len(songs), "songs": sorted(songs, key=lambda item: int(item["logical_song_key"]))})

    train_keys = [record["logical_song_key"] for record in records if record["population_role"] == "TRAIN_PRIMARY" and record["status"] == "RETAINED"]
    test_keys = [record["logical_song_key"] for record in records if record["population_role"] == "TEST_LONG_SONG"]
    excluded_missing = [record["logical_song_key"] for record in records if record["exclusion_reason"] == "MISSING_WORKER_ID"]
    short_records = [record for record in records if record["logical_song_key"] in EXCLUDED_SHORT_PRIMARY]
    dmer_path = Path(output_root) / "dmer_targets.jsonl"
    pdmer_path = Path(output_root) / "pdmer_tasks.jsonl"
    _write_once(dmer_path, "".join(canonical_json(item) + "\n" for item in dmer))
    _write_once(pdmer_path, "".join(canonical_json(item) + "\n" for item in pdmer))
    artifacts = {
        "dmer_targets": {"logical_path": "processed/deam-pdmer-clean-v3/dmer_targets.jsonl", "file_sha256": _sha256_file(dmer_path), "record_count": len(dmer)},
        "pdmer_tasks": {"logical_path": "processed/deam-pdmer-clean-v3/pdmer_tasks.jsonl", "file_sha256": _sha256_file(pdmer_path), "record_count": len(pdmer), "song_assignment_count": assignment_count},
    }
    unsigned = {
        "schema_version": "1.0", "manifest_id": V3_MANIFEST_ID, "supersedes_manifest_id": V2_MANIFEST_ID, "supersedes_manifest_sha256": v2["manifest_sha256"], "dataset_name": "DEAM", "dataset_version": V3_VERSION, "dataset_id": dataset_id, "cleaning_rule_version": "deam-cleaning-workerid-15s-v2", "short_audio_exclusion_policy_version": "exclude-short-primary-exact-45s-v1", "source_inventory_logical_path": v2["source_inventory_logical_path"], "source_inventory_sha256": v2["source_inventory_sha256"], "data_root_config_key": "SC_MV_DMER_DATA_ROOT", "raw_data_policy": "READ_ONLY",
        "population": {"source_audio_count": 1802, "source_primary_count": 1744, "source_long_song_count": 58, "retained_train_primary_count": TRAIN_PRIMARY_COUNT, "retained_test_long_song_count": TEST_LONG_COUNT, "excluded_primary_missing_worker_id_count": len(excluded_missing), "excluded_primary_short_audio_count": len(short_records), "retained_record_count": len(dmer)},
        "split_policy": {"train": "995 exact-45s retained primary songs", "test": "all 58 long songs", "validation": "UNDEFINED_PENDING_SEPARATE_FREEZE", "old_split_status": "DEAM-PRIMARY-v1 and primary-song-80-10-10-v1 are historical/superseded and are not consumed"},
        "annotation_policy": v2["annotation_policy"],
        "audio_policy": {"primary_excerpt_seconds": 45, "short_primary_policy": "EXCLUDE_SHORT_PRIMARY_FROM_TRAINING_POPULATION", "excluded_short_primary_logical_song_keys": list(EXCLUDED_SHORT_PRIMARY), "forbidden_operations": ["TRIM", "PADDING", "INTERPOLATION", "RESHAPE", "ADAPTIVE_POOLING"]},
        "retained_train_primary_song_keys": train_keys, "retained_test_long_song_keys": test_keys, "excluded_primary_song_keys": excluded_missing, "records": records, "artifacts": artifacts,
        "audit_summary": {"worker_set_mismatch_song_count": v2["audit_summary"]["worker_set_mismatch_song_count"], "duplicate_worker_detail_count": v2["audit_summary"]["duplicate_worker_detail_count"], "short_audio_count_in_retained_primary": 0, "excluded_short_audio_count": len(short_records), "feature_materialization_verdict": "READY_FOR_EXACT_45S_PREFLIGHT"},
    }
    manifest = unsigned | {"manifest_sha256": sha256_canonical(unsigned)}
    _write_once(repo_manifest_path, canonical_json(manifest) + "\n")
    audit_unsigned = {"schema_version": "1.0", "audit_id": "DEAM-PDMER-CLEANING-AUDIT-v3", "dataset_manifest_id": V3_MANIFEST_ID, "dataset_manifest_sha256": manifest["manifest_sha256"], "supersedes_audit_id": "DEAM-PDMER-CLEANING-AUDIT-v2", "supersedes_audit_sha256": "f27366b020ba1a0282f8c19293f03021dd8d0c7cf8f6f7135bb9ee4807aa38e4", "decision_id": "DEC-0030", "cleaning_rule_version": "deam-cleaning-workerid-15s-v2", "short_audio_exclusion_policy_version": "exclude-short-primary-exact-45s-v1", "source_inventory_logical_path": v2["source_inventory_logical_path"], "source_inventory_sha256": v2["source_inventory_sha256"], "counts": unsigned["population"], "excluded_primary_missing_worker_id": excluded_missing, "excluded_primary_short_audio": [{"logical_song_key": item["logical_song_key"], "duration_seconds": item["source_audio"]["duration_seconds"], "source_sha256": item["source_audio"]["source_sha256"]} for item in short_records], "artifact_checksums": artifacts, "annotation_contract": {"discard_before_ms": 15000, "dmer_rule_version": "dmer-all-valid-rows-mean-v1", "pdmer_rule_version": "pdmer-worker-meta-task-v2", "pdmer_task_granularity": "one cross-song meta-task per WorkerId"}, "validation_policy": "UNDEFINED_PENDING_SEPARATE_FREEZE", "feature_materialization": {"status": "READY_FOR_EXACT_45S_PREFLIGHT", "no_cache_published": True, "reason": "all retained primary audio is exact-45s eligible"}}
    audit = audit_unsigned | {"audit_sha256": sha256_canonical(audit_unsigned)}
    _write_once(cleaning_audit_path, canonical_json(audit) + "\n")
    processed_manifest_path = Path(output_root) / "manifest.json"
    _write_once(processed_manifest_path, canonical_json(manifest) + "\n")
    return {"manifest": manifest, "audit": audit, "dmer_count": len(dmer), "pdmer_task_count": len(pdmer), "pdmer_song_assignment_count": assignment_count, "short_excluded": list(EXCLUDED_SHORT_PRIMARY)}


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
