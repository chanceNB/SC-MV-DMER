"""Read-only Step 2/RG-02 preflight for DEAM-PDMER-CLEAN-v2."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sc_mv_dmer.foundation.canonical import sha256_canonical
from sc_mv_dmer.foundation.identity import stable_id


DATASET_MANIFEST_ID = "DEAM-PDMER-CLEAN-v2"
DATASET_VERSION = "deam-pdmer-clean-v2"
VALIDATION_POLICY = "UNDEFINED_PENDING_SEPARATE_FREEZE"
DATA_ROOT_CONFIG_KEY = "SC_MV_DMER_DATA_ROOT"
EXPECTED_SHORT_KEYS = ("1174", "1200", "1273", "1493", "1789")
FORBIDDEN_OPERATIONS = ("TRIM", "PADDING", "INTERPOLATION", "RESHAPE", "ADAPTIVE_POOLING")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_internal_checksum(payload: dict[str, Any], field: str) -> bool:
    observed = payload.get(field)
    if not isinstance(observed, str):
        return False
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return sha256_canonical(unsigned) == observed


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _check_raw_audio(data_root: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    import soundfile as sf

    durations: list[dict[str, Any]] = []
    failures: list[str] = []
    seen_paths: set[str] = set()
    for record in manifest["records"]:
        audio = record["source_audio"]
        relative = audio["source_relative_path"]
        if relative in seen_paths:
            continue
        seen_paths.add(relative)
        path = Path(data_root) / Path(relative)
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        info = sf.info(str(path))
        duration = float(info.duration)
        expected_hash = audio.get("source_sha256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            failures.append(f"checksum:{relative}")
        durations.append({"logical_song_key": record["logical_song_key"], "population_role": record.get("population_role"), "status": record.get("status"), "duration_seconds": duration, "exact_45s_eligible": duration >= 45.0, "source_relative_path": relative, "source_sha256": actual_hash})
    return durations, failures


def _check_processed_artifacts(repo_manifest: dict[str, Any], processed_root: Path) -> tuple[dict[str, Any], list[str]]:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name, expected in repo_manifest["artifacts"].items():
        path = Path(processed_root) / Path(expected["logical_path"]).relative_to("processed/deam-pdmer-clean-v2")
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        checks[name] = {"logical_path": expected["logical_path"], "exists": exists, "expected_sha256": expected["file_sha256"], "actual_sha256": actual_hash, "record_count": expected["record_count"]}
        if not exists:
            failures.append(f"missing_artifact:{name}")
        elif actual_hash != expected["file_sha256"]:
            failures.append(f"artifact_checksum:{name}")
    return checks, failures


def evaluate_step2_rg02_v2(*, repo_root: Path, data_root: Path, manifest_path: Path | None = None, audit_path: Path | None = None, processed_root: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root)
    data_root = Path(data_root)
    manifest_path = manifest_path or repo_root / "manifests/datasets/deam-pdmer-clean-v2.json"
    audit_path = audit_path or repo_root / "reports/data/deam-pdmer-clean-v2-audit.json"
    manifest = _load(manifest_path)
    audit = _load(audit_path)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    checks["dataset_manifest_identity"] = manifest.get("manifest_id") == DATASET_MANIFEST_ID and manifest.get("dataset_version") == DATASET_VERSION
    checks["dataset_manifest_internal_checksum"] = _verify_internal_checksum(manifest, "manifest_sha256")
    checks["audit_identity"] = audit.get("audit_id") == "DEAM-PDMER-CLEANING-AUDIT-v2" and audit.get("dataset_manifest_id") == DATASET_MANIFEST_ID
    checks["audit_internal_checksum"] = _verify_internal_checksum(audit, "audit_sha256")
    checks["manifest_file_checksum"] = sha256_file(manifest_path)
    checks["audit_file_checksum"] = sha256_file(audit_path)
    if not checks["dataset_manifest_identity"] or not checks["dataset_manifest_internal_checksum"]:
        failures.append("dataset_manifest_identity_or_checksum")
    if not checks["audit_identity"] or not checks["audit_internal_checksum"]:
        failures.append("cleaning_audit_identity_or_checksum")

    population = manifest.get("population", {})
    checks["population"] = {"train_primary": population.get("retained_train_primary_count"), "test_long_song": population.get("retained_test_long_song_count"), "excluded_missing_worker_id": population.get("excluded_primary_missing_worker_id_count"), "retained_records": population.get("retained_record_count")}
    if checks["population"] != {"train_primary": 1000, "test_long_song": 58, "excluded_missing_worker_id": 744, "retained_records": 1058}:
        failures.append("population_counts")
    split_policy = manifest.get("split_policy", {})
    checks["validation_policy"] = split_policy.get("validation")
    if checks["validation_policy"] != VALIDATION_POLICY:
        failures.append("validation_policy")
    checks["annotation_start_ms"] = manifest.get("annotation_policy", {}).get("discard_before_ms")
    checks["pdmer_rule_version"] = manifest.get("annotation_policy", {}).get("pdmer_rule_version")
    checks["dmer_rule_version"] = manifest.get("annotation_policy", {}).get("dmer_rule_version")
    if checks["annotation_start_ms"] != 15000:
        failures.append("annotation_before_15s_not_excluded")
    if checks["pdmer_rule_version"] != "pdmer-worker-meta-task-v2":
        failures.append("pdmer_not_worker_meta_task_v2")

    short_expected = list(EXPECTED_SHORT_KEYS)
    short_declared = manifest.get("audio_policy", {}).get("short_primary_logical_song_keys")
    checks["declared_short_audio_keys"] = short_declared
    if sorted(short_declared or []) != short_expected:
        failures.append("declared_short_audio_set")
    if tuple(manifest.get("audio_policy", {}).get("forbidden_operations", ())) != FORBIDDEN_OPERATIONS:
        failures.append("forbidden_audio_operation_contract")

    durations, audio_failures = _check_raw_audio(data_root, manifest)
    all_short = sorted(item["logical_song_key"] for item in durations if not item["exact_45s_eligible"])
    observed_short = sorted(item["logical_song_key"] for item in durations if not item["exact_45s_eligible"] and item.get("status") == "RETAINED" and item.get("population_role") == "TRAIN_PRIMARY")
    checks["raw_audio"] = {"data_root_config_key": DATA_ROOT_CONFIG_KEY, "source_root_runtime": "external_configured_root", "audio_count": len(durations), "all_short_audio_keys": all_short, "observed_short_audio_keys": observed_short, "source_failures": audio_failures}
    if audio_failures:
        failures.extend(audio_failures)
    if observed_short != short_expected:
        failures.append("observed_short_audio_set")

    processed_root = Path(processed_root) if processed_root is not None else data_root / "processed/deam-pdmer-clean-v2"
    processed_manifest_path = processed_root / "manifest.json"
    checks["processed_manifest"] = {"path": "processed/deam-pdmer-clean-v2/manifest.json", "exists": processed_manifest_path.is_file()}
    if processed_manifest_path.is_file():
        processed_manifest = _load(processed_manifest_path)
        checks["processed_manifest"]["identity_match"] = processed_manifest.get("manifest_id") == DATASET_MANIFEST_ID and processed_manifest.get("manifest_sha256") == manifest.get("manifest_sha256")
        checks["processed_manifest"]["internal_checksum_valid"] = _verify_internal_checksum(processed_manifest, "manifest_sha256")
        if not checks["processed_manifest"]["identity_match"] or not checks["processed_manifest"]["internal_checksum_valid"]:
            failures.append("processed_manifest_lineage")
    else:
        failures.append("missing_processed_manifest")
    artifact_checks, artifact_failures = _check_processed_artifacts(manifest, processed_root)
    checks["processed_artifacts"] = artifact_checks
    failures.extend(artifact_failures)

    if processed_manifest_path.is_file() and not artifact_failures:
        dmer = _jsonl(processed_root / "dmer_targets.jsonl")
        pdmer = _jsonl(processed_root / "pdmer_tasks.jsonl")
        dmer_roles = Counter(row.get("population_role") for row in dmer)
        pdmer_song_roles = Counter(song.get("population_role") for task in pdmer for song in task.get("songs", []))
        pdmer_unique_songs = {(song.get("logical_song_key"), song.get("population_role")) for task in pdmer for song in task.get("songs", [])}
        pdmer_unique_roles = Counter(role for _key, role in pdmer_unique_songs)
        checks["dmer_pdmer_records"] = {"dmer_count": len(dmer), "pdmer_task_count": len(pdmer), "pdmer_song_assignment_count": sum(int(row.get("song_count", 0)) for row in pdmer), "dmer_role_counts": dict(sorted(dmer_roles.items())), "pdmer_song_assignment_role_counts": dict(sorted(pdmer_song_roles.items())), "pdmer_unique_song_role_counts": dict(sorted(pdmer_unique_roles.items())), "dmer_dataset_ids": sorted({row.get("dataset_id") for row in dmer}), "pdmer_dataset_ids": sorted({row.get("dataset_id") for row in pdmer}), "pdmer_task_ids_unique": len({row.get("task_id") for row in pdmer}) == len(pdmer), "pdmer_task_identity_valid": all(row.get("task_id") == stable_id("pdmer_task", (manifest["dataset_id"], row.get("worker_id", ""))) for row in pdmer)}
        if len(dmer) != 1058 or len(pdmer) != 99 or checks["dmer_pdmer_records"]["pdmer_song_assignment_count"] != 10293 or dmer_roles != Counter({"TRAIN_PRIMARY": 1000, "TEST_LONG_SONG": 58}) or pdmer_unique_roles != Counter({"TRAIN_PRIMARY": 1000, "TEST_LONG_SONG": 58}):
            failures.append("dmer_pdmer_counts")
        if not checks["dmer_pdmer_records"]["pdmer_task_ids_unique"] or not checks["dmer_pdmer_records"]["pdmer_task_identity_valid"]:
            failures.append("pdmer_cross_song_worker_identity")
        if any(row.get("annotation_start_ms") != 15000 for row in dmer):
            failures.append("dmer_annotation_start")
        if any(min(view.get("time_ms", [0])) < 15000 for row in dmer for view in row.get("views", {}).values()):
            failures.append("dmer_pre_15s_values")
        if any(min((view or {}).get("time_ms", [15000])) < 15000 for task in pdmer for song in task.get("songs", []) for view in song.get("views", {}).values()):
            failures.append("pdmer_pre_15s_values")
        if any(row.get("dataset_id") != manifest["dataset_id"] for row in dmer + pdmer):
            failures.append("artifact_dataset_id")

    checks["legacy_inputs"] = {"old_split_consumed": False, "old_rg02_evidence_consumed": False, "replacement_split_generated": False}
    checks["feature_materialization"] = {"cache_published": False, "pseudo_label_cache_published": False, "training_artifacts_published": False}
    verdict = "BLOCKED" if failures or observed_short else "PASS"
    evidence = {"schema_version": "1.0", "evidence_id": "E2_STEP2_RG02_DEAM_PDMER_CLEAN_V2_PREFLIGHT-v1", "gate": "RG-02", "verdict": verdict, "dataset_manifest": {"logical_path": "manifests/datasets/deam-pdmer-clean-v2.json", "manifest_id": manifest.get("manifest_id"), "dataset_id": manifest.get("dataset_id"), "internal_sha256": manifest.get("manifest_sha256"), "file_sha256": checks["manifest_file_checksum"]}, "cleaning_audit": {"logical_path": "reports/data/deam-pdmer-clean-v2-audit.json", "audit_id": audit.get("audit_id"), "internal_sha256": audit.get("audit_sha256"), "file_sha256": checks["audit_file_checksum"]}, "lineage": {"decisions": ["DEC-0028", "DEC-0029"], "cleaning_rule_version": manifest.get("cleaning_rule_version"), "dmer_rule_version": checks["dmer_rule_version"], "pdmer_rule_version": checks["pdmer_rule_version"], "annotation_discard_before_ms": checks["annotation_start_ms"], "validation_policy": checks["validation_policy"]}, "data_root": {"config_key": DATA_ROOT_CONFIG_KEY, "runtime_root_external": True, "absolute_path_serialized": False}, "checks": checks, "blockers": sorted(set(failures + (["short_primary_audio_exact_45s"] if observed_short else []))), "forbidden_operations": list(FORBIDDEN_OPERATIONS), "source_artifacts_consumed": ["manifests/datasets/deam-pdmer-clean-v2.json", "reports/data/deam-pdmer-clean-v2-audit.json", "processed/deam-pdmer-clean-v2/manifest.json", "processed/deam-pdmer-clean-v2/dmer_targets.jsonl", "processed/deam-pdmer-clean-v2/pdmer_tasks.jsonl"]}
    evidence["evidence_sha256"] = sha256_canonical(evidence)
    return evidence
