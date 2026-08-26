"""Read-only RG-02 data preflight for DEAM-PDMER-CLEAN-v3."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from sc_mv_dmer.data.step2_preflight import _jsonl, _load, _verify_internal_checksum, sha256_file
from sc_mv_dmer.foundation.canonical import sha256_canonical
from sc_mv_dmer.foundation.identity import stable_id


V3_MANIFEST_ID = "DEAM-PDMER-CLEAN-v3"
V3_VERSION = "deam-pdmer-clean-v3"
V3_TRAIN_COUNT = 995
V3_TEST_COUNT = 58
V3_DMER_COUNT = 1053
V3_PDMER_TASK_COUNT = 99
V3_PDMER_ASSIGNMENTS = 10243
V3_SHORT_EXCLUDED = ["1174", "1200", "1273", "1493", "1789"]
V3_VALIDATION_POLICY = "UNDEFINED_PENDING_SEPARATE_FREEZE"


def _check_audio(data_root: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    import soundfile as sf

    durations: list[dict[str, Any]] = []
    failures: list[str] = []
    seen: set[str] = set()
    for record in manifest.get("records", []):
        relative = record["source_audio"]["source_relative_path"]
        if relative in seen:
            continue
        seen.add(relative)
        path = Path(data_root) / Path(relative)
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual = sha256_file(path)
        if actual != record["source_audio"]["source_sha256"]:
            failures.append(f"checksum:{relative}")
        info = sf.info(str(path))
        durations.append({"logical_song_key": record["logical_song_key"], "population_role": record["population_role"], "status": record["status"], "duration_seconds": float(info.duration), "exact_45s_eligible": float(info.duration) >= 45.0, "source_sha256": actual})
    return durations, failures


def evaluate_step2_rg02_v3(*, repo_root: Path, data_root: Path, manifest_path: Path | None = None, audit_path: Path | None = None, processed_root: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root)
    data_root = Path(data_root)
    manifest_path = manifest_path or repo_root / "manifests/datasets/deam-pdmer-clean-v3.json"
    audit_path = audit_path or repo_root / "reports/data/deam-pdmer-clean-v3-audit.json"
    processed_root = Path(processed_root) if processed_root is not None else data_root / "processed/deam-pdmer-clean-v3"
    manifest = _load(manifest_path)
    audit = _load(audit_path)
    failures: list[str] = []
    checks: dict[str, Any] = {}
    checks["dataset_manifest_identity"] = manifest.get("manifest_id") == V3_MANIFEST_ID and manifest.get("dataset_version") == V3_VERSION
    checks["dataset_manifest_internal_checksum"] = _verify_internal_checksum(manifest, "manifest_sha256")
    checks["dataset_manifest_file_sha256"] = sha256_file(manifest_path)
    checks["cleaning_audit_identity"] = audit.get("audit_id") == "DEAM-PDMER-CLEANING-AUDIT-v3" and audit.get("dataset_manifest_id") == V3_MANIFEST_ID
    checks["cleaning_audit_internal_checksum"] = _verify_internal_checksum(audit, "audit_sha256")
    checks["cleaning_audit_file_sha256"] = sha256_file(audit_path)
    if not all((checks["dataset_manifest_identity"], checks["dataset_manifest_internal_checksum"], checks["cleaning_audit_identity"], checks["cleaning_audit_internal_checksum"])):
        failures.append("manifest_or_cleaning_audit_identity_checksum")

    population = manifest.get("population", {})
    checks["population"] = population
    expected_population = {"source_audio_count": 1802, "source_primary_count": 1744, "source_long_song_count": 58, "retained_train_primary_count": V3_TRAIN_COUNT, "retained_test_long_song_count": V3_TEST_COUNT, "excluded_primary_missing_worker_id_count": 744, "excluded_primary_short_audio_count": 5, "retained_record_count": V3_DMER_COUNT}
    if population != expected_population:
        failures.append("population_counts")
    records = manifest.get("records", [])
    primary_records = [record for record in records if record.get("population_role") == "TRAIN_PRIMARY"]
    long_records = [record for record in records if record.get("population_role") == "TEST_LONG_SONG"]
    checks["manifest_coverage"] = {"record_count": len(records), "unique_logical_song_keys": len({record.get("logical_song_key") for record in records}), "unique_source_paths": len({record.get("source_audio", {}).get("source_relative_path") for record in records}), "primary_source_count": len(primary_records), "long_source_count": len(long_records), "retained_train_count": sum(record.get("status") == "RETAINED" for record in primary_records), "retained_test_count": sum(record.get("status") == "RETAINED" for record in long_records)}
    if checks["manifest_coverage"] != {"record_count": 1802, "unique_logical_song_keys": 1802, "unique_source_paths": 1802, "primary_source_count": 1744, "long_source_count": 58, "retained_train_count": V3_TRAIN_COUNT, "retained_test_count": V3_TEST_COUNT}:
        failures.append("manifest_full_coverage")
    checks["validation_policy"] = manifest.get("split_policy", {}).get("validation")
    if checks["validation_policy"] != V3_VALIDATION_POLICY:
        failures.append("validation_policy")
    checks["annotation_contract"] = {"discard_before_ms": manifest.get("annotation_policy", {}).get("discard_before_ms"), "dmer_rule_version": manifest.get("annotation_policy", {}).get("dmer_rule_version"), "pdmer_rule_version": manifest.get("annotation_policy", {}).get("pdmer_rule_version")}
    if checks["annotation_contract"] != {"discard_before_ms": 15000, "dmer_rule_version": "dmer-all-valid-rows-mean-v1", "pdmer_rule_version": "pdmer-worker-meta-task-v2"}:
        failures.append("annotation_contract")
    checks["short_audio_exclusion"] = {"expected_keys": V3_SHORT_EXCLUDED, "manifest_keys": manifest.get("audio_policy", {}).get("excluded_short_primary_logical_song_keys"), "reason": "SHORT_AUDIO_EXACT_45S", "operations_forbidden": manifest.get("audio_policy", {}).get("forbidden_operations")}
    if sorted(checks["short_audio_exclusion"]["manifest_keys"] or []) != V3_SHORT_EXCLUDED:
        failures.append("short_audio_exclusion_set")

    durations, audio_failures = _check_audio(data_root, manifest)
    retained_short = sorted(item["logical_song_key"] for item in durations if item["status"] == "RETAINED" and item["population_role"] == "TRAIN_PRIMARY" and not item["exact_45s_eligible"])
    excluded_short = sorted(item["logical_song_key"] for item in durations if item["status"] == "EXCLUDED" and item["logical_song_key"] in V3_SHORT_EXCLUDED and not item["exact_45s_eligible"])
    checks["raw_audio"] = {"data_root_config_key": "SC_MV_DMER_DATA_ROOT", "runtime_root_external": True, "audio_count": len(durations), "retained_train_short_keys": retained_short, "excluded_short_keys": excluded_short, "source_failures": audio_failures}
    failures.extend(audio_failures)
    if retained_short or excluded_short != V3_SHORT_EXCLUDED:
        failures.append("exact_45s_qualification")

    processed_manifest_path = processed_root / "manifest.json"
    processed_manifest = _load(processed_manifest_path) if processed_manifest_path.is_file() else None
    checks["processed_manifest"] = {"logical_path": "processed/deam-pdmer-clean-v3/manifest.json", "exists": processed_manifest is not None, "identity_match": bool(processed_manifest and processed_manifest.get("manifest_id") == V3_MANIFEST_ID and processed_manifest.get("manifest_sha256") == manifest.get("manifest_sha256")), "internal_checksum_valid": bool(processed_manifest and _verify_internal_checksum(processed_manifest, "manifest_sha256"))}
    if not checks["processed_manifest"]["exists"] or not checks["processed_manifest"]["identity_match"] or not checks["processed_manifest"]["internal_checksum_valid"]:
        failures.append("processed_manifest_lineage")

    artifact_checks: dict[str, Any] = {}
    for name, expected in manifest.get("artifacts", {}).items():
        path = processed_root / Path(expected["logical_path"]).name
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        artifact_checks[name] = {"logical_path": expected["logical_path"], "exists": exists, "expected_sha256": expected["file_sha256"], "actual_sha256": actual}
        if not exists or actual != expected["file_sha256"]:
            failures.append(f"artifact_{name}_checksum")

    if not any(key.startswith("artifact_") for key in failures):
        dmer = _jsonl(processed_root / "dmer_targets.jsonl")
        pdmer = _jsonl(processed_root / "pdmer_tasks.jsonl")
        dmer_roles = Counter(row.get("population_role") for row in dmer)
        assignments = [song for task in pdmer for song in task.get("songs", [])]
        unique_roles = Counter((song.get("logical_song_key"), song.get("population_role")) for song in assignments)
        unique_role_counts = Counter(role for (_key, role) in unique_roles)
        checks["dmer_pdmer"] = {"dmer_count": len(dmer), "dmer_role_counts": dict(sorted(dmer_roles.items())), "pdmer_task_count": len(pdmer), "pdmer_song_assignment_count": len(assignments), "pdmer_unique_song_role_counts": dict(sorted(unique_role_counts.items())), "dataset_ids_match": all(row.get("dataset_id") == manifest.get("dataset_id") for row in dmer + pdmer), "pdmer_task_ids_unique": len({row.get("task_id") for row in pdmer}) == len(pdmer), "pdmer_task_identity_valid": all(row.get("task_id") == stable_id("pdmer_task", (manifest["dataset_id"], row.get("worker_id", ""))) for row in pdmer)}
        if len(dmer) != V3_DMER_COUNT or dmer_roles != Counter({"TRAIN_PRIMARY": V3_TRAIN_COUNT, "TEST_LONG_SONG": V3_TEST_COUNT}) or len(pdmer) != V3_PDMER_TASK_COUNT or len(assignments) != V3_PDMER_ASSIGNMENTS or unique_role_counts != Counter({"TRAIN_PRIMARY": V3_TRAIN_COUNT, "TEST_LONG_SONG": V3_TEST_COUNT}) or not checks["dmer_pdmer"]["dataset_ids_match"] or not checks["dmer_pdmer"]["pdmer_task_ids_unique"] or not checks["dmer_pdmer"]["pdmer_task_identity_valid"]:
            failures.append("dmer_pdmer_coverage_or_identity")
        if any(row.get("logical_song_key") in V3_SHORT_EXCLUDED for row in dmer + assignments):
            failures.append("excluded_short_song_in_artifacts")
        if any(row.get("annotation_start_ms") != 15000 for row in dmer + assignments):
            failures.append("annotation_before_15s")
        if any(view is not None and min(view.get("time_ms", [15000])) < 15000 for row in dmer for view in row.get("views", {}).values()):
            failures.append("dmer_pre_15s_values")
        if any(view is not None and min(view.get("time_ms", [15000])) < 15000 for song in assignments for view in song.get("views", {}).values()):
            failures.append("pdmer_pre_15s_values")
    checks["processed_artifacts"] = artifact_checks
    checks["legacy_inputs"] = {"old_split_consumed": False, "old_rg02_evidence_consumed": False, "replacement_split_generated": False}
    checks["feature_cache_binding"] = {"cache_id": "deam-four-view-v3", "cache_version": "v3", "dataset_manifest_logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "dataset_manifest_sha256": manifest.get("manifest_sha256"), "published": False}
    checks["pseudo_label_cache_binding"] = {"bundle_id": "deam-pseudo-label-v3", "dataset_manifest_sha256": manifest.get("manifest_sha256"), "published": False}
    checks["training_artifacts"] = {"published": False, "architecture_identity": "A1-v1", "loss_definition_unchanged": True}
    verdict = "PASS" if not failures else "BLOCKED"
    evidence = {"schema_version": "1.0", "evidence_id": "E2_STEP2_RG02_DEAM_PDMER_CLEAN_V3_PREFLIGHT-v1", "gate": "RG-02", "scope": "DATA_PREFLIGHT_ONLY", "verdict": verdict, "dataset_manifest": {"logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "manifest_id": manifest.get("manifest_id"), "dataset_id": manifest.get("dataset_id"), "internal_sha256": manifest.get("manifest_sha256"), "file_sha256": checks["dataset_manifest_file_sha256"]}, "cleaning_audit": {"logical_path": "reports/data/deam-pdmer-clean-v3-audit.json", "audit_id": audit.get("audit_id"), "internal_sha256": audit.get("audit_sha256"), "file_sha256": checks["cleaning_audit_file_sha256"]}, "lineage": {"decisions": ["DEC-0028", "DEC-0029", "DEC-0030"], "supersedes_dataset_manifest_id": "DEAM-PDMER-CLEAN-v2", "cleaning_rule_version": manifest.get("cleaning_rule_version"), "short_audio_exclusion_policy_version": manifest.get("short_audio_exclusion_policy_version"), "validation_policy": checks["validation_policy"], "annotation_discard_before_ms": checks["annotation_contract"]["discard_before_ms"], "dmer_rule_version": checks["annotation_contract"]["dmer_rule_version"], "pdmer_rule_version": checks["annotation_contract"]["pdmer_rule_version"]}, "data_root": {"config_key": "SC_MV_DMER_DATA_ROOT", "runtime_root_external": True, "absolute_path_serialized": False}, "checks": checks, "blockers": sorted(set(failures)), "forbidden_operations": ["TRIM", "PADDING", "INTERPOLATION", "RESHAPE", "ADAPTIVE_POOLING"], "source_artifacts_consumed": ["manifests/datasets/deam-pdmer-clean-v3.json", "reports/data/deam-pdmer-clean-v3-audit.json", "processed/deam-pdmer-clean-v3/manifest.json", "processed/deam-pdmer-clean-v3/dmer_targets.jsonl", "processed/deam-pdmer-clean-v3/pdmer_tasks.jsonl"]}
    evidence["evidence_sha256"] = sha256_canonical(evidence)
    return evidence
