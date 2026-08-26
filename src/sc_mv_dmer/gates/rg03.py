"""RG-03 Sensor-only formal pre-flight and dry-run runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_mv_dmer.foundation.canonical import sha256_canonical


DATASET_ID = "dataset_480f00e52c71f8a4f7c2681f2a59d3b11c105d915e3faf508bd06970ee846dae"
DATASET_MANIFEST_SHA256 = "d4c539945c5477e5ca325e9535f6f100918341a9292460d533377c493ee8473d"
VALIDATION_CONTRACT_ID = "DEAM-PDMER-CLEAN-v3-SONG-VALIDATION/v1"
FEATURE_CACHE_SHA256 = "0f7b0e699184208bcb0e719d8288bfc3c1f5c9ade989a83f90d1e892985a7089"
PSEUDO_CACHE_SHA256 = "8979085e00668295c992c506d04ff4ab7231a0728f78f41d6dcb29af023345dd"
FEATURE_MANIFEST = "manifests/features/deam-four-view-cache-v3.json"
PSEUDO_MANIFEST = "manifests/features/deam-pseudo-label-bundle-v3.json"
VALIDATION_MANIFEST = "manifests/splits/deam-pdmer-clean-v3-validation-v1.json"
EVIDENCE_FILES = (
    "evidence/data/deam-pdmer-clean-v3-validation-ranking-v1.json",
    "evidence/data/deam-pdmer-clean-v3-pdmer-role-masking-v1.json",
    "evidence/data/deam-pdmer-clean-v3-normalization-baseline-v1.json",
    "evidence/data/deam-pdmer-clean-v3-mode-ccc-prerequisite-v1.json",
    "evidence/data/deam-pdmer-clean-v3-key-contract-v1.json",
)


@dataclass(frozen=True)
class PreflightReport:
    verdict: str
    checks: dict[str, bool]
    blockers: tuple[str, ...]
    provenance: dict[str, Any]


def _load(root: Path, logical_path: str) -> dict[str, Any]:
    return json.loads((root / logical_path).read_text(encoding="utf-8"))


def _internal_ok(root: Path, logical_path: str, checksum_key: str) -> bool:
    payload = _load(root, logical_path)
    observed = payload.get(checksum_key)
    unsigned = dict(payload)
    unsigned.pop(checksum_key, None)
    return isinstance(observed, str) and sha256_canonical(unsigned) == observed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_payloads_available(repo_root: Path, artifact_root: Path | None, cache: dict[str, Any]) -> bool:
    missing = []
    for record in cache.get("records", []):
        logical = record.get("payload_logical_path", "")
        candidate = (artifact_root / logical) if artifact_root is not None and logical.startswith("artifacts/") else (repo_root / logical)
        if not candidate.is_file() or _file_sha256(candidate) != record.get("payload_sha256"):
            missing.append(logical)
            if len(missing) >= 5:
                break
    return not missing


def run_preflight(repo_root: Path, *, artifact_root: Path | None = None) -> PreflightReport:
    repo_root = Path(repo_root).resolve()
    checks: dict[str, bool] = {}
    blockers: list[str] = []
    manifest = _load(repo_root, "manifests/datasets/deam-pdmer-clean-v3.json")
    checks["dataset_identity"] = manifest.get("dataset_id") == DATASET_ID and manifest.get("manifest_sha256") == DATASET_MANIFEST_SHA256 and _internal_ok(repo_root, "manifests/datasets/deam-pdmer-clean-v3.json", "manifest_sha256")
    validation = _load(repo_root, VALIDATION_MANIFEST)
    checks["validation_manifest"] = validation.get("validation_contract_id") == VALIDATION_CONTRACT_ID and validation.get("dataset_id") == DATASET_ID and validation.get("dataset_manifest_sha256") == DATASET_MANIFEST_SHA256 and _internal_ok(repo_root, VALIDATION_MANIFEST, "validation_manifest_sha256")
    checks["population"] = validation.get("population") == {"eligible_primary": 995, "optimization_train": 896, "validation": 99, "test_long_songs": 58} and len(validation.get("records", [])) == 995 and len(validation.get("test_records", [])) == 58
    records = validation.get("records", [])
    expected = sorted((sha256_canonical({"dataset_id": DATASET_ID, "song_id": row["song_id"], "validation_contract_id": VALIDATION_CONTRACT_ID}), row["song_id"]) for row in records)
    checks["canonical_ranking"] = all((row.get("rank_digest"), row.get("song_id")) == expected[index] and row.get("rank_index") == index for index, row in enumerate(records))
    train = set(validation.get("optimization_train_song_ids", [])); valid = set(validation.get("validation_song_ids", [])); test = set(validation.get("test_song_ids", []))
    checks["membership_disjointness"] = len(train) == 896 and len(valid) == 99 and len(test) == 58 and not train & valid and not (train | valid) & test and len(train | valid) == 995
    for logical in EVIDENCE_FILES:
        checks[f"checksum:{logical}"] = _internal_ok(repo_root, logical, "evidence_sha256")
    pdmer = _load(repo_root, EVIDENCE_FILES[1])
    checks["pdmer_masking"] = pdmer.get("checks", {}).get("test_excluded_from_train_validation") is True and pdmer.get("policy", {}).get("worker_identity_cross_role_reuse") == "ALLOWED" and pdmer.get("policy", {}).get("test_worker_presence_does_not_delete_train_episode") is True
    norm = _load(repo_root, EVIDENCE_FILES[2])
    checks["normalization_baseline"] = norm.get("fit_population", {}).get("song_count") == 896 and norm.get("constant_baseline", {}).get("energy", {}).get("source_field") == "targets.rms" and norm.get("constant_baseline", {}).get("energy", {}).get("strict_rms_correspondence") is True and norm.get("test_isolation") == "test targets/statistics never used"
    mode = _load(repo_root, EVIDENCE_FILES[3])
    mode_reasons = {"MODE_CCC_NO_VALID_PAIRS", "MODE_CCC_LT2_VALID_PAIRS", "MODE_CCC_ZERO_VARIANCE_TARGET", "MODE_CCC_ZERO_VARIANCE_PREDICTION", "MODE_CCC_DENOMINATOR_ZERO", "MODE_CCC_NONFINITE_RESULT"}
    checks["mode_contract"] = mode.get("metric", {}).get("implementation") == "MEP-CCC2-S5-3R+" and mode.get("metric", {}).get("moments_precision") == "FP64" and mode.get("metric", {}).get("aggregation") == "pooled valid frame" and mode.get("metric", {}).get("criterion") == "CCC > 0.7" and mode_reasons.issubset(set(mode.get("metric", {}).get("invalid_reason_codes", [])))
    key = _load(repo_root, EVIDENCE_FILES[4])
    checks["key_contract"] = key.get("vocabulary") == ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] and key.get("segments", {}).get("valid_rule") == "finite valid frames >= 5/10" and key.get("segments", {}).get("frames_per_segment") == 10 and key.get("ce_contract", {}).get("reduction") == "pooled valid-segment mean" and key.get("prior", {}).get("class_count_C") == 12 and key.get("test_excluded_from_prior") is True
    feature = _load(repo_root, FEATURE_MANIFEST); pseudo = _load(repo_root, PSEUDO_MANIFEST)
    checks["cache_binding"] = feature.get("manifest_sha256") == FEATURE_CACHE_SHA256 and pseudo.get("manifest_sha256") == PSEUDO_CACHE_SHA256 and feature.get("dataset_manifest_sha256") == DATASET_MANIFEST_SHA256 and pseudo.get("dataset_manifest_sha256") == DATASET_MANIFEST_SHA256
    rejected_artifact_root = artifact_root is not None and ((artifact_root / ".git").is_file() or (artifact_root / ".git").is_dir())
    if rejected_artifact_root:
        checks["immutable_external_artifact_root"] = False
        blockers.append("MISSING_IMMUTABLE_EXECUTION_ARTIFACT")
    else:
        checks["immutable_external_artifact_root"] = True
    payload_available = False if rejected_artifact_root else _cache_payloads_available(repo_root, artifact_root, feature) and _cache_payloads_available(repo_root, artifact_root, pseudo)
    checks["cache_payload_checksums"] = payload_available
    if not payload_available:
        blockers.append("MISSING_IMMUTABLE_EXECUTION_ARTIFACT")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=False)
    checks["formal_workspace_clean"] = status.returncode == 0 and not status.stdout.strip()
    if not checks["formal_workspace_clean"]:
        blockers.append("FORMAL_PRE_FLIGHT_BLOCKED_DIRTY_RESEARCH_WORKSPACE")
    checks["test_isolation"] = checks["normalization_baseline"] and checks["pdmer_masking"] and validation.get("population", {}).get("test_long_songs") == 58
    blockers = sorted(set(blockers))
    verdict = "PASS" if not blockers and all(checks.values()) else "BLOCKED"
    return PreflightReport(verdict=verdict, checks=checks, blockers=tuple(blockers), provenance={"repo_root": str(repo_root), "head": status.returncode == 0 and subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip() or None, "artifact_root_config": "SC_MV_DMER_ARTIFACT_ROOT", "absolute_paths_serialized": False, "dataset_manifest": DATASET_MANIFEST_SHA256, "feature_cache_manifest": FEATURE_CACHE_SHA256, "pseudo_label_cache_manifest": PSEUDO_CACHE_SHA256, "validation_contract": VALIDATION_CONTRACT_ID})


def dry_run(repo_root: Path, *, artifact_root: Path | None = None) -> dict[str, Any]:
    report = run_preflight(repo_root, artifact_root=artifact_root)
    return {"runner": "RG-03-SENSOR-ONLY/v1", "dry_run": True, "training_started": False, "checkpoint_published": False, "rg03_pass_evidence_published": False, "preflight_verdict": report.verdict, "checks": report.checks, "blockers": list(report.blockers), "provenance": report.provenance, "planned_population": {"optimization_train": 896, "validation": 99, "test": 58}, "selection": {"metric": "validation CE", "cadence": "every epoch", "patience": 10, "min_delta": 1e-4, "tie_break": ["earliest epoch", "smallest global step"]}, "seed_set": "SC-MV-DMER-FORMAL-S5/v1"}
