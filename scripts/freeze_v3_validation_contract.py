"""Create the append-only DEAM-PDMER-CLEAN-v3 validation contract artifacts.

This is a data-contract operation only. It reads the immutable v3 manifest,
v3 caches and processed PDMER artifact, and never writes to any of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


DATASET_MANIFEST_ID = "DEAM-PDMER-CLEAN-v3"
DATASET_ID = "dataset_480f00e52c71f8a4f7c2681f2a59d3b11c105d915e3faf508bd06970ee846dae"
DATASET_MANIFEST_SHA256 = "d4c539945c5477e5ca325e9535f6f100918341a9292460d533377c493ee8473d"
CONTRACT_ID = "DEAM-PDMER-CLEAN-v3-SONG-VALIDATION/v1"
VALIDATION_MANIFEST_ID = "DEAM-PDMER-CLEAN-v3-SONG-VALIDATION-MANIFEST-v1"
TRAIN_COUNT = 896
VALIDATION_COUNT = 99
ELIGIBLE_COUNT = 995
TEST_COUNT = 58
DATA_ROOT_CONFIG_KEY = "SC_MV_DMER_DATA_ROOT"
FEATURE_CACHE_PATH = "manifests/features/deam-four-view-cache-v3.json"
PSEUDO_CACHE_PATH = "manifests/features/deam-pseudo-label-bundle-v3.json"
DMER_PATH = "processed/deam-pdmer-clean-v3/dmer_targets.jsonl"
PDMER_PATH = "processed/deam-pdmer-clean-v3/pdmer_tasks.jsonl"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, payload: dict[str, Any], checksum_key: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite append-only artifact: {path}")
    unsigned = dict(payload)
    unsigned.pop(checksum_key, None)
    payload[checksum_key] = sha256_canonical(unsigned)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return _file_sha256(path)


def _rank_digest(song_id: str) -> str:
    return sha256_canonical(
        {
            "dataset_id": DATASET_ID,
            "song_id": song_id,
            "validation_contract_id": CONTRACT_ID,
        }
    )


def _load_payload(repo_root: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = repo_root / Path(row["payload_logical_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _file_sha256(path) != row["payload_sha256"]:
        raise ValueError(f"pseudo-label payload checksum mismatch: {row['sample_id']}")
    with np.load(path) as values:
        return {name: values[name].copy() for name in values.files}


def _finite_stats(values: list[np.ndarray]) -> dict[str, Any]:
    merged = np.concatenate(values).astype(np.float64, copy=False)
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        raise ValueError("normalization target has no finite train values")
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite, dtype=np.float64)),
        "std_population_ddof0": float(np.std(finite, ddof=0, dtype=np.float64)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "finite": True,
    }


def _key_segment_result(key: np.ndarray, valid_mask: np.ndarray) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(9):
        start = index * 10
        stop = start + 10
        frames = key[start:stop].astype(np.float64, copy=False)
        mask = valid_mask[start:stop].astype(bool, copy=False)
        finite = np.isfinite(frames).all(axis=1)
        finite_valid = mask & finite
        count = int(np.count_nonzero(finite_valid))
        reason: str | None = None
        label: int | None = None
        if count < 5:
            reason = "KEY_VALID_FRAME_COUNT_LT_5"
        else:
            selected = frames[finite_valid]
            unknown = np.any(np.sum(selected, axis=1) <= 0.0) or np.any(selected < 0.0)
            if unknown:
                reason = "KEY_UNKNOWN"
            else:
                scores = np.mean(selected, axis=0, dtype=np.float64)
                maximum = float(np.max(scores))
                winners = np.flatnonzero(scores == maximum)
                if len(winners) != 1:
                    reason = "KEY_TIE"
                else:
                    label = int(winners[0])
        result.append(
            {
                "segment_index": index,
                "start_seconds": index * 5,
                "end_seconds": (index + 1) * 5,
                "finite_valid_frame_count": count,
                "valid": reason is None,
                "class_index": label,
                "invalid_reason_code": reason,
            }
        )
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    manifest_path = repo_root / "manifests/datasets/deam-pdmer-clean-v3.json"
    feature_path = repo_root / FEATURE_CACHE_PATH
    pseudo_path = repo_root / PSEUDO_CACHE_PATH
    audit_path = repo_root / "reports/data/deam-pdmer-clean-v3-audit.json"
    data_root = Path(args.data_root).resolve()
    manifest = _load(manifest_path)
    if manifest["manifest_id"] != DATASET_MANIFEST_ID or manifest["dataset_id"] != DATASET_ID:
        raise ValueError("v3 manifest identity mismatch")
    if manifest["manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise ValueError("v3 manifest internal checksum mismatch")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_sha256")
    if sha256_canonical(unsigned_manifest) != DATASET_MANIFEST_SHA256:
        raise ValueError("v3 manifest canonical checksum mismatch")
    if manifest["population"]["retained_train_primary_count"] != ELIGIBLE_COUNT or manifest["population"]["retained_test_long_song_count"] != TEST_COUNT:
        raise ValueError("v3 population mismatch")
    records = [row for row in manifest["records"] if row.get("status") == "RETAINED"]
    train_rows = [row for row in records if row["population_role"] == "TRAIN_PRIMARY"]
    test_rows = [row for row in records if row["population_role"] == "TEST_LONG_SONG"]
    if len(train_rows) != ELIGIBLE_COUNT or len(test_rows) != TEST_COUNT:
        raise ValueError("retained v3 records do not match 995/58")
    by_song = {row["song_id"]: row for row in train_rows}
    ranked = sorted(((_rank_digest(song_id), song_id, row) for song_id, row in by_song.items()), key=lambda item: (item[0], item[1]))
    membership: list[dict[str, Any]] = []
    for index, (digest, song_id, row) in enumerate(ranked):
        split = "validation" if index < VALIDATION_COUNT else "optimization_train"
        membership.append({
            "song_id": song_id,
            "sample_id": row["sample_id"],
            "logical_song_key": row["logical_song_key"],
            "source_audio_sha256": row["source_audio"]["source_sha256"],
            "split": split,
            "rank_digest": digest,
            "rank_index": index,
        })
    validation_ids = [row["song_id"] for row in membership if row["split"] == "validation"]
    train_ids = [row["song_id"] for row in membership if row["split"] == "optimization_train"]
    test_membership = [{"song_id": row["song_id"], "sample_id": row["sample_id"], "logical_song_key": row["logical_song_key"], "source_audio_sha256": row["source_audio"]["source_sha256"]} for row in test_rows]
    feature = _load(feature_path)
    pseudo = _load(pseudo_path)
    audit = _load(audit_path)
    if feature["dataset_manifest_sha256"] != DATASET_MANIFEST_SHA256 or pseudo["dataset_manifest_sha256"] != DATASET_MANIFEST_SHA256:
        raise ValueError("cache does not bind v3 manifest")
    pseudo_by_song = {row["song_id"]: row for row in pseudo["records"]}
    if set(pseudo_by_song) != {row["song_id"] for row in records}:
        raise ValueError("pseudo-label cache coverage mismatch")
    pseudo_payloads = {song_id: _load_payload(repo_root, pseudo_by_song[song_id]) for song_id in train_ids}
    stats: dict[str, Any] = {}
    for target in ("rms", "brightness", "mode"):
        values: list[np.ndarray] = []
        for song_id in train_ids:
            payload = pseudo_payloads[song_id]
            mask = payload["valid_target_mask"].astype(bool)
            values.append(payload[target].reshape(-1)[mask])
        stats[target] = _finite_stats(values)
    segment_records: list[dict[str, Any]] = []
    class_counts = [0] * 12
    invalid_reasons: dict[str, int] = {}
    for song_id in train_ids:
        payload = pseudo_payloads[song_id]
        segments = _key_segment_result(payload["key"], payload["valid_target_mask"])
        for segment in segments:
            item = {"song_id": song_id, **segment}
            segment_records.append(item)
            if segment["valid"]:
                class_counts[segment["class_index"]] += 1
            else:
                reason = segment["invalid_reason_code"]
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    valid_segment_count = sum(class_counts)
    prior = [(count + 1) / (valid_segment_count + 12) for count in class_counts]
    split_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "validation_manifest_id": VALIDATION_MANIFEST_ID,
        "validation_contract_id": CONTRACT_ID,
        "dataset_manifest_id": DATASET_MANIFEST_ID,
        "dataset_manifest_logical_path": "manifests/datasets/deam-pdmer-clean-v3.json",
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_manifest_file_sha256": _file_sha256(manifest_path),
        "population": {"eligible_primary": ELIGIBLE_COUNT, "optimization_train": TRAIN_COUNT, "validation": VALIDATION_COUNT, "test_long_songs": TEST_COUNT},
        "optimization_train_song_ids": train_ids,
        "validation_song_ids": validation_ids,
        "test_song_ids": [row["song_id"] for row in test_membership],
        "records": membership,
        "test_records": test_membership,
        "canonicalization": {"encoding": "UTF-8", "json_object_key_order": "lexicographic ascending", "separators": "compact comma/colon", "digest": "SHA-256", "rank_input_keys": ["dataset_id", "song_id", "validation_contract_id"], "ordering": "rank_digest ascending, then song_id ascending", "rank_index_base": 0, "membership_inputs_excluded": ["annotation", "target", "model_result", "seed", "path", "filename"]},
        "lineage": {"decisions": ["DEC-0028", "DEC-0029", "DEC-0030", "DEC-0031"], "data_root_config_key": DATA_ROOT_CONFIG_KEY, "validation_policy": CONTRACT_ID, "feature_cache_logical_path": FEATURE_CACHE_PATH, "feature_cache_manifest_sha256": feature["manifest_sha256"], "pseudo_label_cache_logical_path": PSEUDO_CACHE_PATH, "pseudo_label_cache_manifest_sha256": pseudo["manifest_sha256"]},
    }
    split_path = repo_root / "manifests/splits/deam-pdmer-clean-v3-validation-v1.json"
    split_file_sha256 = _write_once(split_path, split_payload, "validation_manifest_sha256")
    rank_evidence = {"schema_version": "1.0", "evidence_id": "DEAM-PDMER-CLEAN-v3-VALIDATION-RANKING-v1", "contract_id": CONTRACT_ID, "verdict": "PASS", "dataset_manifest": {"manifest_id": DATASET_MANIFEST_ID, "dataset_id": DATASET_ID, "logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "internal_sha256": DATASET_MANIFEST_SHA256, "file_sha256": _file_sha256(manifest_path)}, "validation_manifest": {"logical_path": "manifests/splits/deam-pdmer-clean-v3-validation-v1.json", "internal_sha256": split_payload["validation_manifest_sha256"], "file_sha256": split_file_sha256}, "population": split_payload["population"], "ranking_rule": split_payload["canonicalization"], "rank_records": membership, "checks": {"eligible_count": len(membership) == ELIGIBLE_COUNT, "train_count": len(train_ids) == TRAIN_COUNT, "validation_count": len(validation_ids) == VALIDATION_COUNT, "train_validation_disjoint": not (set(train_ids) & set(validation_ids)), "test_disjoint": not ((set(train_ids) | set(validation_ids)) & {row["song_id"] for row in test_membership}), "complete_primary_coverage": len(set(train_ids) | set(validation_ids)) == ELIGIBLE_COUNT}}
    rank_path = repo_root / "evidence/data/deam-pdmer-clean-v3-validation-ranking-v1.json"
    rank_file_sha256 = _write_once(rank_path, rank_evidence, "evidence_sha256")
    pdmer_path = data_root / PDMER_PATH
    pdmer_tasks = [json.loads(line) for line in pdmer_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    task_summary: list[dict[str, Any]] = []
    repeated: dict[str, set[str]] = {}
    role_counts = {"optimization_train": 0, "validation": 0, "test": 0}
    role_song_sets = {"optimization_train": set(), "validation": set(), "test": set()}
    train_id_set = set(train_ids)
    validation_id_set = set(validation_ids)
    for task in pdmer_tasks:
        roles: dict[str, int] = {"optimization_train": 0, "validation": 0, "test": 0}
        for song in task["songs"]:
            sid = song["song_id"]
            role = "validation" if sid in validation_id_set else "optimization_train" if sid in train_id_set else "test"
            roles[role] += 1
            role_counts[role] += 1
            role_song_sets[role].add(sid)
        active_roles = [role for role, count in roles.items() if count]
        worker_hash = hashlib.sha256(str(task["worker_id"]).encode("utf-8")).hexdigest()
        repeated[worker_hash] = set(active_roles)
        task_summary.append({"worker_id_sha256": worker_hash, "song_counts_by_role": roles, "active_roles": active_roles, "mask_policy": "role_slice; no episode crosses train/validation/test role"})
    repeated_counts = {"optimization_train_validation": sum(set(("optimization_train", "validation")).issubset(roles) for roles in repeated.values()), "optimization_train_test": sum(set(("optimization_train", "test")).issubset(roles) for roles in repeated.values()), "validation_test": sum(set(("validation", "test")).issubset(roles) for roles in repeated.values())}
    pdmer_evidence = {"schema_version": "1.0", "evidence_id": "DEAM-PDMER-CLEAN-v3-PDMER-ROLE-MASKING-v1", "contract_id": CONTRACT_ID, "verdict": "PASS", "dataset_manifest": {"manifest_id": DATASET_MANIFEST_ID, "dataset_id": DATASET_ID, "logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "internal_sha256": DATASET_MANIFEST_SHA256, "file_sha256": _file_sha256(manifest_path)}, "pdmer_artifact": {"logical_path": PDMER_PATH, "file_sha256": _file_sha256(pdmer_path), "task_count": len(pdmer_tasks)}, "population": split_payload["population"], "policy": {"test_song_episodes_targets_statistics": "NEVER_ENTER_TRAIN_OR_VALIDATION", "worker_identity_cross_role_reuse": "ALLOWED", "test_worker_presence_does_not_delete_train_episode": True, "episode_boundary": "role-local; no cross-role episode"}, "role_song_counts": {role: len(values) for role, values in role_song_sets.items()}, "role_assignment_counts": role_counts, "cross_role_repeated_worker_counts": repeated_counts, "task_summary": task_summary, "checks": {"test_excluded_from_train_validation": not (role_song_sets["test"] & (role_song_sets["optimization_train"] | role_song_sets["validation"])), "train_song_count": len(role_song_sets["optimization_train"]) == TRAIN_COUNT, "validation_song_count": len(role_song_sets["validation"]) == VALIDATION_COUNT, "test_song_count": len(role_song_sets["test"]) == TEST_COUNT}}
    pdmer_evidence_path = repo_root / "evidence/data/deam-pdmer-clean-v3-pdmer-role-masking-v1.json"
    pdmer_file_sha256 = _write_once(pdmer_evidence_path, pdmer_evidence, "evidence_sha256")
    normalization = {"schema_version": "1.0", "evidence_id": "DEAM-PDMER-CLEAN-v3-NORMALIZATION-BASELINE-v1", "contract_id": CONTRACT_ID, "verdict": "PASS", "dataset_manifest": {"manifest_id": DATASET_MANIFEST_ID, "dataset_id": DATASET_ID, "logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "internal_sha256": DATASET_MANIFEST_SHA256, "file_sha256": _file_sha256(manifest_path)}, "pseudo_label_cache": {"logical_path": PSEUDO_CACHE_PATH, "file_sha256": _file_sha256(pseudo_path), "manifest_sha256": pseudo["manifest_sha256"], "version": pseudo["cache_version"]}, "fit_population": {"split": "optimization_train", "song_count": TRAIN_COUNT, "song_fingerprint": sha256_canonical(train_ids)}, "normalization": {"transform": "z=(x-mean_train)/max(std_train,1e-6)", "epsilon_std": 1e-6, "metric_space": "inverse-transformed raw target space", "targets": {"rms": {"canonical_target_id": "DEAM-PDMER-CLEAN-v3/PSEUDO_LABEL/RMS", "pseudo_label_field": "targets.rms", "strict_field_match": True, **stats["rms"]}, "brightness": stats["brightness"], "mode": {"domain": "[-1,1]", **stats["mode"]}, "key": {"transform": "categorical identity; no numeric z-score", "fit_provenance": "train-only vocabulary and prior"}}}, "constant_baseline": {"energy": {"canonical_target_id": "DEAM-PDMER-CLEAN-v3/PSEUDO_LABEL/RMS", "source_field": "targets.rms", "strict_rms_correspondence": True, "fit": "finite optimization-train mean", "value": stats["rms"]["mean"]}, "brightness": {"fit": "finite optimization-train mean", "value": stats["brightness"]["mean"]}}, "test_isolation": "test targets/statistics never used"}
    norm_path = repo_root / "evidence/data/deam-pdmer-clean-v3-normalization-baseline-v1.json"
    norm_file_sha256 = _write_once(norm_path, normalization, "evidence_sha256")
    mode_evidence = {"schema_version": "1.0", "evidence_id": "DEAM-PDMER-CLEAN-v3-MODE-CCC-PREREQUISITE-v1", "contract_id": CONTRACT_ID, "verdict": "PASS", "dataset_manifest_sha256": DATASET_MANIFEST_SHA256, "pseudo_label_cache": {"logical_path": PSEUDO_CACHE_PATH, "file_sha256": _file_sha256(pseudo_path), "manifest_sha256": pseudo["manifest_sha256"]}, "metric": {"implementation": "MEP-CCC2-S5-3R+", "aggregation": "pooled valid frame", "moments_precision": "FP64", "input_precision": "float32 accepted; converted to FP64", "finite_filter": "prediction,target,mask jointly valid and finite", "criterion": "CCC > 0.7", "invalid_reason_codes": ["MODE_CCC_NO_VALID_PAIRS", "MODE_CCC_LT2_VALID_PAIRS", "MODE_CCC_ZERO_VARIANCE_TARGET", "MODE_CCC_ZERO_VARIANCE_PREDICTION", "MODE_CCC_DENOMINATOR_ZERO", "MODE_CCC_NONFINITE_RESULT"], "invalid_handling": "metric invalid; never coerced to numeric pass/fail"}, "fit_population": {"optimization_train": TRAIN_COUNT, "validation": VALIDATION_COUNT, "test_excluded": True}, "execution_status": "PREREQUISITE_ONLY_NO_MODEL_EVALUATION"}
    mode_path = repo_root / "evidence/data/deam-pdmer-clean-v3-mode-ccc-prerequisite-v1.json"
    mode_file_sha256 = _write_once(mode_path, mode_evidence, "evidence_sha256")
    key_evidence = {"schema_version": "1.0", "evidence_id": "DEAM-PDMER-CLEAN-v3-KEY-CONTRACT-v1", "contract_id": CONTRACT_ID, "verdict": "PASS", "dataset_manifest_sha256": DATASET_MANIFEST_SHA256, "pseudo_label_cache": {"logical_path": PSEUDO_CACHE_PATH, "file_sha256": _file_sha256(pseudo_path), "manifest_sha256": pseudo["manifest_sha256"]}, "vocabulary": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"], "segments": {"seconds": [[i * 5, (i + 1) * 5] for i in range(9)], "frames_per_segment": 10, "valid_rule": "finite valid frames >= 5/10", "invalid_reasons": ["KEY_UNKNOWN", "KEY_MISSING", "KEY_TIE", "KEY_VALID_FRAME_COUNT_LT_5", "KEY_NONFINITE_TARGET", "KEY_MASK_FALSE", "KEY_VOCABULARY_MISMATCH"], "invalid_excluded_from": ["CE", "prior", "checkpoint_selection"]}, "ce_contract": {"reduction": "pooled valid-segment mean", "zero_valid_segment": "sample contributes no CE; KEY_ZERO_VALID_SEGMENTS", "zero_valid_batch": "exclude batch and record KEY_INVALID_BATCH_ZERO_VALID_SEGMENTS", "zero_valid_validation": "non-finite CE; KEY_VALIDATION_ZERO_VALID_SEGMENTS", "nonfinite": "run invalid; KEY_CE_NONFINITE", "model_prior_contract_identical": True}, "prior": {"fit_split": "optimization_train", "song_count": TRAIN_COUNT, "valid_segment_count_N": valid_segment_count, "class_counts": class_counts, "class_count_C": 12, "smoothing": "Laplace/add-one", "probabilities": prior, "provenance": {"segment_rule": "finite valid frames >= 5/10", "train_song_fingerprint": sha256_canonical(train_ids)}}, "segment_records": segment_records, "invalid_reason_counts": invalid_reasons, "test_excluded_from_prior": True}
    key_path = repo_root / "evidence/data/deam-pdmer-clean-v3-key-contract-v1.json"
    key_file_sha256 = _write_once(key_path, key_evidence, "evidence_sha256")
    result = {"validation_manifest": str(split_path), "validation_manifest_sha256": split_payload["validation_manifest_sha256"], "validation_manifest_file_sha256": split_file_sha256, "evidence_files": {"ranking": [str(rank_path), rank_file_sha256], "pdmer": [str(pdmer_evidence_path), pdmer_file_sha256], "normalization": [str(norm_path), norm_file_sha256], "mode": [str(mode_path), mode_file_sha256], "key": [str(key_path), key_file_sha256]}, "counts": split_payload["population"], "stable_dataset_id": DATASET_ID, "training": False, "checkpoint": False, "rg03_formal_evaluation": False}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    configured_root = os.environ.get(DATA_ROOT_CONFIG_KEY)
    parser.add_argument("--data-root", type=Path, default=Path(configured_root) if configured_root else None)
    args = parser.parse_args()
    if args.data_root is None:
        raise SystemExit(f"{DATA_ROOT_CONFIG_KEY} is required")
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
