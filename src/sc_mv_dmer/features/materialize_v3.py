"""Materialize the approved DEAM-PDMER-CLEAN-v3 Step 2 caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from sc_mv_dmer.features.binning import bin_feature
from sc_mv_dmer.features.cache import CacheSpec, FourViewFeatureCacheManifest, write_cache
from sc_mv_dmer.features.deep import extract_deep
from sc_mv_dmer.features.handcrafted import extract_handcrafted
from sc_mv_dmer.features.materialize import _deep_forward, _load_pinned_mert
from sc_mv_dmer.features.pseudo_labels import derive_pseudo_labels
from sc_mv_dmer.features.resample import load_waveform
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


EXCLUDED = {"1174", "1200", "1273", "1493", "1789"}
TRAIN_ROLE = "TRAIN_PRIMARY"
TEST_ROLE = "TEST_LONG_SONG"
EXPECTED_DIMS = {"deep": 768, "mel": 128, "mfcc": 40, "chroma": 12}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_internal(payload: dict[str, Any], key: str) -> None:
    observed = payload.get(key)
    if not isinstance(observed, str) or sha256_canonical({k: v for k, v in payload.items() if k != key}) != observed:
        raise ValueError(f"internal checksum mismatch for {key}")


def _write_pseudo_cache(items: list[dict[str, Any]], *, repo_root: Path, destination: Path, spec: dict[str, Any]) -> dict[str, Any]:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable pseudo-label cache: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".deam-pseudo-label-v3-staging-", dir=str(destination.parent)))
    logical_root = "artifacts/pseudo_labels/deam-pseudo-label-v3"
    records: list[dict[str, Any]] = []
    try:
        for item in items:
            arrays = {name: np.ascontiguousarray(value) for name, value in item["targets"].items()}
            arrays["valid_target_mask"] = np.ascontiguousarray(item["valid_target_mask"])
            filename = f"{item['sample_id']}.npz"
            staged = staging / filename
            with staged.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
            records.append({
                "dataset_id": item["dataset_id"],
                "song_id": item["song_id"],
                "sample_id": item["sample_id"],
                "population_role": item["population_role"],
                "source_audio_sha256": item["source_audio_sha256"],
                "payload_logical_path": f"{logical_root}/{filename}",
                "payload_sha256": _sha256_file(staged),
                "targets": item["target_meta"],
                "valid_target_mask": {"shape": [90], "dtype": str(item["valid_target_mask"].dtype), "sha256": hashlib.sha256(item["valid_target_mask"].tobytes()).hexdigest()},
                "timegrid_binding_sha256": item["timegrid_binding_sha256"],
                "rule_version": item["rule_version"],
                "segment_boundaries_seconds": item["segment_boundaries_seconds"],
            })
        records.sort(key=lambda row: row["song_id"])
        unsigned = {**spec, "record_count": len(records), "records": records}
        manifest = unsigned | {"manifest_sha256": sha256_canonical(unsigned)}
        final_root = repo_root / logical_root
        final_root.mkdir(parents=True, exist_ok=True)
        for payload in staging.glob("*.npz"):
            target = final_root / payload.name
            if target.exists():
                if _sha256_file(target) != _sha256_file(payload):
                    raise ValueError(f"immutable pseudo payload collision: {target}")
            else:
                os.replace(payload, target)
        destination.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        return manifest
    finally:
        for child in staging.glob("*"):
            child.unlink(missing_ok=True)
        staging.rmdir()


def materialize_v3_step2(
    *,
    repo_root: Path,
    data_root: Path,
    mert_model_root: Path,
    dataset_manifest_path: Path,
    upstream_manifest_path: Path,
    dimensions_path: Path,
    timegrid_path: Path,
    rg01_path: Path,
    feature_cache_path: Path,
    pseudo_cache_path: Path,
    alignment_evidence_path: Path,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    data_root = Path(data_root).resolve()
    dataset_path = Path(dataset_manifest_path)
    dataset = _json(dataset_path)
    _verify_internal(dataset, "manifest_sha256")
    if dataset.get("manifest_id") != "DEAM-PDMER-CLEAN-v3" or dataset.get("dataset_version") != "deam-pdmer-clean-v3":
        raise ValueError("Step 2 requires DEAM-PDMER-CLEAN-v3")
    if dataset.get("split_policy", {}).get("validation") != "UNDEFINED_PENDING_SEPARATE_FREEZE":
        raise ValueError("validation policy must remain undefined")
    if sorted(dataset.get("audio_policy", {}).get("excluded_short_primary_logical_song_keys", [])) != sorted(EXCLUDED):
        raise ValueError("v3 short-audio exclusion set mismatch")
    upstream = _json(upstream_manifest_path)
    if upstream.get("revision") != "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5":
        raise ValueError("unpinned MERT upstream manifest")
    dimensions = _json(dimensions_path)
    if dimensions.get("observed_t_raw") != 3374 or dimensions.get("canonical_downstream_t") != 90:
        raise ValueError("dimensions manifest is not pinned E1.2 v2")
    timegrid = _json(timegrid_path)
    if timegrid.get("raw_frame_count") != 3374 or timegrid.get("bin_count") != 90:
        raise ValueError("timegrid manifest is not pinned E1.3 v2")
    rg01 = _json(rg01_path)
    if rg01.get("verdict") != "PASS":
        raise ValueError("RG-01 is not PASS")
    retained = [row for row in dataset["records"] if row.get("status") == "RETAINED"]
    retained.sort(key=lambda row: int(row["logical_song_key"]))
    if len(retained) != 1053 or sum(row["population_role"] == TRAIN_ROLE for row in retained) != 995 or sum(row["population_role"] == TEST_ROLE for row in retained) != 58:
        raise ValueError("v3 eligible population is not 995 train + 58 test")
    if any(row["logical_song_key"] in EXCLUDED for row in retained):
        raise ValueError("excluded short song entered eligible population")

    dimension_sha256 = _sha256_file(dimensions_path)
    timegrid_sha256 = _sha256_file(timegrid_path)
    dataset_file_sha256 = _sha256_file(dataset_path)
    upstream_file_sha256 = _sha256_file(upstream_manifest_path)
    rg01_file_sha256 = _sha256_file(rg01_path)
    source_inventory_path = dataset.get("source_inventory_logical_path")
    processor, model = _load_pinned_mert(Path(mert_model_root))
    feature_records: list[dict[str, Any]] = []
    pseudo_items: list[dict[str, Any]] = []
    source_checksums: dict[str, str] = {}
    for index, row in enumerate(retained, start=1):
        source = row["source_audio"]
        audio_path = data_root / Path(source["source_relative_path"])
        actual_source_sha256 = _sha256_file(audio_path)
        if actual_source_sha256 != source["source_sha256"]:
            raise ValueError(f"source checksum mismatch: {row['logical_song_key']}")
        if float(source["duration_seconds"]) < 45.0:
            raise ValueError(f"retained source is shorter than exact 45s: {row['logical_song_key']}")
        source_checksums[row["logical_song_key"]] = actual_source_sha256
        wave22 = load_waveform(audio_path, song_id=row["song_id"], sample_id=row["sample_id"], source_relative_path=source["source_relative_path"], target_sample_rate_hz=22050, source_audio_sha256=actual_source_sha256)
        wave24 = load_waveform(audio_path, song_id=row["song_id"], sample_id=row["sample_id"], source_relative_path=source["source_relative_path"], target_sample_rate_hz=24000, source_audio_sha256=actual_source_sha256)
        wave22 = wave22.model_copy(update={"dataset_id": dataset["dataset_id"]})
        wave24 = wave24.model_copy(update={"dataset_id": dataset["dataset_id"]})
        handcrafted = extract_handcrafted(wave22)
        binned = {name: bin_feature(record, timegrid_binding_sha256=timegrid_sha256).model_copy(update={"dimension_binding_sha256": dimension_sha256}) for name, record in handcrafted.items()}
        deep = extract_deep(wave24, _deep_forward(wave24.waveform, processor, model), timegrid_sha256).model_copy(update={"dataset_id": dataset["dataset_id"], "dimension_binding_sha256": dimension_sha256})
        feature_records.append({"dataset_id": dataset["dataset_id"], "song_id": row["song_id"], "sample_id": row["sample_id"], "population_role": row["population_role"], "source_audio_sha256": actual_source_sha256, "views": {**binned, "deep": deep}})
        labels = derive_pseudo_labels(wave22, timegrid_binding_sha256=timegrid_sha256, rule_version="deam-pseudo-label-v3")
        pseudo_items.append({"dataset_id": dataset["dataset_id"], "song_id": row["song_id"], "sample_id": row["sample_id"], "population_role": row["population_role"], "source_audio_sha256": actual_source_sha256, "targets": labels.targets, "valid_target_mask": labels.valid_target_mask, "target_meta": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()} for name, value in sorted(labels.targets.items())}, "timegrid_binding_sha256": labels.timegrid_binding_sha256, "rule_version": labels.rule_version, "segment_boundaries_seconds": [list(boundary) for boundary in labels.segment_boundaries_seconds]})
        if index % 25 == 0:
            print(f"materialized {index}/{len(retained)}", flush=True)

    feature_spec = CacheSpec(cache_id="deam-four-view", version="v3", logical_root="artifacts/features/deam-four-view-v3", dataset_manifest_logical_path="manifests/datasets/deam-pdmer-clean-v3.json", dataset_manifest_sha256=dataset["manifest_sha256"], dataset_manifest_file_sha256=dataset_file_sha256, source_inventory_logical_path=source_inventory_path, source_inventory_sha256=dataset.get("source_inventory_sha256"), split_manifest_logical_path=None, split_manifest_sha256=None, dimensions_logical_path="manifests/dimensions/mert-downstream-dimensions-v2.json", dimensions_sha256=dimension_sha256, timegrid_logical_path="manifests/timegrid/mert-2hz-timegrid-v2.json", timegrid_sha256=timegrid_sha256, upstream_manifest_logical_path="manifests/upstream/mert-v1-95m-primary-v1.json", upstream_manifest_sha256=upstream_file_sha256, rg01_evidence_logical_path="evidence/gates/rg-01/rg-01-attempt-v1.json", rg01_evidence_sha256=rg01_file_sha256, population_counts={TRAIN_ROLE: 995, TEST_ROLE: 58}, validation_policy="UNDEFINED_PENDING_SEPARATE_FREEZE")
    feature_manifest = write_cache(feature_records, feature_spec, feature_cache_path)
    pseudo_spec = {"schema_version": "1.0", "manifest_id": "DEAM-PSEUDO-LABEL-BUNDLE-v3", "cache_version": "v3", "rule_version": "deam-pseudo-label-v3", "dataset_manifest_logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "dataset_manifest_sha256": dataset["manifest_sha256"], "dataset_manifest_file_sha256": dataset_file_sha256, "source_inventory_logical_path": source_inventory_path, "source_inventory_sha256": dataset.get("source_inventory_sha256"), "dimensions_logical_path": "manifests/dimensions/mert-downstream-dimensions-v2.json", "dimensions_sha256": dimension_sha256, "timegrid_logical_path": "manifests/timegrid/mert-2hz-timegrid-v2.json", "timegrid_sha256": timegrid_sha256, "upstream_manifest_logical_path": "manifests/upstream/mert-v1-95m-primary-v1.json", "upstream_manifest_sha256": upstream_file_sha256, "rg01_evidence_logical_path": "evidence/gates/rg-01/rg-01-attempt-v1.json", "rg01_evidence_sha256": rg01_file_sha256, "population_counts": {TRAIN_ROLE: 995, TEST_ROLE: 58}, "validation_policy": "UNDEFINED_PENDING_SEPARATE_FREEZE"}
    pseudo_manifest = _write_pseudo_cache(pseudo_items, repo_root=repo_root, destination=pseudo_cache_path, spec=pseudo_spec)
    alignment = _build_alignment_evidence(dataset, feature_manifest, pseudo_manifest, source_checksums, feature_cache_path, pseudo_cache_path, alignment_evidence_path)
    return {"feature_manifest": feature_manifest.model_dump(mode="json"), "pseudo_manifest": pseudo_manifest, "alignment": alignment}


def _build_alignment_evidence(dataset: dict[str, Any], feature: FourViewFeatureCacheManifest, pseudo: dict[str, Any], source_checksums: dict[str, str], feature_path: Path, pseudo_path: Path, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable alignment evidence: {destination}")
    train = [row for row in dataset["records"] if row.get("status") == "RETAINED" and row.get("population_role") == TRAIN_ROLE]
    train.sort(key=lambda row: (hashlib.sha256(row["song_id"].encode("utf-8")).hexdigest(), row["song_id"]))
    sample = [row["song_id"] for row in train[:3]]
    feature_records = {row.song_id: row for row in feature.records}
    pseudo_records = {row["song_id"]: row for row in pseudo["records"]}
    checks = {"feature_record_count": feature.record_count == 1053, "feature_population_counts": feature.population_counts == {TRAIN_ROLE: 995, TEST_ROLE: 58}, "pseudo_record_count": pseudo["record_count"] == 1053, "train_test_disjoint": set(row.population_role for row in feature.records) == {TRAIN_ROLE, TEST_ROLE}, "all_sample_source_checksums_bound": all(row.source_audio_sha256 == source_checksums.get(next(item["logical_song_key"] for item in dataset["records"] if item["song_id"] == row.song_id)) for row in feature.records), "sample_records_present": all(song_id in feature_records and song_id in pseudo_records for song_id in sample), "sample_shapes": all(all(feature_records[song_id].views[name]["shape"] == [90, dim] for name, dim in EXPECTED_DIMS.items()) for song_id in sample), "sample_pseudo_shapes": all(all(pseudo_records[song_id]["targets"][name]["shape"][0] == 90 for name in ("rms", "brightness", "mode", "key")) for song_id in sample)}
    verdict = "PASS" if all(checks.values()) else "FAIL"
    payload = {"schema_version": "1.0", "evidence_id": "E2_STEP2_DEAM_PDMER_CLEAN_V3_ALIGNMENT-v1", "scope": "FEATURE_AND_PSEUDO_LABEL_ALIGNMENT_AUDIT", "verdict": verdict, "dataset_manifest": {"logical_path": "manifests/datasets/deam-pdmer-clean-v3.json", "manifest_id": dataset["manifest_id"], "dataset_id": dataset["dataset_id"], "internal_sha256": dataset["manifest_sha256"]}, "feature_cache": {"logical_path": "manifests/features/deam-four-view-cache-v3.json", "file_sha256": _sha256_file(feature_path), "manifest_sha256": feature.manifest_sha256, "version": feature.cache_version}, "pseudo_label_cache": {"logical_path": "manifests/features/deam-pseudo-label-bundle-v3.json", "file_sha256": _sha256_file(pseudo_path), "manifest_sha256": pseudo["manifest_sha256"], "version": pseudo["cache_version"]}, "population": {"train_primary": 995, "test_long_song": 58, "validation_policy": "UNDEFINED_PENDING_SEPARATE_FREEZE"}, "alignment_rule": {"timegrid_hz": 2, "bin_count": 90, "duration_seconds": 45, "boundary": "HALF_OPEN_SAMPLE_DOMAIN", "forbidden_operations": ["TRIM", "PADDING", "INTERPOLATION", "RESHAPE", "ADAPTIVE_POOLING"]}, "review_sample": {"selection_rule": "SHA256(song_id UTF-8) ascending, song_id tie-break", "train_song_ids": sample}, "checks": checks}
    payload["evidence_sha256"] = sha256_canonical(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload
