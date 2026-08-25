"""RG-02 automatic cache audit; manual review remains a separate authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sc_mv_dmer.features.cache import FourViewFeatureCacheManifest, verify_cache_manifest
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


def evaluate_rg02_automatic(cache: FourViewFeatureCacheManifest | Path, *, expected_count: int = 1744, evidence_path: Path | None = None) -> dict[str, Any]:
    manifest = verify_cache_manifest(cache) if isinstance(cache, Path) else cache
    failures: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in manifest.records:
        if record.song_id in seen:
            failures.append({"sample_id": record.sample_id, "reason": "duplicate_song_id"})
        seen.add(record.song_id)
        payload_path = Path(cache).parent.parent.parent / record.payload_logical_path if isinstance(cache, Path) else None
        if payload_path is not None and not payload_path.is_file():
            failures.append({"sample_id": record.sample_id, "reason": "payload_missing"})
            continue
        if payload_path is not None:
            actual = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            if actual != record.payload_sha256:
                failures.append({"sample_id": record.sample_id, "reason": "payload_checksum_mismatch"})
            try:
                with np.load(payload_path) as payload:
                    for view in ("deep", "mel", "mfcc", "chroma"):
                        array = payload[view]
                        expected_dim = {"deep": 768, "mel": 128, "mfcc": 40, "chroma": 12}[view]
                        if array.shape != (90, expected_dim) or not np.isfinite(array).all():
                            failures.append({"sample_id": record.sample_id, "reason": f"{view}_shape_or_finite"})
            except Exception as exc:
                failures.append({"sample_id": record.sample_id, "reason": f"payload_read:{type(exc).__name__}"})
    passed = manifest.record_count == expected_count and len(manifest.records) == expected_count and not failures and len(seen) == expected_count
    result: dict[str, Any] = {"schema_version": "1.0", "gate": "RG-02", "authority_kind": "automatic", "verdict": "PASS" if passed else "FAIL", "record_count": manifest.record_count, "expected_record_count": expected_count, "failure_count": len(failures), "failure_sample_ids": [item["sample_id"] for item in failures], "failures": failures, "cache_manifest_sha256": manifest.manifest_sha256}
    result["evidence_sha256"] = sha256_canonical(result)
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result
