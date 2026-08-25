"""Deterministic RG-02 human-review package generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


def select_alignment_sample(train_song_ids: list[str] | tuple[str, ...], *, count: int = 3) -> tuple[str, ...]:
    if count != 3:
        raise ValueError("RG-02 review sample is fixed at three train songs")
    if len(set(train_song_ids)) != len(train_song_ids):
        raise ValueError("train song IDs must be unique")
    return tuple(sorted(train_song_ids, key=lambda song_id: (hashlib.sha256(song_id.encode("utf-8")).hexdigest(), song_id))[:3])


def build_alignment_package(*, cache_manifest_path: Path, split_manifest_path: Path, output_dir: Path, automatic_audit: dict[str, Any]) -> dict[str, Any]:
    split = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    sample_ids = select_alignment_sample(tuple(split["train_song_ids"]))
    payload = {"schema_version": "1.0", "package_id": "RG-02-MANUAL-REVIEW-v1", "selection_rule": "SHA256(song_id UTF-8) ascending, song_id tie-break", "selection_population": "frozen train split only", "song_ids": list(sample_ids), "cache_manifest_logical_path": "manifests/features/deam-four-view-cache-v1.json", "cache_manifest_sha256": hashlib.sha256(Path(cache_manifest_path).read_bytes()).hexdigest(), "automatic_audit_verdict": automatic_audit.get("verdict"), "reviewer_records": [], "status": "PENDING_MANUAL_REVIEW"}
    payload["package_sha256"] = sha256_canonical(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "package.json").write_text(canonical_json(payload) + "\n", encoding="utf-8")
    (output_dir / "README.txt").write_text("RG-02 reviewer fields intentionally empty. Human review is required; this package does not constitute PASS.\n", encoding="utf-8")
    return payload


class HumanAuthorityRequired(RuntimeError):
    pass


def finalize_rg02(automatic_audit: dict[str, Any], reviewer_records: list[dict[str, Any]]) -> str:
    if not reviewer_records:
        raise HumanAuthorityRequired("real human reviewer records are required for RG-02 PASS")
    if automatic_audit.get("verdict") != "PASS":
        return "FAIL"
    return "PASS"
