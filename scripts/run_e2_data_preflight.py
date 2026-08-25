"""Discover DEAM sources and emit E2 split-readiness evidence.

This runner never creates split membership.  A missing approved split is a
machine-readable blocker, as required by the frozen data contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from sc_mv_dmer.data.deam import bind_deam_primary, discover_deam, write_json_once
from sc_mv_dmer.data.manifests import verify_dataset_manifest_checksum, verify_inventory_checksum
from sc_mv_dmer.data.splits import FrozenSplitManifestUnavailable, load_frozen_split, verify_frozen_split
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout.strip())
    return commit, dirty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inventory_path = root / "reports/preflight/deam-source-inventory.json"
    dataset_path = root / "manifests/datasets/deam-primary-v1.json"
    split_path = root / "manifests/splits/deam-primary-song-80-10-10-v1.json"
    evidence_path = root / "evidence/data/e2-deam-primary-split-ready-v2.json"
    report_path = root / "reports/data/e2-deam-primary-split-ready-v2-report.txt"

    commit, dirty = _git_state(root)
    inventory = discover_deam(args.data_root)
    verify_inventory_checksum(inventory)
    if inventory_path.exists():
        from sc_mv_dmer.data.manifests import SourceInventory
        registered_inventory = SourceInventory.model_validate_json(inventory_path.read_text(encoding="utf-8"))
        verify_inventory_checksum(registered_inventory)
        if registered_inventory.inventory_sha256 != inventory.inventory_sha256:
            raise RuntimeError("registered source inventory checksum differs from current read-only audit")
    else:
        write_json_once(inventory.model_dump(mode="json"), inventory_path)
    manifest = bind_deam_primary(inventory, inventory_logical_path="reports/preflight/deam-source-inventory.json")
    verify_dataset_manifest_checksum(manifest)
    if dataset_path.exists():
        from sc_mv_dmer.data.manifests import DatasetManifest
        registered_manifest = DatasetManifest.model_validate_json(dataset_path.read_text(encoding="utf-8"))
        verify_dataset_manifest_checksum(registered_manifest)
        if registered_manifest.manifest_sha256 != manifest.manifest_sha256:
            raise RuntimeError("registered dataset manifest checksum differs from current read-only binding")
        manifest = registered_manifest
    else:
        write_json_once(manifest.model_dump(mode="json"), dataset_path)

    blocker = None
    split_verification = None
    try:
        split = load_frozen_split(split_path)
        split_verification = verify_frozen_split(
            manifest,
            split,
            dataset_manifest_file_sha256=_file_sha256(dataset_path),
        )
    except FrozenSplitManifestUnavailable as exc:
        blocker = str(exc)
    verdict = "PASS" if split_verification is not None and not dirty else "BLOCKED"
    payload = {
        "schema_version": "1.0",
        "qualification_id": "E2_DEAM_PRIMARY_SPLIT_READY-v1",
        "verdict": verdict,
        "current_effective": verdict,
        "blocker": blocker,
        "evidence_commit": commit,
        "observed_git_dirty": dirty,
        "data_root_is_external": True,
        "inventory_logical_path": "reports/preflight/deam-source-inventory.json",
        "inventory_file_sha256": _file_sha256(inventory_path),
        "inventory_sha256": inventory.inventory_sha256,
        "dataset_manifest_logical_path": "manifests/datasets/deam-primary-v1.json",
        "dataset_manifest_file_sha256": _file_sha256(dataset_path),
        "dataset_manifest_sha256": manifest.manifest_sha256,
        "split_manifest_logical_path": "manifests/splits/deam-primary-song-80-10-10-v1.json",
        "split_manifest_present": split_path.is_file(),
        "split_manifest_file_sha256": _file_sha256(split_path) if split_path.is_file() else None,
        "split_manifest_sha256": split.get("split_sha256") if split_path.is_file() else None,
        "primary_population_count": len(manifest.primary_records),
        "long_song_count": len(manifest.long_song_keys),
        "audio_source_count": len(inventory.audio_song_keys),
        "annotation_song_count": len(inventory.annotation_song_keys),
        "primary_song_keys_sha256": sha256_canonical(list(manifest.primary_song_keys)),
        "long_song_keys_sha256": sha256_canonical(list(manifest.long_song_keys)),
        "split_verification": split_verification,
        "forbidden_actions": ["GENERATE_REPLACEMENT_SPLIT", "MODIFY_FROZEN_SPLIT_MEMBERS", "FEATURE_EXTRACTION", "FEATURE_CACHE", "TRAINING"],
    }
    payload["qualification_sha256"] = sha256_canonical(payload)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("x", encoding="utf-8") as output:
        output.write(canonical_json(payload) + "\n")
    lines = [
        "=== E2_DEAM_PRIMARY_SPLIT_READY PRE-FLIGHT ===",
        f"verdict={verdict}",
        f"primary_population_count={len(manifest.primary_records)}",
        f"long_song_count={len(manifest.long_song_keys)}",
        f"audio_source_count={len(inventory.audio_song_keys)}",
        f"annotation_song_count={len(inventory.annotation_song_keys)}",
        f"split_manifest_present={split_path.is_file()}",
        f"blocker={blocker or 'none'}",
    ]
    with report_path.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
