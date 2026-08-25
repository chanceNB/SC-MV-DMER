"""Freeze the first authorized DEAM primary song split."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from sc_mv_dmer.data.manifests import DatasetManifest, verify_dataset_manifest_checksum
from sc_mv_dmer.data.splits import freeze_primary_split, write_split_once


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = Path(args.dataset_manifest)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    verify_dataset_manifest_checksum(manifest)
    split = freeze_primary_split(
        manifest,
        dataset_manifest_logical_path="manifests/datasets/deam-primary-v1.json",
        dataset_manifest_file_sha256=_sha256(manifest_path),
    )
    write_split_once(split, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
