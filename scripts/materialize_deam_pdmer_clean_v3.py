"""Materialize the approved DEAM-PDMER-CLEAN-v3 data snapshot only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc_mv_dmer.data.deam_clean_v3 import materialize_v3_from_v2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    data_root = args.data_root
    result = materialize_v3_from_v2(
        v2_manifest_path=repo / "manifests/datasets/deam-pdmer-clean-v2.json",
        v2_processed_root=data_root / "processed/deam-pdmer-clean-v2",
        output_root=data_root / "processed/deam-pdmer-clean-v3",
        repo_manifest_path=repo / "manifests/datasets/deam-pdmer-clean-v3.json",
        cleaning_audit_path=repo / "reports/data/deam-pdmer-clean-v3-audit.json",
    )
    print(json.dumps({"manifest_id": result["manifest"]["manifest_id"], "dataset_id": result["manifest"]["dataset_id"], "train_primary": result["manifest"]["population"]["retained_train_primary_count"], "test_long_song": result["manifest"]["population"]["retained_test_long_song_count"], "dmer_count": result["dmer_count"], "pdmer_task_count": result["pdmer_task_count"], "pdmer_song_assignment_count": result["pdmer_song_assignment_count"], "short_excluded": result["short_excluded"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
