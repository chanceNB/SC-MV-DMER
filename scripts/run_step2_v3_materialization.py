"""Run Step 2 v3 feature/pseudo-label cache materialization only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sc_mv_dmer.features.materialize_v3 import materialize_v3_step2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("SC_MV_DMER_DATA_ROOT", r"E:\数据集\_pdmer_compat")))
    parser.add_argument("--mert-model-root", type=Path, default=Path(os.environ.get("SC_MV_DMER_MERT_MODEL_ROOT", r"D:\SC-MV-DMER-data\upstream_models\mert-v1-95m\12af15fef9d0ac838c3f475bfbbf26d2060dd4f5")))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    result = materialize_v3_step2(
        repo_root=repo,
        data_root=args.data_root,
        mert_model_root=args.mert_model_root,
        dataset_manifest_path=repo / "manifests/datasets/deam-pdmer-clean-v3.json",
        upstream_manifest_path=repo / "manifests/upstream/mert-v1-95m-primary-v1.json",
        dimensions_path=repo / "manifests/dimensions/mert-downstream-dimensions-v2.json",
        timegrid_path=repo / "manifests/timegrid/mert-2hz-timegrid-v2.json",
        rg01_path=repo / "evidence/gates/rg-01/rg-01-attempt-v1.json",
        feature_cache_path=repo / "manifests/features/deam-four-view-cache-v3.json",
        pseudo_cache_path=repo / "manifests/features/deam-pseudo-label-bundle-v3.json",
        alignment_evidence_path=repo / "evidence/gates/rg-02/deam-pdmer-clean-v3-alignment-v1.json",
    )
    feature = result["feature_manifest"]
    pseudo = result["pseudo_manifest"]
    print(json.dumps({"feature_cache_manifest_sha256": feature["manifest_sha256"], "pseudo_label_manifest_sha256": pseudo["manifest_sha256"], "alignment_verdict": result["alignment"]["verdict"], "alignment_evidence_sha256": result["alignment"]["evidence_sha256"], "record_count": feature["record_count"], "population_counts": feature["population_counts"], "training": False, "checkpoint": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
