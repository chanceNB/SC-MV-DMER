from pathlib import Path
import json
import os

from sc_mv_dmer.features.materialize import materialize_population
from sc_mv_dmer.gates.rg02 import evaluate_rg02_automatic
from sc_mv_dmer.features.alignment_review import build_alignment_package

repo = Path(__file__).resolve().parents[1]
data_root = Path(os.environ.get("SC_MV_DMER_DATA_ROOT", r"E:\数据集\_pdmer_compat"))
cache_path = repo / "manifests/features/deam-four-view-cache-v1.json"
pseudo_path = repo / "manifests/features/deam-pseudo-label-bundle-v1.json"
cache, pseudo = materialize_population(
    repo_root=repo,
    data_root=data_root,
    dataset_manifest_path=repo / "manifests/datasets/deam-primary-v1.json",
    split_manifest_path=repo / "manifests/splits/deam-primary-song-80-10-10-v1.json",
    upstream_manifest_path=repo / "manifests/upstream/mert-v1-95m-primary-v1.json",
    dimensions_path=repo / "manifests/dimensions/mert-downstream-dimensions-v2.json",
    timegrid_path=repo / "manifests/timegrid/mert-2hz-timegrid-v2.json",
    rg01_path=repo / "evidence/gates/rg-01/rg-01-attempt-v1.json",
    mert_model_root=Path(os.environ.get("SC_MV_DMER_MERT_MODEL_ROOT", r"D:\SC-MV-DMER-data\upstream_models\mert-v1-95m\12af15fef9d0ac838c3f475bfbbf26d2060dd4f5")),
    cache_manifest_path=cache_path,
    pseudo_manifest_path=pseudo_path,
)
audit = evaluate_rg02_automatic(cache_path, evidence_path=repo / "evidence/gates/rg-02/automatic-audit-v1.json")
package = build_alignment_package(cache_manifest_path=cache_path, split_manifest_path=repo / "manifests/splits/deam-primary-song-80-10-10-v1.json", output_dir=repo / "evidence/gates/rg-02/review-package-v1", automatic_audit=audit)
print(json.dumps({"cache_manifest_sha256": cache.manifest_sha256, "pseudo_manifest_sha256": pseudo["manifest_sha256"], "audit": audit, "review_status": package["status"]}, sort_keys=True))
