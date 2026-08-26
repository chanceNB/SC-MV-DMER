"""Run the approved v3 data-only RG-02 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc_mv_dmer.data.step2_preflight_v3 import evaluate_step2_rg02_v3
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    evidence = evaluate_step2_rg02_v3(repo_root=repo, data_root=args.data_root)
    evidence_path = repo / "evidence/gates/rg-02/deam-pdmer-clean-v3-preflight-v1.json"
    audit_path = repo / "reports/data/deam-pdmer-clean-v3-preflight-audit.json"
    for path in (evidence_path, audit_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite immutable v3 preflight artifact: {path}")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
    audit_unsigned = {"schema_version": "1.0", "audit_id": "DEAM-PDMER-PREFLIGHT-AUDIT-v3", "dataset_manifest_id": evidence["dataset_manifest"]["manifest_id"], "dataset_manifest_sha256": evidence["dataset_manifest"]["internal_sha256"], "rg02_evidence_logical_path": "evidence/gates/rg-02/deam-pdmer-clean-v3-preflight-v1.json", "rg02_evidence_sha256": evidence["evidence_sha256"], "verdict": evidence["verdict"], "blockers": evidence["blockers"], "checks": evidence["checks"], "data_root_config_key": "SC_MV_DMER_DATA_ROOT", "cache_published": False, "pseudo_label_cache_published": False, "training_artifacts_published": False}
    audit = audit_unsigned | {"audit_sha256": sha256_canonical(audit_unsigned)}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(evidence_path), "preflight_audit": str(audit_path), "verdict": evidence["verdict"], "evidence_sha256": evidence["evidence_sha256"], "audit_sha256": audit["audit_sha256"], "blockers": evidence["blockers"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
