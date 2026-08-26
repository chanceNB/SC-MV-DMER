"""Run the read-only DEAM-PDMER-CLEAN-v2 Step 2/RG-02 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc_mv_dmer.data.step2_preflight import evaluate_step2_rg02_v2
from sc_mv_dmer.foundation.canonical import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output or repo_root / "evidence/gates/rg-02/deam-pdmer-clean-v2-preflight-v1.json"
    evidence = evaluate_step2_rg02_v2(repo_root=repo_root, data_root=args.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v2 preflight evidence: {output}")
    output.write_text(canonical_json(evidence) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(output), "verdict": evidence["verdict"], "evidence_sha256": evidence["evidence_sha256"], "blockers": evidence["blockers"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
