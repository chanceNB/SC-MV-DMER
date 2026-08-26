"""Write 995-population sampler and schedule calculations without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc_mv_dmer.data.training_contract_preflight import build_v3_training_contract
from sc_mv_dmer.foundation.canonical import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    manifest = json.loads((repo / "manifests/datasets/deam-pdmer-clean-v3.json").read_text(encoding="utf-8"))
    artifact = build_v3_training_contract(manifest_sha256=manifest["manifest_sha256"])
    output = repo / "reports/data/deam-pdmer-clean-v3-training-contract-preflight.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable training preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(artifact) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "semantic_config_hash": artifact["semantic_config_hash"], "resolved_config_hash": artifact["resolved_config_hash"], "budget": artifact["resolved"]["budget"], "training_authorized": artifact["training_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
