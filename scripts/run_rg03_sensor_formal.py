"""RG-03 Sensor-only runner; default operation is contract-only dry-run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sc_mv_dmer.gates.rg03 import dry_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    artifact_root = args.artifact_root
    if artifact_root is None and os.environ.get("SC_MV_DMER_ARTIFACT_ROOT"):
        artifact_root = Path(os.environ["SC_MV_DMER_ARTIFACT_ROOT"])
    print(json.dumps(dry_run(args.repo_root, artifact_root=artifact_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
