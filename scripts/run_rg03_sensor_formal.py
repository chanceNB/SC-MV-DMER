"""RG-03 Sensor-only runner; default operation is contract-only dry-run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sc_mv_dmer.gates.rg03 import dry_run
from sc_mv_dmer.training.sensor_formal import train_formal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--train", action="store_true", help="run authorized Sensor-only formal training")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    artifact_root = args.artifact_root
    if artifact_root is None and os.environ.get("SC_MV_DMER_ARTIFACT_ROOT"):
        artifact_root = Path(os.environ["SC_MV_DMER_ARTIFACT_ROOT"])
    if args.train:
        if artifact_root is None:
            parser.error("--artifact-root or SC_MV_DMER_ARTIFACT_ROOT is required for training")
        if args.output_root is None:
            parser.error("--output-root is required for training")
        result = train_formal(repo_root=args.repo_root, artifact_root=artifact_root, output_root=args.output_root, device=args.device, max_epochs=args.max_epochs, batch_size=args.batch_size)
    else:
        result = dry_run(args.repo_root, artifact_root=artifact_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
