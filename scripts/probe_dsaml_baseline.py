"""Read-only readiness probe for the pinned official-feature DSAML baseline."""

import argparse
import json
from pathlib import Path

from sc_mv_dmer.models.dsaml import probe_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.feature_manifest.read_text(encoding="utf-8")) if args.feature_manifest else None
    print(json.dumps(probe_readiness(metadata), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
