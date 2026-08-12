from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sc_mv_dmer import app_name
from sc_mv_dmer.data.discovery import discover_deam_source
from sc_mv_dmer.data.probes import build_probe, write_probe_manifest
from sc_mv_dmer.features.mert_upstream import discover_mert_local


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sc-mv-dmer")
    subcommands = parser.add_subparsers(dest="command")
    register = subcommands.add_parser("register-deam-probe")
    register.add_argument("--data-root", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)
    mert_status = subcommands.add_parser("mert-status")
    mert_status.add_argument("--model-root", type=Path)
    mert_status.add_argument("--output", type=Path, required=True)
    mert_status.add_argument("--supersedes")
    mert_status.add_argument("--correction-reason")
    args = parser.parse_args(argv)
    if args.command == "register-deam-probe":
        write_probe_manifest(build_probe(discover_deam_source(args.data_root)), args.output)
        return 0
    if args.command == "mert-status":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = discover_mert_local(args.model_root).model_dump(mode="json")
        if args.supersedes:
            payload["supersedes"] = args.supersedes
            payload["correction_reason"] = args.correction_reason or "checksum authentication correction"
        with args.output.open("x", encoding="utf-8") as artifact:
            artifact.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    print(app_name())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
