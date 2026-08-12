from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sc_mv_dmer import app_name
from sc_mv_dmer.data.discovery import discover_deam_source
from sc_mv_dmer.data.probes import build_probe, write_probe_manifest
from sc_mv_dmer.features.mert_upstream import discover_mert_local
from sc_mv_dmer.foundation.capabilities import CapabilityRole, DependencyValidity
from sc_mv_dmer.foundation.config import RunMode, resolve_config
from sc_mv_dmer.foundation.preflight import observe_preflight
from sc_mv_dmer.foundation.qualification import (
    load_mert_status,
    load_probe_manifest,
    qualify_e0_minimum_bootstrap,
    write_terminal_qualification,
)


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
    qualify = subcommands.add_parser("qualify-step")
    qualify.add_argument("--step", type=int, required=True)
    qualify.add_argument("--mode", choices=(RunMode.FORMAL.value, RunMode.DEBUG.value), required=True)
    qualify.add_argument("--probe-manifest", type=Path, required=True)
    qualify.add_argument("--mert-status", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
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
    if args.command == "qualify-step":
        if args.step != 0:
            raise ValueError("only E0 minimum bootstrap qualification is implemented")
        repository_root = Path(__file__).resolve().parents[2]
        config = resolve_config(
            [repository_root / "configs" / "research" / "defaults.yaml"],
            {},
            RunMode(args.mode),
        )
        observation = observe_preflight(repository_root, config)
        bundle = qualify_e0_minimum_bootstrap(
            observation,
            load_probe_manifest(args.probe_manifest),
            load_mert_status(args.mert_status),
            {
                "E0_MINIMUM_BOOTSTRAP": DependencyValidity.VALID,
                CapabilityRole.MERT_UPSTREAM_IDENTITY: DependencyValidity.VALID,
                CapabilityRole.MERT_PROBE_IDENTITY: DependencyValidity.VALID,
                CapabilityRole.MERT_PROBE_REPRESENTATION: DependencyValidity.VALID,
            },
        )
        write_terminal_qualification(bundle, args.output)
        return 0
    print(app_name())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
