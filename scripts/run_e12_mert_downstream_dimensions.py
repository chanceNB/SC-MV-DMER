"""Execute E1.2 MERT downstream dimension derivation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from sc_mv_dmer.features.mert_temporal_geometry import load_pinned_mert_geometry
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.dimensions import (
    E12DimensionQualification,
    compile_downstream_dimensions,
    file_sha256,
    load_upstream_manifest,
    verify_downstream_binding,
    write_immutable_dimension_binding,
)
from sc_mv_dmer.features.mert_snapshot import (
    PinnedMertUpstreamManifest,
    verify_pinned_mert_manifest,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_preflight(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _verify_terminal_hash(path: Path, field: str) -> tuple[dict[str, Any], str]:
    payload = _load_json(path)
    expected = payload.get(field)
    if not isinstance(expected, str):
        raise ValueError(f"missing {field}: {path}")
    unsigned = dict(payload)
    del unsigned[field]
    if sha256_canonical(unsigned) != expected:
        raise ValueError(f"terminal checksum mismatch: {path}")
    return payload, file_sha256(path)


def _write_exclusive(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(canonical_json(payload) + "\n")


def run_e12(
    *,
    repository_root: Path,
    e11_evidence_path: Path,
    e11_qualification_path: Path,
    upstream_manifest_path: Path,
    snapshot_root: Path,
    binding_path: Path,
    qualification_path: Path,
    report_path: Path,
    report_json_path: Path,
) -> E12DimensionQualification:
    targets = (binding_path, qualification_path, report_path, report_json_path)
    existing = [str(path) for path in targets if Path(path).exists()]
    if existing:
        raise FileExistsError("E1.2 terminal artifact already exists: " + ", ".join(existing))

    commit, dirty = _git_preflight(repository_root)
    e11_evidence, e11_file_sha256 = _verify_terminal_hash(e11_evidence_path, "report_sha256")
    e11_qualification, e11_qualification_file_sha256 = _verify_terminal_hash(
        e11_qualification_path, "qualification_sha256"
    )
    if e11_qualification.get("e1_1_verdict") != "PASS":
        raise ValueError("E1.2 requires current-effective E1.1 PASS")
    # E1.1 keeps the machine-readable forward report and qualification
    # separate.  Pass the qualification verdict alongside the report facts;
    # the recorded evidence checksum still refers to the untouched report.
    e11_evidence["e1_1_verdict"] = e11_qualification["e1_1_verdict"]
    upstream_payload = load_upstream_manifest(upstream_manifest_path)
    pinned_manifest = PinnedMertUpstreamManifest.model_validate(upstream_payload)
    verify_pinned_mert_manifest(pinned_manifest, snapshot_root)
    geometry = load_pinned_mert_geometry(
        snapshot_root,
        input_sample_rate_hz=24_000,
        input_sample_count=1_080_000,
        expected_config_sha256=pinned_manifest.config_sha256,
        expected_custom_code_aggregate_sha256=pinned_manifest.custom_code_aggregate_sha256,
    )
    binding = compile_downstream_dimensions(
        e11_evidence=e11_evidence,
        e11_evidence_sha256=e11_file_sha256,
        predecessor_qualification=str(e11_qualification_path.relative_to(repository_root)).replace(
            "\\", "/"
        ),
        predecessor_qualification_sha256=e11_qualification_file_sha256,
        upstream_manifest=upstream_payload,
        upstream_manifest_sha256=file_sha256(upstream_manifest_path),
        temporal_geometry=geometry,
    )
    verify_downstream_binding(binding)
    write_immutable_dimension_binding(binding, binding_path)
    binding_file_sha256 = file_sha256(binding_path)

    report_text = "\n".join(
        (
            "E1.2 MERT downstream dimension derivation",
            "verdict = PASS",
            f"T_raw = {binding.observed_t_raw}",
            f"observed_fps = {binding.observed_fps:.17g} Hz",
            f"effective_stride_samples = {geometry.effective_stride_samples}",
            f"effective_receptive_field_samples = {geometry.effective_receptive_field_samples}",
            f"expected_frame_count_from_contract = {geometry.expected_frame_count}",
            f"TimesNet raw temporal input = {binding.timesnet_raw_temporal_input_dimension}",
            "TimesNet period interpretation = runtime FFT top-k over measured raw frames",
            f"MERT_to_2Hz_ratio = {binding.required_temporal_reduction.ratio_numerator}/{binding.required_temporal_reduction.ratio_denominator}",
            f"canonical_downstream_T = {binding.canonical_downstream_t}",
            "forbidden = trim,padding,interpolation,reshape,adaptive_pooling",
            f"binding_sha256 = {binding.binding_sha256}",
            f"binding_file_sha256 = {binding_file_sha256}",
        )
    ) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as output:
        output.write(report_text)
    report_json_payload = {
        "schema_version": "1.0",
        "qualification_id": "E1-2-MERT-DOWNSTREAM-DIMENSION-DERIVATION-v1",
        "binding_logical_path": str(binding_path.relative_to(repository_root)).replace("\\", "/"),
        "binding_sha256": binding.binding_sha256,
        "binding_file_sha256": binding_file_sha256,
        "observed_t_raw": binding.observed_t_raw,
        "observed_fps": binding.observed_fps,
        "canonical_downstream_t": binding.canonical_downstream_t,
        "forbidden_actions": list(binding.forbidden_downstream_actions),
    }
    _write_exclusive(report_json_payload, report_json_path)

    payload = {
        "schema_version": "1.0",
        "qualification_id": "E1-2-MERT-DOWNSTREAM-DIMENSION-DERIVATION-v1",
        "implementation_git_commit": commit,
        "observed_git_dirty": dirty,
        "predecessor_qualification": str(e11_qualification_path.relative_to(repository_root)).replace(
            "\\", "/"
        ),
        "predecessor_qualification_sha256": e11_qualification_file_sha256,
        "e1_1_evidence_logical_path": str(e11_evidence_path.relative_to(repository_root)).replace(
            "\\", "/"
        ),
        "e1_1_evidence_sha256": e11_file_sha256,
        "binding_logical_path": str(binding_path.relative_to(repository_root)).replace("\\", "/"),
        "binding_file_sha256": binding_file_sha256,
        "binding_sha256": binding.binding_sha256,
        "temporal_geometry_sha256": binding.temporal_geometry_sha256,
        "observed_t_raw": binding.observed_t_raw,
        "observed_fps": binding.observed_fps,
        "canonical_downstream_t": 90,
        "forbidden_actions": list(binding.forbidden_downstream_actions),
        "e1_2_verdict": "PASS",
        "e1_3_readiness": "READY",
    }
    qualification = E12DimensionQualification(
        **payload,
        qualification_sha256=sha256_canonical(payload),
    )
    _write_exclusive(qualification.model_dump(mode="json"), qualification_path)
    return qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    root = _repo_root()
    parser.add_argument("--e11-evidence", type=Path, default=root / "reports/experiments/e1-1-mert-real-frame-rate.json")
    parser.add_argument("--e11-qualification", type=Path, default=root / "evidence/qualifications/e1-1-mert-frame-rate-verification.json")
    parser.add_argument("--upstream-manifest", type=Path, default=root / "manifests/upstream/mert-v1-95m-primary-v1.json")
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--binding", type=Path, default=root / "manifests/dimensions/mert-downstream-dimensions-v1.json")
    parser.add_argument("--qualification", type=Path, default=root / "evidence/qualifications/e1-2-mert-downstream-dimension-derivation.json")
    parser.add_argument("--report", type=Path, default=root / "reports/experiments/e1-2-mert-downstream-dimension-report.txt")
    parser.add_argument("--report-json", type=Path, default=root / "reports/experiments/e1-2-mert-downstream-dimension.json")
    args = parser.parse_args()
    snapshot_root = args.snapshot_root
    if snapshot_root is None:
        inventory = _load_json(root / "reports/preflight/mert-pinned-snapshot-inventory.json")
        snapshot_root = Path(str(inventory["model_root_runtime"]))
    run_e12(
        repository_root=root,
        e11_evidence_path=args.e11_evidence,
        e11_qualification_path=args.e11_qualification,
        upstream_manifest_path=args.upstream_manifest,
        snapshot_root=snapshot_root,
        binding_path=args.binding,
        qualification_path=args.qualification,
        report_path=args.report,
        report_json_path=args.report_json,
    )
    print("E1.2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
