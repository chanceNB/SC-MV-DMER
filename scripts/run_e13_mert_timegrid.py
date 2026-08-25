"""Execute E1.3 MERT 2 Hz/90-bin TimeGrid binding."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from sc_mv_dmer.features.mert_timegrid import (
    E13TimeGridQualification,
    MertTimeGridBinding,
    compile_mert_timegrid,
    verify_timegrid_binding,
    write_immutable_timegrid_binding,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.dimensions import (
    DownstreamDimensionBinding,
    file_sha256,
    verify_downstream_binding,
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


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _verify_terminal_hash(path: Path, field: str) -> tuple[dict[str, Any], str]:
    payload = _load_object(path)
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


def run_e13(
    *,
    repository_root: Path,
    dimension_binding_path: Path,
    e12_qualification_path: Path,
    timegrid_path: Path,
    qualification_path: Path,
    report_path: Path,
    report_json_path: Path,
) -> E13TimeGridQualification:
    targets = (timegrid_path, qualification_path, report_path, report_json_path)
    existing = [str(path) for path in targets if Path(path).exists()]
    if existing:
        raise FileExistsError("E1.3 terminal artifact already exists: " + ", ".join(existing))

    commit, dirty = _git_preflight(repository_root)
    binding_payload, binding_file_sha256 = _verify_terminal_hash(
        dimension_binding_path, "binding_sha256"
    )
    e12_payload, e12_file_sha256 = _verify_terminal_hash(
        e12_qualification_path, "qualification_sha256"
    )
    if e12_payload.get("e1_2_verdict") != "PASS" or e12_payload.get("e1_3_readiness") != "READY":
        raise ValueError("E1.3 requires current-effective E1.2 PASS/READY")
    binding = DownstreamDimensionBinding.model_validate(binding_payload)
    verify_downstream_binding(binding)
    timegrid = compile_mert_timegrid(
        geometry=binding.temporal_geometry,
        dimension_binding_logical_path=str(dimension_binding_path.relative_to(repository_root)).replace(
            "\\", "/"
        ),
        dimension_binding_sha256=binding_file_sha256,
        predecessor_qualification=str(e12_qualification_path.relative_to(repository_root)).replace(
            "\\", "/"
        ),
        predecessor_qualification_sha256=e12_file_sha256,
        temporal_geometry_sha256=binding.temporal_geometry_sha256,
    )
    verify_timegrid_binding(timegrid)
    write_immutable_timegrid_binding(timegrid, timegrid_path)
    timegrid_file_sha256 = file_sha256(timegrid_path)

    report_text = "\n".join(
        (
            "E1.3 MERT 2 Hz/90-bin TimeGrid binding",
            "verdict = PASS",
            f"expected_frame_count_from_contract = {timegrid.expected_frame_count}",
            f"observed/raw_frame_count = {timegrid.raw_frame_count}",
            f"effective_stride_samples = {timegrid.temporal_geometry.effective_stride_samples}",
            f"effective_receptive_field_samples = {timegrid.temporal_geometry.effective_receptive_field_samples}",
            f"assigned_frame_count = {timegrid.assigned_frame_count}",
            f"unassigned_frame_count = {timegrid.unassigned_frame_count}",
            f"duplicate_assignment_count = {timegrid.duplicate_assignment_count}",
            f"valid_bin_count = {timegrid.valid_bin_count}",
            f"bin_frame_count_min_max = {timegrid.min_bin_frame_count},{timegrid.max_bin_frame_count}",
            "boundary_rule = half-open sample-domain bins [0,0.5),...,[44.5,45.0)",
            "aggregation = mean over assigned valid frames",
            "forbidden = trim,padding,drop,clamp,copy,fill,interpolation,adaptive_pooling,reshape",
            f"binding_sha256 = {timegrid.binding_sha256}",
            f"binding_file_sha256 = {timegrid_file_sha256}",
        )
    ) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as output:
        output.write(report_text)
    report_json_payload = {
        "schema_version": "1.0",
        "qualification_id": "E1-3-MERT-TIMEGRID-2HZ-BINDING-v1",
        "timegrid_logical_path": str(timegrid_path.relative_to(repository_root)).replace("\\", "/"),
        "timegrid_binding_sha256": timegrid.binding_sha256,
        "timegrid_file_sha256": timegrid_file_sha256,
        "raw_frame_count": timegrid.raw_frame_count,
        "expected_frame_count": timegrid.expected_frame_count,
        "bin_frame_counts": list(timegrid.bin_frame_counts),
        "unassigned_frame_count": timegrid.unassigned_frame_count,
        "duplicate_assignment_count": timegrid.duplicate_assignment_count,
    }
    _write_exclusive(report_json_payload, report_json_path)

    payload = {
        "schema_version": "1.0",
        "qualification_id": "E1-3-MERT-TIMEGRID-2HZ-BINDING-v1",
        "implementation_git_commit": commit,
        "observed_git_dirty": dirty,
        "predecessor_qualification": str(e12_qualification_path.relative_to(repository_root)).replace(
            "\\", "/"
        ),
        "predecessor_qualification_sha256": e12_file_sha256,
        "dimension_binding_logical_path": str(dimension_binding_path.relative_to(repository_root)).replace(
            "\\", "/"
        ),
        "dimension_binding_sha256": binding_file_sha256,
        "timegrid_logical_path": str(timegrid_path.relative_to(repository_root)).replace("\\", "/"),
        "timegrid_file_sha256": timegrid_file_sha256,
        "timegrid_binding_sha256": timegrid.binding_sha256,
        "temporal_geometry_sha256": timegrid.temporal_geometry_sha256,
        "raw_frame_count": timegrid.raw_frame_count,
        "expected_frame_count": timegrid.expected_frame_count,
        "bin_count": 90,
        "assigned_frame_count": timegrid.assigned_frame_count,
        "unassigned_frame_count": 0,
        "duplicate_assignment_count": 0,
        "min_bin_frame_count": timegrid.min_bin_frame_count,
        "max_bin_frame_count": timegrid.max_bin_frame_count,
        "aggregation_rule": timegrid.aggregation_rule,
        "forbidden_actions": list(timegrid.forbidden_operations),
        "e1_3_verdict": "PASS",
        "e1_4_readiness": "READY",
    }
    qualification = E13TimeGridQualification(
        **payload,
        qualification_sha256=sha256_canonical(payload),
    )
    _write_exclusive(qualification.model_dump(mode="json"), qualification_path)
    return qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    root = _repo_root()
    parser.add_argument("--dimension-binding", type=Path, default=root / "manifests/dimensions/mert-downstream-dimensions-v1.json")
    parser.add_argument("--e12-qualification", type=Path, default=root / "evidence/qualifications/e1-2-mert-downstream-dimension-derivation.json")
    parser.add_argument("--timegrid", type=Path, default=root / "manifests/timegrid/mert-2hz-timegrid-v1.json")
    parser.add_argument("--qualification", type=Path, default=root / "evidence/qualifications/e1-3-mert-timegrid-binding.json")
    parser.add_argument("--report", type=Path, default=root / "reports/experiments/e1-3-mert-timegrid-report.txt")
    parser.add_argument("--report-json", type=Path, default=root / "reports/experiments/e1-3-mert-timegrid.json")
    args = parser.parse_args()
    run_e13(
        repository_root=root,
        dimension_binding_path=args.dimension_binding,
        e12_qualification_path=args.e12_qualification,
        timegrid_path=args.timegrid,
        qualification_path=args.qualification,
        report_path=args.report,
        report_json_path=args.report_json,
    )
    print("E1.3 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
