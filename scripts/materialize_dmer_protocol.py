"""Write a new full-population split and audited 45s/30s window inventory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from sc_mv_dmer.data.dmer_protocol import (
    DIMENSIONS, SPLIT_MANIFEST_ID, WINDOW_CONTRACT_ID, align_targets,
    build_split, make_windows, verify_split, verify_target_file,
)
from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical


AUDIT_NAME = "deam-dmer-full-v1-window-coverage-v1"


def materialize_protocol(data_directory: Path, repo_root: Path, output_directory: Path) -> dict:
    data_directory, repo_root, output_directory = map(Path, (data_directory, repo_root, output_directory))
    registered_split = repo_root / "manifests/splits" / f"{SPLIT_MANIFEST_ID}.json"
    registered_audit = repo_root / "reports/data" / f"{AUDIT_NAME}.json"
    for path in (output_directory, registered_split, registered_audit):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite protocol artifact: {path}")
    manifest_path = data_directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    split = build_split(manifest)
    split_verification = verify_split(manifest, split)
    targets_path = data_directory / "dmer_targets.jsonl"
    target_sha256 = verify_target_file(manifest, targets_path)
    targets = {}
    with targets_path.open(encoding="utf-8") as source:
        for line in source:
            target = json.loads(line)
            song = target["song_id"]
            if song in targets:
                raise ValueError(f"duplicate target song: {song}")
            targets[song] = target
    if set(targets) != {r["song_id"] for r in manifest["records"]}:
        raise ValueError("target file does not cover exactly the manifest song population")
    memberships = {r["song_id"]: r["split"] for r in split["records"]}
    all_windows, per_song = [], []
    total_points = dict.fromkeys(DIMENSIONS, 0)
    uncovered = duplicates = 0
    partial_bin_count = 0
    dimension_tail_mismatches = []
    by_split = {name: {"song_count": 0, "window_count": 0, "valid_point_counts": dict.fromkeys(DIMENSIONS, 0)} for name in ("train", "validation", "test", "long_test")}
    for record in manifest["records"]:
        target = targets[record["song_id"]]
        windows = make_windows(record)
        membership = memberships[record["song_id"]]
        covered = {name: Counter() for name in DIMENSIONS}
        output_audio_times = Counter()
        for window in windows:
            _, masks = align_targets(target, window)
            for index, time in enumerate(window["output_time_ms"]):
                if window["output_audio_mask"][index]: output_audio_times[time] += 1
                for dimension, name in enumerate(DIMENSIONS):
                    if masks[index, dimension]: covered[name][time] += 1
            partial_bin_count += sum(0 < coverage < 1 for coverage in window["audio_coverage"])
            all_windows.append(window | {"split": membership})
        counts = {}
        for name in DIMENSIONS:
            view = target["views"][name]
            expected_times = [t for t, valid in zip(view["time_ms"], view["supervision_mask"]) if valid]
            if not expected_times:
                raise ValueError(f"zero supervised points: {record['song_id']} {name}")
            for time in expected_times:
                uncovered += covered[name][time] == 0
                duplicates += max(0, covered[name][time] - 1)
            counts[name] = len(expected_times)
            total_points[name] += len(expected_times)
            by_split[membership]["valid_point_counts"][name] += len(expected_times)
        if target["views"]["valence"]["time_ms"] != target["views"]["arousal"]["time_ms"]:
            dimension_tail_mismatches.append(record["logical_song_key"])
        if any(count != 1 for count in output_audio_times.values()):
            raise ValueError("audio-derived output timestamps overlap")
        by_split[membership]["song_count"] += 1
        by_split[membership]["window_count"] += len(windows)
        per_song.append({"song_id": record["song_id"], "logical_song_key": record["logical_song_key"], "split": membership, "window_count": len(windows), "valid_point_counts": counts, "output_audio_point_count": len(output_audio_times)})
    if uncovered or duplicates:
        raise ValueError(f"incomplete time coverage: {uncovered} uncovered, {duplicates} duplicate supervised points")
    if len({w["window_id"] for w in all_windows}) != len(all_windows):
        raise ValueError("duplicate window identities")
    window_bytes = ("\n".join(canonical_json(w) for w in all_windows) + "\n").encode("utf-8")
    audit = {
        "schema_version": "1.0", "audit_id": AUDIT_NAME, "verdict": "PASS",
        "dataset_id": manifest["dataset_id"], "dataset_manifest_sha256": manifest["manifest_sha256"],
        "dataset_manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "target_file_sha256": target_sha256, "split_sha256": split["split_sha256"],
        "split_verification": split_verification, "window_contract_id": WINDOW_CONTRACT_ID,
        "windows_file_sha256": hashlib.sha256(window_bytes).hexdigest(),
        "song_count": len(per_song), "window_count": len(all_windows),
        "valid_point_counts": total_points, "uncovered_supervised_point_count": uncovered,
        "duplicate_supervised_point_count": duplicates, "partial_audio_bin_count": partial_bin_count,
        "dimension_timestamp_mismatch_song_keys": dimension_tail_mismatches,
        "shorter_than_45s_retained_song_keys": [r["logical_song_key"] for r in manifest["records"] if r["population_kind"] == "SHORT_CLIP" and r["audio_duration_seconds"] < 45],
        "by_split": by_split, "per_song": per_song,
        "features_generated": False, "training_started": False, "formal_training_ready": False,
        "policy": "45s inputs, final 30s outputs at 2Hz; 30s full-song stride; audio-only windows; independent valence/arousal masks; native labels unchanged",
    }
    audit["audit_sha256"] = sha256_canonical(audit)
    output_directory.mkdir(parents=True, exist_ok=False)
    (output_directory / "windows.jsonl").write_bytes(window_bytes)
    for payload, local_name, registration in ((split, "split.json", registered_split), (audit, "audit.json", registered_audit)):
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        with (output_directory / local_name).open("xb") as destination: destination.write(encoded)
        registration.parent.mkdir(parents=True, exist_ok=True)
        with registration.open("xb") as destination: destination.write(encoded)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory or args.data_directory.with_name("deam-dmer-full-v1-protocol-v1")
    audit = materialize_protocol(args.data_directory, args.repo_root, output)
    print(json.dumps({key: audit[key] for key in ("verdict", "song_count", "window_count", "valid_point_counts", "uncovered_supervised_point_count", "duplicate_supervised_point_count")}, ensure_ascii=False))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
