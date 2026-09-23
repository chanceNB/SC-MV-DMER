import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from sc_mv_dmer.foundation.canonical import sha256_canonical


def api():
    assert importlib.util.find_spec("sc_mv_dmer.data.dmer_protocol") is not None, "full DMER protocol is not implemented"
    from sc_mv_dmer.data import dmer_protocol
    return dmer_protocol


def manifest():
    result = {
        "manifest_id": "DEAM-DMER-FULL-v1", "dataset_id": "full-dataset",
        "population": {"short_clips": 1744, "full_songs": 58, "total": 1802, "excluded": 0},
        "artifacts": {"dmer_targets": {"file_sha256": "a" * 64, "record_count": 1802}},
        "records": [
            {"song_id": f"song-{i}", "sample_id": f"sample-{i}", "logical_song_key": str(i),
             "population_kind": "SHORT_CLIP" if i < 1744 else "FULL_SONG",
             "audio_duration_seconds": 45 if i < 1744 else 105}
            for i in range(1802)
        ],
    }
    result["manifest_sha256"] = sha256_canonical(result)
    return result


def test_split_exact_population_and_original_canonical_ranking():
    p = api()
    m = manifest()
    split = p.build_split(m)
    assert p.verify_split(m, split)["verdict"] == "PASS"
    assert [split[k] for k in ("train_count", "validation_count", "test_count", "long_test_count")] == [1395, 174, 175, 58]
    assert len(split["records"]) == 1802
    assert len({r["song_id"] for r in split["records"]}) == 1802
    ranked = sorted((sha256_canonical({"dataset_id": m["dataset_id"], "song_id": r["song_id"], "split_contract_id": "DEAM-PRIMARY-SONG-SPLIT/v1"}), r["song_id"]) for r in m["records"][:1744])
    assert split["train_song_ids"] == [song for _, song in ranked[:1395]]
    assert set(split["long_test_song_ids"]) == {r["song_id"] for r in m["records"][1744:]}
    assert p.build_split(m) == split


def test_split_rejects_rehashed_membership_and_source_tampering():
    p = api()
    m = manifest()
    split = p.build_split(m)
    altered = copy.deepcopy(split)
    altered["records"][0]["split"] = "test"
    altered["split_sha256"] = sha256_canonical({k: v for k, v in altered.items() if k != "split_sha256"})
    with pytest.raises(ValueError): p.verify_split(m, altered)
    m["records"][0]["audio_duration_seconds"] = 44
    with pytest.raises(ValueError, match="manifest.*checksum"): p.verify_split(m, split)


def test_target_hash_is_bound_to_split():
    p = api()
    m = manifest()
    split = p.build_split(m)
    m["artifacts"]["dmer_targets"]["file_sha256"] = "b" * 64
    m["manifest_sha256"] = sha256_canonical({k: v for k, v in m.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError): p.verify_split(m, split)


def test_short_audio_kept_with_partial_last_input_bin():
    p = api()
    r = {"song_id": "s", "logical_song_key": "1174", "population_kind": "SHORT_CLIP", "audio_duration_seconds": 44.86}
    windows = p.make_windows(r)
    assert len(windows) == 1
    w = windows[0]
    assert w["window_id"] == "1174@000000000"
    assert w["input_time_ms"] == list(range(0, 45000, 500))
    assert w["output_time_ms"] == list(range(15000, 45000, 500))
    assert w["audio_coverage"][-1] == pytest.approx(0.72)
    assert all(w["audio_mask"]) and all(w["output_audio_mask"])


def test_long_windows_depend_on_audio_not_labels_and_cover_tail_once():
    p = api()
    r = {"song_id": "s", "logical_song_key": "2002", "population_kind": "FULL_SONG", "audio_duration_seconds": 75.1}
    windows = p.make_windows(r)
    assert [w["start_ms"] for w in windows] == [0, 30000, 60000]
    times = [t for w in windows for t, valid in zip(w["output_time_ms"], w["output_audio_mask"]) if valid]
    assert times == list(range(15000, 75500, 500))
    assert len(times) == len(set(times))
    assert windows[-1]["audio_coverage"][30] == pytest.approx(0.2)


def test_align_keeps_dimension_specific_last_point_and_no_interpolation():
    p = api()
    w = p.make_windows({"song_id": "s", "logical_song_key": "2002", "population_kind": "FULL_SONG", "audio_duration_seconds": 75.1})[-1]
    target = {"song_id": "s", "logical_song_key": "2002", "views": {
        "valence": {"time_ms": [74500], "values": [0.4], "supervision_mask": [True]},
        "arousal": {"time_ms": [74500, 75000], "values": [0.3, -0.2], "supervision_mask": [True, True]},
    }}
    y, mask = p.align_targets(target, w)
    assert y.shape == mask.shape == (60, 2)
    assert y.dtype == np.float32 and mask.dtype == np.bool_
    assert mask[0].tolist() == [False, True]
    assert y[0, 1] == pytest.approx(-0.2)
    assert mask.sum() == 1


def test_file_hash_check_rejects_changed_target_file(tmp_path):
    p = api()
    path = tmp_path / "targets.jsonl"
    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="target.*checksum"):
        p.verify_target_file(manifest(), path)


def test_materialization_audits_all_targets_and_refuses_overwrite(tmp_path):
    script = Path(__file__).parents[2] / "scripts/materialize_dmer_protocol.py"
    assert script.is_file(), "protocol materialization CLI is not implemented"
    spec = importlib.util.spec_from_file_location("materialize_dmer_protocol", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, repo, output = tmp_path / "data", tmp_path / "repo", tmp_path / "output"
    source.mkdir()
    m = manifest()
    with (source / "dmer_targets.jsonl").open("w", encoding="utf-8") as f:
        for r in m["records"]:
            times = list(range(15000, int(r["audio_duration_seconds"] * 1000), 500))
            target = {"song_id": r["song_id"], "logical_song_key": r["logical_song_key"], "views": {name: {"time_ms": times, "values": [0.2] * len(times), "supervision_mask": [True] * len(times)} for name in ("valence", "arousal")}}
            f.write(json.dumps(target) + "\n")
    m["artifacts"]["dmer_targets"]["file_sha256"] = hashlib.sha256((source / "dmer_targets.jsonl").read_bytes()).hexdigest()
    m["manifest_sha256"] = sha256_canonical({k: v for k, v in m.items() if k != "manifest_sha256"})
    (source / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    result = module.materialize_protocol(source, repo, output)
    assert result["verdict"] == "PASS"
    assert result["song_count"] == 1802
    assert result["uncovered_supervised_point_count"] == result["duplicate_supervised_point_count"] == 0
    assert result["window_count"] == 1744 + 3 * 58
    assert (output / "windows.jsonl").is_file()
    assert len(list((repo / "manifests/splits").glob("*.json"))) == 1
    with pytest.raises(FileExistsError): module.materialize_protocol(source, repo, output)
