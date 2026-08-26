from pathlib import Path

from sc_mv_dmer.data.step2_preflight import evaluate_step2_rg02_v2


def test_v2_preflight_binds_v2_and_short_audio_policy(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    data_root = Path(r"E:\数据集\_pdmer_compat")
    evidence = evaluate_step2_rg02_v2(repo_root=repo, data_root=data_root)
    assert evidence["dataset_manifest"]["manifest_id"] == "DEAM-PDMER-CLEAN-v2"
    assert evidence["lineage"]["validation_policy"] == "UNDEFINED_PENDING_SEPARATE_FREEZE"
    assert evidence["checks"]["raw_audio"]["observed_short_audio_keys"] == ["1174", "1200", "1273", "1493", "1789"]
    assert evidence["verdict"] == "BLOCKED"
    assert evidence["checks"]["feature_materialization"]["cache_published"] is False
    assert all("path_configured" not in item for item in evidence["checks"]["processed_artifacts"].values())
    assert all(not str(value).startswith(("E:\\", "C:\\", "/")) for value in evidence["checks"]["processed_artifacts"].values())
