import numpy as np
import pytest
import torch
import json
from pathlib import Path

import sc_mv_dmer.training.dmer_runner as dmer_runner
from sc_mv_dmer.training.dmer_runner import WindowDataset, stitch_song_predictions, train_epoch
from sc_mv_dmer.training.dmer_acoustic import TrainNormalizer


def test_stitch_groups_windows_before_song_metrics_and_keeps_asymmetric_tail():
    rows = []
    for start, end in [(15000, 45000), (45000, 75000)]:
        time = list(range(start, end, 500))
        mask = np.ones((60, 2), dtype=bool)
        if start == 45000: mask[-1, 1] = False
        rows.append({"song_id": "long", "output_time_ms": time, "prediction": np.ones((60, 2)),
                     "target": np.ones((60, 2)), "mask": mask})
    result = stitch_song_predictions(rows)
    assert len(result) == 1
    assert result[0]["mask"].sum(axis=0).tolist() == [120, 119]
    with pytest.raises(ValueError, match="duplicate"):
        stitch_song_predictions([rows[0], rows[0]])


def test_real_optimizer_step_updates_predictor_without_sensor_gate():
    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(2))
        def forward(self, views, audio_mask, **kwargs):
            return {"prediction": self.value[None, None].expand(len(audio_mask), 60, 2)}
    model = SmallModel()
    batch = {"views": {"deep": torch.ones(2, 90, 768)}, "audio_mask": torch.ones(2, 90, dtype=torch.bool),
             "target": torch.ones(2, 60, 2) * .4, "mask": torch.ones(2, 60, 2, dtype=torch.bool)}
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    loss = train_epoch(model, [batch], opt, torch.device("cpu"))
    assert loss > 0
    assert torch.all(model.value > 0)


def test_window_dataset_does_not_return_targets_as_model_features(tmp_path):
    p = tmp_path / "one.npz"
    np.savez(p, mel=np.ones((90, 128)), mfcc=np.ones((90, 40)), chroma=np.ones((90, 12)),
             deep=np.ones((90, 768)), audio_mask=np.ones(90, dtype=bool))
    views = {k: np.ones((90, d)) for k, d in [("mel",128),("mfcc",40),("chroma",12),("deep",768)]}
    norm = TrainNormalizer.fit([{"split":"train","views":views,"audio_mask":np.ones(90,dtype=bool)}])
    row = {"song_id":"song", "window_id":"one", "path":str(p), "split":"train", "output_time_ms":list(range(15000,45000,500)),
           "target":np.full((60,2), .7), "mask":np.ones((60,2),bool)}
    item = WindowDataset([row], norm)[0]
    assert set(item["views"]) == {"mel","mfcc","chroma","deep"}
    assert item["target"].shape == (60, 2)


def test_window_dataset_accepts_stricter_decoded_mask_but_never_masks_a_label(tmp_path):
    p = tmp_path / "decoded.npz"
    actual = np.ones(90, dtype=bool); actual[89] = False
    np.savez(p, mel=np.ones((90,128)), mfcc=np.ones((90,40)), chroma=np.ones((90,12)),
             deep=np.ones((90,768)), audio_mask=actual)
    views={k:np.ones((90,d)) for k,d in [("mel",128),("mfcc",40),("chroma",12),("deep",768)]}
    norm=TrainNormalizer.fit([{"split":"train","views":views,"audio_mask":actual}])
    target_mask=np.ones((60,2),dtype=bool); target_mask[-1]=False
    row={"song_id":"song","window_id":"decoded","path":str(p),"split":"train",
         "audio_mask":[True]*90,"output_time_ms":list(range(15000,45000,500)),
         "target":np.zeros((60,2)),"mask":target_mask}
    assert WindowDataset([row],norm)[0]["audio_mask"].sum()==89
    row["mask"][-1]=True
    with pytest.raises(ValueError,match="target.*decoded audio"):
        WindowDataset([row],norm)


def test_normalized_transition_entropy_is_not_normalized_a_second_time():
    class UniformModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(2))
        def forward(self, views, audio_mask, **kwargs):
            return {"prediction":self.value[None,None].expand(1,60,2),
                    "row_entropy":{"shared":torch.ones(8)}}
    model = UniformModel()
    batch = {"views":{"deep":torch.ones(1,90,768)},"audio_mask":torch.ones(1,90,dtype=torch.bool),
             "target":torch.zeros(1,60,2),"mask":torch.ones(1,60,2,dtype=torch.bool)}
    loss = train_epoch(model,[batch],torch.optim.SGD(model.parameters(),lr=.1),torch.device("cpu"))
    assert loss == pytest.approx(0.)


def test_load_rows_does_not_decode_test_targets_or_touch_test_features(tmp_path, monkeypatch):
    manifest_path = tmp_path / "dataset.json"
    split_path = tmp_path / "split.json"
    targets_path = tmp_path / "targets.jsonl"
    cache_path = tmp_path / "cache.json"
    records = [{"song_id": name} for name in ("train-song", "validation-song", "test-song", "long-song")]
    manifest = {"records": records, "artifacts": {"dmer_targets": {"file_sha256": "target-hash"}}}
    split = {"records": [{"song_id": "train-song", "split": "train"},
                         {"song_id": "validation-song", "split": "validation"},
                         {"song_id": "test-song", "split": "test"},
                         {"song_id": "long-song", "split": "long_test"}]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    train_target = {"song_id": "train-song", "values": [1]}
    validation_target = {"song_id": "validation-song", "values": [2]}
    targets_path.write_text(
        json.dumps(train_target) + "\n" + json.dumps(validation_target) +
        '\n{"song_id":"test-song","values": THIS_IS_NOT_JSON}\n' +
        '{"song_id":"long-song","values": THIS_IS_NOT_JSON}\n', encoding="utf-8")
    train_feature, validation_feature = tmp_path / "train.npz", tmp_path / "validation.npz"
    train_feature.write_bytes(b"train")
    validation_feature.write_bytes(b"validation")
    cache = {"dataset_manifest_sha256": "dataset-file-hash", "manifest_sha256": "cache-hash",
             "windows": [{"window_id": name, "path": filename, "sha256": digest}
                         for name, filename, digest in (("train-song", "train.npz", "train-hash"),
                                                        ("validation-song", "validation.npz", "validation-hash"),
                                                        ("test-song", "test-does-not-exist.npz", "test-hash"),
                                                        ("long-song", "long-does-not-exist.npz", "long-hash"))]}
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    monkeypatch.setattr(dmer_runner, "verify_split", lambda *_: None)
    monkeypatch.setattr(dmer_runner, "validate_feature_binding", lambda *_: None)
    monkeypatch.setattr(dmer_runner, "sha256_canonical", lambda _: "cache-hash")
    monkeypatch.setattr(dmer_runner, "make_windows", lambda record: [{
        "song_id": record["song_id"], "window_id": record["song_id"], "output_time_ms": list(range(60))}])
    monkeypatch.setattr(dmer_runner, "align_targets", lambda *_: (np.zeros((60, 2)), np.ones((60, 2), dtype=bool)))
    touched = []

    def fake_hash(path):
        path = Path(path)
        touched.append(path.name)
        if path.name in ("test-does-not-exist.npz", "long-does-not-exist.npz"):
            raise AssertionError("test/long-test feature was touched")
        return {"targets.jsonl": "target-hash", "dataset.json": "dataset-file-hash",
                "train.npz": "train-hash", "validation.npz": "validation-hash"}.get(path.name, "unused")

    monkeypatch.setattr(dmer_runner, "file_sha256", fake_hash)
    _, _, rows = dmer_runner.load_rows(manifest_path, split_path, targets_path, cache_path)
    assert [row["song_id"] for row in rows] == ["train-song", "validation-song"]
    assert "test-does-not-exist.npz" not in touched
    assert "long-does-not-exist.npz" not in touched
