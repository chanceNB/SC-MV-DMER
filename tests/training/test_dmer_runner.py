import numpy as np
import pytest
import torch

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
