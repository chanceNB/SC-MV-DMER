import numpy as np
import pytest
import torch

from sc_mv_dmer.training.dmer_acoustic import (
    TrainNormalizer, masked_va_loss, smoothness_loss, gate_against_crnn,
    fit_constant, validate_feature_binding,
)


def test_normalization_never_fits_validation_or_padding():
    rows = [{"split": "train", "views": {"mel": np.array([[1., 3.], [3., 5.], [999., 999.]])},
             "audio_mask": np.array([True, True, False])}]
    norm = TrainNormalizer.fit(rows)
    np.testing.assert_allclose(norm.mean["mel"], [2., 4.])
    transformed = norm.transform(rows[0]["views"], rows[0]["audio_mask"])
    np.testing.assert_allclose(transformed["mel"], [[-1., -1.], [1., 1.], [0., 0.]])
    with pytest.raises(ValueError, match="train"):
        TrainNormalizer.fit([dict(rows[0], split="validation")])


def test_va_loss_is_song_and_dimension_equal_and_masks_before_arithmetic():
    p = torch.tensor([[[1., 2.], [1., 2.]], [[3., 4.], [99., 99.]]], requires_grad=True)
    y = torch.zeros_like(p)
    y[1, 1] = float("nan")
    mask = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [0, 0]]], dtype=torch.bool)
    loss = masked_va_loss(p, y, mask)
    assert loss.item() == pytest.approx((1 + 4 + 9 + 16) / 4)
    loss.backward()
    assert torch.isfinite(p.grad).all()
    assert torch.equal(p.grad[1, 1], torch.zeros(2))


def test_invalid_prediction_cannot_disappear_from_loss():
    p = torch.full((1, 3, 2), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        masked_va_loss(p, torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool))


def test_smoothness_does_not_bridge_missing_timestamps():
    p = torch.tensor([[[0., 0.], [20., 20.], [1., 1.], [2., 2.]]])
    mask = torch.tensor([[[1, 1], [0, 0], [1, 1], [1, 1]]], dtype=torch.bool)
    assert smoothness_loss(p, mask).item() == pytest.approx(1.)


def test_constant_fits_only_train_and_both_dimensions_independently():
    rows = [{"split": "train", "target": np.array([[1., 2.], [3., np.nan]]),
             "mask": np.array([[1, 1], [1, 0]], dtype=bool)}]
    np.testing.assert_array_equal(fit_constant(rows), [2., 2.])
    with pytest.raises(ValueError, match="train"):
        fit_constant([dict(rows[0], split="long_test")])


def test_gate_does_not_average_away_failed_valence():
    def report(v, a):
        return {"macro": {"valence": {"ccc": v}, "arousal": {"ccc": a}}}
    assert gate_against_crnn(report(.40, .80), report(.50, .50))["verdict"] == "FAIL"
    assert gate_against_crnn(report(.49, .51), report(.50, .50))["verdict"] == "NEAR"


def test_legacy_or_partial_feature_cache_cannot_start_new_experiment():
    dataset = {"manifest_sha256": "new-full-population"}
    with pytest.raises(ValueError):
        validate_feature_binding(dataset, {"feature_version": "old-995", "complete": True})
    with pytest.raises(ValueError):
        validate_feature_binding(dataset, {"feature_version": "deam-dmer-features-v2", "complete": False})


def test_gate_invalid_dimension_is_invalid_not_exception_or_average():
    bad = {"status": "INVALID", "macro": {"valence": {"ccc": None}, "arousal": {"ccc": .8}}}
    good = {"macro": {"valence": {"ccc": .4}, "arousal": {"ccc": .6}}}
    assert gate_against_crnn(bad, good)["verdict"] == "INVALID"


def test_feature_binding_accepts_real_schema_and_rejects_swapped_song_identity():
    from sc_mv_dmer.data.dmer_protocol import make_windows
    dataset = {"manifest_sha256": "new", "records": [
        {"song_id": "a", "logical_song_key": "2", "population_kind": "SHORT_CLIP", "audio_duration_seconds": 45.},
        {"song_id": "b", "logical_song_key": "3", "population_kind": "SHORT_CLIP", "audio_duration_seconds": 45.},
    ]}
    windows = [make_windows(r)[0] for r in dataset["records"]]
    cache = {"feature_version":"deam-dmer-features-v2", "dataset_internal_sha256":"new",
             "full_population_ready":True, "four_view_ready":True, "windows":windows}
    validate_feature_binding(dataset, cache)
    windows[0]["song_id"], windows[1]["song_id"] = windows[1]["song_id"], windows[0]["song_id"]
    with pytest.raises(ValueError, match="identity"):
        validate_feature_binding(dataset, cache)
