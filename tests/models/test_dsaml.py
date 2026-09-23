import importlib.util

import pytest
import torch


def api():
    assert importlib.util.find_spec("sc_mv_dmer.models.dsaml") is not None, "pinned DSAML wrapper is not implemented"
    from sc_mv_dmer.models import dsaml
    return dsaml


def test_current_four_views_cannot_claim_official_dsaml_features():
    d = api()
    report = d.probe_readiness({"feature_version": "deam-dmer-features-v2", "feature_keys": ["deep", "mel", "mfcc", "chroma"]})
    assert report["status"] == "BLOCKED"
    assert "OFFICIAL_FEATURE_CACHE_MISSING" in report["blockers"]
    assert report["upstream_commit"] == "20b12ea1dd8a41ba0e3072a15a19f88d8fa28496"
    assert report["required_features"]["log_mel_spectrogram"]["shape"] == ["B", 60, 51, 128]
    assert report["required_features"]["global_imagebind_audio_embedding"]["shape"] == ["B", 1, 1024]


def test_upstream_sources_match_recorded_hashes_and_include_license():
    report = api().verify_upstream_sources()
    assert report["status"] == "PASS"
    assert report["file_count"] >= 8


def test_official_objective_contains_all_six_metrics_and_attention_penalty():
    d = api()
    prediction = torch.tensor([[[0., 0.], [1., 1.], [2., 2.]]], requires_grad=True)
    target = prediction.detach().clone()
    penalty = torch.tensor(.25, requires_grad=True)
    loss = d.official_objective({"prediction": prediction, "attention_regularization": penalty}, target, torch.ones_like(target, dtype=torch.bool))
    assert loss.item() == pytest.approx(.25, abs=1e-7)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert penalty.grad.item() == 1


def test_attention_penalty_matches_official_last_layer_diagonals():
    d = api()
    maps = torch.zeros(2, 6, 4, 60, 60)
    mask = torch.ones(2, 60, dtype=torch.bool)
    assert d.attention_regularization(maps, mask, layer_count=3).item() == pytest.approx(.9 ** 2 + .1 ** 2)


def test_source_objective_is_not_ccc_only_or_mse():
    d = api()
    prediction = torch.tensor([[[0., 0.], [1., 1.]]], requires_grad=True)
    target = torch.tensor([[[1., 1.], [0., 0.]]])
    loss = d.official_objective({"prediction": prediction, "attention_regularization": torch.tensor(0.)}, target, torch.ones_like(target, dtype=torch.bool))
    assert loss.item() == pytest.approx(5 / 3)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_wrapper_real_cpu_forward_and_eval_reproducibility():
    d = api()
    if not d.runtime_ready():
        pytest.skip("upstream DSAML needs isolated transformers 4.37.2 runtime; current MERT runtime is preserved")
    torch.set_num_threads(2)
    model = d.DSAMLModel().eval()
    views = {"log_mel_spectrogram": torch.randn(1, 60, 51, 128), "global_imagebind_audio_embedding": torch.randn(1, 1, 1024)}
    mask = torch.ones(1, 90, dtype=torch.bool)
    with torch.no_grad():
        first, second = model(views, mask), model(views, mask)
    assert first["prediction"].shape == (1, 60, 2)
    assert first["attention_maps"].shape == (1, 6, 16, 60, 60)
    assert torch.isfinite(first["prediction"]).all()
    torch.testing.assert_close(first["prediction"], second["prediction"], atol=0, rtol=0)
    assert first["dimension_order"] == ["valence", "arousal"]
    with pytest.raises(ValueError, match="official"):
        model({"deep": torch.randn(1, 90, 768), "mel": torch.randn(1, 90, 128)}, mask)


def test_partial_tail_uses_only_valid_output_features():
    d = api()
    if not d.runtime_ready(): pytest.skip("needs isolated upstream runtime")
    torch.set_num_threads(2)
    model = d.DSAMLModel().eval()
    mask = torch.ones(1, 90, dtype=torch.bool)
    mask[:, 35:] = False
    views = {"log_mel_spectrogram": torch.randn(1, 60, 51, 128), "global_imagebind_audio_embedding": torch.randn(1, 1, 1024)}
    with torch.no_grad():
        first = model(views, mask)
        views["log_mel_spectrogram"][:, 5:] = float("nan")
        second = model(views, mask)
    torch.testing.assert_close(first["prediction"], second["prediction"])
    assert first["output_audio_mask"].sum() == 5
