import pytest
import torch

from sc_mv_dmer.models import AcousticModel, MarkovStateBridge, MertTemporalFrontend


def test_sub_convolution_tail_retains_output_instead_of_rejecting_song():
    # DEAM2053 has a 4.263ms final half-second bin: audio exists, but no
    # complete MERT receptive field has a centre in that final bin.
    frontend = MertTemporalFrontend(use_timesnet=False, feature_dim=4)
    centers = (399 + 640 * torch.arange(3374, dtype=torch.float64)) / 48000
    mask = (centers + 400 / 48000 <= 40.004263)[None]
    audio = (torch.arange(90) * .5 < 40.004263)[None]
    result = frontend(torch.ones(1, 2, 3374, 4), centers, mask, audio)
    assert result.shape == (1, 90, 4)
    assert audio[0, 80]
    assert torch.equal(result[0, 80], torch.zeros(4))
    assert torch.isfinite(result).all()


torch.set_num_threads(2)


def views(batch=2):
    generator = torch.Generator().manual_seed(73)
    return {k: torch.randn(batch, 90, d, generator=generator)
            for k, d in {'deep': 768, 'mel': 128, 'mfcc': 40, 'chroma': 12}.items()}


@pytest.mark.parametrize('variant', ['mert_mlp', 'concat', 'shared_state',
    'independent_state', 'no_state_bias', 'deep_only', 'crnn', 'bilstm_attention'])
def test_variants_emit_fixed_grid_and_ignore_masked_feature_values(variant):
    model = AcousticModel(variant=variant, hidden_dim=32).eval()
    inputs = views()
    mask = torch.ones(2, 90, dtype=torch.bool)
    mask[:, 86:] = False
    with torch.no_grad():
        result = model(inputs, mask)
        altered = {k: value.masked_fill(~mask[..., None], float('nan'))
                   for k, value in inputs.items()}
        other = model(altered, mask)
    assert result['prediction'].shape == (2, 60, 2)
    assert result['fused'].shape == (2, 90, 32)
    assert torch.isfinite(result['prediction']).all()
    assert result['prediction'].abs().max() <= 1
    torch.testing.assert_close(result['prediction'], other['prediction'])
    assert result['feature_source'] == ('handcrafted_only_no_mert' if variant in ('crnn', 'bilstm_attention')
                                        else 'mert_layers_5_6_2hz_cache_no_timesnet')


def test_all_invalid_audio_rejected():
    with pytest.raises(ValueError, match='valid audio'):
        AcousticModel(hidden_dim=32)(views(), torch.zeros(2, 90, dtype=torch.bool))


def test_shared_bridge_retains_sharp_evidence_and_skips_padding():
    bridge = MarkovStateBridge(hidden_dim=8, state_count=8)
    evidence = torch.full((1, 3, 8), -40.)
    evidence[0, 0, 0] = 0
    evidence[0, 1, 7] = 0
    evidence[0, 2, 1] = 0
    mask = torch.tensor([[True, False, True]])
    posterior = bridge.filter_log_emissions(evidence, mask, view='deep')
    assert posterior[0, 0, 0] > .99
    torch.testing.assert_close(posterior[:, 1], posterior[:, 0])
    torch.testing.assert_close(posterior.sum(-1), torch.ones(1, 3))
    assert posterior[0, 2, 1] > .99
    torch.testing.assert_close(bridge.transition('deep').sum(-1), torch.ones(8))
    assert bridge.transition_logits_for('deep') is bridge.transition_logits_for('mel')


def test_independent_variant_changes_A_but_keeps_shared_pi():
    bridge = MarkovStateBridge(hidden_dim=8, independent_transitions=True)
    assert bridge.transition_logits_for('deep') is not bridge.transition_logits_for('mel')
    assert bridge.initial_logits.shape == (8,)


def test_state_prediction_and_sensors_backpropagate_to_intended_modules():
    model = AcousticModel(hidden_dim=32, view_dropout_p=0)
    output = model(views(), torch.ones(2, 90, dtype=torch.bool))
    loss = output['prediction'].square().mean() + sum(x.square().mean() for x in output['sensors'].values())
    loss.backward()
    for name in ['deep', 'mel', 'mfcc', 'chroma']:
        grad = model.encoders[name].projection.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert model.bridge.transition_logits_for('deep').grad.abs().sum() > 0
    assert model.bridge.prototypes['chroma'].grad.abs().sum() > 0


def test_view_dropout_leaves_clean_sensor_predictions_unchanged():
    model = AcousticModel(hidden_dim=32, view_dropout_p=0)
    inputs, mask = views(), torch.ones(2, 90, dtype=torch.bool)
    first = model(inputs, mask)
    model.view_dropout_p = 1
    dropped = model(inputs, mask)
    for name in first['sensors']:
        torch.testing.assert_close(first['sensors'][name], dropped['sensors'][name])
    assert dropped['dropped_views'].all()


def test_raw_layers_gate_timesnet_before_real_time_pooling_and_receive_gradient():
    model = AcousticModel(hidden_dim=32, use_timesnet=True, view_dropout_p=0)
    layers = torch.randn(2, 2, 180, 768, requires_grad=True)
    times = torch.arange(180, dtype=torch.float32) / 4 + .125
    raw_mask = torch.ones(2, 180, dtype=torch.bool)
    raw_mask[:, 176:] = False
    mask = torch.ones(2, 90, dtype=torch.bool)
    mask[:, 88:] = False
    result = model(views(), mask, mert_layers=layers,
                   mert_time_seconds=times, mert_mask=raw_mask)
    assert result['feature_source'] == 'raw_mert_layers_5_6_trainable_gate_timesnet'
    result['prediction'][:, :58].square().mean().backward()
    assert model.mert_frontend.layer_logits.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.mert_frontend.timesnet.parameters())
    assert layers.grad[:, :, :176].abs().sum() > 0
    assert layers.grad[:, :, 176:].abs().sum() == 0


def test_timesnet_never_silently_consumes_old_cache():
    with pytest.raises(ValueError, match='raw MERT'):
        AcousticModel(hidden_dim=32, use_timesnet=True)(views(), torch.ones(2, 90, dtype=torch.bool))


def test_real_time_pooling_uses_frame_centers_and_rejects_missing_covered_bins():
    frontend = MertTemporalFrontend(use_timesnet=False, feature_dim=4)
    layers = torch.stack([torch.arange(360).reshape(90, 4).float()] * 2)[None]
    time = torch.arange(90) * .5 + .25
    mask = torch.ones(1, 90, dtype=torch.bool)
    result = frontend(layers, time, mask, mask)
    torch.testing.assert_close(result, layers[:, 0])
    broken = mask.clone()
    broken[:, 15] = False
    with pytest.raises(ValueError, match='empty MERT bin'):
        frontend(layers, time, broken, mask)


def test_mode_sensor_respects_declared_range():
    model = AcousticModel(hidden_dim=32).eval()
    with torch.no_grad():
        model.sensor_heads['mode'][-1].bias.fill_(100)
        result = model(views(), torch.ones(2, 90, dtype=torch.bool))
    assert result['sensors']['mode'].abs().max() <= 1


def test_masked_raw_values_cannot_influence_timesnet_output():
    frontend = MertTemporalFrontend(use_timesnet=True, feature_dim=4).eval()
    layers = torch.randn(1, 2, 180, 4)
    times = torch.arange(180).float() / 4 + .125
    raw_mask = torch.ones(1, 180, dtype=torch.bool)
    raw_mask[:, 174:] = False
    audio_mask = raw_mask.reshape(1, 90, 2).any(-1)
    with torch.no_grad():
        first = frontend(layers, times, raw_mask, audio_mask)
        changed = layers.masked_fill(~raw_mask[:, None, :, None], float('nan'))
        second = frontend(changed, times, raw_mask, audio_mask)
    torch.testing.assert_close(first, second)
