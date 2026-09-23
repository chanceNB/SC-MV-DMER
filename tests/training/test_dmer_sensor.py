"""Train-only sensor fitting, masked gradients and evidence-grounded events."""
import importlib
import json

import numpy as np
import pytest
import torch


def sensor():
    return importlib.import_module('sc_mv_dmer.training.dmer_sensor')


def events():
    return importlib.import_module('sc_mv_dmer.models.events')


def test_target_normalization_rejects_validation_and_ignores_masked_nan():
    row = {'split': 'train', 'regression': np.array([[1., 2., -1.], [3., 4., 1.], [np.nan] * 3]),
           'regression_mask': np.array([[True] * 3, [True] * 3, [False] * 3])}
    stats = sensor().SensorNormalizer.fit([row])
    np.testing.assert_allclose(stats.mean, [2., 3., 0.])
    np.testing.assert_allclose(stats.std, [1., 1., 1.])
    np.testing.assert_allclose(stats.transform(row['regression'], row['regression_mask'])[-1], 0.)
    with pytest.raises(ValueError, match='train'):
        sensor().SensorNormalizer.fit([row | {'split': 'validation'}])


def test_real_sensor_encoder_receives_gradients_and_masks_padded_input():
    torch.set_num_threads(2)
    torch.manual_seed(52)
    model = sensor().SensorNetwork()
    views = {k: torch.randn(2, 90, d) for k, d in {'mel': 128, 'mfcc': 40, 'chroma': 12}.items()}
    mask = torch.ones(2, 90, dtype=torch.bool)
    mask[1, 87:] = False
    for value in views.values(): value[1, 87:] = float('nan')
    result = model(views, mask)
    assert result['regression'].shape == (2, 90, 3)
    assert result['key_logits'].shape == (2, 9, 12)
    target = torch.randn(2, 90, 3)
    valid = mask[..., None].expand_as(target)
    target[~valid] = float('nan')
    loss = sensor().sensor_loss(result, target, valid, torch.zeros(2, 9, dtype=torch.long), torch.ones(2, 9, dtype=torch.bool))
    loss.backward()
    for encoder in model.encoders.values():
        assert encoder.projection.weight.grad.abs().sum() > 0
        assert encoder.projection.out_features == 256
    assert torch.isfinite(loss)


def test_nonfinite_valid_sensor_prediction_fails_instead_of_disappearing():
    out = {'regression': torch.zeros(1, 90, 3, requires_grad=True),
           'key_logits': torch.zeros(1, 9, 12, requires_grad=True)}
    out['regression'] = out['regression'] + float('nan')
    with pytest.raises(ValueError, match='finite'):
        sensor().sensor_loss(out, torch.zeros(1, 90, 3), torch.ones(1, 90, 3, dtype=torch.bool),
                             torch.zeros(1, 9, dtype=torch.long), torch.ones(1, 9, dtype=torch.bool))


def test_ks_diagnostic_never_passes_formal_event_gate_but_does_not_block_acoustic():
    metrics = {'mode_ccc': .9, 'rms_mse': .01, 'rms_baseline_mse': 1.,
               'brightness_mse': .01, 'brightness_baseline_mse': 1., 'key_ce': .1, 'key_prior_ce': 2.}
    result = sensor().sensor_gate(metrics, 'KRUMHANSL_SCHMUCKLER_DIAGNOSTIC_NOT_ESSENTIA')
    assert result['mode_ccc_gt_0_7']
    assert result['events_llm_blocked']
    assert result['acoustic_blocked'] is False
    assert 'ESSENTIA_KEY_TARGET_UNAVAILABLE' in result['blocking_reasons']
    metrics['mode_ccc'] = .7
    assert not sensor().sensor_gate(metrics, 'ESSENTIA_EDMA')['mode_ccc_gt_0_7']


def test_predicted_events_keep_absolute_time_and_unknown_tonic():
    api = events()
    predictions = np.tile([.2, .3, .05], (90, 1))
    mask = np.ones((90, 3), dtype=bool)
    mask[87:] = False
    keys = np.full((9, 12), 1 / 12)
    keys[0] = 0.; keys[0, 9] = 1.
    result = api.make_predicted_events(predictions, mask, keys, np.ones(9, bool),
                                       start_ms=30000, audio_coverage=np.r_[np.ones(87), np.zeros(3)],
                                       thresholds={'rms': [.1, .3], 'brightness': [.2, .4]})
    assert result[0]['start_ms'] == 30000 and result[0]['end_ms'] == 35000
    assert result[0]['tonic'] == 'A'
    assert result[1]['tonic'] is None
    assert result[-1]['observed_seconds'] == 3.5
    text = api.serialize_events(result)
    assert 'A minor' not in text and 'A小调' not in text
    assert 'unknown' in text


def test_cot_candidates_are_train_only_exact_grid_and_review_is_unscored(tmp_path):
    api = events()
    row = {'split': 'train', 'window_id': 'song@000000000', 'song_id': 'song',
           'output_time_ms': list(range(15000, 45000, 500)),
           'target': np.tile([.25, -.125], (60, 1)), 'mask': np.ones((60, 2), bool)}
    result = api.write_cot_candidates([row], {row['window_id']: []}, tmp_path / 'cot')
    assert result['human_quality_pass'] is False
    assert result['review_required_count'] == 100
    assert result['review_available_count'] == 1
    candidate = json.loads((tmp_path / 'cot' / 'candidates.jsonl').read_text(encoding='utf-8'))
    assert candidate['synthetic'] and candidate['teacher_status'] == 'PENDING_HUMAN_REVIEW'
    assert candidate['output_time_ms'] == row['output_time_ms']
    from sc_mv_dmer.models.qwen import parse_answer
    np.testing.assert_allclose(parse_answer(candidate['teacher_text']).numpy(), row['target'])
    assert '<ANSWER>' not in candidate['inference_prompt']
    assert candidate['inference_prompt'] == ''  # Exact match to the empty predicted-event snapshot.
    review = json.loads((tmp_path / 'cot' / 'review_template.json').read_text(encoding='utf-8'))
    assert review['items'][0]['approved'] is None
    with pytest.raises(ValueError, match='train'):
        api.write_cot_candidates([row | {'split': 'validation'}], {}, tmp_path / 'invalid')


def test_missing_cot_label_keeps_record_without_fabricating_answer(tmp_path):
    row = {'split': 'train', 'window_id': 'song@000000000', 'song_id': 'song',
           'output_time_ms': list(range(15000, 45000, 500)),
           'target': np.zeros((60, 2)), 'mask': np.ones((60, 2), bool)}
    row['mask'][59, 0] = False
    events().write_cot_candidates([row], {row['window_id']: []}, tmp_path / 'cot')
    candidate = json.loads((tmp_path / 'cot' / 'candidates.jsonl').read_text(encoding='utf-8'))
    assert candidate['teacher_text'] is None
    assert candidate['teacher_status'] == 'MISSING_TARGET_KEEP_VA_ONLY'


def test_sensor_epoch_updates_weights_and_pooled_evaluation_uses_raw_units():
    torch.set_num_threads(2)
    model = sensor().SensorNetwork()
    batch = {'views': {k: torch.randn(1, 90, d) for k, d in {'mel':128,'mfcc':40,'chroma':12}.items()},
             'audio_mask': torch.ones(1,90,dtype=torch.bool), 'regression': torch.randn(1,90,3),
             'regression_mask': torch.ones(1,90,3,dtype=torch.bool),
             'key_target': torch.zeros(1,9,dtype=torch.long),'key_mask':torch.ones(1,9,dtype=torch.bool)}
    initial = model.heads['mode'][0].weight.detach().clone()
    loss = sensor().train_sensor_epoch(model, [batch], torch.optim.AdamW(model.parameters(), lr=.001), torch.device('cpu'))
    assert np.isfinite(loss)
    assert not torch.equal(initial, model.heads['mode'][0].weight)
    target = np.tile(np.linspace(-.2,.3,90)[:,None],(1,3))
    record = {'prediction':target.copy(),'regression':target,'regression_mask':np.ones((90,3),bool),
              'key_probability':np.tile(np.r_[.9,np.repeat(.1/11,11)],(9,1)),
              'key_target':np.zeros(9,int),'key_mask':np.ones(9,bool)}
    scores = sensor().evaluate_sensor_records([record], np.zeros(3), np.repeat(1/12,12))
    assert scores['mode_ccc'] == pytest.approx(1.)
    assert scores['rms_mse'] == pytest.approx(0.)
    assert scores['key_ce'] == pytest.approx(-np.log(.9))


def test_event_out_of_domain_predictions_are_unknown_and_constant_threshold_is_medium():
    prediction = np.tile([.2, .3, .1], (90,1))
    prediction[:10,1] = 1.5
    result = events().make_predicted_events(prediction, np.ones((90,3),bool),
                    np.full((9,12),1/12),np.ones(9,bool),start_ms=0,
                    audio_coverage=np.ones(90),thresholds={'rms':[.2,.2],'brightness':[.2,.4]})
    assert result[0]['brightness'] is None
    assert result[0]['brightness_level'] is None
    assert result[0]['rms_level'] == 'medium'
