import copy
import json
from pathlib import Path

import pytest
import torch

from sc_mv_dmer.training.dmer_qwen import (
    QwenInputs, dry_run, build_inference_prompt, train_step, inject_lora,
    STAGE_EPOCHS, STAGE_LR, validate_human_review,
)


def test_missing_prerequisites_are_explicit_and_dry_run_never_creates_outputs(tmp_path):
    inputs = QwenInputs(**{name: tmp_path / name for name in QwenInputs.__dataclass_fields__})
    report = dry_run(inputs)
    assert report['status'] == 'BLOCKED'
    assert report['training_started'] is False
    assert len(report['blockers']) >= 8
    assert list(tmp_path.iterdir()) == []


def test_prompt_uses_only_frozen_predicted_events_and_not_targets():
    event = {'source': 'PREDICTED_AUDIO_SENSORS', 'inference_prompt': 'Energy rises at 20s.'}
    first = build_inference_prompt(event)
    event.update(target=[[1, -1]] * 60, teacher_text='GT secret answer')
    assert build_inference_prompt(event) == first
    assert 'secret' not in first


def test_pending_or_unbound_review_cannot_pass():
    items = [{'window_id': str(i), 'approved': True, 'evidence_correct': True,
              'va_grid_correct': True, 'unsupported_claim': False} for i in range(100)]
    review = {'human_reviewed': False, 'reviewer': None, 'items': items}
    with pytest.raises(ValueError, match='human'):
        validate_human_review(review, cot_hash='x', event_hash='y', allowed_windows=set(map(str, range(100))))
    review.update(human_reviewed=True, reviewer='human-reviewer', cot_file_sha256='bad', event_snapshot_sha256='y')
    with pytest.raises(ValueError, match='binding'):
        validate_human_review(review, cot_hash='x', event_hash='y', allowed_windows=set(map(str, range(100))))


def test_review_requires_100_distinct_samples_and_85_percent_actual_passes():
    review = {'human_reviewed': True, 'reviewer': 'named-human', 'cot_file_sha256': 'cot',
              'event_snapshot_sha256': 'events', 'items': [
        {'window_id': str(i), 'approved': i < 85, 'evidence_correct': True,
         'va_grid_correct': True, 'unsupported_claim': False} for i in range(100)]}
    result = validate_human_review(review, cot_hash='cot', event_hash='events', allowed_windows=set(map(str, range(100))))
    assert result['passed'] == 85
    review['items'][0]['evidence_correct'] = False
    with pytest.raises(ValueError, match='85'):
        validate_human_review(review, cot_hash='cot', event_hash='events', allowed_windows=set(map(str, range(100))))


def test_two_real_cpu_steps_freeze_then_update_lora_without_changing_formal_schedule():
    transformers = pytest.importorskip('transformers')
    if not hasattr(transformers, 'Qwen2Config'):
        pytest.skip('requires isolated Qwen transformers environment')
    pytest.importorskip('peft')
    from sc_mv_dmer.models import AcousticModel, QwenDualHeadAdapter
    torch.set_num_threads(2)
    torch.manual_seed(74)
    base = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(vocab_size=128,
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=512, pad_token_id=0))
    adapter = QwenDualHeadAdapter(base, acoustic_dim=16)
    acoustic = AcousticModel(hidden_dim=16, use_timesnet=True, view_dropout_p=0)
    batch = {'views': {k: torch.randn(1,90,d) for k,d in {'deep':768,'mel':128,'mfcc':40,'chroma':12}.items()},
        'audio_mask': torch.ones(1,90,dtype=torch.bool), 'target': torch.zeros(1,60,2),
        'mask': torch.ones(1,60,2,dtype=torch.bool), 'mert_layers': torch.randn(1,2,180,768),
        'mert_time_seconds': torch.arange(180).float()[None] / 4 + .125,
        'mert_mask': torch.ones(1,180,dtype=torch.bool), 'prompt_input_ids': torch.tensor([[1,3,4]]),
        'prompt_attention_mask': torch.ones(1,3,dtype=torch.bool), 'answer_labels': torch.tensor([[7,8,9,2]]),
        'sensor_target': torch.zeros(1,90,3), 'sensor_mask': torch.ones(1,90,3,dtype=torch.bool),
        'key_target': torch.zeros(1,9,dtype=torch.long), 'key_mask': torch.ones(1,9,dtype=torch.bool)}
    original = {n:p.detach().clone() for n,p in base.named_parameters()}
    before_patch = adapter.patch_projection.weight.detach().clone()
    opt = torch.optim.AdamW([p for m in (acoustic,adapter) for p in m.parameters() if p.requires_grad], lr=1e-3)
    first = train_step(acoustic, adapter, batch, opt, stage=1, device='cpu')
    assert 'language' not in first
    assert not torch.equal(before_patch, adapter.patch_projection.weight)
    assert all(torch.equal(original[n],p) for n,p in base.named_parameters())
    adapter.backbone = inject_lora(adapter.backbone)
    adapter.configure_stage('lora')
    before_lora = {n:p.detach().clone() for n,p in adapter.backbone.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for m in (acoustic,adapter) for p in m.parameters() if p.requires_grad], lr=1e-3)
    second = train_step(acoustic, adapter, batch, opt, stage=2, device='cpu')
    assert second['language'] > 0 and 'consistency' not in second
    assert any(not torch.equal(before_lora[n],p) for n,p in adapter.backbone.named_parameters() if n in before_lora)
    assert STAGE_EPOCHS == (5,5) and STAGE_LR == (1e-4,3e-5)
