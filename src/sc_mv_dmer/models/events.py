"""Masked predicted audio evidence and explicitly unreviewed training teachers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TONICS = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def fit_event_thresholds(rows) -> dict:
    values = {name: [] for name in ('rms', 'brightness')}
    for row in rows:
        if row['split'] != 'train':
            raise ValueError('event thresholds may fit train rows only')
        y, mask = np.asarray(row['regression']), np.asarray(row['regression_mask'], dtype=bool)
        if y.shape != mask.shape or y.shape[-1] != 3 or not np.isfinite(y[mask]).all():
            raise ValueError('invalid train sensor targets')
        for index, name in enumerate(values):
            values[name].extend(y[mask[:, index], index].tolist())
    if any(not value for value in values.values()):
        raise ValueError('event threshold channel has no train evidence')
    return {name: np.quantile(value, [1 / 3, 2 / 3]).tolist() for name, value in values.items()}


def make_predicted_events(prediction: np.ndarray, valid_mask: np.ndarray, key_probabilities: np.ndarray,
                          key_mask: np.ndarray, *, start_ms: int, audio_coverage: np.ndarray,
                          thresholds: dict, minimum_key_probability: float = .5) -> list[dict]:
    prediction, mask = np.asarray(prediction), np.asarray(valid_mask, dtype=bool)
    coverage, key_p, key_valid = np.asarray(audio_coverage), np.asarray(key_probabilities), np.asarray(key_mask, dtype=bool)
    if prediction.shape != (90, 3) or mask.shape != prediction.shape or coverage.shape != (90,):
        raise ValueError('events require 90 input times and three sensor dimensions')
    if key_p.shape != (9, 12) or key_valid.shape != (9,):
        raise ValueError('tonic events require nine 12-class segments')
    if not np.isfinite(coverage).all() or ((coverage < 0) | (coverage > 1)).any():
        raise ValueError('invalid audio coverage')
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(key_p[key_valid]).all():
        raise ValueError('nonfinite predicted evidence')
    physical = np.stack([prediction[:, 0] >= 0,
                         (prediction[:, 1] >= 0) & (prediction[:, 1] <= 1),
                         np.abs(prediction[:, 2]) <= 1], -1)
    out_of_domain = mask & ~physical
    mask = mask & physical
    if key_valid.any() and ((key_p[key_valid] < 0).any() or not np.allclose(key_p[key_valid].sum(-1), 1.)):
        raise ValueError('key probabilities must be normalized')
    if not 0 < minimum_key_probability <= 1:
        raise ValueError('key probability threshold must be in (0,1]')
    for name in ('rms', 'brightness'):
        limits = np.asarray(thresholds[name])
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
            raise ValueError('invalid fitted event thresholds')
    result = []
    for segment in range(9):
        sl = slice(segment * 10, (segment + 1) * 10)
        event = {'start_ms': start_ms + segment * 5000, 'end_ms': start_ms + (segment + 1) * 5000,
                 'observed_seconds': float(coverage[sl].sum() * .5),
                 'source': 'PREDICTED_AUDIO_SENSORS', 'valid': {}, 'tonic': None,
                 'out_of_domain_count': {name: int(out_of_domain[sl, i].sum())
                                         for i, name in enumerate(('rms', 'brightness', 'mode'))},
                 'tonic_probability': None, 'tonic_status': 'unknown',
                 'tonic_semantics': 'pitch_class_only_not_chord_or_major_minor_key'}
        for index, name in enumerate(('rms', 'brightness', 'mode')):
            take = mask[sl, index] & (coverage[sl] > 0)
            event['valid'][name] = bool(take.any())
            value = float(np.average(prediction[sl, index][take], weights=coverage[sl][take])) if take.any() else None
            event[name] = value
            if name in thresholds:
                limits = thresholds[name]
                if value is None: level = None
                elif limits[0] == limits[1] and np.isclose(value, limits[0], rtol=0, atol=1e-12): level = 'medium'
                else: level = ('low', 'medium', 'high')[int(np.searchsorted(limits, value, side='right'))]
                event[name + '_level'] = level
        if key_valid[segment] and event['observed_seconds'] > 0:
            index = int(key_p[segment].argmax())
            probability = float(key_p[segment, index])
            event['tonic_probability'] = probability
            if probability >= minimum_key_probability:
                event['tonic'], event['tonic_status'] = TONICS[index], 'predicted_uncalibrated'
        result.append(event)
    return result


def serialize_events(events: list[dict]) -> str:
    """Compact evidence text. Token qualification belongs to the pinned tokenizer."""
    lines = []
    for event in events:
        mode = 'unknown' if event.get('mode') is None else f'{event["mode"]:+.3f}'
        lines.append(f'{event["start_ms"]/1000:g}-{event["end_ms"]/1000:g}s '
                     f'energy={event.get("rms_level") or "unknown"} '
                     f'brightness={event.get("brightness_level") or "unknown"} '
                     f'KS-margin={mode} tonic={event.get("tonic") or "unknown"} '
                     f'coverage={event["observed_seconds"]:g}s')
    return '\n'.join(lines)


def write_cot_candidates(rows: list[dict], events_by_window: dict, output: Path,
                         review_count: int = 100, binding: dict | None = None) -> dict:
    """Write synthetic training-only explanation candidates, never quality PASS.

    Labels appear only in teacher output; the inference prompt contains solely
    predicted audio evidence and the requested time grid. Missing label records
    are retained explicitly for VA-only training without fabricating answers.
    """
    from sc_mv_dmer.models.qwen import format_answer

    if review_count != 100:
        raise ValueError('the registered explanation review requires 100 candidates')
    if any(row['split'] != 'train' for row in rows):
        raise ValueError('synthetic teacher creation accepts train rows only')
    if len({r['window_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate teacher window')
    candidates = []
    for row in rows:
        y, mask = np.asarray(row['target']), np.asarray(row['mask'], dtype=bool)
        times = row['output_time_ms']
        if y.shape != (60, 2) or mask.shape != y.shape or len(times) != 60 or np.any(np.diff(times) != 500):
            raise ValueError('teacher requires the native sixty-point grid')
        if not np.isfinite(y[mask]).all() or (np.abs(y[mask]) > 1).any():
            raise ValueError('teacher target outside finite [-1,1] domain')
        evidence = events_by_window[row['window_id']]
        evidence_text = serialize_events(evidence)
        instruction = ('根据音频时序与以下预测事件输出情感轨迹及简短依据。'
                  '事件中的tonic只表示音级估计，不代表和弦或大小调。'
                  f'输出时刻为{times[0]}至{times[-1]}毫秒，间隔500毫秒，共60对(V,A)。')
        item = {'window_id': row['window_id'], 'song_id': row['song_id'], 'split': 'train',
                'synthetic': True, 'generator': 'GROUNDED_DETERMINISTIC_THREE_STEP_TEMPLATE/v1',
                'teacher_status': 'MISSING_TARGET_KEEP_VA_ONLY', 'output_time_ms': times,
                'target_mask': mask.tolist(), 'inference_prompt': evidence_text, 'teacher_instruction': instruction, 'teacher_text': None,
                'events': evidence, 'human_reviewed': False,
                'grounding_limit': 'descriptive_association_only_no_causal_or_semantic_state_claim'}
        if mask.all():
            available = [e for e in evidence if e.get('observed_seconds', 0) > 0]
            first = available[0] if available else None
            observation = ('没有有效的事件描述。' if first is None else
                           f'首个有效区间的能量为{first.get("rms_level") or "unknown"}、亮度为{first.get("brightness_level") or "unknown"}；共记录{len(available)}段事件。')
            explanation = ('1. 音频证据：' + observation +
                           f' 2. 情感轨迹：V均值{y[:,0].mean():+.3f}，A均值{y[:,1].mean():+.3f}；'
                           f'首尾变化分别为{y[-1,0]-y[0,0]:+.3f}和{y[-1,1]-y[0,1]:+.3f}。'
                           ' 3. 对齐说明：按指定的60个时刻逐点给出结果；声学描述不构成情感变化的因果证明。')
            item.update(teacher_status='PENDING_HUMAN_REVIEW', teacher_text=f'<THINK>{explanation}</THINK>\n{format_answer(y)}')
        candidates.append(item)
    selected = sorted((r for r in candidates if r['teacher_text'] is not None),
                      key=lambda r: (hashlib.sha256(r['window_id'].encode()).hexdigest(), r['window_id']))[:100]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'candidates.jsonl').open('x', encoding='utf-8') as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    review = {'status': 'PENDING_HUMAN_REVIEW', 'required_count': 100, 'minimum_pass_fraction': .85,
              'cot_file_sha256': hashlib.sha256((output / 'candidates.jsonl').read_bytes()).hexdigest(),
              'event_snapshot_sha256': (binding or {}).get('event_snapshot_sha256'),
              'binding': binding or {}, 'reviewer': None, 'human_reviewed': False,
              'items': [{'window_id': r['window_id'], 'approved': None, 'evidence_correct': None,
                         'va_grid_correct': None, 'unsupported_claim': None, 'notes': ''} for r in selected]}
    (output / 'review_template.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {'candidate_count': len(candidates), 'eligible_teacher_count': sum(r['teacher_text'] is not None for r in candidates),
               'review_required_count': 100, 'review_available_count': len(selected), 'human_quality_pass': False,
               'training_authorized_by_this_artifact': False, 'token_budget_qualified': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary
