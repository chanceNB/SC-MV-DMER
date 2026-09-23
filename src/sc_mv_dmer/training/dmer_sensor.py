"""Corrected sensor development training; its gates never block acoustics."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from sc_mv_dmer.models.acoustic import ViewEncoder

VIEWS = {'mel': 128, 'mfcc': 40, 'chroma': 12}
CHANNELS = ('rms', 'brightness', 'mode')


@dataclass
class SensorNormalizer:
    mean: np.ndarray
    std: np.ndarray
    counts: np.ndarray

    @classmethod
    def fit(cls, rows):
        sums, squares, counts = np.zeros(3), np.zeros(3), np.zeros(3, dtype=np.int64)
        for row in rows:
            if row['split'] != 'train':
                raise ValueError('sensor normalization may fit train rows only')
            y, mask = np.asarray(row['regression'], dtype=np.float64), np.asarray(row['regression_mask'], dtype=bool)
            if y.shape != mask.shape or y.ndim != 2 or y.shape[-1] != 3 or not np.isfinite(y[mask]).all():
                raise ValueError('invalid sensor targets or masks')
            y = np.where(mask, y, 0)
            sums += y.sum(0); squares += (y * y).sum(0); counts += mask.sum(0)
        if not counts.any():
            raise ValueError('no train sensor supervision')
        mean = sums / np.maximum(counts, 1)
        std = np.maximum(np.sqrt(np.maximum(squares / np.maximum(counts, 1) - mean ** 2, 0)), 1e-6)
        std[counts == 0] = 1.
        return cls(mean, std, counts)

    def transform(self, values, mask):
        mask = np.asarray(mask, dtype=bool)
        values = np.asarray(values)
        if not np.isfinite(values[mask]).all():
            raise ValueError('nonfinite valid target')
        return np.where(mask, (np.where(mask, values, 0) - self.mean) / self.std, 0).astype(np.float32)

    def inverse(self, values):
        return np.asarray(values) * self.std + self.mean

    def to_dict(self):
        return {'fit_role': 'train', 'channels': list(CHANNELS), 'mean': self.mean.tolist(),
                'std': self.std.tolist(), 'counts': self.counts.tolist()}


class SensorNetwork(nn.Module):
    """Independent 256-d Transformer encoders, clean inputs, 256→64 heads."""
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({name: ViewEncoder(dim, 256) for name, dim in VIEWS.items()})
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, output))
                                   for name, output in [('rms', 1), ('brightness', 1), ('mode', 1), ('key', 12)]})

    def forward(self, views, audio_mask):
        mask = audio_mask.bool()
        if mask.ndim != 2 or mask.shape[1] != 90 or not mask.any(-1).all():
            raise ValueError('every sensor sequence needs valid audio on the 90-frame grid')
        encoded = {}
        for name, dimensions in VIEWS.items():
            if views[name].shape != (*mask.shape, dimensions):
                raise ValueError(f'invalid {name} feature shape')
            clean = views[name].masked_fill(~mask[..., None], 0)
            if not torch.isfinite(clean).all():
                raise ValueError('nonfinite valid input feature')
            encoded[name] = self.encoders[name](clean, mask)
        regression = torch.cat([self.heads[name](encoded[view]) for name, view in zip(CHANNELS, VIEWS)], -1)
        segments = encoded['chroma'].reshape(-1, 9, 10, 256)
        segment_mask = mask.reshape(-1, 9, 10)
        pooled = (segments * segment_mask[..., None]).sum(2) / segment_mask.sum(2).clamp_min(1)[..., None]
        return {'regression': regression.masked_fill(~mask[..., None], 0),
                'key_logits': self.heads['key'](pooled), 'key_audio_mask': segment_mask.any(-1)}


def sensor_loss(output, target, regression_mask, key_target, key_mask):
    prediction = output['regression']
    mask, key_mask = regression_mask.bool(), key_mask.bool()
    if prediction.shape != target.shape or mask.shape != target.shape or target.shape[-1] != 3:
        raise ValueError('sensor loss shape mismatch')
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('valid sensor predictions and targets must be finite')
    delta = prediction.masked_fill(~mask, 0) - target.masked_fill(~mask, 0)
    # Equal song/channel weighting, while unavailable pseudo labels stay masked.
    count = mask.sum(1)
    observed = count > 0
    terms = []
    for dim in range(3):
        per_song = delta[..., dim].square().sum(1) / count[:, dim].clamp_min(1)
        if observed[:, dim].any(): terms.append(per_song[observed[:, dim]].mean())
    logits = output['key_logits']
    if logits.shape[:-1] != key_mask.shape or key_target.shape != key_mask.shape or logits.shape[-1] != 12:
        raise ValueError('key target shape mismatch')
    if key_mask.any():
        if not torch.isfinite(logits[key_mask]).all() or ((key_target[key_mask] < 0) | (key_target[key_mask] >= 12)).any():
            raise ValueError('invalid finite key logits or class')
        terms.append(F.cross_entropy(logits[key_mask], key_target[key_mask]))
    return torch.stack(terms).sum() if terms else prediction.masked_fill(~mask, 0).sum() * 0


def sensor_gate(metrics: dict, key_source: str) -> dict:
    def improved(name):
        current, baseline = metrics.get(name), metrics.get({'rms_mse': 'rms_baseline_mse', 'brightness_mse': 'brightness_baseline_mse', 'key_ce': 'key_prior_ce'}[name])
        return current is not None and baseline is not None and np.isfinite(current) and np.isfinite(baseline) and current <= baseline - max(1e-8, 1e-6 * abs(baseline))
    ccc = metrics.get('mode_ccc')
    checks = {'mode_ccc_gt_0_7': ccc is not None and np.isfinite(ccc) and ccc > .7,
              'energy_mse_improved': improved('rms_mse'), 'brightness_mse_improved': improved('brightness_mse'),
              'key_ce_improved': improved('key_ce')}
    reasons = [name.upper() + '_FAILED' for name, passed in checks.items() if not passed]
    if key_source != 'ESSENTIA_EDMA': reasons.append('ESSENTIA_KEY_TARGET_UNAVAILABLE')
    # A single development run cannot issue the five-seed formal gate.
    reasons.append('FORMAL_FIVE_SEED_AGGREGATION_NOT_COMPLETED')
    return {**{k: bool(v) for k, v in checks.items()}, 'blocking_reasons': reasons,
            'events_llm_blocked': bool(reasons), 'acoustic_blocked': False,
            'mode_aggregation': 'pooled_valid_frames', 'formal_pass': False}


def load_sensor_rows(rows):
    result = []
    for row in rows:
        with np.load(row['path'], allow_pickle=False) as payload:
            raw = dict(row, views={k: payload[k].astype(np.float32) for k in VIEWS},
                       audio_mask=payload['audio_mask'].astype(bool), audio_coverage=payload['audio_coverage'],
                       regression=np.stack([payload['sensor_' + k] for k in CHANNELS], -1),
                       regression_mask=payload['sensor_valid_mask'].astype(bool),
                       key_target=payload['sensor_key'].astype(np.int64), key_mask=payload['sensor_key_mask'].astype(bool))
        if raw['regression'].shape != (90, 3) or raw['regression_mask'].shape != (90, 3):
            raise ValueError('invalid raw sensor regression shape')
        if raw['key_target'].shape != (9,) or raw['key_mask'].shape != (9,):
            raise ValueError('invalid raw sensor key shape')
        if not np.array_equal(raw['audio_mask'], row['audio_mask']):
            raise ValueError('sensor feature/protocol audio masks differ')
        raw['regression_mask'] &= raw['audio_mask'][:, None]
        raw['key_mask'] &= raw['audio_mask'].reshape(9, 10).any(-1)
        result.append(raw)
    return result


class SensorDataset(Dataset):
    def __init__(self, rows, feature_normalizer, sensor_normalizer):
        self.rows, self.feature_normalizer, self.sensor_normalizer = rows, feature_normalizer, sensor_normalizer

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {'views': self.feature_normalizer.transform(row['views'], row['audio_mask']),
                'audio_mask': row['audio_mask'], 'regression': self.sensor_normalizer.transform(row['regression'], row['regression_mask']),
                'regression_mask': row['regression_mask'], 'key_target': row['key_target'], 'key_mask': row['key_mask'],
                'index': index}


def _sensor_batch(batch, device):
    views = {k: v.to(device=device, dtype=torch.float32) for k, v in batch['views'].items()}
    return views, batch['audio_mask'].to(device), [batch[k].to(device) for k in ('regression', 'regression_mask', 'key_target', 'key_mask')]


def train_sensor_epoch(model, loader, optimizer, device):
    model.train()
    total, count = 0., 0
    for batch in loader:
        views, mask, targets = _sensor_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = sensor_loss(model(views, mask), *targets)
        if not torch.isfinite(loss): raise ValueError('nonfinite sensor loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(mask); count += len(mask)
    if not count: raise ValueError('empty sensor training population')
    return total / count


def evaluate_sensor_records(records, training_mean, training_key_prior):
    all_p, all_y = {k: [] for k in CHANNELS}, {k: [] for k in CHANNELS}
    key_losses, prior_losses = [], []
    for row in records:
        p, y, mask = np.asarray(row['prediction'], dtype=np.float64), np.asarray(row['regression'], dtype=np.float64), np.asarray(row['regression_mask'], bool)
        if p.shape != y.shape or y.shape != mask.shape or not np.isfinite(p[mask]).all() or not np.isfinite(y[mask]).all():
            raise ValueError('nonfinite or malformed valid sensor predictions')
        for dim, name in enumerate(CHANNELS):
            all_p[name].append(p[mask[:, dim], dim]); all_y[name].append(y[mask[:, dim], dim])
        key_mask = np.asarray(row['key_mask'], bool)
        if key_mask.any():
            key = np.asarray(row['key_target'])[key_mask]
            probability = np.asarray(row['key_probability'])[key_mask]
            if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(-1), 1.):
                raise ValueError('invalid key probabilities')
            key_losses.extend((-np.log(np.clip(probability[np.arange(len(key)), key], 1e-12, 1))).tolist())
            prior_losses.extend((-np.log(np.asarray(training_key_prior)[key])).tolist())
    result = {'song_count': len(records), 'mode_ccc': None, 'mode_invalid_reason': None}
    for dim, name in enumerate(CHANNELS):
        y = np.concatenate(all_y[name]) if all_y[name] else np.array([])
        p = np.concatenate(all_p[name]) if all_p[name] else np.array([])
        result[name + '_valid_count'] = len(y)
        result[name + '_mse'] = float(np.mean((p-y)**2)) if len(y) else None
        result[name + '_baseline_mse'] = float(np.mean((training_mean[dim]-y)**2)) if len(y) else None
        if name == 'mode':
            if len(y) < 2: result['mode_invalid_reason'] = 'MODE_CCC_LT2_VALID_PAIRS'
            elif np.all(y == y[0]): result['mode_invalid_reason'] = 'MODE_CCC_ZERO_VARIANCE_TARGET'
            elif np.all(p == p[0]): result['mode_invalid_reason'] = 'MODE_CCC_ZERO_VARIANCE_PREDICTION'
            else:
                denominator = np.var(p) + np.var(y) + (p.mean() - y.mean())**2
                result['mode_ccc'] = float(2 * np.mean((p-p.mean()) * (y-y.mean())) / denominator)
    result['key_valid_count'] = len(key_losses)
    result['key_ce'] = float(np.mean(key_losses)) if key_losses else None
    result['key_prior_ce'] = float(np.mean(prior_losses)) if prior_losses else None
    return result


@torch.no_grad()
def predict_sensor(model, loader, device, normalizer):
    model.eval()
    records, total, count = [], 0., 0
    for batch in loader:
        views, mask, targets = _sensor_batch(batch, device)
        output = model(views, mask)
        loss = sensor_loss(output, *targets)
        if not torch.isfinite(loss): raise ValueError('nonfinite validation loss')
        total += float(loss) * len(mask); count += len(mask)
        raw = normalizer.inverse(output['regression'].cpu().numpy())
        probability = output['key_logits'].softmax(-1).cpu().numpy()
        for offset, index in enumerate(batch['index']):
            row = loader.dataset.rows[int(index)]
            records.append(dict(row, prediction=raw[offset], key_probability=probability[offset]))
    if not count: raise ValueError('empty sensor evaluation population')
    return total / count, records


def write_event_snapshot(output, records_by_split, thresholds, binding):
    from sc_mv_dmer.models.events import make_predicted_events, serialize_events
    from sc_mv_dmer.data.deam_dmer_full import file_sha256
    from sc_mv_dmer.foundation.canonical import sha256_canonical

    artifacts, events_by_window = {}, {}
    for split, records in records_by_split.items():
        if split not in ('train', 'validation'): raise ValueError('sensor development snapshots use train/validation only')
        path = output / f'events_{split}.jsonl'
        with path.open('x', encoding='utf-8') as f:
            for row in records:
                if row['split'] != split: raise ValueError('event split mismatch')
                predicted = make_predicted_events(row['prediction'], row['regression_mask'], row['key_probability'], row['key_mask'],
                                                   start_ms=row['start_ms'], audio_coverage=row['audio_coverage'], thresholds=thresholds)
                events_by_window[row['window_id']] = predicted
                item = {k: row[k] for k in ('window_id', 'song_id', 'split', 'output_time_ms')}
                item.update(events=predicted, inference_prompt=serialize_events(predicted))
                f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        artifacts[split] = {'path': path.name, 'sha256': file_sha256(path), 'count': len(records)}
    result = dict(binding, schema_version='1.0', source='PREDICTED_AUDIO_SENSORS', artifacts=artifacts,
                  formal_llm_ready=False, thresholds=thresholds,
                  mask_source='AUDIO_COVERAGE_AND_OBSERVED_AUDIO_QUALITY_NO_VA_LABELS')
    result['snapshot_sha256'] = sha256_canonical(result)
    (output / 'event_snapshot_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result, events_by_window


def run_sensor(*, manifest_path, split_path, targets_path, cache_path, output, seed=52826381,
               epochs=50, patience=10, batch_size=16, lr=3e-4, device='cuda', write_cot=False):
    """Development optimization with frozen train-only statistics and val selection."""
    import os
    from sc_mv_dmer.training.dmer_runner import load_rows
    from sc_mv_dmer.training.dmer_acoustic import SEEDS, TrainNormalizer
    from sc_mv_dmer.models.events import fit_event_thresholds, write_cot_candidates
    from sc_mv_dmer.data.deam_dmer_full import file_sha256

    output = Path(output)
    if output.exists(): raise FileExistsError('sensor output is immutable; use a new run directory')
    if seed not in SEEDS or min(epochs, patience, batch_size) < 1 or lr <= 0:
        raise ValueError('invalid sensor experiment settings')
    manifest, split, rows = load_rows(Path(manifest_path), Path(split_path), Path(targets_path), Path(cache_path))
    train = load_sensor_rows([r for r in rows if r['split'] == 'train'])
    validation = load_sensor_rows([r for r in rows if r['split'] == 'validation'])
    if (len(train), len(validation)) != (1395, 174): raise ValueError('unexpected sensor population')
    feature_norm, target_norm = TrainNormalizer.fit(train), SensorNormalizer.fit(train)
    thresholds = fit_event_thresholds(train)
    key_counts = np.ones(12, dtype=np.float64)
    for row in train: key_counts += np.bincount(row['key_target'][row['key_mask']], minlength=12)
    key_prior = key_counts / key_counts.sum()
    cache = json.loads(Path(cache_path).read_text(encoding='utf-8'))
    spec = {'seed': seed, 'epochs': epochs, 'patience': patience, 'batch_size': batch_size, 'lr': lr,
            'hidden_dim': 256, 'loss': 'three train-normalized per-song MSE terms plus valid-segment CE',
            'selection': 'lowest validation total normalized sensor loss; earliest exact tie',
            'key_prior': 'train-valid segments Laplace alpha=1', 'dataset_manifest_sha256': manifest['manifest_sha256'],
            'split_sha256': split['split_sha256'], 'feature_manifest_file_sha256': file_sha256(Path(cache_path)),
            'sensor_key_source': cache['sensor_key_source'], 'test_used': False, 'development_only': True,
            'code_sha256': file_sha256(Path(__file__))}
    output.mkdir(parents=True)
    dump = lambda name, data: (output / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    dump('run_spec.json', spec)
    dump('normalization.json', {'features': feature_norm.to_dict(), 'sensors': target_norm.to_dict(),
                               'key_prior': key_prior.tolist(), 'event_thresholds': thresholds})
    started = time.monotonic()
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        dev = torch.device(device)
        train_data, val_data = SensorDataset(train, feature_norm, target_norm), SensorDataset(validation, feature_norm, target_norm)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_data, batch_size=batch_size)
        model = SensorNetwork().to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best_loss, best_epoch, stale = float('inf'), 0, 0
        for epoch in range(1, epochs + 1):
            loss = train_sensor_epoch(model, train_loader, optimizer, dev)
            validation_loss, records = predict_sensor(model, val_loader, dev, target_norm)
            scores = evaluate_sensor_records(records, target_norm.mean, key_prior)
            line = {'epoch': epoch, 'train_loss': loss, 'validation_loss': validation_loss, 'mode_ccc': scores['mode_ccc']}
            with (output / 'curve.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(line, allow_nan=False) + '\n')
            print(json.dumps(line, allow_nan=False), flush=True)
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'spec': spec,
                          'feature_normalizer': feature_norm.to_dict(), 'sensor_normalizer': target_norm.to_dict()}
            torch.save(checkpoint, output / 'last.pt')
            if validation_loss < best_loss:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                torch.save(checkpoint, output / 'best.pt')
            else: stale += 1
            if stale >= patience: break
        model.load_state_dict(torch.load(output / 'best.pt', map_location=dev, weights_only=False)['state_dict'])
        _, val_records = predict_sensor(model, val_loader, dev, target_norm)
        _, train_records = predict_sensor(model, DataLoader(train_data, batch_size=batch_size), dev, target_norm)
        metrics = evaluate_sensor_records(val_records, target_norm.mean, key_prior)
        binding = {k: spec[k] for k in ('dataset_manifest_sha256', 'split_sha256', 'feature_manifest_file_sha256')}
        binding['sensor_checkpoint_sha256'] = file_sha256(output / 'best.pt')
        snapshot, event_map = write_event_snapshot(output, {'train': train_records, 'validation': val_records}, thresholds, binding)
        if write_cot:
            write_cot_candidates(train, event_map, output / 'cot', binding=binding | {'event_snapshot_sha256': snapshot['snapshot_sha256']})
        report = dict(binding, status='DEVELOPMENT_COMPLETED', sensor_key_source=cache['sensor_key_source'], seed=seed,
                      best_epoch=best_epoch, metrics=metrics, gate=sensor_gate(metrics, cache['sensor_key_source']), formal_pass=False,
                      event_snapshot_sha256=snapshot['snapshot_sha256'], test_evaluated=False,
                      elapsed_seconds=time.monotonic()-started, cot_candidates_written=write_cot)
        dump('sensor_report.json', report)
        return report
    except BaseException as error:
        dump('failure.json', {'status': 'INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED', 'error': str(error)})
        raise
