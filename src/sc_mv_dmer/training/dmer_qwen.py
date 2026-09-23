"""Strict readiness and executable 5+5 epoch DMER/Qwen training.

No pretrained weights are downloaded and dry_run never writes files. The
frozen event snapshot is model-visible; train-only teacher text is supervision
only. Parsed numeric consistency is not part of either stage's objective.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import itertools
import json
import math
import re
import socket
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sc_mv_dmer.data.deam_dmer_full import file_sha256
from sc_mv_dmer.data.dmer_protocol import verify_split
from sc_mv_dmer.evaluation.dmer_metrics import evaluate_songs
from sc_mv_dmer.foundation.canonical import sha256_canonical
from sc_mv_dmer.models import AcousticModel, QwenDualHeadAdapter, parse_answer, generation_token_budget
from sc_mv_dmer.training.dmer_acoustic import TrainNormalizer, VIEW_DIMS, SEEDS, masked_va_loss, smoothness_loss
from sc_mv_dmer.training.dmer_runner import load_rows, WindowDataset, stitch_song_predictions, save_predictions

STAGE_EPOCHS = (5, 5)
STAGE_LR = (1e-4, 3e-5)
MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'


@dataclass(frozen=True)
class QwenInputs:
    dataset_manifest: Path
    split_manifest: Path
    targets: Path
    feature_manifest: Path
    sensor_report: Path
    event_manifest: Path
    cot_manifest: Path
    human_review: Path
    qwen_snapshot: Path
    resource_authorization: Path


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _bound_path(manifest_path: Path, relative: str) -> Path:
    path = (manifest_path.parent / relative).resolve()
    if not path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError('artifact path escapes its manifest directory')
    return path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_human_review(review: dict, *, cot_hash: str, event_hash: str,
                          allowed_windows: set[str]) -> dict:
    if review.get('human_reviewed') is not True or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        raise ValueError('actual named human review is required; pending templates do not pass')
    if review.get('cot_file_sha256') != cot_hash or review.get('event_snapshot_sha256') != event_hash:
        raise ValueError('human review binding does not match CoT and frozen events')
    items = review.get('items', [])
    identities = [x.get('window_id') for x in items]
    if len(items) < 100 or len(set(identities)) != len(items) or not set(identities) <= allowed_windows:
        raise ValueError('human review requires at least 100 distinct train CoT windows')
    fields = ('approved', 'evidence_correct', 'va_grid_correct', 'unsupported_claim')
    if any(any(type(row.get(field)) is not bool for field in fields) for row in items):
        raise ValueError('human review has incomplete decisions')
    passed = sum(row['approved'] and row['evidence_correct'] and row['va_grid_correct']
                 and not row['unsupported_claim'] for row in items)
    if 100 * passed < 85 * len(items):
        raise ValueError('human quality is below the required 85 percent')
    return {'reviewed': len(items), 'passed': passed, 'fraction': passed / len(items)}


def build_inference_prompt(event: dict) -> str:
    if event.get('source', 'PREDICTED_AUDIO_SENSORS') != 'PREDICTED_AUDIO_SENSORS':
        raise ValueError('inference events must originate from frozen predicted audio sensors')
    text = event.get('inference_prompt')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('predicted event serialization is missing')
    # No target, teacher text, or annotation field is read in this function.
    return ('Predict Valence and Arousal for the 60 local times 15.0, 15.5, ..., 44.5 seconds. '
            'Use the audio time tokens and the following measured/predicted acoustic evidence. '
            'Give a concise evidence-based explanation in <THINK>, then <ANSWER>VA: '
            '[(v1,a1), ..., (v60,a60)]</ANSWER>. Values must be in [-1,1].\n'
            'Frozen acoustic events:\n' + text)


def _load_events(path: Path, manifest: dict) -> dict[str, dict]:
    unsigned = {k: v for k, v in manifest.items() if k != 'snapshot_sha256'}
    if manifest.get('snapshot_sha256') != sha256_canonical(unsigned):
        raise ValueError('event snapshot checksum mismatch')
    if manifest.get('source') != 'PREDICTED_AUDIO_SENSORS':
        raise ValueError('events must be produced by a frozen sensor checkpoint')
    result = {}
    for role in ('train', 'validation'):
        artifact = manifest['artifacts'][role]
        source = _bound_path(path, artifact['path'])
        if file_sha256(source) != artifact['sha256']:
            raise ValueError(f'{role} event file checksum mismatch')
        records = _jsonl(source)
        if len(records) != artifact['count']:
            raise ValueError('event record count mismatch')
        for row in records:
            if row.get('split') != role or row['window_id'] in result:
                raise ValueError('duplicated or cross-role event window')
            build_inference_prompt(row)
            result[row['window_id']] = row
    return result


def _load_cot(path: Path, manifest: dict, events: dict[str, dict]) -> tuple[dict[str, dict], str]:
    source = _bound_path(path, manifest['records_path'])
    actual_hash = file_sha256(source)
    if actual_hash != manifest['records_sha256']:
        raise ValueError('CoT file checksum mismatch')
    records = _jsonl(source)
    result = {}
    for row in records:
        identity = row['window_id']
        if row.get('split') != 'train' or identity not in events or events[identity]['split'] != 'train':
            raise ValueError('CoT must contain training windows only')
        if identity in result or row.get('song_id') != events[identity]['song_id']:
            raise ValueError('CoT identity is duplicated or mismatched')
        if row.get('inference_prompt') != events[identity]['inference_prompt']:
            raise ValueError('CoT model-visible prompt differs from frozen sensor events')
        teacher = row.get('teacher_text')
        if not isinstance(teacher, str):
            raise ValueError('CoT artifact contains pending/missing teacher text')
        parse_answer(teacher)
        if not re.search(r'<THINK>\s*\S.*?</THINK>', teacher, re.S):
            raise ValueError('teacher explanation is missing')
        result[identity] = row
    if not result:
        raise ValueError('CoT artifact is empty')
    return result, actual_hash


def _snapshot_check(path: Path) -> dict:
    config = _json(path / 'config.json')
    expected = {'model_type': 'qwen2', 'hidden_size': 3584, 'vocab_size': 152064,
                'num_hidden_layers': 28, 'num_attention_heads': 28, 'num_key_value_heads': 4,
                'intermediate_size': 18944}
    if any(config.get(k) != value for k, value in expected.items()):
        raise ValueError('snapshot is not the expected Qwen2.5-7B architecture')
    for file in ('tokenizer.json', 'tokenizer_config.json'):
        if not (path / file).is_file():
            raise ValueError(f'local snapshot is missing {file}')
    index = path / 'model.safetensors.index.json'
    names = set(_json(index)['weight_map'].values()) if index.is_file() else {'model.safetensors'}
    for name in names:
        resolved = (path / name).resolve()
        if not resolved.is_relative_to(path.resolve()) or not resolved.is_file() or resolved.stat().st_size == 0:
            raise ValueError('local Qwen weight shard missing or invalid')
    provenance = _json(path / 'snapshot_provenance.json')
    if provenance.get('model_id') != MODEL_ID or not re.fullmatch(r'[0-9a-f]{40}', provenance.get('revision', '')):
        raise ValueError('snapshot needs a pinned official model ID and revision')
    checksums = provenance.get('files', {})
    for name in names | {'config.json', 'tokenizer.json', 'tokenizer_config.json'}:
        if checksums.get(name) != file_sha256(path / name):
            raise ValueError(f'Qwen snapshot file hash is not pinned: {name}')
    return {'model_id': MODEL_ID, 'revision': provenance['revision'], 'config': expected}


def _validate_resource(record: dict) -> None:
    required = ('source_reference', 'confirmation_text', 'host_name', 'confirmed_at')
    if record.get('authorized_by') != 'USER' or any(not isinstance(record.get(k), str) or not record[k].strip() for k in required):
        raise ValueError('external GPU execution needs a real user confirmation and source reference')
    if record.get('model_id') != MODEL_ID or record.get('stages') != [1, 2] or record.get('epochs') != [5, 5]:
        raise ValueError('resource authorization does not cover this model and fixed 5+5 stages')
    if record.get('external_resource') is not True:
        raise ValueError('this formal profile requires explicitly authorized external GPU resources')


def dry_run(inputs: QwenInputs) -> dict:
    blockers, checks, documents = [], {}, {}
    def check(name, function):
        try:
            result = function()
            checks[name] = result if result is not None else 'PASS'
            return result
        except Exception as error:
            blockers.append({'code': name, 'detail': str(error)})
            return None
    for field in dataclasses.fields(inputs):
        if field.name in ('qwen_snapshot', 'targets'):
            continue
        value = check('READ_' + field.name.upper(), lambda name=field.name: _json(Path(getattr(inputs, name))))
        if value is not None:
            documents[field.name] = value
    dataset, split, cache = (documents.get(k) for k in ('dataset_manifest', 'split_manifest', 'feature_manifest'))
    rows = None
    if dataset is not None and split is not None and cache is not None:
        def verify_data():
            if len(dataset['records']) != 1802:
                raise ValueError('all 1802 source songs are required')
            _, _, loaded = load_rows(inputs.dataset_manifest, inputs.split_manifest, inputs.targets, inputs.feature_manifest)
            if sum(r['split'] == 'train' for r in loaded) != 1395 or sum(r['split'] == 'validation' for r in loaded) != 174:
                raise ValueError('unexpected revised 80/10/10 training population')
            for row in loaded:
                with zipfile.ZipFile(row['path']) as archive:
                    if not {'mert_layers.npy', 'mert_time_seconds.npy', 'mert_mask.npy'} <= set(archive.namelist()):
                        raise ValueError('full raw two-layer MERT cache is required')
                    with archive.open('mert_layers.npy') as member:
                        version = np.lib.format.read_magic(member)
                        shape, _, _ = np.lib.format._read_array_header(member, version)
                    if len(shape) != 3 or shape[0] != 2 or shape[-1] != 768:
                        raise ValueError('raw MERT cache has incorrect layer shape')
            return loaded
        rows = check('FULL_DATA_SPLIT_RAW_CACHE', verify_data)
        if rows is not None:
            checks['FULL_DATA_SPLIT_RAW_CACHE'] = {'source_songs': 1802, 'train_windows': 1395, 'validation_windows': 174}
    sensor, event = documents.get('sensor_report'), documents.get('event_manifest')
    if sensor is not None:
        def verify_sensor():
            gate = sensor.get('gate', {})
            if sensor.get('formal_pass') is not True or gate.get('formal_pass') is not True:
                raise ValueError('sensor formal PASS is missing; this does not block acoustic baselines')
            if 'ESSENTIA' not in str(sensor.get('sensor_key_source', '')).upper():
                raise ValueError('sensor key source must be genuine Essentia, not diagnostic K-S')
            if not all(gate.get(k) is True for k in ('mode_ccc_gt_0_7', 'energy_mse_improved', 'brightness_mse_improved', 'key_ce_improved')):
                raise ValueError('one or more sensor quality criteria failed')
            if dataset is None or split is None or sensor.get('dataset_manifest_sha256') != dataset['manifest_sha256'] or sensor.get('split_sha256') != split['split_sha256']:
                raise ValueError('sensor dataset/split binding mismatch')
            if sensor.get('feature_manifest_file_sha256') != file_sha256(inputs.feature_manifest):
                raise ValueError('sensor feature cache binding mismatch')
            checkpoint = _bound_path(inputs.sensor_report, sensor.get('checkpoint_path', 'best.pt'))
            if file_sha256(checkpoint) != sensor.get('sensor_checkpoint_sha256'):
                raise ValueError('qualified sensor checkpoint checksum mismatch')
        check('SENSOR_FORMAL_ESSENTIA', verify_sensor)
    events = None
    if event is not None:
        events = check('FROZEN_EVENT_SNAPSHOT', lambda: _load_events(inputs.event_manifest, event))
        if events is not None:
            checks['FROZEN_EVENT_SNAPSHOT'] = {'windows': len(events), 'snapshot_sha256': event['snapshot_sha256']}
        def verify_event_bindings():
            if sensor is None or split is None:
                raise ValueError('sensor and split bindings are missing')
            if event.get('sensor_checkpoint_sha256') != sensor.get('sensor_checkpoint_sha256') or event.get('split_sha256') != split['split_sha256'] or event.get('feature_manifest_file_sha256') != file_sha256(inputs.feature_manifest):
                raise ValueError('event lineage mismatch')
            if rows is None or events is None:
                raise ValueError('full train/validation event population cannot be verified')
            expected = {row['window_id']: row for row in rows if row['split'] in ('train', 'validation')}
            if set(events) != set(expected) or any(events[k]['song_id'] != expected[k]['song_id'] or events[k]['split'] != expected[k]['split'] for k in events):
                raise ValueError('event snapshot does not cover exact train/validation windows')
        check('EVENT_LINEAGE_AND_POPULATION', verify_event_bindings)
    cot_result = None
    cot_manifest = documents.get('cot_manifest')
    if cot_manifest is not None:
        def verify_cot():
            if event is None or split is None or events is None:
                raise ValueError('CoT prerequisite event/split bindings missing')
            if cot_manifest.get('event_snapshot_sha256') != event['snapshot_sha256'] or cot_manifest.get('split_sha256') != split['split_sha256']:
                raise ValueError('CoT lineage does not match frozen events/split')
            return _load_cot(inputs.cot_manifest, cot_manifest, events)
        cot_result = check('TRAIN_ONLY_COT', verify_cot)
        if cot_result:
            checks['TRAIN_ONLY_COT'] = {'windows': len(cot_result[0]), 'records_sha256': cot_result[1]}
    review = documents.get('human_review')
    if review is not None:
        def verify_review():
            if cot_result is None or event is None:
                raise ValueError('CoT and event bindings needed before human review can pass')
            return validate_human_review(review, cot_hash=cot_result[1], event_hash=event['snapshot_sha256'], allowed_windows=set(cot_result[0]))
        check('HUMAN_REVIEW_100_85', verify_review)
    snapshot = check('LOCAL_PINNED_QWEN_SNAPSHOT', lambda: _snapshot_check(Path(inputs.qwen_snapshot)))
    if snapshot and cot_result:
        def verify_tokens():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(inputs.qwen_snapshot), local_files_only=True, trust_remote_code=False)
            largest_teacher, largest_prompt = 0, 0
            for row in cot_result[0].values():
                explanation = re.search(r'<THINK>(.*?)</THINK>', row['teacher_text'], re.S).group(1)
                if len(tokenizer.encode(explanation, add_special_tokens=False)) > 300:
                    raise ValueError('teacher THINK exceeds 300 tokens; never truncate the numeric ANSWER')
                largest_teacher = max(largest_teacher, len(tokenizer.encode(row['teacher_text'], add_special_tokens=False)))
            for row in events.values():
                largest_prompt = max(largest_prompt, len(tokenizer.encode(build_inference_prompt(row), add_special_tokens=False)))
            return generation_token_budget(tokenizer) | {'measured_teacher_max': largest_teacher, 'measured_prompt_max': largest_prompt,
                'measured_training_sequence_max': largest_teacher + largest_prompt + 90}
        check('REAL_TOKENIZER_FULL_60_PAIR_BUDGET', verify_tokens)
    resource = documents.get('resource_authorization')
    if resource is not None:
        check('EXPLICIT_EXTERNAL_GPU_AUTHORIZATION', lambda: _validate_resource(resource))
    for dependency in ('transformers', 'peft', 'accelerate', 'bitsandbytes'):
        if importlib.util.find_spec(dependency) is None:
            blockers.append({'code': 'DEPENDENCY_' + dependency.upper(), 'detail': 'not installed in the selected runtime'})
    if importlib.util.find_spec('transformers'):
        check('TRANSFORMERS_PIN', lambda: _require_transformers_pin())
    return {'status': 'BLOCKED' if blockers else 'READY', 'training_started': False,
            'acoustic_training_blocked': False, 'stages': list(STAGE_EPOCHS), 'learning_rates': list(STAGE_LR),
            'blockers': blockers, 'checks': checks, 'test_evaluated': False}


def _require_transformers_pin():
    import transformers
    if transformers.__version__ != '4.51.3':
        raise ValueError('Qwen execution uses isolated transformers==4.51.3')
    return transformers.__version__


def inject_lora(backbone):
    from peft import LoraConfig, TaskType, get_peft_model
    return get_peft_model(backbone, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], bias='none'))


def _sensor_loss(acoustic, output: dict, batch: dict, device) -> torch.Tensor:
    # Qualified sensor checkpoints predict standardized regression channels.
    # Use clean encoder outputs directly; the public bounded mode display
    # tensor is deliberately not used as standardized training evidence.
    encoded = output['encoded_views']
    prediction = torch.cat([acoustic.sensor_heads[name](encoded[view]) for name, view in
                            (('rms', 'mel'), ('brightness', 'mfcc'), ('mode', 'chroma'))], -1)
    target, mask = batch['sensor_target'].to(device), batch['sensor_mask'].to(device).bool()
    if not torch.isfinite(target[mask]).all() or not torch.isfinite(prediction[mask]).all():
        raise ValueError('nonfinite sensor supervision')
    counts = mask.sum(1)
    values = (prediction - target.masked_fill(~mask, 0)).square().masked_fill(~mask, 0).sum(1) / counts.clamp_min(1)
    eligible = counts > 0
    terms = [values[:, c][eligible[:, c]].mean() for c in range(3) if eligible[:, c].any()]
    frames = encoded['chroma'].reshape(-1, 9, 10, acoustic.hidden_dim)
    frame_mask = batch['audio_mask'].to(device).bool().reshape(-1, 9, 10)
    pooled = (frames * frame_mask[..., None]).sum(2) / frame_mask.sum(2).clamp_min(1)[..., None]
    logits = acoustic.sensor_heads['key'](pooled)
    key_mask = batch['key_mask'].to(device).bool() & frame_mask.any(2)
    if key_mask.any():
        terms.append(nn.functional.cross_entropy(logits[key_mask], batch['key_target'].to(device)[key_mask]))
    if not terms:
        raise ValueError('batch has no eligible sensor supervision')
    return torch.stack(terms).sum()


def train_step(acoustic, adapter, batch: dict, optimizer, *, stage: int, device,
               backward_scale: float = 1., zero_grad: bool = True,
               step_optimizer: bool = True) -> dict[str, float]:
    if stage not in (1, 2):
        raise ValueError('training stage must be 1 or 2')
    acoustic.train(); adapter.train()
    if stage == 1:
        adapter.backbone.eval()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    device = torch.device(device)
    views = {name: value.to(device) for name, value in batch['views'].items()}
    audio_mask = batch['audio_mask'].to(device)
    raw = {k: batch[k].to(device) for k in ('mert_layers', 'mert_time_seconds', 'mert_mask')}
    acoustic_output = acoustic(views, audio_mask, **raw)
    labels = batch.get('answer_labels') if stage == 2 else None
    if labels is not None and not (labels != -100).any():
        labels = None
    output = adapter(batch['prompt_input_ids'].to(device), acoustic_output['fused'], audio_mask,
        prompt_attention_mask=batch['prompt_attention_mask'].to(device),
        answer_labels=labels.to(device) if labels is not None else None)
    target, mask = batch['target'].to(device), batch['mask'].to(device)
    terms = {'va': masked_va_loss(output['prediction'], target, mask),
             'smooth': smoothness_loss(output['prediction'], mask),
             'state': torch.stack([torch.relu(.5 - h).mean() for h in acoustic_output['row_entropy'].values()]).mean(),
             'sensor': _sensor_loss(acoustic, acoustic_output, batch, device)}
    loss = terms['va'] + .05 * terms['smooth'] + .01 * terms['state'] + .1 * terms['sensor']
    if stage == 2 and output['lm_loss'] is not None:
        terms['language'] = output['lm_loss']
        loss = loss + terms['language']
    if not torch.isfinite(loss):
        raise ValueError('nonfinite Qwen training objective')
    (loss * backward_scale).backward()
    parameters = [p for model in (acoustic, adapter) for p in model.parameters() if p.requires_grad]
    if step_optimizer:
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
    return {name: float(value.detach()) for name, value in terms.items()} | {'total': float(loss.detach())}


class QwenWindowDataset(Dataset):
    def __init__(self, rows, normalizer, sensor_normalizer, events, cot):
        self.rows, self.events, self.cot = rows, events, cot
        self.base = WindowDataset(rows, normalizer, raw_mert=True, preload=False)
        self.sensor_normalizer = sensor_normalizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = self.base[index]
        with np.load(row['path'], allow_pickle=False) as payload:
            raw = np.stack([payload['sensor_' + name] for name in ('rms', 'brightness', 'mode')], -1)
            mask = payload['sensor_valid_mask'].astype(bool) & item['audio_mask'][:, None]
            item.update(sensor_target=self.sensor_normalizer.transform(raw, mask), sensor_mask=mask,
                key_target=payload['sensor_key'].astype(np.int64), key_mask=payload['sensor_key_mask'].astype(bool))
        item['prompt_text'] = build_inference_prompt(self.events[row['window_id']])
        item['teacher_text'] = self.cot.get(row['window_id'], {}).get('teacher_text') if row['split'] == 'train' else None
        item['window_id'] = row['window_id']
        return item


class QwenCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        from torch.utils.data import default_collate
        excluded = {'prompt_text', 'teacher_text'}
        result = default_collate([{k:v for k,v in sample.items() if k not in excluded} for sample in samples])
        encoded = self.tokenizer([sample['prompt_text'] for sample in samples], padding=True,
                                 truncation=False, add_special_tokens=False, return_tensors='pt')
        result['prompt_input_ids'], result['prompt_attention_mask'] = encoded['input_ids'], encoded['attention_mask']
        teachers = [self.tokenizer.encode(sample['teacher_text'], add_special_tokens=False)
                    if sample['teacher_text'] is not None else [] for sample in samples]
        maximum = max(map(len, teachers), default=0)
        if maximum:
            labels = torch.full((len(samples), maximum), -100, dtype=torch.long)
            for index, tokens in enumerate(teachers):
                if tokens:
                    labels[index, :len(tokens)] = torch.tensor(tokens)
            result['answer_labels'] = labels
        return result


@torch.no_grad()
def predict_validation(acoustic, adapter, loader, device):
    acoustic.eval(); adapter.eval()
    records = []
    for batch in loader:
        views = {k:v.to(device) for k,v in batch['views'].items()}
        audio = batch['audio_mask'].to(device)
        raw = {k:batch[k].to(device) for k in ('mert_layers', 'mert_time_seconds', 'mert_mask')}
        fused = acoustic(views, audio, **raw)['fused']
        prediction = adapter(batch['prompt_input_ids'].to(device), fused, audio,
            prompt_attention_mask=batch['prompt_attention_mask'].to(device))['prediction'].cpu().numpy()
        for i, song_id in enumerate(batch['song_id']):
            records.append({'song_id': song_id, 'prediction': prediction[i], 'target': batch['target'][i].numpy(),
                'mask': batch['mask'][i].numpy(), 'output_time_ms': batch['output_time_ms'][i].tolist()})
    songs = stitch_song_predictions(records)
    return evaluate_songs(songs), songs


def _checkpoint(acoustic, adapter, optimizer, stage, epoch, score, spec):
    from peft import get_peft_model_state_dict
    return {'acoustic_state': acoustic.state_dict(),
        'adapter_state': {k:v for k,v in adapter.state_dict().items() if not k.startswith('backbone.')},
        'lora_state': get_peft_model_state_dict(adapter.backbone) if stage == 2 else None,
        'optimizer': optimizer.state_dict(), 'stage': stage, 'epoch': epoch, 'validation_score': score,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'spec': spec}


def _restore(checkpoint, acoustic, adapter):
    acoustic.load_state_dict(checkpoint['acoustic_state'], strict=True)
    expected = {k for k in adapter.state_dict() if not k.startswith('backbone.')}
    if set(checkpoint['adapter_state']) != expected:
        raise ValueError('checkpoint adapter keys do not match executable architecture')
    adapter.load_state_dict(checkpoint['adapter_state'], strict=False)
    if checkpoint['lora_state'] is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(adapter.backbone, checkpoint['lora_state'])


def _save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def _write(path: Path, value: dict):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


@torch.no_grad()
def generate_validation(acoustic, adapter, loader, tokenizer, device, output_path: Path, budget: int):
    from sc_mv_dmer.models import evaluate_answer_consistency
    acoustic.eval(); adapter.eval()
    total, parsed = 0, 0
    with output_path.open('x', encoding='utf-8') as stream:
        for batch in loader:
            # The validation generation loader has batch size one. No teacher
            # tokens or label values are passed to the model.
            audio = batch['audio_mask'].to(device)
            fused = acoustic({k:v.to(device) for k,v in batch['views'].items()}, audio,
                **{k:batch[k].to(device) for k in ('mert_layers','mert_time_seconds','mert_mask')})['fused']
            prompt_ids, prompt_mask = batch['prompt_input_ids'].to(device), batch['prompt_attention_mask'].to(device)
            prediction = adapter(prompt_ids, fused, audio, prompt_attention_mask=prompt_mask)['prediction'][0]
            prefix = adapter.prepare_prefix(prompt_ids, fused, audio, prompt_mask)
            token_ids = adapter.backbone.generate(**prefix, max_new_tokens=budget, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
            text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
            evaluation = evaluate_answer_consistency(prediction, text, audio[0,30:90])
            total += 1; parsed += int(evaluation['parse_success'])
            stream.write(json.dumps({'window_id': batch['window_id'][0], 'song_id': batch['song_id'][0],
                'raw_generated_text': text, 'prediction': prediction.cpu().tolist(),
                'consistency_evaluation': evaluation}, ensure_ascii=False, allow_nan=False) + '\n')
    return {'population': total, 'parsed': parsed, 'parse_rate': parsed / total if total else None,
            'role': 'validation', 'consistency_is_loss': False}


def run_training(inputs: QwenInputs, output: Path, *, seed: int = SEEDS[0],
                 device: str = 'cuda:0', micro_batch: int = 1, accumulation: int = 16,
                 generate_explanations: bool = True) -> dict:
    """Execute fixed formal stages only when all external evidence is present.

    There is deliberately no formal epoch override or automatic approval,
    teacher creation, snapshot download, test evaluation, or gate bypass.
    """
    readiness = dry_run(inputs)
    if readiness['status'] != 'READY':
        return readiness
    if seed not in SEEDS or micro_batch < 1 or accumulation < 1:
        raise ValueError('invalid registered seed or runtime batching')
    resource = _json(inputs.resource_authorization)
    if resource['host_name'].casefold() != socket.gethostname().casefold():
        raise ValueError('execution host is not the user-authorized external resource')
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('formal QLoRA training requires the authorized CUDA resource')
    repository = Path(__file__).resolve().parents[3]
    status = subprocess.check_output(['git','status','--porcelain'], cwd=repository, text=True)
    if status.strip():
        raise ValueError('formal Qwen execution requires a clean versioned research checkout')
    if output.exists():
        raise FileExistsError('run directories are immutable; use a new output directory')
    manifest, split, rows = load_rows(inputs.dataset_manifest, inputs.split_manifest, inputs.targets, inputs.feature_manifest)
    train = [row for row in rows if row['split'] == 'train']
    validation = [row for row in rows if row['split'] == 'validation']
    sensor_report, event_manifest = _json(inputs.sensor_report), _json(inputs.event_manifest)
    events = _load_events(inputs.event_manifest, event_manifest)
    cot, _ = _load_cot(inputs.cot_manifest, _json(inputs.cot_manifest), events)
    sensor_path = _bound_path(inputs.sensor_report, sensor_report.get('checkpoint_path', 'best.pt'))
    sensor_checkpoint = torch.load(sensor_path, map_location='cpu', weights_only=False)
    from sc_mv_dmer.training.dmer_sensor import SensorNormalizer
    sn = sensor_checkpoint['sensor_normalizer']
    fn = sensor_checkpoint['feature_normalizer']
    if sn.get('fit_role') != 'train' or fn.get('fit_role') != 'train':
        raise ValueError('sensor and feature normalizers must be fitted on train only')
    sensor_normalizer = SensorNormalizer(np.asarray(sn['mean']), np.asarray(sn['std']), np.asarray(sn['counts']))
    def normalization_rows():
        for row in train:
            with np.load(row['path'], allow_pickle=False) as payload:
                yield {'split':'train', 'audio_mask':payload['audio_mask'], 'views':{k:payload[k] for k in VIEW_DIMS}}
    normalizer = TrainNormalizer.fit(normalization_rows())
    for name in ('mel','mfcc','chroma'):
        if not np.allclose(normalizer.mean[name], fn['mean'][name], rtol=1e-6, atol=1e-6) or not np.allclose(normalizer.std[name], fn['std'][name], rtol=1e-6, atol=1e-6):
            raise ValueError('qualified sensor normalization differs from the current train population')
    spec = {'model_id': MODEL_ID, 'seed':seed, 'epochs':list(STAGE_EPOCHS), 'learning_rates':list(STAGE_LR),
        'micro_batch':micro_batch, 'accumulation':accumulation, 'device':device, 'hidden_dim':256,
        'raw_mert_layers':True, 'timesnet':True, 'event_source':'FROZEN_PREDICTED_AUDIO_SENSORS',
        'selection':'maximum song-equal validation mean CCC, earliest exact tie; stage-1 best initializes stage-2',
        'loss_stage1':'MSE_VA+0.05*smooth+0.01*normalized_entropy_floor+0.1*sensor',
        'loss_stage2':'stage1 + teacher_forced_LM_cross_entropy', 'parsed_consistency':'evaluation_only',
        'test_evaluated':False, 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip(),
        'input_files':{field.name:{'path':str(getattr(inputs,field.name)), 'sha256':file_sha256(getattr(inputs,field.name))}
                       for field in dataclasses.fields(inputs) if field.name != 'qwen_snapshot'},
        'snapshot':readiness['checks']['LOCAL_PINNED_QWEN_SNAPSHOT'], 'code_sha256':file_sha256(Path(__file__))}
    output.mkdir(parents=True)
    _write(output / 'run_spec.json', spec)
    _write(output / 'readiness.json', readiness)
    _write(output / 'normalization.json', normalizer.to_dict() | {'sensor':sn, 'split_sha256':split['split_sha256']})
    started = time.monotonic()
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
        tokenizer = AutoTokenizer.from_pretrained(str(inputs.qwen_snapshot), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        backbone = AutoModelForCausalLM.from_pretrained(str(inputs.qwen_snapshot), local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16), device_map={'':device})
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
        backbone.config.use_cache = False
        acoustic = AcousticModel('shared_state', use_timesnet=True, hidden_dim=256).to(device)
        state = {('sensor_heads.'+key[len('heads.'):]) if key.startswith('heads.') else key:value
                 for key,value in sensor_checkpoint['state_dict'].items()}
        required = {key for key in acoustic.state_dict() if key.startswith(('encoders.mel.','encoders.mfcc.','encoders.chroma.','sensor_heads.'))}
        if set(state) != required:
            raise ValueError('qualified sensor weights do not match all required clean-view encoders and heads')
        acoustic.load_state_dict(state, strict=False)
        acoustic.va_head.requires_grad_(False)
        adapter = QwenDualHeadAdapter(backbone, acoustic_dim=256, freeze_backbone=True)
        adapter.patch_encoder.to(device); adapter.patch_projection.to(device); adapter.va_head.to(device)
        adapter.positions = adapter.positions.to(device)
        train_data = QwenWindowDataset(train, normalizer, sensor_normalizer, events, cot)
        val_data = QwenWindowDataset(validation, normalizer, sensor_normalizer, events, {})
        collator = QwenCollator(tokenizer)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_data, batch_size=micro_batch, shuffle=True, generator=generator, collate_fn=collator)
        val_loader = DataLoader(val_data, batch_size=micro_batch, shuffle=False, collate_fn=collator)
        history = []
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        for stage, (epochs, lr) in enumerate(zip(STAGE_EPOCHS, STAGE_LR), 1):
            if stage == 2:
                adapter.backbone = inject_lora(adapter.backbone)
                adapter.configure_stage('lora')
            parameters = [p for model in (acoustic,adapter) for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)
            best_score, best_epoch = -math.inf, None
            for epoch in range(1, epochs + 1):
                iterator, losses = iter(train_loader), []
                while True:
                    group = list(itertools.islice(iterator, accumulation))
                    if not group:
                        break
                    songs = sum(len(batch['target']) for batch in group)
                    for index, batch in enumerate(group):
                        losses.append(train_step(acoustic,adapter,batch,optimizer,stage=stage,device=device,
                            backward_scale=len(batch['target'])/songs, zero_grad=index==0,
                            step_optimizer=index==len(group)-1))
                metrics, predicted = predict_validation(acoustic,adapter,val_loader,device)
                score = metrics['selection_score']
                if score is None or not math.isfinite(score):
                    raise ValueError('validation CCC cannot select a checkpoint')
                record = {'stage':stage,'epoch':epoch,'train_loss':float(np.mean([x['total'] for x in losses])),
                    'validation_score':score,'macro':metrics['macro'],'elapsed_seconds':time.monotonic()-started}
                history.append(record)
                with (output / 'curve.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
                print(json.dumps(record,allow_nan=False),flush=True)
                state = _checkpoint(acoustic,adapter,optimizer,stage,epoch,score,spec)
                _save_checkpoint(output / f'stage{stage}_last.pt',state)
                if score > best_score:
                    best_score,best_epoch = score,epoch
                    _save_checkpoint(output / f'stage{stage}_best.pt',state)
            best = torch.load(output / f'stage{stage}_best.pt',map_location=device,weights_only=False)
            _restore(best,acoustic,adapter)
            metrics,predicted = predict_validation(acoustic,adapter,val_loader,device)
            _write(output / f'stage{stage}_validation_metrics.json',metrics | {'best_epoch':best_epoch,'last_epoch':epochs})
            save_predictions(output / f'stage{stage}_validation_predictions.jsonl',predicted)
        explanations = {'status':'NOT_REQUESTED'}
        if generate_explanations:
            generation_loader = DataLoader(val_data,batch_size=1,shuffle=False,collate_fn=collator)
            budget = readiness['checks']['REAL_TOKENIZER_FULL_60_PAIR_BUDGET']['max_new_tokens']
            explanations = generate_validation(acoustic,adapter,generation_loader,tokenizer,device,
                output / 'validation_generated_answers.jsonl',budget)
        result = {'status':'SUCCEEDED','training_started':True,'completed_epochs':[5,5],
            'final_checkpoint':'stage2_best.pt','last_checkpoint':'stage2_last.pt','test_evaluated':False,
            'validation_explanations':explanations,'elapsed_seconds':time.monotonic()-started,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(torch.device(device))}
        _write(output / 'final.json',result)
        return result
    except BaseException as error:
        _write(output / 'final.json',{'status':'INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'FAILED',
            'error':str(error),'elapsed_seconds':time.monotonic()-started,'test_evaluated':False})
        raise
