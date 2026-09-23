"""Lossless song-population rebuild for ordinary DEAM DMER.

Raw audio remains untouched. Annotation means use finite original rater rows
at original timestamps >=15s, independently of WorkerId. This publishes no
feature cache, split, normalization statistics, or training authorization.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Callable

from sc_mv_dmer.foundation.canonical import canonical_json, sha256_canonical
from sc_mv_dmer.foundation.identity import stable_id

VERSION = "deam-dmer-full-v1"
MANIFEST_ID = "DEAM-DMER-FULL-v1"
TIME_COLUMN = re.compile(r"sample_(\d+)ms")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_mean_annotations(path: Path) -> dict:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        fields = [x.strip() for x in next(reader)]
        if len(set(fields)) != len(fields):
            raise ValueError(f'duplicate columns: {path.name}')
        columns = sorted((int(m[1]), i) for i, name in enumerate(fields)
                         if (m := TIME_COLUMN.fullmatch(name)) and int(m[1]) >= 15000)
        if not columns:
            raise ValueError(f'no post-15s annotations: {path.name}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'no annotation rows: {path.name}')
    means, counts, invalid = [], [], []
    for _, col in columns:
        values = []
        for row in rows:
            try:
                value = float(row[col])
            except (ValueError, IndexError):
                continue
            if math.isfinite(value):
                values.append(value)
        counts.append(len(values))
        invalid.append(len(rows) - len(values))
        means.append(math.fsum(values) / len(values) if values else None)
    return {'time_ms': [ms for ms, _ in columns], 'values': means,
            'valid_mask': [n > 0 for n in counts], 'valid_rater_count': counts,
            'invalid_rater_count': invalid, 'row_count': len(rows),
            'worker_id_present': 'WorkerId' in fields,
            'aggregation': 'MEAN_OVER_ALL_FINITE_ORIGINAL_ROWS_NO_DEDUPLICATION',
            'target_domain': 'ORIGINAL_VALUES_NO_RESCALE_NO_CLIP'}


def _duration(path: Path) -> float:
    import soundfile as sf
    return float(sf.info(str(path)).duration)


def _source_path(root: Path, relative: str) -> Path:
    logical = Path(relative)
    # DEAM's compatibility tree contains registered Windows directory links.
    # Reject traversal in the manifest, but allow these read-only source links;
    # the actual contents must also match the registered source checksum.
    if logical.is_absolute() or logical.drive or '..' in logical.parts:
        raise ValueError('source path escapes data root')
    return root / logical


def materialize(data_root: Path, source_manifest: Path, output_root: Path, *,
                expected_counts: tuple[int, int] = (1744, 58),
                duration_reader: Callable[[Path], float] = _duration,
                progress: Callable[[str], None] | None = None) -> dict:
    data_root, output_root = Path(data_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError(f'immutable output already exists: {output_root}')
    source = json.loads(Path(source_manifest).read_text(encoding='utf-8'))
    originals = sorted(source['records'], key=lambda r: int(r['logical_song_key']))
    keys = [r['logical_song_key'] for r in originals]
    actual = (sum(int(k) <= 2000 for k in keys), sum(int(k) > 2000 for k in keys))
    if actual != expected_counts or len(set(keys)) != len(keys):
        raise ValueError(f'population mismatch: {actual}, expected {expected_counts}')
    dataset_id = stable_id('dataset', ('DEAM', VERSION, 'ORDINARY_DMER_FULL_POPULATION'))
    records, targets = [], []
    short_keys, missing_worker, restored = [], [], []
    invalid_values = missing_target_points = uncovered_points = 0
    raw_files = {}
    for i, old in enumerate(originals, 1):
        key = old['logical_song_key']
        kind = 'SHORT_CLIP' if int(key) <= 2000 else 'FULL_SONG'
        song_id = stable_id('song', ('DEAM', 'source-song', key))
        sample_id = stable_id('sample', (dataset_id, song_id, 'full-source-timeline'))
        audio = old['source_audio']
        path = _source_path(data_root, audio['source_relative_path'])
        checksum = file_sha256(path)
        if checksum != audio['source_sha256']:
            raise ValueError(f'raw audio changed: {key}')
        duration = duration_reader(path)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'invalid audio duration: {key}')
        if kind == 'SHORT_CLIP' and duration < 45:
            short_keys.append(key)
        raw_files[audio['source_relative_path']] = checksum
        views, sources = {}, {}
        for view in ('arousal', 'valence'):
            ref = old['annotation_views'][view]
            p = _source_path(data_root, ref['source_relative_path'])
            digest = file_sha256(p)
            if digest != ref['source_sha256']:
                raise ValueError(f'raw annotation changed: {key}/{view}')
            v = read_mean_annotations(p)
            v['audio_timestamp_covered_mask'] = [ms < duration * 1000 for ms in v['time_ms']]
            v['supervision_mask'] = [valid and covered for valid, covered in
                                     zip(v['valid_mask'], v['audio_timestamp_covered_mask'])]
            invalid_values += sum(v['invalid_rater_count'])
            missing_target_points += sum(not valid for valid in v['valid_mask'])
            uncovered_points += sum(not valid for valid in v['audio_timestamp_covered_mask'])
            views[view] = v
            sources[view] = {'source_relative_path': ref['source_relative_path'], 'source_sha256': digest}
            raw_files[ref['source_relative_path']] = digest
        if not all(v['worker_id_present'] for v in views.values()):
            missing_worker.append(key)
        if old.get('status') == 'EXCLUDED':
            restored.append(key)
        identity = {'dataset_id': dataset_id, 'song_id': song_id, 'sample_id': sample_id,
                    'logical_song_key': key, 'population_kind': kind}
        records.append(identity | {'status': 'RETAINED', 'split': 'UNASSIGNED',
                       'legacy_song_id': old.get('song_id'), 'legacy_status': old.get('status'),
                       'audio_duration_seconds': duration, 'source_audio': {
                           'source_relative_path': audio['source_relative_path'], 'source_sha256': checksum,
                           'source_size_bytes': path.stat().st_size}, 'annotation_sources': sources,
                       'audio_processing': 'REFERENCE_FULL_ORIGINAL_NO_TRIM_NO_PADDING',
                       'shorter_than_nominal_45s': kind == 'SHORT_CLIP' and duration < 45,
                       'worker_identity_required': False})
        targets.append(identity | {'annotation_start_ms': 15000, 'views': views})
        if progress and (i % 200 == 0 or i == len(originals)):
            progress(f'verified {i}/{len(originals)} songs')
    population = {'short_clips': actual[0], 'full_songs': actual[1], 'total': len(records), 'excluded': 0}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{VERSION}-staging-', dir=output_root.parent))
    target_path = stage / 'dmer_targets.jsonl'
    target_path.write_text(''.join(canonical_json(t) + '\n' for t in targets), encoding='utf-8')
    manifest = {'schema_version': '1.0', 'manifest_id': MANIFEST_ID, 'dataset_id': dataset_id,
                'dataset_version': VERSION, 'task': 'DMER', 'population': population,
                'split_policy': 'UNASSIGNED_PENDING_USER_DECISION', 'formal_training_ready': False,
                'readiness_blockers': ['SPLIT_PROTOCOL_UNCONFIRMED', 'NEW_FEATURE_BINDING_REQUIRED'],
                'historical_manifest_file_sha256': file_sha256(Path(source_manifest)),
                'processing_code_sha256': file_sha256(Path(__file__)),
                'label_policy': 'ORIGINAL_TIMESTAMPS_GE_15000MS_FINITE_RATER_MEAN',
                'audio_policy': 'ALL_SONGS_RETAINED_FULL_ORIGINAL_DURATION_NO_FEATURES_GENERATED',
                'feature_caches_bound': [], 'records': records,
                'artifacts': {'dmer_targets': {'logical_path': f'processed/{VERSION}/dmer_targets.jsonl',
                    'record_count': len(targets), 'file_sha256': file_sha256(target_path)}}}
    manifest['manifest_sha256'] = sha256_canonical(manifest)
    audit = {'schema_version': '1.0', 'audit_id': 'DEAM-DMER-FULL-REBUILD-v1',
             'verdict': 'PASS_DATA_REBUILD_ONLY', 'manifest_sha256': manifest['manifest_sha256'],
             'population': population, 'restored_song_count': len(restored), 'restored_song_keys': restored,
             'missing_worker_id_retained_count': len(missing_worker),
             'missing_worker_id_retained_keys': missing_worker, 'short_audio_retained_keys': short_keys,
             'invalid_annotation_value_count': invalid_values,
             'target_points_without_finite_raters': missing_target_points,
             'target_timestamps_outside_audio_count': uncovered_points,
             'all_source_checksums_match_historical': True, 'verified_raw_file_count': len(raw_files),
             'raw_file_inventory_sha256': sha256_canonical(raw_files),
             'new_split_created': False, 'audio_written': False, 'features_generated': False,
             'training_started': False, 'old_artifacts_modified': False}
    audit['audit_sha256'] = sha256_canonical(audit)
    (stage / 'manifest.json').write_text(canonical_json(manifest) + '\n', encoding='utf-8')
    (stage / 'audit.json').write_text(canonical_json(audit) + '\n', encoding='utf-8')
    if output_root.exists():
        raise FileExistsError(f'output appeared during build: {output_root}')
    stage.rename(output_root)
    return manifest
