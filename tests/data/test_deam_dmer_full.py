import csv
import json
from pathlib import Path

import pytest

from sc_mv_dmer.data import deam_dmer_full as full


def write_annotations(path, worker=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['sample_14500ms', 'sample_15000ms', 'sample_15500ms']
        w.writerow((['WorkerId'] if worker else []) + header)
        w.writerow((['person'] if worker else []) + [99, 0.2, 0.4])
        w.writerow((['person'] if worker else []) + [99, 0.6, 0.8])


def test_no_worker_identity_required_and_duplicate_rows_preserved(tmp_path):
    for worker in (False, True):
        p = tmp_path / f'{worker}.csv'
        write_annotations(p, worker)
        v = full.read_mean_annotations(p)
        assert v['time_ms'] == [15000, 15500]
        assert v['values'] == pytest.approx([0.4, 0.6])
        assert v['valid_mask'] == [True, True]
        assert v['valid_rater_count'] == [2, 2]


def test_invalid_values_masked_without_fabrication_or_song_deletion(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_text('sample_15000ms,sample_15500ms\nnan,0.2\n,0.6\n', encoding='utf-8')
    v = full.read_mean_annotations(p)
    assert v['values'] == [None, 0.4]
    assert v['valid_mask'] == [False, True]
    assert v['valid_rater_count'] == [0, 2]


def test_retain_short_audio_and_full_long_song_without_assigning_split(tmp_path):
    root = tmp_path / 'raw'
    records = []
    for key, duration, worker in [('2', 45.0, False), ('1174', 44.94, True), ('2001', 240.0, True)]:
        audio = root / f'audio/{key}.mp3'
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b'fixture')
        views = {}
        for view in ('arousal', 'valence'):
            rel = f'annotations/{view}/{key}.csv'
            write_annotations(root / rel, worker)
            views[view] = {'source_relative_path': rel, 'source_sha256': full.file_sha256(root / rel)}
        records.append({'logical_song_key': key, 'status': 'EXCLUDED', 'song_id': f'old-{key}',
                        'source_audio': {'source_relative_path': f'audio/{key}.mp3', 'source_sha256': full.file_sha256(audio)},
                        'annotation_views': views})
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'records': records}), encoding='utf-8')
    durations = {'2': 45., '1174': 44.94, '2001': 240.}
    output = tmp_path / 'result'
    result = full.materialize(root, source, output, expected_counts=(2, 1),
                              duration_reader=lambda p: durations[p.stem])
    assert result['population'] == {'short_clips': 2, 'full_songs': 1, 'total': 3, 'excluded': 0}
    assert result['split_policy'] == 'UNASSIGNED_PENDING_USER_DECISION'
    assert len(result['records']) == 3
    assert all(r['status'] == 'RETAINED' for r in result['records'])
    assert result['records'][1]['audio_duration_seconds'] == 44.94
    assert result['records'][2]['audio_duration_seconds'] == 240.0
    assert result['formal_training_ready'] is False
    before = (output / 'dmer_targets.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        full.materialize(root, source, output, expected_counts=(2, 1), duration_reader=lambda p: durations[p.stem])
    assert (output / 'dmer_targets.jsonl').read_bytes() == before


def test_wrong_population_fails_without_publishing(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('{"records": []}', encoding='utf-8')
    output = tmp_path / 'result'
    with pytest.raises(ValueError, match='population'):
        full.materialize(tmp_path, source, output)
    assert not output.exists()


def test_registered_relative_source_can_use_directory_link(tmp_path):
    root, storage = tmp_path / 'raw', tmp_path / 'storage'
    root.mkdir()
    storage.mkdir()
    (storage / 'audio.mp3').write_bytes(b'original')
    try:
        (root / 'audio').symlink_to(storage, target_is_directory=True)
    except OSError:
        pytest.skip('directory symlinks unavailable')
    assert full._source_path(root, 'audio/audio.mp3').read_bytes() == b'original'
    with pytest.raises(ValueError):
        full._source_path(root, '../storage/audio.mp3')
